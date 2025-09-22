#  Copyright 2025 zstar1003. All Rights Reserved.
#  Project source code: https://github.com/zstar1003/ragflow-plus

import json
import os
import re
import shutil
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime
from io import StringIO
from urllib.parse import urlparse

import requests
from database import MINIO_CONFIG, get_es_client, get_minio_client
# mineru 패키지 사용
from mineru.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2, read_fn
from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json
from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
from mineru.utils.enum_class import MakeMode

from . import logger
from .excel_parser import parse_excel_file
from .rag_tokenizer import RagTokenizer
from .korean_tokenizer import KoreanTokenizer
from .utils import _create_task_record, _update_document_progress, _update_kb_chunk_count, generate_uuid, get_bbox_from_block
from bs4 import BeautifulSoup

tknzr = RagTokenizer()
korean_tokenizer = KoreanTokenizer()

# HTML 테이블을 마크다운 테이블로 변환하는 함수
def html_table_to_markdown(html):
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if not table:
        return html  # 테이블이 아니면 원본 반환

    rows = table.find_all("tr")
    md_rows = []
    for i, row in enumerate(rows):
        cols = [col.get_text(strip=True) for col in row.find_all(["td", "th"])]
        md_rows.append("| " + " | ".join(cols) + " |")
        if i == 0:  # 헤더 다음 구분선
            md_rows.append("|" + "|".join([" --- "]*len(cols)) + "|")
    return "\n".join(md_rows)

# HTML 태그 제거 함수 (정규식 사용)
def html_to_text(html):
    # <...> 형태의 태그를 모두 제거
    return re.sub(r'<[^>]+>', ' ', html)

def tokenize_text(text):
    """토크나이저로 텍스트를 토큰화합니다."""
    logger.info(f"[Parser-INFO] KoreanTokenizer 사용 전: {text[:100]}")
    # HTML 태그가 포함된 경우 태그 제거 (정규식 사용)
    if '<' in text and '>' in text:
        text = html_to_text(text)
        # text = tknzr.tokenize(text)
        # logger.info(f"[Parser-INFO] 기본Tokenizer 사용: {text[:100]}")
    tokens = korean_tokenizer.tokenize(text)
    if isinstance(tokens, list):
        text = ' '.join(tokens)
        logger.info(f"[Parser-INFO] KoreanTokenizer 사용: {text[:100]}")
        return text
    # logger.info(f"[Parser-INFO] 기본Tokenizer 사용 전: {text[:100]}")
    # logger.info(f"[Parser-INFO] 기본Tokenizer 사용: {text[:100]}")
    # return text

def merge_title_text_blocks(content_list, block_info_list, middle_json_blocks=None):
    """
    middle_json_blocks의 block_type을 사용해서 title 블록과 바로 다음 text 블록을 하나의 청크로 합치는 함수
    
    Args:
        content_list: MinerU pipeline_union_make 결과
        block_info_list: 블록 정보 리스트 (page_idx, bbox 포함)
        middle_json_blocks: middle_json에서 추출된 블록 타입 정보 리스트 [{"block_type": "title"}, {"block_type": "text"}, ...]
    
    Returns:
        merged_content_list: 병합된 콘텐츠 리스트
        merged_block_info_list: 병합된 블록 정보 리스트
    """
    merged_content_list = []
    merged_block_info_list = []
    skip_next = False
    title_count = 0
    text_count = 0
    merged_count = 0
    
    # middle_json_blocks가 있으면 block_type 정보를 사용, 없으면 기존 방식 사용
    use_middle_json = middle_json_blocks is not None and len(middle_json_blocks) == len(content_list)
    
    if use_middle_json:
        logger.info(f"[Parser-INFO] middle_json block_type 사용: {len(middle_json_blocks)} blocks")
    else:
        logger.info(f"[Parser-INFO] content_list type 사용 (middle_json 없음)")
    
    for i, chunk_data in enumerate(content_list):
        # 이전에 병합되어 건너뛸 블록인 경우
        if skip_next:
            skip_next = False
            continue
        
        # middle_json_blocks에서 block_type 정보 가져오기
        if use_middle_json and i < len(middle_json_blocks):
            current_middle_type = middle_json_blocks[i].get('block_type', '').lower()
            next_middle_type = middle_json_blocks[i + 1].get('block_type', '').lower() if i + 1 < len(middle_json_blocks) else ''
            
            # middle_json_blocks의 block_type을 직접 사용
            is_title_block = current_middle_type == 'title'
            is_next_text_block = next_middle_type == 'text' if i + 1 < len(middle_json_blocks) else False
        else:
            # 기존 방식: content_list의 type 사용
            title_types = ["title", "header", "heading", "h1", "h2", "h3", "h4", "h5", "h6"]
            text_types = ["text", "paragraph", "para"]
            
            current_type = chunk_data.get("type", "").lower()
            next_type = content_list[i + 1].get("type", "").lower() if i + 1 < len(content_list) else ""
            
            is_title_block = any(title_type in current_type for title_type in title_types)
            is_next_text_block = any(text_type in next_type for text_type in text_types) if i + 1 < len(content_list) else False
        
        # 블록 타입별 카운트
        if is_title_block:
            title_count += 1
        elif (use_middle_json and i < len(middle_json_blocks) and middle_json_blocks[i].get('block_type', '').lower() == 'text') or \
             (not use_middle_json and any(text_type in chunk_data.get("type", "").lower() for text_type in ["text", "paragraph", "para"])):
            text_count += 1
            
        # title 블록이고 다음 블록이 text인 경우 병합
        if (is_title_block and i + 1 < len(content_list) and is_next_text_block):
            
            title_chunk = chunk_data
            text_chunk = content_list[i + 1]
            
            # title과 text 내용 병합
            title_content = title_chunk.get("text", "").strip()
            text_content = text_chunk.get("text", "").strip()
            
            logger.info(f"[Parser-INFO] ✅ Title-Text 병합 발견: idx={i}")
            
            # 병합된 내용이 비어있지 않은 경우만 처리
            if title_content or text_content:
                merged_content = f"{title_content}\n{text_content}".strip()
                
                # 새로운 병합 블록 생성
                merged_chunk = {
                    "type": "text",  # 병합된 블록은 text 타입으로 설정
                    "text": merged_content
                }
                
                # bbox 정보 병합
                title_bbox = [0, 0, 0, 0]
                text_bbox = [0, 0, 0, 0]
                title_page = 0
                text_page = 0
                
                # title 블록의 정보 가져오기
                if i < len(block_info_list):
                    title_info = block_info_list[i]
                    title_page = title_info.get("page_idx", 0)
                    title_bbox = title_info.get("bbox", [0, 0, 0, 0])
                
                # text 블록의 정보 가져오기
                if i + 1 < len(block_info_list):
                    text_info = block_info_list[i + 1]
                    text_page = text_info.get("page_idx", 0)
                    text_bbox = text_info.get("bbox", [0, 0, 0, 0])
                
                # bbox 병합: 두 블록을 포함하는 최소 경계 상자 계산
                if title_bbox != [0, 0, 0, 0] and text_bbox != [0, 0, 0, 0]:
                    # 여러 페이지에 걸친 경우 첫 번째 페이지의 bbox 사용
                    if title_page == text_page:
                        # 같은 페이지: 두 bbox를 포함하는 경계 상자 계산
                        merged_bbox = [
                            min(title_bbox[0], text_bbox[0]),  # x1 최소값
                            min(title_bbox[1], text_bbox[1]),  # y1 최소값
                            max(title_bbox[2], text_bbox[2]),  # x2 최대값
                            max(title_bbox[3], text_bbox[3])   # y2 최대값
                        ]
                        merged_page = title_page
                    else:
                        # 다른 페이지: 첫 번째 블록(title)의 정보 사용
                        merged_bbox = title_bbox
                        merged_page = title_page
                else:
                    # bbox 정보가 없는 경우 title 블록 정보 사용
                    merged_bbox = title_bbox
                    merged_page = title_page
                
                # 병합된 블록 정보 생성
                merged_block_info = {
                    "page_idx": merged_page,
                    "bbox": merged_bbox
                }
                
                merged_content_list.append(merged_chunk)
                merged_block_info_list.append(merged_block_info)
                
                # 다음 텍스트 블록은 건너뛰기
                skip_next = True
                merged_count += 1
                
                logger.info(f"[Parser-INFO] ✅ Title과 Text 블록 병합 완료: page={merged_page}")
            else:
                # 내용이 비어있는 경우 원본 블록들을 개별적으로 추가
                merged_content_list.append(title_chunk)
                if i < len(block_info_list):
                    merged_block_info_list.append(block_info_list[i])
        else:
            # title이 아니거나 다음이 text가 아닌 경우 그대로 추가
            merged_content_list.append(chunk_data)
            if i < len(block_info_list):
                merged_block_info_list.append(block_info_list[i])
            else:
                # block_info_list가 부족한 경우 기본값 추가
                merged_block_info_list.append({"page_idx": 0, "bbox": [0, 0, 0, 0]})
    
    logger.info(f"[Parser-INFO] 블록 병합 통계: total={len(content_list)}, title={title_count}, text={text_count}, merged={merged_count}")
    logger.info(f"[Parser-INFO] ✅ 블록 병합 완료: {len(content_list)} -> {len(merged_content_list)} 블록")
    return merged_content_list, merged_block_info_list


@contextmanager
def capture_stdout_stderr(doc_id):
    """표준 출력과 표준 에러를 캡처하여 실시간으로 데이터베이스에 업데이트합니다."""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    # 문자열 버퍼 생성
    stdout_buffer = StringIO()
    stderr_buffer = StringIO()
    
    # 사용자 정의 출력 클래스로 실시간 캡처 및 진행률 업데이트
    class ProgressCapture:
        def __init__(self, original, buffer, doc_id):
            self.original = original
            self.buffer = buffer
            self.doc_id = doc_id
            self.last_update = time.time()
            # 표준 출력 스트림과 호환되도록 필요한 속성 추가
            self.encoding = getattr(original, 'encoding', 'utf-8')
            self.errors = getattr(original, 'errors', 'strict')
            self.mode = getattr(original, 'mode', 'w')
            
        def write(self, text):
            self.original.write(text)  # 기존 출력을 유지
            self.buffer.write(text)
            
            # 진행 정보가 포함되어 있는지 확인
            if any(keyword in text for keyword in ['Predict:', '%|', 'Processing pages:', 'OCR-', 'MFD', 'MFR', 'Table', 'it/s]', 'INFO']):
                # 텍스트 정리, ANSI 이스케이프 시퀀스 및 불필요한 공백 제거
                clean_text = re.sub(r'\x1b\[[0-9;]*m', '', text.strip())
                clean_text = re.sub(r'\s+', ' ', clean_text)  # 여러 공백을 하나로 합침
                
                if clean_text and len(clean_text) > 5:  # 너무 짧은 텍스트는 필터링
                    current_time = time.time()
                    # 업데이트 빈도 제한, 너무 자주 DB 작업 방지
                    if current_time - self.last_update > 0.3:  # 0.3초마다 한 번만 업데이트
                        try:
                            # 주요 정보 추출, 우선적으로 진행률 정보 표시
                            if '%|' in clean_text and ('Predict:' in clean_text or 'Processing' in clean_text):
                                # 진행률 정보, 바로 사용
                                _update_document_progress(self.doc_id, message=clean_text[:500])
                            elif 'INFO' in clean_text and any(x in clean_text for x in ['처리', '분석', '추출']):
                                # 중요한 처리 정보
                                _update_document_progress(self.doc_id, message=clean_text[:500])
                            else:
                                # 기타 정보도 업데이트, 우선순위는 낮음
                                _update_document_progress(self.doc_id, message=clean_text[:500])
                            self.last_update = current_time
                        except Exception as e:
                            logger.error(f"[Parser-ERROR] 진행 메시지 업데이트 실패: {e}")
            
        def flush(self):
            self.original.flush()
            
        def __getattr__(self, name):
            # 다른 속성들을 원본 출력 스트림으로 프록시
            return getattr(self.original, name)
    
    try:
        # 표준 출력과 오류 출력을 대체
        sys.stdout = ProgressCapture(old_stdout, stdout_buffer, doc_id)
        sys.stderr = ProgressCapture(old_stderr, stderr_buffer, doc_id)
        yield stdout_buffer, stderr_buffer
    finally:
        # 원본 출력 복원
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def perform_parse(doc_id, doc_info, file_info, embedding_config, kb_info):
    """
    문서 파싱의 핵심 로직을 수행합니다.

    Args:
        doc_id (str): 문서 ID.
        doc_info (dict): 문서 정보가 담긴 딕셔너리 (name, location, type, kb_id, parser_config, created_by).
        file_info (dict): 파일 정보가 담긴 딕셔너리 (parent_id/bucket_name).
        kb_info (dict): 지식베이스 정보가 담긴 딕셔너리 (created_by).

    Returns:
        dict: 파싱 결과가 담긴 딕셔너리 (success, chunk_count).
    """
    temp_pdf_path = None
    middle_json_data = None  # middle_json 데이터 초기화
    temp_image_dir = None
    start_time = time.time()

    middle_json_content = None  # 중간 JSON 내용 초기화
    image_info_list = []  # 이미지 정보 리스트

    # 기본값 처리
    embedding_model_name = embedding_config.get("llm_name") if embedding_config and embedding_config.get("llm_name") else "bge-m3"  # 기본 모델
    # 모델명 처리
    if embedding_model_name and "___" in embedding_model_name:
        embedding_model_name = embedding_model_name.split("___")[0]

    # 실리콘플로우 플랫폼의 특수 처리를 제거하고 원래 모델명을 유지
    # 아래 코드는 주석 처리하여 사용자가 설정한 실제 모델을 사용하도록 함
    # if embedding_model_name == "netease-youdao/bce-embedding-base_v1":
    #     embedding_model_name = "BAAI/bge-m3"

    embedding_api_base = embedding_config.get("api_base") if embedding_config and embedding_config.get("api_base") else "http://localhost:11434"  # 기본 API URL

    # API 기본 주소가 빈 문자열이면 실리콘플로우 API 주소로 설정
    if embedding_api_base == "":
        embedding_api_base = "https://api.siliconflow.cn/v1/embeddings"
        logger.info(f"[Parser-INFO] API 기본 주소가 비어 있어 실리콘플로우 API 주소로 설정됨: {embedding_api_base}")

    embedding_api_key = embedding_config.get("api_key") if embedding_config else None  # None 또는 빈 문자열일 수 있음

    # Embedding API URL 완성
    embedding_url = None  # 기본값 None
    if embedding_api_base:
        # embedding_api_base에 프로토콜이 포함되어 있는지 확인 (http:// 또는 https://)
        if not embedding_api_base.startswith(("http://", "https://")):
            embedding_api_base = "http://" + embedding_api_base

        # 끝의 슬래시 제거
        normalized_base_url = embedding_api_base.rstrip("/")

        # 요청 URL에 11434 포트가 있으면 ollama 모델로 간주, ollama 전용 API 사용
        is_ollama = "11434" in normalized_base_url
        if is_ollama:
            # Ollama 전용 엔드포인트
            embedding_url = normalized_base_url + "/api/embeddings"
        elif normalized_base_url.endswith("/v1"):
            embedding_url = normalized_base_url + "/embeddings"
        elif normalized_base_url.endswith("/embeddings"):
            embedding_url = normalized_base_url
        else:
            embedding_url = normalized_base_url + "/v1/embeddings"

    logger.info(f"[Parser-INFO] Embedding 설정 사용: URL='{embedding_url}', Model='{embedding_model_name}', Key={embedding_api_key}")

    try:
        kb_id = doc_info["kb_id"]
        file_location = doc_info["location"]
        # 파일 경로에서 원래 확장자 추출
        _, file_extension = os.path.splitext(file_location)
        file_type = doc_info["type"].lower()
        bucket_name = file_info["parent_id"]  # 파일이 저장된 버킷은 parent_id
        tenant_id = kb_info["created_by"]  # 지식베이스 생성자를 tenant_id로 사용

        # 진행 상황 업데이트 콜백 (내부 업데이트 함수 직접 호출)
        def update_progress(prog=None, msg=None):
            _update_document_progress(doc_id, progress=prog, message=msg)
            logger.info(f"[Parser-PROGRESS] Doc: {doc_id}, Progress: {prog}, Message: {msg}")


        # 1. MinIO에서 파일 내용 가져오기
        minio_client = get_minio_client()
        if not minio_client.bucket_exists(bucket_name):
            raise Exception(f"저장소 버킷이 존재하지 않습니다: {bucket_name}")

        update_progress(0.1, f"저장소에서 파일을 가져오는 중: {file_location}")
        response = minio_client.get_object(bucket_name, file_location)
        file_content = response.read()
        response.close()
        update_progress(0.2, "파일 가져오기 성공, 파싱 준비 중")


        # 2. 파일 유형에 따라 파서 선택
        content_list = []
        if file_type.endswith("pdf"):
            update_progress(0.3, "MinerU 파서 사용")

            # 임시 파일에 PDF 내용 저장
            temp_dir = tempfile.gettempdir()
            temp_pdf_path = os.path.join(temp_dir, f"{doc_id}.pdf")
            with open(temp_pdf_path, "wb") as f:
                f.write(file_content)

            # PDF 바이트를 mineru 호환 형식으로 변환
            pdf_bytes = open(temp_pdf_path, "rb").read()
            pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes)
            
            # 임시 출력 디렉토리 설정
            temp_image_dir = os.path.join(temp_dir, f"images_{doc_id}")
            os.makedirs(temp_image_dir, exist_ok=True)
            image_writer = FileBasedDataWriter(temp_image_dir)

            # MinerU로 처리, 상세 출력 캡처
            with capture_stdout_stderr(doc_id):
                # 언어 및 파싱 방법 설정 (기본값 사용)
                lang = "korean"  # 한국어 기본값, 필요에 따라 변경
                parse_method = "auto"  # 자동 감지, 필요에 따라 "txt" 또는 "ocr"로 변경 가능
        
                update_progress(0.4, f"{parse_method}로 PDF 처리 중, 구체적인 진행 상황은 컨테이너 로그 참조")
                # pipeline 백엔드로 문서 분석 수행
                # pipeline_doc_analyze는 여러 문서를 일괄 처리할 수 있으므로 리스트로 전달
                infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list = pipeline_doc_analyze(
                    [pdf_bytes],  # 문서 바이트 리스트
                    [lang],       # 언어 리스트
                    parse_method=parse_method,
                    formula_enable=False,
                    table_enable=True
                )
            
                update_progress(0.6, f"{parse_method} 결과 처리 중")
                # 첫 번째(유일한) 문서의 결과 가져오기
                model_list = infer_results[0]
                images_list = all_image_lists[0]
                pdf_doc = all_pdf_docs[0]
                _lang = lang_list[0]
                _ocr_enable = ocr_enabled_list[0]

                # 중간 JSON 생성
                middle_json = pipeline_result_to_middle_json(
                    model_list, 
                    images_list, 
                    pdf_doc, 
                    image_writer, 
                    _lang, 
                    _ocr_enable, 
                )
                middle_json_data = middle_json  # 병합 함수에서 사용할 데이터 할당
            
                update_progress(0.8, "내용 추출 중")
                # PDF 정보 접근
                pdf_info = middle_json["pdf_info"]
                # content_list 생성
                image_dir = os.path.basename(temp_image_dir)
                content_list = pipeline_union_make(pdf_info, MakeMode.CONTENT_LIST, image_dir)
                # table 블록을 마크다운으로 변환
                for chunk in content_list:
                    if chunk.get("type", "").lower() == "table" and "<table" in chunk.get("text", ""):
                        chunk["text"] = html_table_to_markdown(chunk["text"])
                # 중간 JSON 문자열 직접 가져오기
                middle_json_content = middle_json
                # 로깅
                logger.info(f"[Parser-INFO] 문서 처리 완료, 청크 수: {len(content_list)}")
                
                # 첫 몇 개 블록의 타입과 내용 샘플 로그
                for idx, chunk in enumerate(content_list[:10]):  # 처음 10개만 로그
                    chunk_type = chunk.get("type", "unknown")
                    chunk_text = chunk.get("text", "")[:100] if chunk.get("text") else ""
                    logger.info(f"[Parser-INFO] Block[{idx}]: type='{chunk_type}', text='{chunk_text}...'")
        
        elif file_type.endswith("word") or file_type.endswith("ppt") or file_type.endswith("txt") or file_type.endswith("md") or file_type.endswith("html"):
            update_progress(0.3, f"지원하지 않는 파일 유형: {file_type}")
            raise NotImplementedError(f"파일 유형 '{file_type}'에 대한 파서가 아직 구현되지 않았습니다. MinerU 2.1.0부터는 따로 PDF로 변환 후 처리필요")
        # 엑셀 파일은 별도로 처리
        elif file_type.endswith("excel"):
            update_progress(0.3, "MinerU 파서 사용")
            # 임시 파일에 내용 저장
            temp_dir = tempfile.gettempdir()
            temp_file_path = os.path.join(temp_dir, f"{doc_id}{file_extension}")
            with open(temp_file_path, "wb") as f:
                f.write(file_content)

            logger.info(f"[Parser-INFO] 임시 파일 경로: {temp_file_path}")

            update_progress(0.8, "내용 추출 중")
            # 내용 리스트 처리
            content_list = parse_excel_file(temp_file_path)

        elif file_type.endswith("visual"):
            update_progress(0.3, "MinerU 파서 사용")

            # 임시 파일에 내용 저장
            temp_dir = tempfile.gettempdir()
            temp_file_path = os.path.join(temp_dir, f"{doc_id}{file_extension}")
            with open(temp_file_path, "wb") as f:
                f.write(file_content)
            logger.info(f"[Parser-INFO] 임시 파일 경로: {temp_file_path}")

            # 이미지 바이트 읽기
            image_bytes = read_fn(temp_file_path)
            
            # 임시 출력 디렉토리 설정
            temp_image_dir = os.path.join(temp_dir, f"images_{doc_id}")
            os.makedirs(temp_image_dir, exist_ok=True)
            image_writer = FileBasedDataWriter(temp_image_dir)

            # MinerU로 처리, 상세 출력 캡처
            with capture_stdout_stderr(doc_id):
                # 언어 설정 (기본값)
                lang = "korean"  # 필요에 따라 변경
                
                update_progress(0.4, "이미지 분석 중 (OCR 처리)")
                # pipeline 백엔드로 이미지 분석 - OCR 처리 활성화
                infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list = pipeline_doc_analyze(
                    [image_bytes],  # 이미지 바이트 리스트
                    [lang],        # 언어 리스트
                    parse_method="ocr",  # 이미지 파일은 OCR 모드 사용
                    formula_enable=True,
                    table_enable=True
                )
                
                update_progress(0.6, "결과 처리 중")
                # 첫 번째(유일한) 이미지의 결과 가져오기
                model_list = infer_results[0]
                images_list = all_image_lists[0]
                pdf_doc = all_pdf_docs[0]
                _lang = lang_list[0]
                _ocr_enable = ocr_enabled_list[0]
            
                # 중간 JSON 생성
                middle_json = pipeline_result_to_middle_json(
                    model_list, 
                    images_list, 
                    pdf_doc, 
                    image_writer, 
                    _lang, 
                    _ocr_enable
                )
                middle_json_data = middle_json  # 병합 함수에서 사용할 데이터 할당
                
                update_progress(0.8, "내용 추출 중")
                # PDF 정보 접근
                pdf_info = middle_json["pdf_info"]
                # content_list 생성
                image_dir = os.path.basename(temp_image_dir)
                content_list = pipeline_union_make(pdf_info, MakeMode.CONTENT_LIST, image_dir)
                # 중간 JSON 직접 가져오기
                middle_json_content = middle_json
                
                # 로깅
                logger.info(f"[Parser-INFO] 이미지 처리 완료, 청크 수: {len(content_list)}")
        else:
            update_progress(0.3, f"지원하지 않는 파일 유형: {file_type}")
            raise NotImplementedError(f"파일 유형 '{file_type}'의 파서는 아직 구현되지 않았습니다.")


        # middle_json_content를 파싱하여 블록 정보 추출
        block_info_list = []
        middle_json_blocks = []  # merge_title_text_blocks 함수에서 사용할 블록 리스트
        if middle_json_content:
            if isinstance(middle_json_content, dict):
                middle_data = middle_json_content  # 바로 할당
            else:
                middle_data = None
                logger.warning(f"[Parser-WARNING] middle_json_content가 예상한 딕셔너리 형식이 아닙니다. 실제 타입: {type(middle_json_content)}.")
            try:
                # middle_json의 블록 타입 분석
                middle_block_types = {}
                total_middle_blocks = 0
                
                # 정보 추출
                for page_idx, page_data in enumerate(middle_data.get("pdf_info", [])):
                    page_blocks = page_data.get("preproc_blocks", [])
                    logger.info(f"[Parser-INFO] Page {page_idx}: {len(page_blocks)} blocks in middle_json")
                    
                    for block_idx, block in enumerate(page_blocks):
                        total_middle_blocks += 1
                        block_type = block.get("type", "unknown")
                        middle_block_types[block_type] = middle_block_types.get(block_type, 0) + 1
                        
                        # merge_title_text_blocks 함수에서 사용할 블록 정보 추가
                        middle_json_blocks.append({"block_type": block_type})
                        
                        block_bbox = get_bbox_from_block(block)
                        # 텍스트가 있고 bbox가 있는 블록만 추출
                        if block_bbox != [0, 0, 0, 0]:
                            block_info_list.append({"page_idx": page_idx, "bbox": block_bbox})
                        else:
                            logger.warning("[Parser-WARNING] 블록의 bbox 형식이 유효하지 않아 건너뜀.")

                    logger.info(f"[Parser-INFO] middle_data에서 {len(block_info_list)}개의 블록 정보를 추출함.")
                
                logger.info(f"[Parser-INFO] MiddleJSON 블록 타입 분포 (총 {total_middle_blocks}개): {middle_block_types}")

            except json.JSONDecodeError:
                logger.error("[Parser-ERROR] middle_json_content 파싱 실패.")
                raise Exception("[Parser-ERROR] middle_json_content 파싱 실패.")
            except Exception as e:
                logger.error(f"[Parser-ERROR] middle_json_content 처리 중 오류: {e}")
                raise Exception(f"[Parser-ERROR] middle_json_content 처리 중 오류: {e}")

            # Title과 Text 블록 병합 처리
            if content_list and block_info_list:
                try:
                    logger.info(f"[Parser-INFO] Title-Text 블록 병합 시작: {len(content_list)}개 블록")
                    # middle_json_blocks 사용 (middle_json_content에서 추출된 블록 타입 정보)
                    middle_blocks_for_merge = middle_json_blocks if 'middle_json_blocks' in locals() and middle_json_blocks else None
                    content_list, block_info_list = merge_title_text_blocks(content_list, block_info_list, middle_blocks_for_merge)
                    logger.info(f"[Parser-INFO] Title-Text 블록 병합 완료: {len(content_list)}개 블록")
                except Exception as e:
                    logger.warning(f"[Parser-WARNING] Title-Text 블록 병합 중 오류, 원본 사용: {e}")

        # 3. 파싱 결과 처리 (MinIO 업로드, ES 저장)
        update_progress(0.95, "파싱 결과 저장 중")
        es_client = get_es_client()
        # 참고: MinIO 버킷은 파일의 parent_id가 아니라 kb_id(지식베이스 ID)여야 함
        output_bucket = kb_id
        if not minio_client.bucket_exists(output_bucket):
            minio_client.make_bucket(output_bucket)
            logger.info(f"[Parser-INFO] MinIO 버킷 생성: {output_bucket}")

        # 임베딩 벡터 차원 구하기
        embedding_dim = None
        try:
            # 테스트 텍스트로 벡터 차원 먼저 구하기
            test_content = "test"
            headers = {"Content-Type": "application/json"}
            if embedding_api_key:
                headers["Authorization"] = f"Bearer {embedding_api_key}"

            is_ollama = "11434" in embedding_url if embedding_url else False
            if is_ollama:
                test_resp = requests.post(
                    embedding_url,
                    headers=headers,
                    json={"model": embedding_model_name, "prompt": test_content},
                    timeout=15,
                )
            else:
                test_resp = requests.post(
                    embedding_url,
                    headers=headers,
                    json={"model": embedding_model_name, "input": test_content},
                    timeout=15,
                )
            
            test_resp.raise_for_status()
            test_data = test_resp.json()
            
            if is_ollama:
                test_vec = test_data.get("embedding")
            else:
                test_vec = test_data["data"][0]["embedding"]
            
            embedding_dim = len(test_vec)
            logger.info(f"[Parser-INFO] 임베딩 차원 감지: {embedding_dim}")
            
        except Exception as e:
            logger.error(f"[Parser-ERROR] 임베딩 차원 구하기 실패: {e}")
            raise Exception(f"[Parser-ERROR] 임베딩 차원 구하기 실패: {e}")

        index_name = f"ragflow_{tenant_id}"
        vector_field_name = f"q_{embedding_dim}_vec"
        
        if not es_client.indices.exists(index=index_name):
            # 동적 차원으로 인덱스 생성
            es_client.indices.create(
                index=index_name,
                body={
                    "settings": {"number_of_replicas": 0},
                    "mappings": {
                        "properties": {
                            "doc_id": {"type": "keyword"}, 
                            "kb_id": {"type": "keyword"}, 
                            "content_with_weight": {"type": "text"}, 
                            vector_field_name: {"type": "dense_vector", "dims": embedding_dim}
                        }
                    },
                },
            )
            logger.info(f"[Parser-INFO] Elasticsearch 인덱스 생성: {index_name}, 벡터 차원: {embedding_dim}")
        else:
            # 기존 인덱스에 현재 차원의 벡터 필드가 있는지 확인
            try:
                mapping = es_client.indices.get_mapping(index=index_name)
                existing_properties = mapping[index_name]["mappings"]["properties"]
                
                if vector_field_name not in existing_properties:
                    # 새 벡터 필드 추가
                    es_client.indices.put_mapping(
                        index=index_name,
                        body={
                            "properties": {
                                vector_field_name: {"type": "dense_vector", "dims": embedding_dim}
                            }
                        }
                    )
                    logger.info(f"[Parser-INFO] 인덱스 {index_name}에 새 벡터 필드 추가: {vector_field_name}, 차원: {embedding_dim}")
            except Exception as e:
                logger.error(f"[Parser-ERROR] 인덱스 매핑 업데이트 실패: {e}")
                raise Exception(f"[Parser-ERROR] 인덱스 매핑 업데이트 실패: {e}")

        chunk_count = 0
        chunk_ids_list = []


        for chunk_idx, chunk_data in enumerate(content_list):
            page_idx = 0  # 기본 페이지 인덱스
            bbox = [0, 0, 0, 0]  # 기본 bbox

            # chunk_idx로 block_info_list에서 해당 블록 정보 직접 가져오기 시도
            if chunk_idx < len(block_info_list):
                block_info = block_info_list[chunk_idx]
                page_idx = block_info.get("page_idx", 0)
                bbox = block_info.get("bbox", [0, 0, 0, 0])
                # bbox가 유효하지 않으면 기본값으로 재설정 (필요시)
                if not (isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(n, (int, float)) for n in bbox)):
                    logger.info(f"[Parser-WARNING] Chunk {chunk_idx}의 bbox 형식이 유효하지 않아 기본값 사용: {bbox}")
                    bbox = [0, 0, 0, 0]
            else:
                # block_info_list 길이가 content_list보다 짧으면 경고 출력 (한 번만)
                if chunk_idx == len(block_info_list):
                    logger.warning(f"[Parser-WARNING] block_info_list 길이({len(block_info_list)})가 content_list 길이({len(content_list)})보다 짧음. 이후 블록은 기본 page_idx와 bbox 사용.")


            if chunk_data["type"] == "text" or chunk_data["type"] == "table" or chunk_data["type"] == "equation" or chunk_data["type"] == "title":
                if chunk_data["type"] == "text" or chunk_data["type"] == "title":
                    content = chunk_data["text"]
                    if not content or not content.strip():
                        continue
                    # 마크다운 특수문자 필터링
                    content = re.sub(r"[!#\\$/]", "", content)
                elif chunk_data["type"] == "equation":
                    content = chunk_data["text"]
                    if not content or not content.strip():
                        continue
                elif chunk_data["type"] == "table":
                    caption_list = chunk_data.get("table_caption", [])  # 리스트, 기본값 빈 리스트
                    table_body = chunk_data.get("table_body", "")  # 표 본문, 기본값 빈 문자열

                    # 표 본문이 비어 있으면 실제 내용 없음, 건너뜀
                    if not table_body.strip():
                        continue

                    # caption_list가 문자열 리스트인지 확인
                    if isinstance(caption_list, list) and all(isinstance(item, str) for item in caption_list):
                        # 리스트의 모든 문자열을 공백으로 연결
                        caption_str = " ".join(caption_list)
                    elif isinstance(caption_list, str):
                        # caption이 문자열이면 바로 사용
                        caption_str = caption_list
                    else:
                        # 기타(빈 리스트, None 등)면 빈 문자열 사용
                        caption_str = ""
                    # 처리된 캡션과 표 본문 연결
                    content = caption_str + table_body


                embedding_vec = []  # 빈 리스트로 초기화
                # 임베딩 벡터 구하기
                try:
                    headers = {"Content-Type": "application/json"}
                    if embedding_api_key:
                        headers["Authorization"] = f"Bearer {embedding_api_key}"

                    if is_ollama:
                        embedding_resp = requests.post(
                            embedding_url,  # 동적으로 생성된 URL 사용
                            headers=headers,  # headers 추가 (API Key 포함 가능)
                            json={
                                "model": embedding_model_name,  # 동적으로 가져오거나 기본 모델명 사용
                                "prompt": content,
                            },
                            timeout=15,  # 타임아웃 약간 증가
                        )
                    else:
                        embedding_resp = requests.post(
                            embedding_url,  # 동적으로 생성된 URL 사용
                            headers=headers,  # headers 추가 (API Key 포함 가능)
                            json={
                                "model": embedding_model_name,  # 동적으로 가져오거나 기본 모델명 사용
                                "input": content,
                            },
                            timeout=15,  # 타임아웃 약간 증가
                        )

                    embedding_resp.raise_for_status()
                    embedding_data = embedding_resp.json()

                    # ollama 임베딩 모델의 반환값은 별도 처리
                    if is_ollama:
                        embedding_vec = embedding_data.get("embedding")
                    else:
                        embedding_vec = embedding_data["data"][0]["embedding"]

                    # 벡터 차원이 예상과 다르면 오류
                    if len(embedding_vec) != embedding_dim:
                        error_msg = f"[Parser-ERROR] 임베딩 벡터 차원이 일치하지 않음, 예상: {embedding_dim}, 실제: {len(embedding_vec)}"
                        logger.error(error_msg)
                        update_progress(-5, error_msg)
                        raise ValueError(error_msg)
                    logger.info(f"[Parser-INFO] 임베딩 성공, 차원: {len(embedding_vec)}")
                except Exception as e:
                    logger.error(f"[Parser-ERROR] 임베딩 실패: {e}")
                    raise Exception(f"[Parser-ERROR] 임베딩 실패: {e}")

                chunk_id = generate_uuid()


                try:
                    # ES 문서 준비
                    current_time_es = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    current_timestamp_es = datetime.now().timestamp()

                    # 좌표 포맷 변환
                    x1, y1, x2, y2 = bbox
                    bbox_reordered = [x1, x2, y1, y2]

                    es_doc = {
                        "doc_id": doc_id,
                        "kb_id": kb_id,
                        "docnm_kwd": doc_info["name"],
                        "title_tks": tokenize_text(doc_info["name"]),
                        "title_sm_tks": tokenize_text(doc_info["name"]),
                        "content_with_weight": content,
                        "content_ltks": tokenize_text(content),
                        "content_sm_ltks": tokenize_text(content),
                        "page_num_int": [page_idx + 1],
                        "position_int": [[page_idx + 1] + bbox_reordered],  # 포맷: [[page, x1, x2, y1, y2]]
                        "top_int": [1],
                        "create_time": current_time_es,
                        "create_timestamp_flt": current_timestamp_es,
                        "img_id": "",
                        vector_field_name: embedding_vec,
                    }

                    # Elasticsearch에 저장
                    es_client.index(index=index_name, id=chunk_id, document=es_doc)  # document 파라미터 사용

                    chunk_count += 1
                    chunk_ids_list.append(chunk_id)

                except Exception as e:
                    logger.error(f"[Parser-ERROR] 텍스트 블록 {chunk_idx} (page: {page_idx}, bbox: {bbox}) 처리 실패: {e}")
                    raise Exception(f"[Parser-ERROR] 텍스트 블록 {chunk_idx} (page: {page_idx}, bbox: {bbox}) 처리 실패: {e}")


            elif chunk_data["type"] == "image":
                img_path_relative = chunk_data.get("img_path")
                if not img_path_relative or not temp_image_dir:
                    continue

                img_path_abs = os.path.join(temp_image_dir, os.path.basename(img_path_relative))
                if not os.path.exists(img_path_abs):
                    logger.warning(f"[Parser-WARNING] 이미지 파일이 존재하지 않음: {img_path_abs}")
                    continue

                img_id = generate_uuid()
                img_ext = os.path.splitext(img_path_abs)[1]
                img_key = f"images/{img_id}{img_ext}"  # MinIO 내 오브젝트명
                content_type = f"image/{img_ext[1:].lower()}"
                if content_type == "image/jpg":
                    content_type = "image/jpeg"

                try:
                    # MinIO에 이미지 업로드 (버킷은 kb_id)
                    minio_client.fput_object(bucket_name=output_bucket, object_name=img_key, file_path=img_path_abs, content_type=content_type)

                    # 이미지 공개 접근 권한 설정
                    policy = {"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Principal": {"AWS": "*"}, "Action": ["s3:GetObject"], "Resource": [f"arn:aws:s3:::{kb_id}/images/*"]}]}
                    minio_client.set_bucket_policy(kb_id, json.dumps(policy))

                    logger.info(f"이미지 업로드 성공: {img_key}")
                    minio_endpoint = MINIO_CONFIG["endpoint"]
                    use_ssl = MINIO_CONFIG.get("secure", False)
                    protocol = "https" if use_ssl else "http"
                    img_url = f"{protocol}://{minio_endpoint}/{output_bucket}/{img_key}"

                    # 이미지 정보 기록 (URL 및 위치)
                    image_info = {
                        "url": img_url,
                        "position": chunk_count,  # 현재 처리한 텍스트 블록 수를 위치로 사용
                    }
                    image_info_list.append(image_info)

                    logger.info(f"이미지 접근 링크: {img_url}")

                except Exception as e:
                    logger.error(f"[Parser-ERROR] 이미지 업로드 실패 {img_path_abs}: {e}")
                    raise Exception(f"[Parser-ERROR] 이미지 업로드 실패 {img_path_abs}: {e}")

        # 처리 요약 정보 출력
        logger.info(f"[Parser-INFO] 총 {chunk_count}개의 텍스트 블록 처리함.")

        # 4. 텍스트 블록의 이미지 정보 업데이트
        if image_info_list and chunk_ids_list:

            try:

                # 각 텍스트 블록에 가장 가까운 이미지를 찾음
                for i, chunk_id in enumerate(chunk_ids_list):
                    # 현재 텍스트 블록과 가장 가까운 이미지 찾기
                    nearest_image = None

                    for img_info in image_info_list:
                        # 텍스트 블록과 이미지의 "거리" 계산
                        distance = abs(i - img_info["position"])  # 위치 차이로 거리 측정
                        # 텍스트 블록과 이미지의 거리가 5 미만이면 관련 있다고 간주
                        if distance < 5:
                            nearest_image = img_info

                    # 가장 가까운 이미지가 있으면 텍스트 블록의 img_id 업데이트
                    if nearest_image:
                        # 상대 경로 부분 저장
                        parsed_url = urlparse(nearest_image["url"])
                        relative_path = parsed_url.path.lstrip("/")  # 앞의 슬래시 제거
                        # ES 문서 업데이트
                        direct_update = {"doc": {"img_id": relative_path}}
                        es_client.update(index=index_name, id=chunk_id, body=direct_update, refresh=True)
                        index_name = f"ragflow_{tenant_id}"
                        logger.info(f"[Parser-INFO] 텍스트 블록 {chunk_id}의 이미지 연결 업데이트: {relative_path}")

            except Exception as e:
                logger.error(f"[Parser-ERROR] 텍스트 블록 이미지 연결 업데이트 실패: {e}")
                raise Exception(f"[Parser-ERROR] 텍스트 블록 이미지 연결 업데이트 실패: {e}")


        # 5. 최종 상태 업데이트
        process_duration = time.time() - start_time
        _update_document_progress(doc_id, progress=1.0, message="파싱 완료", status="1", run="3", chunk_count=chunk_count, process_duration=process_duration)
        _update_kb_chunk_count(kb_id, chunk_count)  # 지식베이스 전체 블록 수 업데이트
        _create_task_record(doc_id, chunk_ids_list)  # task 기록 생성

        update_progress(1.0, "파싱 완료")
        logger.info(f"[Parser-INFO] 파싱 완료, 문서ID: {doc_id}, 소요시간: {process_duration:.2f}s, 블록 수: {chunk_count}")

        return {"success": True, "chunk_count": chunk_count}

    except Exception as e:
        process_duration = time.time() - start_time
        logger.error(f"[Parser-ERROR] 문서 {doc_id} 파싱 실패: {e}")
        error_message = f"파싱 실패: {e}"
        # 문서 상태를 실패로 업데이트
        _update_document_progress(doc_id, status="1", run="0", message=error_message, process_duration=process_duration)  # status=1은 완료, run=0은 실패
        return {"success": False, "error": error_message}

    finally:
        # 임시 파일 정리
        try:
            if temp_pdf_path and os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)
            if temp_image_dir and os.path.exists(temp_image_dir):
                shutil.rmtree(temp_image_dir, ignore_errors=True)
        except Exception as clean_e:
            logger.error(f"[Parser-WARNING] 임시 파일 정리 실패: {clean_e}")
