#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import datetime
import json
import logging
import os
import re
from collections import defaultdict
import json_repair
from bs4 import BeautifulSoup

from api.db import LLMType
from api.db.services.document_service import DocumentService
from api.db.services.llm_service import TenantLLMService, LLMBundle
from api.utils.file_utils import get_project_base_directory
from rag.settings import TAG_FLD
from rag.utils import num_tokens_from_string, encoder


def chunks_format(reference):
    def get_value(d, k1, k2):
        return d.get(k1, d.get(k2))

    return [
        {
            "id": get_value(chunk, "chunk_id", "id"),
            "content": get_value(chunk, "content", "content_with_weight"),
            "document_id": get_value(chunk, "doc_id", "document_id"),
            "document_name": get_value(chunk, "docnm_kwd", "document_name"),
            "dataset_id": get_value(chunk, "kb_id", "dataset_id"),
            "image_id": get_value(chunk, "image_id", "img_id"),
            "positions": get_value(chunk, "positions", "position_int"),
            "url": chunk.get("url"),
            "similarity": chunk.get("similarity"),
            "vector_similarity": chunk.get("vector_similarity"),
            "term_similarity": chunk.get("term_similarity"),
            "doc_type": chunk.get("doc_type_kwd"),
        }
        for chunk in reference.get("chunks", [])
    ]


def llm_id2llm_type(llm_id):
    llm_id, _ = TenantLLMService.split_model_name_and_factory(llm_id)
    fnm = os.path.join(get_project_base_directory(), "conf")
    llm_factories = json.load(open(os.path.join(fnm, "llm_factories.json"), "r"))
    for llm_factory in llm_factories["factory_llm_infos"]:
        for llm in llm_factory["llm"]:
            if llm_id == llm["llm_name"]:
                return llm["model_type"].strip(",")[-1]


def message_fit_in(msg, max_length=4000):
    """
    메시지 목록을 조정하여 총 토큰 수가 max_length 제한을 초과하지 않도록 합니다

    매개변수:
        msg: 메시지 목록, 각 요소는 role과 content를 포함하는 딕셔너리입니다
        max_length: 최대 토큰 수 제한, 기본값 4000

    반환:
        tuple: (실제 토큰 수, 조정된 메시지 목록)
    """

    def count():
        """현재 메시지 목록의 총 토큰 수를 계산합니다"""
        nonlocal msg
        tks_cnts = []
        for m in msg:
            tks_cnts.append({"role": m["role"], "count": num_tokens_from_string(m["content"])})
        total = 0
        for m in tks_cnts:
            total += m["count"]
        return total

    c = count()
    # 제한을 초과하지 않으면 직접 반환합니다
    if c < max_length:
        return c, msg

    # 첫 번째 축소: 시스템 메시지와 마지막 메시지를 보존합니다
    msg_ = [m for m in msg if m["role"] == "system"]
    if len(msg) > 1:
        msg_.append(msg[-1])
    msg = msg_
    c = count()
    if c < max_length:
        return c, msg

    # 시스템 메시지와 마지막 메시지의 토큰 수를 계산합니다
    ll = num_tokens_from_string(msg_[0]["content"])
    ll2 = num_tokens_from_string(msg_[-1]["content"])
    # 시스템 메시지 비율이 80%를 초과하면 시스템 메시지를 자릅니다
    if ll / (ll + ll2) > 0.8:
        m = msg_[0]["content"]
        m = encoder.decode(encoder.encode(m)[: max_length - ll2])
        msg[0]["content"] = m
        return max_length, msg

    # 그렇지 않으면 마지막 메시지를 자릅니다
    m = msg_[-1]["content"]
    m = encoder.decode(encoder.encode(m)[: max_length - ll2])
    msg[-1]["content"] = m
    return max_length, msg

def html_table_to_markdown(html):
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if not table:
        return html
    # <table> 이전/이후 텍스트 추출
    before = ""
    after = ""
    # 텍스트 노드/태그 모두 고려
    for elem in table.previous_siblings:
        if hasattr(elem, 'get_text'):
            before = elem.get_text(strip=True) + before
        elif isinstance(elem, str):
            before = elem.strip() + before
    for elem in table.next_siblings:
        if hasattr(elem, 'get_text'):
            after += elem.get_text(strip=True)
        elif isinstance(elem, str):
            after += elem.strip()
    # 마크다운 변환
    rows = table.find_all("tr")
    md_rows = []
    for i, row in enumerate(rows):
        cols = [col.get_text(strip=True) for col in row.find_all(["td", "th"])]
        md_rows.append("| " + " | ".join(cols) + " |")
        if i == 0:
            md_rows.append("|"+"|".join([" --- "]*len(cols))+"|")
    # 앞/뒤 텍스트와 합쳐 반환
    result = (before.strip() + "\n" if before.strip() else "") + "\n".join(md_rows) + ("\n" + after.strip() if after.strip() else "")
    return result

def kb_prompt(kbinfos, max_tokens):
    """
    검색된 지식 베이스 내용을 대규모 언어 모델에 적합한 프롬프트로 형식화합니다

    매개변수:
        kbinfos (dict): 검색 결과, chunks 등의 정보를 포함합니다
        max_tokens (int): 모델의 최대 토큰 제한

    프로세스:
        1. 검색된 모든 문서 조각 내용을 추출합니다
        2. 토큰 수를 계산하여 모델 제한을 초과하지 않도록 합니다
        3. 문서 메타데이터를 가져옵니다
        4. 문서 이름별로 문서 조각을 구성합니다
        5. 구조화된 프롬프트로 형식화합니다

    반환:
        list: 형식화된 지식 베이스 내용 목록, 각 요소는 한 문서의 관련 정보입니다
    """
    knowledges = [ck["content_with_weight"] for ck in kbinfos["chunks"]]
    used_token_count = 0
    chunks_num = 0
    for i, c in enumerate(knowledges):
        used_token_count += num_tokens_from_string(c)
        chunks_num += 1
        if max_tokens * 0.97 < used_token_count:
            knowledges = knowledges[:i]
            logging.warning(f"Not all the retrieval into prompt: {i + 1}/{len(knowledges)}")
            break

    docs = DocumentService.get_by_ids([ck["doc_id"] for ck in kbinfos["chunks"][:chunks_num]])
    docs = {d.id: d.meta_fields for d in docs}

    doc2chunks = defaultdict(lambda: {"chunks": [], "meta": []})
    # for i, ck in enumerate(kbinfos["chunks"][:chunks_num]):
    #     cnt = f"---\nID: {i}\n" + (f"URL: {ck['url']}\n" if "url" in ck else "")
    #     cnt += ck["content_with_weight"]
    #     doc2chunks[ck["docnm_kwd"]]["chunks"].append(cnt)
    #     doc2chunks[ck["docnm_kwd"]]["meta"] = docs.get(ck["doc_id"], {})
    for i, ck in enumerate(kbinfos["chunks"][:chunks_num]):
        chunk_content = ck["content_with_weight"]
        # 테이블이 포함된 경우 마크다운 변환
        if "<table" in chunk_content:
            logging.debug(f"Original chunk ID {ck['chunk_id']} content: {chunk_content}")
            chunk_content = html_table_to_markdown(chunk_content)
            logging.debug(f"Converted HTML table to Markdown in chunk ID {ck['chunk_id']}: {chunk_content}")
        cnt = f"---\nID: {i}\n" + (f"URL: {ck['url']}\n" if "url" in ck else "")
        cnt += chunk_content
        doc2chunks[ck["docnm_kwd"]]["chunks"].append(cnt)
        doc2chunks[ck["docnm_kwd"]]["meta"] = docs.get(ck["doc_id"], {})

    knowledges = []
    for nm, cks_meta in doc2chunks.items():
        txt = f"\n문서: {nm} \n"
        for k, v in cks_meta["meta"].items():
            txt += f"{k}: {v}\n"
        txt += "관련 문서조각은 다음과 같습니다:\n"
        for i, chunk in enumerate(cks_meta["chunks"], 1):
            txt += f"{chunk}\n"
        knowledges.append(txt)
    return knowledges


def citation_prompt():
    return """
# 인용 요구사항:
- '##i$$ ##j$$' 형식으로 인용을 삽입하세요. 여기서 i, j는 인용된 내용의 ID이며 '##'과 '$$'로 묶습니다.
- 문장 끝에 인용을 삽입하고, 각 문장에는 최대 4개의 인용을 사용할 수 있습니다.
- 답변 내용이 검색된 텍스트 블록에서 온 것이 아니라면 인용을 삽입하지 마세요.
- 독립적인 문서 ID(예: `#ID#`)를 사용하지 마세요.
- 어떠한 경우에도 다른 인용 스타일이나 형식(예: `~~i==`, `[i]`, `(i)` 등)을 사용해서는 안 됩니다.
- 인용은 항상 `##i$$` 형식을 사용해야 합니다.
- 형식 오류, 금지된 스타일 사용 또는 지원되지 않는 인용 사용을 포함하되 이에 국한되지 않는 위 규칙을 준수하지 않는 모든 경우는 오류로 간주되며 해당 문장에 대한 인용 추가를 건너뛰어야 합니다.

--- 예시 ---
<SYSTEM>: 다음은 지식 베이스입니다:

문서: 일론 머스크, 암호화폐에 대한 침묵을 깨고 도지코인에 올인하지 말라고 경고하다 ...
URL: https://blockworks.co/news/elon-musk-crypto-dogecoin
ID: 0
테슬라 공동 창업자는 도지코인에 올인하지 말라고 조언했지만, 일론 머스크는 그것이 여전히 자신이 가장 좋아하는 암호화폐라고 말했습니다...

문서: 도지코인에 대한 일론 머스크의 트윗이 소셜 미디어 열풍을 일으키다
ID: 1
머스크는 D.O.G.E. 즉, 도지코인의 약자인 '기꺼이 봉사하겠다'고 밝혔습니다.

문서: 일론 머스크의 트윗이 도지코인 가격에 미치는 인과적 영향
ID: 2
밈 기반 암호화폐인 도지코인을 생각하면 일론 머스크를 빼놓을 수 없습니다...

문서: 일론 머스크의 트윗, 공공 서비스 분야에서 도지코인의 미래 전망에 불을 붙이다
ID: 3
일론 머스크의 도지코인 발표 이후 시장이 뜨거워지고 있습니다. 이것이 암호화폐의 새로운 시대를 의미할까요?...

    이상은 지식 베이스 정보입니다.

<USER>: 일론 머스크는 도지코인에 대해 어떻게 생각하나요?

<ASSISTANT>: 머스크는 도지코인에 대한 애정을 꾸준히 표현하며, 유머 감각과 브랜드의 개 요소를 자주 언급합니다. 그는 이것이 자신이 가장 좋아하는 암호화폐라고 말한 적이 있습니다 ##0$$ ##1$$。
최근 머스크는 도지코인이 미래에 새로운 사용 사례를 가질 수 있음을 암시했습니다. 그의 트윗은 도지코인이 공공 서비스에 통합될 수 있다는 추측을 불러일으켰습니다 ##3$$。
전반적으로 머스크는 도지코인을 좋아하고 자주 홍보하지만, 과도한 투자는 경고하며 투기적 성격에 대한 애정과 신중함을 동시에 보여줍니다.

--- 예시 종료 ---

"""


def keyword_extraction(chat_mdl, content, topn=3):
    prompt = f"""
역할: 텍스트 분석가
작업: 주어진 텍스트 내용에서 가장 중요한 키워드/구를 추출합니다
요구사항:
- 텍스트 내용을 요약하고, 상위 {topn}개의 중요한 키워드/구를 제시합니다
- 키워드는 반드시 원문 언어를 사용해야 합니다
- 키워드는 영어 쉼표로 구분합니다
- 키워드만 출력합니다

### 텍스트 내용
{content}
"""
    msg = [{"role": "system", "content": prompt}, {"role": "user", "content": "Output: "}]
    _, msg = message_fit_in(msg, chat_mdl.max_length)
    kwd = chat_mdl.chat(prompt, msg[1:], {"temperature": 0.2})
    if isinstance(kwd, tuple):
        kwd = kwd[0]
    kwd = re.sub(r"<think>.*</think>", "", kwd, flags=re.DOTALL)
    if kwd.find("**ERROR**") >= 0:
        return ""
    return kwd


def question_proposal(chat_mdl, content, topn=3):
    prompt = f"""
Role: You're a text analyzer. 
Task:  propose {topn} questions about a given piece of text content.
Requirements: 
  - Understand and summarize the text content, and propose top {topn} important questions.
  - The questions SHOULD NOT have overlapping meanings.
  - The questions SHOULD cover the main content of the text as much as possible.
  - The questions MUST be in language of the given piece of text content.
  - One question per line.
  - Question ONLY in output.

### Text Content 
{content}

"""
    msg = [{"role": "system", "content": prompt}, {"role": "user", "content": "Output: "}]
    _, msg = message_fit_in(msg, chat_mdl.max_length)
    kwd = chat_mdl.chat(prompt, msg[1:], {"temperature": 0.2})
    if isinstance(kwd, tuple):
        kwd = kwd[0]
    kwd = re.sub(r"<think>.*</think>", "", kwd, flags=re.DOTALL)
    if kwd.find("**ERROR**") >= 0:
        return ""
    return kwd


def full_question(tenant_id, llm_id, messages, language=None):
    if llm_id2llm_type(llm_id) == "image2text":
        chat_mdl = LLMBundle(tenant_id, LLMType.IMAGE2TEXT, llm_id)
    else:
        chat_mdl = LLMBundle(tenant_id, LLMType.CHAT, llm_id)
    conv = []
    for m in messages:
        if m["role"] not in ["user", "assistant"]:
            continue
        conv.append("{}: {}".format(m["role"].upper(), m["content"]))
    conv = "\n".join(conv)
    today = datetime.date.today().isoformat()
    yesterday = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
    tomorrow = (datetime.date.today() + datetime.timedelta(days=1)).isoformat()
    prompt = f"""
Role: A helpful assistant

Task and steps: 
    1. Generate a full user question that would follow the conversation.
    2. If the user's question involves relative date, you need to convert it into absolute date based on the current date, which is {today}. For example: 'yesterday' would be converted to {yesterday}.

Requirements & Restrictions:
  - If the user's latest question is completely, don't do anything, just return the original question.
  - DON'T generate anything except a refined question."""
    if language:
        prompt += f"""
  - Text generated MUST be in {language}."""
    else:
        prompt += """
  - Text generated MUST be in the same language of the original user's question.
"""
    prompt += f"""

######################
-Examples-
######################

# Example 1
## Conversation
USER: What is the name of Donald Trump's father?
ASSISTANT:  Fred Trump.
USER: And his mother?
###############
Output: What's the name of Donald Trump's mother?

------------
# Example 2
## Conversation
USER: What is the name of Donald Trump's father?
ASSISTANT:  Fred Trump.
USER: And his mother?
ASSISTANT:  Mary Trump.
User: What's her full name?
###############
Output: What's the full name of Donald Trump's mother Mary Trump?

------------
# Example 3
## Conversation
USER: What's the weather today in London?
ASSISTANT:  Cloudy.
USER: What's about tomorrow in Rochester?
###############
Output: What's the weather in Rochester on {tomorrow}?

######################
# Real Data
## Conversation
{conv}
###############
    """
    ans = chat_mdl.chat(prompt, [{"role": "user", "content": "Output: "}], {"temperature": 0.2})
    ans = re.sub(r"<think>.*</think>", "", ans, flags=re.DOTALL)
    return ans if ans.find("**ERROR**") < 0 else messages[-1]["content"]


def content_tagging(chat_mdl, content, all_tags, examples, topn=3):
    prompt = f"""
역할: 당신은 텍스트 분석가입니다.

작업: 예시와 전체 태그 세트를 기반으로 주어진 텍스트 콘텐츠에 태그를 지정합니다(일부 레이블을 지정).

단계:
  - 태그/레이블 세트를 이해합니다.
  - 텍스트 콘텐츠와 할당된 태그 및 관련성 점수가 JSON 형식으로 구성된 모든 예시를 이해합니다.
  - 텍스트 콘텐츠를 요약하고, 태그/레이블 세트에서 가장 관련성이 높은 상위 {topn}개의 태그와 해당 관련성 점수로 태그를 지정합니다.

요구사항
  - 태그는 태그 세트에서 가져와야 합니다.
  - 출력은 JSON 형식이어야만 하며, 키는 태그이고 값은 관련성 점수입니다.
  - 관련성 점수는 1에서 10 사이여야 합니다.
  - 출력에는 키워드만 포함됩니다.

# 태그 세트
{", ".join(all_tags)}

"""
    for i, ex in enumerate(examples):
        prompt += """
# 예시 {}
### 텍스트 내용
{}

출력:
{}

        """.format(i, ex["content"], json.dumps(ex[TAG_FLD], indent=2, ensure_ascii=False))

    prompt += f"""
# 실제 데이터
### 텍스트 내용
{content}

"""
    msg = [{"role": "system", "content": prompt}, {"role": "user", "content": "Output: "}]
    _, msg = message_fit_in(msg, chat_mdl.max_length)
    kwd = chat_mdl.chat(prompt, msg[1:], {"temperature": 0.5})
    if isinstance(kwd, tuple):
        kwd = kwd[0]
    kwd = re.sub(r"<think>.*</think>", "", kwd, flags=re.DOTALL)
    if kwd.find("**ERROR**") >= 0:
        raise Exception(kwd)

    try:
        return json_repair.loads(kwd)
    except json_repair.JSONDecodeError:
        try:
            result = kwd.replace(prompt[:-1], "").replace("user", "").replace("model", "").strip()
            result = "{" + result.split("{")[1].split("}")[0] + "}"
            return json_repair.loads(result)
        except Exception as e:
            logging.exception(f"JSON parsing error: {result} -> {e}")
            raise e


def vision_llm_describe_prompt(page=None) -> str:
    prompt_en = """
INSTRUCTION:
Transcribe the content from the provided PDF page image into clean Markdown format.
- Only output the content transcribed from the image.
- Do NOT output this instruction or any other explanation.
- If the content is missing or you do not understand the input, return an empty string.

RULES:
1. Do NOT generate examples, demonstrations, or templates.
2. Do NOT output any extra text such as 'Example', 'Example Output', or similar.
3. Do NOT generate any tables, headings, or content that is not explicitly present in the image.
4. Transcribe content word-for-word. Do NOT modify, translate, or omit any content.
5. Do NOT explain Markdown or mention that you are using Markdown.
6. Do NOT wrap the output in ```markdown or ``` blocks.
7. Only apply Markdown structure to headings, paragraphs, lists, and tables, strictly based on the layout of the image. Do NOT create tables unless an actual table exists in the image.
8. Preserve the original language, information, and order exactly as shown in the image.
"""

    if page is not None:
        prompt_en += f"\nAt the end of the transcription, add the page divider: `--- Page {page} ---`."

    prompt_en += """
FAILURE HANDLING:
- If you do not detect valid content in the image, return an empty string.
"""
    return prompt_en


def vision_llm_figure_describe_prompt() -> str:
    prompt = """
You are an expert visual data analyst. Analyze the image and provide a comprehensive description of its content. Focus on identifying the type of visual data representation (e.g., bar chart, pie chart, line graph, table, flowchart), its structure, and any text captions or labels included in the image.

Tasks:
1. Describe the overall structure of the visual representation. Specify if it is a chart, graph, table, or diagram.
2. Identify and extract any axes, legends, titles, or labels present in the image. Provide the exact text where available.
3. Extract the data points from the visual elements (e.g., bar heights, line graph coordinates, pie chart segments, table rows and columns).
4. Analyze and explain any trends, comparisons, or patterns shown in the data.
5. Capture any annotations, captions, or footnotes, and explain their relevance to the image.
6. Only include details that are explicitly present in the image. If an element (e.g., axis, legend, or caption) does not exist or is not visible, do not mention it.

Output format (include only sections relevant to the image content):
- Visual Type: [Type]
- Title: [Title text, if available]
- Axes / Legends / Labels: [Details, if available]
- Data Points: [Extracted data]
- Trends / Insights: [Analysis and interpretation]
- Captions / Annotations: [Text and relevance, if available]

Ensure high accuracy, clarity, and completeness in your analysis, and includes only the information present in the image. Avoid unnecessary statements about missing elements.
"""
    return prompt
