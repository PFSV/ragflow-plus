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
import binascii
import logging
import re
import time
from copy import deepcopy
from timeit import default_timer as timer

from api import settings
from api.db import LLMType, ParserType, StatusEnum
from api.db.db_models import DB, Dialog
from api.db.services.common_service import CommonService
from api.db.services.knowledgebase_service import KnowledgebaseService
from api.db.services.llm_service import LLMBundle, TenantLLMService
from rag.app.resume import forbidden_select_fields4resume
from rag.app.tag import label_question
from rag.nlp.search import index_name
from rag.prompts import chunks_format, citation_prompt, kb_prompt, keyword_extraction, llm_id2llm_type, message_fit_in
from rag.utils import num_tokens_from_string, rmSpace

from .database import MINIO_CONFIG


class DialogService(CommonService):
    model = Dialog

    @classmethod
    @DB.connection_context()
    def get_list(cls, tenant_id, page_number, items_per_page, orderby, desc, id, name):
        chats = cls.model.select()
        if id:
            chats = chats.where(cls.model.id == id)
        if name:
            chats = chats.where(cls.model.name == name)
        chats = chats.where((cls.model.tenant_id == tenant_id) & (cls.model.status == StatusEnum.VALID.value))
        if desc:
            chats = chats.order_by(cls.model.getter_by(orderby).desc())
        else:
            chats = chats.order_by(cls.model.getter_by(orderby).asc())

        chats = chats.paginate(page_number, items_per_page)

        return list(chats.dicts())


def chat_solo(dialog, messages, stream=True):
    if llm_id2llm_type(dialog.llm_id) == "image2text":
        chat_mdl = LLMBundle(dialog.tenant_id, LLMType.IMAGE2TEXT, dialog.llm_id)
    else:
        chat_mdl = LLMBundle(dialog.tenant_id, LLMType.CHAT, dialog.llm_id)

    prompt_config = dialog.prompt_config
    tts_mdl = None
    if prompt_config.get("tts"):
        tts_mdl = LLMBundle(dialog.tenant_id, LLMType.TTS)
    msg = [{"role": m["role"], "content": re.sub(r"##\d+\$\$", "", m["content"])} for m in messages if m["role"] != "system"]
    if stream:
        last_ans = ""
        for ans in chat_mdl.chat_streamly(prompt_config.get("system", ""), msg, dialog.llm_setting):
            answer = ans
            delta_ans = ans[len(last_ans) :]
            if num_tokens_from_string(delta_ans) < 16:
                continue
            last_ans = answer
            yield {"answer": answer, "reference": {}, "audio_binary": tts(tts_mdl, delta_ans), "prompt": "", "created_at": time.time()}
        if delta_ans:
            yield {"answer": answer, "reference": {}, "audio_binary": tts(tts_mdl, delta_ans), "prompt": "", "created_at": time.time()}
    else:
        answer = chat_mdl.chat(prompt_config.get("system", ""), msg, dialog.llm_setting)
        user_content = msg[-1].get("content", "[content not available]")
        logging.debug("User: {}|Assistant: {}".format(user_content, answer))
        yield {"answer": answer, "reference": {}, "audio_binary": tts(tts_mdl, answer), "prompt": "", "created_at": time.time()}


def chat(dialog, messages, stream=True, **kwargs):
    assert messages[-1]["role"] == "user", "The last content of this conversation is not from user."
    if not dialog.kb_ids:
        for ans in chat_solo(dialog, messages, stream):
            yield ans
        return

    chat_start_ts = timer()

    if llm_id2llm_type(dialog.llm_id) == "image2text":
        llm_model_config = TenantLLMService.get_model_config(dialog.tenant_id, LLMType.IMAGE2TEXT, dialog.llm_id)
    else:
        llm_model_config = TenantLLMService.get_model_config(dialog.tenant_id, LLMType.CHAT, dialog.llm_id)

    max_tokens = llm_model_config.get("max_tokens", 8192)

    check_llm_ts = timer()

    kbs = KnowledgebaseService.get_by_ids(dialog.kb_ids)
    embedding_list = list(set([kb.embd_id for kb in kbs]))
    if len(embedding_list) != 1:
        yield {"answer": "**ERROR**: Knowledge bases use different embedding models.", "reference": []}
        return {"answer": "**ERROR**: Knowledge bases use different embedding models.", "reference": []}

    embedding_model_name = embedding_list[0]

    retriever = settings.retrievaler

    questions = [m["content"] for m in messages if m["role"] == "user"][-3:]
    attachments = kwargs["doc_ids"].split(",") if "doc_ids" in kwargs else None
    if "doc_ids" in messages[-1]:
        attachments = messages[-1]["doc_ids"]

    # 다국어 검색은 이제 프론트엔드에서 처리됩니다
    if dialog.prompt_config.get("cross_language_search", False):
        logging.debug("Cross-language search is enabled, but translation is handled in frontend")

    create_retriever_ts = timer()

    embd_mdl = LLMBundle(dialog.tenant_id, LLMType.EMBEDDING, embedding_model_name)
    if not embd_mdl:
        raise LookupError("Embedding model(%s) not found" % embedding_model_name)

    bind_embedding_ts = timer()

    if llm_id2llm_type(dialog.llm_id) == "image2text":
        chat_mdl = LLMBundle(dialog.tenant_id, LLMType.IMAGE2TEXT, dialog.llm_id)
    else:
        chat_mdl = LLMBundle(dialog.tenant_id, LLMType.CHAT, dialog.llm_id)

    bind_llm_ts = timer()

    prompt_config = dialog.prompt_config
    field_map = KnowledgebaseService.get_field_map(dialog.kb_ids)
    tts_mdl = None
    if prompt_config.get("tts"):
        tts_mdl = LLMBundle(dialog.tenant_id, LLMType.TTS)
    # 필드 매핑이 준비된 경우 sql을 사용해 보세요
    if field_map:
        logging.debug("Use SQL to retrieval:{}".format(questions[-1]))
        ans = use_sql(questions[-1], field_map, dialog.tenant_id, chat_mdl, prompt_config.get("quote", True))
        if ans:
            yield ans
            return

    for p in prompt_config["parameters"]:
        if p["key"] == "knowledge":
            continue
        if p["key"] not in kwargs and not p["optional"]:
            raise KeyError("Miss parameter: " + p["key"])
        if p["key"] not in kwargs:
            prompt_config["system"] = prompt_config["system"].replace("{%s}" % p["key"], " ")

    questions = questions[-1:]

    refine_question_ts = timer()

    rerank_mdl = None
    if dialog.rerank_id:
        rerank_mdl = LLMBundle(dialog.tenant_id, LLMType.RERANK, dialog.rerank_id)

    bind_reranker_ts = timer()
    generate_keyword_ts = bind_reranker_ts
    thought = ""
    kbinfos = {"total": 0, "chunks": [], "doc_aggs": []}

    if "knowledge" not in [p["key"] for p in prompt_config["parameters"]]:
        knowledges = []
    else:
        if prompt_config.get("keyword", False):
            questions[-1] += keyword_extraction(chat_mdl, questions[-1])
            generate_keyword_ts = timer()

        tenant_ids = list(set([kb.tenant_id for kb in kbs]))

        knowledges = []

        kbinfos = retriever.retrieval(
            " ".join(questions),
            embd_mdl,
            tenant_ids,
            dialog.kb_ids,
            1,
            dialog.top_n,
            dialog.similarity_threshold,
            dialog.vector_similarity_weight,
            doc_ids=attachments,
            top=dialog.top_k,
            aggs=False,
            rerank_mdl=rerank_mdl,
            rank_feature=label_question(" ".join(questions), kbs),
        )
        knowledges = kb_prompt(kbinfos, max_tokens)

    logging.debug("{}->{}".format(" ".join(questions), "\n->".join(knowledges)))

    retrieval_ts = timer()
    if not knowledges and prompt_config.get("empty_response"):
        empty_res = prompt_config["empty_response"]
        yield {"answer": empty_res, "reference": kbinfos, "prompt": "\n\n### Query:\n%s" % " ".join(questions), "audio_binary": tts(tts_mdl, empty_res)}
        return {"answer": prompt_config["empty_response"], "reference": kbinfos}

    kwargs["knowledge"] = "\n------\n" + "\n\n------\n\n".join(knowledges)
    gen_conf = dialog.llm_setting

    msg = [{"role": "system", "content": prompt_config["system"].format(**kwargs)}]
    prompt4citation = ""
    if knowledges and (prompt_config.get("quote", True) and kwargs.get("quote", True)):
        prompt4citation = citation_prompt()
    # 过滤掉 system 角色的消息(因为前面已经单独处理了系统消息)
    msg.extend([{"role": m["role"], "content": re.sub(r"##\d+\$\$", "", m["content"])} for m in messages if m["role"] != "system"])
    used_token_count, msg = message_fit_in(msg, int(max_tokens * 0.95))
    assert len(msg) >= 2, f"message_fit_in has bug: {msg}"
    prompt = msg[0]["content"]

    if "max_tokens" in gen_conf:
        gen_conf["max_tokens"] = min(gen_conf["max_tokens"], max_tokens - used_token_count)

    def decorate_answer(answer):
        nonlocal prompt_config, knowledges, kwargs, kbinfos, prompt, retrieval_ts, questions

        refs = []
        ans = answer.split("</think>")
        think = ""
        if len(ans) == 2:
            think = ans[0] + "</think>"
            answer = ans[1]

        cited_chunk_indices = set()
        inserted_images = {}
        processed_image_urls = set()

        if knowledges and (prompt_config.get("quote", True) and kwargs.get("quote", True)):
            # 获取引用的 chunk 索引
            if not re.search(r"##[0-9]+\$\$", answer):
                answer, idx = retriever.insert_citations(
                    answer,
                    [ck["content_ltks"] for ck in kbinfos["chunks"]],
                    [ck["vector"] for ck in kbinfos["chunks"]],
                    embd_mdl,
                    tkweight=1 - dialog.vector_similarity_weight,
                    vtweight=dialog.vector_similarity_weight,
                )
                cited_chunk_indices = idx
            else:
                for r in re.finditer(r"##([0-9]+)\$\$", answer):
                    i = int(r.group(1))
                    if i < len(kbinfos["chunks"]):
                        cited_chunk_indices.add(i)

            # 处理图片插入
            def insert_image_markdown(match):
                idx = int(match.group(1))
                if idx >= len(kbinfos["chunks"]):
                    return match.group(0)

                chunk = kbinfos["chunks"][idx]
                img_path = chunk.get("image_id")
                if not img_path:
                    return match.group(0)

                protocol = "https" if MINIO_CONFIG.get("secure", False) else "http"
                img_url = f"{protocol}://{MINIO_CONFIG['visit_point']}/{img_path}"

                if img_url in processed_image_urls:
                    return match.group(0)

                processed_image_urls.add(img_url)
                inserted_images[idx] = img_url

                # 插入图片，并限制最大宽度
                return f'{match.group(0)}\n\n<img src="{img_url}" alt="{img_url}" style="max-width:800px;">'

            # 用正则替换插图
            answer = re.sub(r"##(\d+)\$\$", insert_image_markdown, answer)

            # 清理引用文献信息
            idx = set([kbinfos["chunks"][int(i)]["doc_id"] for i in cited_chunk_indices])
            recall_docs = [d for d in kbinfos["doc_aggs"] if d["doc_id"] in idx]
            if not recall_docs:
                recall_docs = kbinfos["doc_aggs"]
            kbinfos["doc_aggs"] = recall_docs

            refs = deepcopy(kbinfos)
            for c in refs["chunks"]:
                if c.get("vector"):
                    del c["vector"]

        # 特殊错误提示
        if "invalid key" in answer.lower() or "invalid api" in answer.lower():
            answer += " Please set LLM API-Key in 'User Setting -> Model providers -> API-Key'"

        # 时间信息拼接
        finish_chat_ts = timer()
        total_time_cost = (finish_chat_ts - chat_start_ts) * 1000
        check_llm_time_cost = (check_llm_ts - chat_start_ts) * 1000
        create_retriever_time_cost = (create_retriever_ts - check_llm_ts) * 1000
        bind_embedding_time_cost = (bind_embedding_ts - create_retriever_ts) * 1000
        bind_llm_time_cost = (bind_llm_ts - bind_embedding_ts) * 1000
        refine_question_time_cost = (refine_question_ts - bind_llm_ts) * 1000
        bind_reranker_time_cost = (bind_reranker_ts - refine_question_ts) * 1000
        generate_keyword_time_cost = (generate_keyword_ts - bind_reranker_ts) * 1000
        retrieval_time_cost = (retrieval_ts - generate_keyword_ts) * 1000
        generate_result_time_cost = (finish_chat_ts - retrieval_ts) * 1000

        prompt += "\n\n### Query:\n%s" % " ".join(questions)
        prompt = f"{prompt}\n\n - Total: {total_time_cost:.1f}ms\n  - Check LLM: {check_llm_time_cost:.1f}ms\n  - Create retriever: {create_retriever_time_cost:.1f}ms\n  - Bind embedding: {bind_embedding_time_cost:.1f}ms\n  - Bind LLM: {bind_llm_time_cost:.1f}ms\n  - Tune question: {refine_question_time_cost:.1f}ms\n  - Bind reranker: {bind_reranker_time_cost:.1f}ms\n  - Generate keyword: {generate_keyword_time_cost:.1f}ms\n  - Retrieval: {retrieval_time_cost:.1f}ms\n  - Generate answer: {generate_result_time_cost:.1f}ms"

        # Create citation mapping
        citation_mapping = []
        if cited_chunk_indices:
            # Sort indices for a deterministic output order
            for i in sorted(list(map(int, cited_chunk_indices))):
                if i < len(kbinfos["chunks"]):
                    chunk_info = kbinfos["chunks"][i]
                    chunk_id = chunk_info.get("chunk_id", chunk_info.get("id"))
                    doc_id = chunk_info.get("doc_id")
                    if chunk_id:
                        citation_mapping.append({
                            "citation_index": i,
                            "chunk_id": chunk_id,
                            "doc_id": doc_id
                        })

        result = {
            "answer": think + answer,
            "reference": refs,
            "prompt": re.sub(r"\n", "  \n", prompt),
            "created_at": time.time()
        }
        if citation_mapping:
            result["citation_mapping"] = citation_mapping

        return result

    if stream:
        last_ans = ""  # 마지막으로 반환된 전체 답변 기록
        answer = ""  # 현재 누적된 전체 답변
        for ans in chat_mdl.chat_streamly(prompt + prompt4citation, msg[1:], gen_conf):
            # 생각 과정(thought)이 있으면 관련 태그 제거
            if thought:
                ans = re.sub(r"<think>.*</think>", "", ans, flags=re.DOTALL)
            answer = ans
            # 새로 추가된 텍스트 조각(delta) 계산
            delta_ans = ans[len(last_ans) :]
            # 새로 추가된 토큰이 너무 적으면(16개 미만) 이번 반환은 건너뜁니다(작은 조각을 자주 보내는 것을 방지)
            if num_tokens_from_string(delta_ans) < 16:
                continue
            last_ans = answer
            # 현재 누적된 답변(생각 과정 포함) + 새로 추가된 조각 반환
            yield {"answer": thought + answer, "reference": {}, "audio_binary": tts(tts_mdl, delta_ans)}
        delta_ans = answer[len(last_ans) :]
        if delta_ans:
            yield {"answer": thought + answer, "reference": {}, "audio_binary": tts(tts_mdl, delta_ans)}
        yield decorate_answer(thought + answer)
    else:
        answer = chat_mdl.chat(prompt + prompt4citation, msg[1:], gen_conf)
        user_content = msg[-1].get("content", "[content not available]")
        logging.debug("User: {}|Assistant: {}".format(user_content, answer))
        res = decorate_answer(answer)
        res["audio_binary"] = tts(tts_mdl, answer)
        yield res


def use_sql(question, field_map, tenant_id, chat_mdl, quota=True):
    sys_prompt = "You are a Database Administrator. You need to check the fields of the following tables based on the user's list of questions and write the SQL corresponding to the last question."
    user_prompt = """
Table name: {};
Table of database fields are as follows:
{}

Question are as follows:
{}
Please write the SQL, only SQL, without any other explanations or text.
""".format(index_name(tenant_id), "\n".join([f"{k}: {v}" for k, v in field_map.items()]), question)
    tried_times = 0

    def get_table():
        nonlocal sys_prompt, user_prompt, question, tried_times
        sql = chat_mdl.chat(sys_prompt, [{"role": "user", "content": user_prompt}], {"temperature": 0.06})
        sql = re.sub(r"<think>.*</think>", "", sql, flags=re.DOTALL)
        logging.debug(f"{question} ==> {user_prompt} get SQL: {sql}")
        sql = re.sub(r"[\r\n]+", " ", sql.lower())
        sql = re.sub(r".*select ", "select ", sql.lower())
        sql = re.sub(r" +", " ", sql)
        sql = re.sub(r"([;；]|```).*", "", sql)
        if sql[: len("select ")] != "select ":
            return None, None
        if not re.search(r"((sum|avg|max|min)\(|group by )", sql.lower()):
            if sql[: len("select *")] != "select *":
                sql = "select doc_id,docnm_kwd," + sql[6:]
            else:
                flds = []
                for k in field_map.keys():
                    if k in forbidden_select_fields4resume:
                        continue
                    if len(flds) > 11:
                        break
                    flds.append(k)
                sql = "select doc_id,docnm_kwd," + ",".join(flds) + sql[8:]

        logging.debug(f"{question} get SQL(refined): {sql}")
        tried_times += 1
        return settings.retrievaler.sql_retrieval(sql, format="json"), sql

    tbl, sql = get_table()
    if tbl is None:
        return None
    if tbl.get("error") and tried_times <= 2:
        user_prompt = """
        Table name: {};
        Table of database fields are as follows:
        {}
        
        Question are as follows:
        {}
        Please write the SQL, only SQL, without any other explanations or text.
        

        The SQL error you provided last time is as follows:
        {}

        Error issued by database as follows:
        {}

        Please correct the error and write SQL again, only SQL, without any other explanations or text.
        """.format(index_name(tenant_id), "\n".join([f"{k}: {v}" for k, v in field_map.items()]), question, sql, tbl["error"])
        tbl, sql = get_table()
        logging.debug("TRY it again: {}".format(sql))

    logging.debug("GET table: {}".format(tbl))
    if tbl.get("error") or len(tbl["rows"]) == 0:
        return None

    docid_idx = set([ii for ii, c in enumerate(tbl["columns"]) if c["name"] == "doc_id"])
    doc_name_idx = set([ii for ii, c in enumerate(tbl["columns"]) if c["name"] == "docnm_kwd"])
    column_idx = [ii for ii in range(len(tbl["columns"])) if ii not in (docid_idx | doc_name_idx)]

    # 마크다운 테이블 작성
    columns = (
        "|" + "|".join([re.sub(r"(/.*|（[^（）]+）)", "", field_map.get(tbl["columns"][i]["name"], tbl["columns"][i]["name"])) for i in column_idx]) + ("|Source|" if docid_idx and docid_idx else "|")
    )

    line = "|" + "|".join(["------" for _ in range(len(column_idx))]) + ("|------|" if docid_idx and docid_idx else "")

    rows = ["|" + "|".join([rmSpace(str(r[i])) for i in column_idx]).replace("None", " ") + "|" for r in tbl["rows"]]
    rows = [r for r in rows if re.sub(r"[ |]+", "", r)]
    if quota:
        rows = "\n".join([r + f" ##{ii}$$ |" for ii, r in enumerate(rows)])
    else:
        rows = "\n".join([r + f" ##{ii}$$ |" for ii, r in enumerate(rows)])
    rows = re.sub(r"T[0-9]{2}:[0-9]{2}:[0-9]{2}(\.[0-9]+Z)?\|", "|", rows)

    if not docid_idx or not doc_name_idx:
        logging.warning("SQL missing field: " + sql)
        return {"answer": "\n".join([columns, line, rows]), "reference": {"chunks": [], "doc_aggs": []}, "prompt": sys_prompt}

    docid_idx = list(docid_idx)[0]
    doc_name_idx = list(doc_name_idx)[0]
    doc_aggs = {}
    for r in tbl["rows"]:
        if r[docid_idx] not in doc_aggs:
            doc_aggs[r[docid_idx]] = {"doc_name": r[doc_name_idx], "count": 0}
        doc_aggs[r[docid_idx]]["count"] += 1
    return {
        "answer": "\n".join([columns, line, rows]),
        "reference": {
            "chunks": [{"doc_id": r[docid_idx], "docnm_kwd": r[doc_name_idx]} for r in tbl["rows"]],
            "doc_aggs": [{"doc_id": did, "doc_name": d["doc_name"], "count": d["count"]} for did, d in doc_aggs.items()],
        },
        "prompt": sys_prompt,
    }


def tts(tts_mdl, text):
    if not tts_mdl or not text:
        return
    bin = b""
    for chunk in tts_mdl.tts(text):
        bin += chunk
    return binascii.hexlify(bin).decode("utf-8")


# 번역 관련 함수는 프론트엔드에서 처리하도록 이동되었으므로 여기서는 더 이상 필요하지 않습니다


def ask(question, kb_ids, tenant_id):
    """
    사용자 검색 요청을 처리하고, 지식 베이스에서 관련 정보를 검색하여 답변을 생성합니다

    매개변수:
        question (str): 사용자의 질문 또는 쿼리
        kb_ids (list): 검색할 지식 베이스를 지정하는 지식 베이스 ID 목록
        tenant_id (str): 권한 제어 및 리소스 격리를 위한 테넌트 ID

    프로세스:
        1. 지정된 지식 베이스 정보 가져오기
        2. 사용할 임베딩 모델 결정
        3. 지식 베이스 유형에 따라 검색기 선택 (일반 검색기 또는 지식 그래프 검색기)
        4. 임베딩 모델 및 채팅 모델 초기화
        5. 검색 작업을 실행하여 관련 문서 조각 가져오기
        6. 지식 베이스 내용을 컨텍스트로 형식화
        7. 시스템 프롬프트 구성
        8. 답변 생성 및 인용 태그 추가
        9. 생성된 답변을 스트리밍으로 반환

    반환:
        generator: 답변 및 인용 정보가 포함된 사전을 생성하는 제너레이터 객체
    """

    kbs = KnowledgebaseService.get_by_ids(kb_ids)
    embedding_list = list(set([kb.embd_id for kb in kbs]))

    is_knowledge_graph = all([kb.parser_id == ParserType.KG for kb in kbs])
    retriever = settings.retrievaler if not is_knowledge_graph else settings.kg_retrievaler
    # 텍스트를 벡터 표현으로 변환하기 위한 임베딩 모델 초기화
    embd_mdl = LLMBundle(tenant_id, LLMType.EMBEDDING, embedding_list[0])
    # 답변 생성을 위한 채팅 모델 초기화
    chat_mdl = LLMBundle(tenant_id, LLMType.CHAT)
    # 컨텍스트 길이를 제어하기 위해 채팅 모델의 최대 토큰 길이 가져오기
    max_tokens = chat_mdl.max_length
    # 모든 지식 베이스의 테넌트 ID를 가져와 중복 제거
    tenant_ids = list(set([kb.tenant_id for kb in kbs]))
    # 더 나은 효과를 위해 더 작은 유사도 임계값 설정 (원래 값 0.1)
    similarity_threshold = 0.01
    # 검색기를 호출하여 관련 문서 조각 검색
    kbinfos = retriever.retrieval(question, embd_mdl, tenant_ids, kb_ids, 1, 12, similarity_threshold, 0.3, aggs=False, rank_feature=label_question(question, kbs))
    # 검색 결과를 프롬프트로 형식화하고 모델의 최대 토큰 제한을 초과하지 않도록 보장
    knowledges = kb_prompt(kbinfos, max_tokens)
    prompt = """
    역할: 당신은 똑똑한 조수입니다.
    작업: 지식 베이스의 정보를 요약하고 사용자의 질문에 답변합니다.
    요구 사항 및 제한 사항:
    - 절대로 내용을 조작하지 마십시오. 특히 숫자는 더욱 그렇습니다.
    - 지식 베이스의 정보가 사용자의 질문과 관련이 없는 경우, **"죄송합니다, 관련 정보가 제공되지 않았습니다."라고만 답변하십시오.
    - 답변은 마크다운 형식을 사용하십시오.
    - 사용자가 질문한 언어로 답변하십시오.
    - 절대로 내용을 조작하지 마십시오. 특히 숫자는 더욱 그렇습니다.

    ### 지식 베이스 정보
    %s

    이상은 지식 베이스 정보입니다.

    """ % "\n".join(knowledges)
    msg = [{"role": "user", "content": question}]

    # 생성 완료 후 답변에 인용 태그 추가
    def decorate_answer(answer):
        nonlocal knowledges, kbinfos, prompt
        answer, idx = retriever.insert_citations(answer, [ck["content_ltks"] for ck in kbinfos["chunks"]], [ck["vector"] for ck in kbinfos["chunks"]], embd_mdl, tkweight=0.7, vtweight=0.3)
        idx = set([kbinfos["chunks"][int(i)]["doc_id"] for i in idx])
        recall_docs = [d for d in kbinfos["doc_aggs"] if d["doc_id"] in idx]
        if not recall_docs:
            recall_docs = kbinfos["doc_aggs"]
        kbinfos["doc_aggs"] = recall_docs
        refs = deepcopy(kbinfos)
        for c in refs["chunks"]:
            if c.get("vector"):
                del c["vector"]

        if answer.lower().find("invalid key") >= 0 or answer.lower().find("invalid api") >= 0:
            answer += " Please set LLM API-Key in 'User Setting -> Model Providers -> API-Key'"
        refs["chunks"] = chunks_format(refs)
        return {"answer": answer, "reference": refs}

    answer = ""
    for ans in chat_mdl.chat_streamly(prompt, msg, {"temperature": 0.1}):
        answer = ans
        yield {"answer": answer, "reference": {}}
    yield decorate_answer(answer)
