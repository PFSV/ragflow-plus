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
import logging
import re
import math
from dataclasses import dataclass

from rag.settings import TAG_FLD, PAGERANK_FLD
from rag.utils import rmSpace
from rag.nlp import rag_tokenizer, query
import numpy as np
from rag.utils.doc_store_conn import DocStoreConnection, MatchDenseExpr, FusionExpr, OrderByExpr


def index_name(uid):
    return f"ragflow_{uid}"


class Dealer:
    def __init__(self, dataStore: DocStoreConnection):
        self.qryr = query.FulltextQueryer()
        self.dataStore = dataStore

    @dataclass
    class SearchResult:
        total: int
        ids: list[str]
        query_vector: list[float] | None = None
        field: dict | None = None
        highlight: dict | None = None
        aggregation: list | dict | None = None
        keywords: list[str] | None = None
        group_docs: list[list] | None = None

    def get_vector(self, txt, emb_mdl, topk=10, similarity=0.1):
        qv, _ = emb_mdl.encode_queries(txt)
        shape = np.array(qv).shape
        if len(shape) > 1:
            raise Exception(f"Dealer.get_vector returned array's shape {shape} doesn't match expectation(exact one dimension).")
        embedding_data = [float(v) for v in qv]
        vector_column_name = f"q_{len(embedding_data)}_vec"
        return MatchDenseExpr(vector_column_name, embedding_data, "float", "cosine", topk, {"similarity": similarity})

    def get_filters(self, req):
        condition = dict()
        for key, field in {"kb_ids": "kb_id", "doc_ids": "doc_id"}.items():
            if key in req and req[key] is not None:
                condition[field] = req[key]
        # TODO(yzc): `available_int` is nullable however infinity doesn't support nullable columns.
        for key in ["knowledge_graph_kwd", "available_int", "entity_kwd", "from_entity_kwd", "to_entity_kwd", "removed_kwd"]:
            if key in req and req[key] is not None:
                condition[key] = req[key]
        return condition

    def search(self, req, idx_names: str | list[str], kb_ids: list[str], emb_mdl=None, highlight=False, rank_feature: dict | None = None):
        """
        혼합 검색 실행 (전문 검색 + 벡터 검색)

        매개변수:
            req: 요청 매개변수 딕셔너리, 포함사항:
                - page: 페이지 번호
                - topk: 반환 결과 최대 수량
                - size: 페이지당 크기
                - fields: 지정 반환 필드
                - question: 질의 문제 텍스트
                - similarity: 벡터 유사도 임계값
            idx_names: 인덱스 이름 또는 목록
            kb_ids: 지식베이스 ID 목록
            emb_mdl: 임베딩 모델, 벡터 검색용
            highlight: 하이라이트 내용 반환 여부
            rank_feature: 순위 특성 설정

        반환:
            SearchResult 객체, 포함사항:
                - total: 매칭 총수
                - ids: 매칭된 chunk ID 목록
                - query_vector: 질의 벡터
                - field: 각 chunk의 필드값
                - highlight: 하이라이트 내용
                - aggregation: 집계 결과
                - keywords: 추출된 키워드
        """
        # 1. 필터 조건과 정렬 규칙 초기화
        filters = self.get_filters(req)
        orderBy = OrderByExpr()

        # 2. 페이징 매개변수 처리
        pg = int(req.get("page", 1)) - 1
        topk = int(req.get("topk", 1024))
        ps = int(req.get("size", topk))
        offset, limit = pg * ps, ps

        # 3. 반환 필드 설정 (기본적으로 문서명, 내용 등 핵심 필드 포함)
        src = req.get(
            "fields",
            [
                "docnm_kwd",
                "content_ltks",
                "kb_id",
                "img_id",
                "title_tks",
                "important_kwd",
                "position_int",
                "doc_id",
                "page_num_int",
                "top_int",
                "create_timestamp_flt",
                "knowledge_graph_kwd",
                "question_kwd",
                "question_tks",
                "available_int",
                "content_with_weight",
                PAGERANK_FLD,
                TAG_FLD,
            ],
        )
        kwds = set([])  # 키워드 집합 초기화
        #kwds = list()  # 키워드 집합 초기화

        # 4. 질의 문제 처리
        qst = req.get("question", "")  # 질의 문제 텍스트 획득
        logging.info(f"수신된 프론트엔드 질의: {qst}")
        q_vec = []  # 질의 벡터 초기화 (벡터 검색 필요시)
        if not qst:
            # 4.1 질의 텍스트가 비어있으면 기본 정렬 검색 실행 (보통 검색 조건이 없는 브라우징용) (주: 프론트엔드 테스트 검색시 빈 텍스트 제출 금지)
            if req.get("sort"):
                orderBy.asc("page_num_int")
                orderBy.asc("top_int")
                orderBy.desc("create_timestamp_flt")
            res = self.dataStore.search(src, [], filters, [], orderBy, offset, limit, idx_names, kb_ids)
            total = self.dataStore.getTotal(res)
            logging.debug("Dealer.search TOTAL: {}".format(total))
        else:
            # 4.2 질의 텍스트가 존재하면 전문/혼합 검색 흐름 진입
            highlightFields = ["content_ltks", "title_tks", "question_kwd"] if highlight else []  # highlight는 현재 항상 False로 작동하지 않음
            # 4.2.1 전문 검색 표현식과 키워드 생성
            matchText, keywords = self.qryr.question(qst, min_match=0.1)
            logging.debug(f"matchText.matching_text: {matchText.matching_text}")
            logging.info(f"keywords: {keywords}\n")
            if emb_mdl is None:
                # 4.2.2 순수 전문 검색 모드 (벡터 모델이 제공되지 않음, 정상 상황에서는 진입하지 않음)
                matchExprs = [matchText]
                res = self.dataStore.search(src, highlightFields, filters, matchExprs, orderBy, offset, limit, idx_names, kb_ids, rank_feature=rank_feature)
                total = self.dataStore.getTotal(res)
                logging.debug("Dealer.search TOTAL: {}".format(total))
            else:
                # 4.2.3 혼합 검색 모드 (전문+벡터)
                # 질의 벡터 생성
                matchDense = self.get_vector(qst, emb_mdl, topk, req.get("similarity", 0.1))
                q_vec = matchDense.embedding_data
                # 반환 필드에 질의 벡터 필드 추가
                src.append(f"q_{len(q_vec)}_vec")
                # 융합 표현식 생성: 벡터 매칭 95%, 전문 5% 설정
                fusionExpr = FusionExpr("weighted_sum", topk, {"weights": "0.05, 0.95"})
#                fusionExpr = FusionExpr("weighted_sum", topk, {"weights": "0.3, 0.7"})
                # 혼합 질의 표현식 구축
                matchExprs = [matchText, matchDense, fusionExpr]

                # 혼합 검색 실행
                res = self.dataStore.search(src, highlightFields, filters, matchExprs, orderBy, offset, limit, idx_names, kb_ids, rank_feature=rank_feature)
                total = self.dataStore.getTotal(res)
                logging.debug("Dealer.search TOTAL: {}".format(total))
                logging.info(f"총 조회된 정보: {total}건")
                # print(f"질의 정보 결과: {res}\n")

                # 결과를 찾지 못했으면 매칭 임계값을 낮춰서 재시도
                #if total == 0:
                if total < 10:
                    if filters.get("doc_id"):
                        # 특정 문서 ID가 있을 때 무조건 질의 실행
                        res = self.dataStore.search(src, [], filters, [], orderBy, offset, limit, idx_names, kb_ids)
                        total = self.dataStore.getTotal(res)
                        logging.info(f"선택된 문서 대상, 총 조회된 정보: {total}건")
                        # print(f"질의 정보 결과: {res}\n")
                    else:
                        # 그렇지 않으면 전문과 벡터 매칭 매개변수를 조정하여 재검색
                        matchText, _ = self.qryr.question(qst, min_match=0.1)
                        filters.pop("doc_id", None)
                        matchDense.extra_options["similarity"] = 0.17
                        res = self.dataStore.search(src, highlightFields, filters, [matchText, matchDense, fusionExpr], orderBy, offset, limit, idx_names, kb_ids, rank_feature=rank_feature)
                        total = self.dataStore.getTotal(res)
                        logging.debug("Dealer.search 2 TOTAL: {}".format(total))
                        logging.info(f"재질의, 총 조회된 정보: {total}건")
                        # print(f"질의 정보 결과: {res}\n")

            # 4.3 키워드 처리 (키워드에 대해 더 세분화된 분할 실행)
            for k in keywords:
                #kwds.append(k)
                kwds.add(k)
                for kk in rag_tokenizer.fine_grained_tokenize(k).split():
                    if len(kk) < 2:
                        continue
                    if kk in kwds:
                        continue
                    kwds.add(kk)

        # 5. 검색 결과에서 ID, 필드, 집계 및 하이라이트 정보 추출
        logging.debug(f"TOTAL: {total}")
        ids = self.dataStore.getChunkIds(res)  # 매칭된 chunk의 ID 추출
        keywords = list(kwds)  # 리스트 형식으로 변환하여 반환
        highlight = self.dataStore.getHighlight(res, keywords, "content_with_weight")  # 하이라이트 내용 획득
        aggs = self.dataStore.getAggregation(res, "docnm_kwd")  # 문서명 기반 집계 분석 실행
        return self.SearchResult(total=total, ids=ids, query_vector=q_vec, aggregation=aggs, highlight=highlight, field=self.dataStore.getFields(res, src), keywords=keywords)

    @staticmethod
    def trans2floats(txt):
        return [float(t) for t in txt.split("\t")]

    def insert_citations(self, answer, chunks, chunk_v, embd_mdl, tkweight=0.1, vtweight=0.9):
        assert len(chunks) == len(chunk_v)
        if not chunks:
            return answer, set([])
        pieces = re.split(r"(```)", answer)
        if len(pieces) >= 3:
            i = 0
            pieces_ = []
            while i < len(pieces):
                if pieces[i] == "```":
                    st = i
                    i += 1
                    while i < len(pieces) and pieces[i] != "```":
                        i += 1
                    if i < len(pieces):
                        i += 1
                    pieces_.append("".join(pieces[st:i]) + "\n")
                else:
                    pieces_.extend(re.split(r"([^\|][；。？!！\n]|[a-z][.?;!][ \n])", pieces[i]))
                    i += 1
            pieces = pieces_
        else:
            pieces = re.split(r"([^\|][；。？!！\n]|[a-z][.?;!][ \n])", answer)
        for i in range(1, len(pieces)):
            if re.match(r"([^\|][；。？!！\n]|[a-z][.?;!][ \n])", pieces[i]):
                pieces[i - 1] += pieces[i][0]
                pieces[i] = pieces[i][1:]
        idx = []
        pieces_ = []
        for i, t in enumerate(pieces):
            if len(t) < 5:
                continue
            idx.append(i)
            pieces_.append(t)
        logging.debug("{} => {}".format(answer, pieces_))
        if not pieces_:
            return answer, set([])

        ans_v, _ = embd_mdl.encode(pieces_)
        for i in range(len(chunk_v)):
            if len(ans_v[0]) != len(chunk_v[i]):
                chunk_v[i] = [0.0] * len(ans_v[0])
                logging.warning("The dimension of query and chunk do not match: {} vs. {}".format(len(ans_v[0]), len(chunk_v[i])))

        assert len(ans_v[0]) == len(chunk_v[0]), "The dimension of query and chunk do not match: {} vs. {}".format(len(ans_v[0]), len(chunk_v[0]))

        chunks_tks = [rag_tokenizer.tokenize(self.qryr.rmWWW(ck)).split() for ck in chunks]
        cites = {}
        thr = 0.63
        while thr > 0.3 and len(cites.keys()) == 0 and pieces_ and chunks_tks:
            for i, a in enumerate(pieces_):
                sim, tksim, vtsim = self.qryr.hybrid_similarity(ans_v[i], chunk_v, rag_tokenizer.tokenize(self.qryr.rmWWW(pieces_[i])).split(), chunks_tks, tkweight, vtweight)
                mx = np.max(sim) * 0.99
                logging.debug("{} SIM: {}".format(pieces_[i], mx))
                if mx < thr:
                    continue
                cites[idx[i]] = list(set([str(ii) for ii in range(len(chunk_v)) if sim[ii] > mx]))[:4]
            thr *= 0.8

        res = ""
        seted = set([])
        for i, p in enumerate(pieces):
            res += p
            if i not in idx:
                continue
            if i not in cites:
                continue
            for c in cites[i]:
                assert int(c) < len(chunk_v)
            for c in cites[i]:
                if c in seted:
                    continue
                res += f" ##{c}$$"
                seted.add(c)

        return res, seted

    def _rank_feature_scores(self, query_rfea, search_res):
        ## For rank feature(tag_fea) scores.
        rank_fea = []
        pageranks = []
        for chunk_id in search_res.ids:
            pageranks.append(search_res.field[chunk_id].get(PAGERANK_FLD, 0))
        pageranks = np.array(pageranks, dtype=float)

        if not query_rfea:
            return np.array([0 for _ in range(len(search_res.ids))]) + pageranks

        q_denor = np.sqrt(np.sum([s * s for t, s in query_rfea.items() if t != PAGERANK_FLD]))
        for i in search_res.ids:
            nor, denor = 0, 0
            for t, sc in eval(search_res.field[i].get(TAG_FLD, "{}")).items():
                if t in query_rfea:
                    nor += query_rfea[t] * sc
                denor += sc * sc
            if denor == 0:
                rank_fea.append(0)
            else:
                rank_fea.append(nor / np.sqrt(denor) / q_denor)
        return np.array(rank_fea) * 10.0 + pageranks

    def rerank(self, sres, query, tkweight=0.3, vtweight=0.7, cfield="content_ltks", rank_feature: dict | None = None):
        """
        초기 검색 결과 (sres)에 대해 재순위 정렬을 수행합니다.

        이 방법은 여러 유사도/특성을 결합하여 각 결과의 새로운 순위 점수를 계산합니다:
        1. 텍스트 유사도 (Token Similarity): 질의 키워드와 문서 내용의 토큰 매칭 기반.
        2. 벡터 유사도 (Vector Similarity): 질의 벡터와 문서 벡터의 코사인 유사도 기반.
        3. 순위 특성 점수 (Rank Feature Scores): 문서의 PageRank 값이나 질의 관련 태그 특성 점수.

        최종 순위 점수는 이러한 여러 점수의 가중 조합(또는 직접 합산)입니다.

        Args:
            sres (SearchResult): 초기 검색 결과 객체, 질의 벡터, 문서 ID, 필드 내용 등 포함.
            query (str): 원본 사용자 질의 문자열.
            tkweight (float): 혼합 유사도 계산에서 텍스트 유사도의 가중치.
            vtweight (float): 혼합 유사도 계산에서 벡터 유사도의 가중치.
            cfield (str): 토큰 매칭을 위한 주요 텍스트 내용 추출용 필드명, 기본값 "content_ltks".
            rank_feature (dict | None): 순위 특성 점수 계산용 질의측 특성,
                                        예: {PAGERANK_FLD: 10}는 PageRank 가중치를 의미,
                                        또는 기타 태그 및 가중치를 포함한 딕셔너리.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]:
                - sim (np.ndarray): 각 문서의 최종 재순위 점수 (혼합 유사도 + 순위 특성 점수).
                - tksim (np.ndarray): 각 문서의 순수 텍스트 유사도 점수.
                - vtsim (np.ndarray): 각 문서의 순수 벡터 유사도 점수.
                초기 검색 결과가 비어있으면 (sres.ids is empty), 세 개의 빈 리스트를 반환.
        """
        _, keywords = self.qryr.question(query)
        vector_size = len(sres.query_vector)
        vector_column = f"q_{vector_size}_vec"
        zero_vector = [0.0] * vector_size
        ins_embd = []
        for chunk_id in sres.ids:
            vector = sres.field[chunk_id].get(vector_column, zero_vector)
            if isinstance(vector, str):
                vector = [float(v) for v in vector.split("\t")]
            ins_embd.append(vector)
        if not ins_embd:
            return [], [], []

        for i in sres.ids:
            if isinstance(sres.field[i].get("important_kwd", []), str):
                sres.field[i]["important_kwd"] = [sres.field[i]["important_kwd"]]
        ins_tw = []
        for i in sres.ids:
            content_ltks = sres.field[i][cfield].split()
            title_tks = [t for t in sres.field[i].get("title_tks", "").split() if t]
            question_tks = [t for t in sres.field[i].get("question_tks", "").split() if t]
            important_kwd = sres.field[i].get("important_kwd", [])
            tks = content_ltks + title_tks * 2 + important_kwd * 5 + question_tks * 6
            ins_tw.append(tks)

        ## For rank feature(tag_fea) scores.
        rank_fea = self._rank_feature_scores(rank_feature, sres)

        sim, tksim, vtsim = self.qryr.hybrid_similarity(sres.query_vector, ins_embd, keywords, ins_tw, tkweight, vtweight)

        return sim + rank_fea, tksim, vtsim

    def rerank_by_model(self, rerank_mdl, sres, query, tkweight=0.3, vtweight=0.7, cfield="content_ltks", rank_feature: dict | None = None):
        _, keywords = self.qryr.question(query)

        for i in sres.ids:
            if isinstance(sres.field[i].get("important_kwd", []), str):
                sres.field[i]["important_kwd"] = [sres.field[i]["important_kwd"]]
        ins_tw = []
        for i in sres.ids:
            content_ltks = sres.field[i][cfield].split()
            title_tks = [t for t in sres.field[i].get("title_tks", "").split() if t]
            important_kwd = sres.field[i].get("important_kwd", [])
            tks = content_ltks + title_tks + important_kwd
            ins_tw.append(tks)

        tksim = self.qryr.token_similarity(keywords, ins_tw)
        vtsim, _ = rerank_mdl.similarity(query, [rmSpace(" ".join(tks)) for tks in ins_tw])
        ## For rank feature(tag_fea) scores.
        rank_fea = self._rank_feature_scores(rank_feature, sres)

        return tkweight * (np.array(tksim) + rank_fea) + vtweight * vtsim, tksim, vtsim

    def hybrid_similarity(self, ans_embd, ins_embd, ans, inst):
        return self.qryr.hybrid_similarity(ans_embd, ins_embd, rag_tokenizer.tokenize(ans).split(), rag_tokenizer.tokenize(inst).split())

    def retrieval(
        self,
        question,
        embd_mdl,
        tenant_ids,
        kb_ids,
        page,
        page_size,
        similarity_threshold=0.2,
        vector_similarity_weight=0.3,
        top=1024,
        doc_ids=None,
        aggs=True,
        rerank_mdl=None,
        highlight=False,
        rank_feature: dict | None = {PAGERANK_FLD: 10},
    ):
        """
        검색 작업을 실행하여 문제에 따라 관련 문서 조각을 질의

        매개변수 설명:
        - question: 사용자 입력 질의 문제
        - embd_mdl: 임베딩 모델, 텍스트를 벡터로 변환하는 데 사용
        - tenant_ids: 테넌트 ID, 문자열 또는 리스트 가능
        - kb_ids: 지식베이스 ID 리스트
        - page: 현재 페이지 번호
        - page_size: 페이지당 결과 수량
        - similarity_threshold: 유사도 임계값, 이 값보다 낮은 결과는 필터링됨
        - vector_similarity_weight: 벡터 유사도 가중치
        - top: 검색 최대 결과 수
        - doc_ids: 문서 ID 리스트, 검색 범위 제한용
        - aggs: 문서 정보 집계 여부
        - rerank_mdl: 재순위 모델
        - highlight: 매칭 내용 하이라이트 여부
        - rank_feature: 순위 특성, PageRank 값 등

        반환:
        검색 결과를 포함한 딕셔너리, 총수, 문서 조각 및 문서 집계 정보 포함
        """
        # 결과 딕셔너리 초기화
        ranks = {"total": 0, "chunks": [], "doc_aggs": {}}
        if not question:
            return ranks
        # 재순위 페이지 제한 설정
        RERANK_LIMIT = 64
        RERANK_LIMIT = int(RERANK_LIMIT // page_size + ((RERANK_LIMIT % page_size) / (page_size * 1.0) + 0.5)) * page_size if page_size > 1 else 1
        if RERANK_LIMIT < 1:
            RERANK_LIMIT = 1
        # 검색 요청 매개변수 구축
        req = {
            "kb_ids": kb_ids,
            "doc_ids": doc_ids,
            "page": math.ceil(page_size * page / RERANK_LIMIT),
            "size": RERANK_LIMIT,
            "question": question,
            "vector": True,
            "topk": top,
            "similarity": similarity_threshold,
            "available_int": 1,
        }

        # 테넌트 ID 형식 처리
        if isinstance(tenant_ids, str):
            tenant_ids = tenant_ids.split(",")

        # 검색 작업 실행
        sres = self.search(req, [index_name(tid) for tid in tenant_ids], kb_ids, embd_mdl, highlight, rank_feature=rank_feature)

        # 재순위 작업 실행
        if rerank_mdl and sres.total > 0:
            sim, tsim, vsim = self.rerank_by_model(rerank_mdl, sres, question, 1 - vector_similarity_weight, vector_similarity_weight, rank_feature=rank_feature)
        else:
            sim, tsim, vsim = self.rerank(sres, question, 1 - vector_similarity_weight, vector_similarity_weight, rank_feature=rank_feature)
        # 검색 함수에서 이미 페이지네이션됨
        idx = np.argsort(sim * -1)[(page - 1) * page_size : page * page_size]

        dim = len(sres.query_vector)
        vector_column = f"q_{dim}_vec"
        zero_vector = [0.0] * dim
        if doc_ids:
            similarity_threshold = 0
            page_size = 30
        sim_np = np.array(sim)
        filtered_count = (sim_np >= similarity_threshold).sum()
        ranks["total"] = int(filtered_count)  # Convert from np.int64 to Python int otherwise JSON serializable error
        for i in idx:
            if sim[i] < similarity_threshold:
                break
            if len(ranks["chunks"]) >= page_size:
                if aggs:
                    continue
                break
            id = sres.ids[i]
            chunk = sres.field[id]
            dnm = chunk.get("docnm_kwd", "")
            did = chunk.get("doc_id", "")
            position_int = chunk.get("position_int", [])
            d = {
                "chunk_id": id,
                "content_ltks": chunk["content_ltks"],
                "content_with_weight": chunk["content_with_weight"],
                "doc_id": did,
                "docnm_kwd": dnm,
                "kb_id": chunk["kb_id"],
                "important_kwd": chunk.get("important_kwd", []),
                "image_id": chunk.get("img_id", ""),
                "similarity": sim[i],
                "vector_similarity": vsim[i],
                "term_similarity": tsim[i],
                "vector": chunk.get(vector_column, zero_vector),
                "positions": position_int,
                "doc_type_kwd": chunk.get("doc_type_kwd", ""),
            }
            if highlight and sres.highlight:
                if id in sres.highlight:
                    d["highlight"] = rmSpace(sres.highlight[id])
                else:
                    d["highlight"] = d["content_with_weight"]
            ranks["chunks"].append(d)
            if dnm not in ranks["doc_aggs"]:
                ranks["doc_aggs"][dnm] = {"doc_id": did, "count": 0}
            ranks["doc_aggs"][dnm]["count"] += 1
        ranks["doc_aggs"] = [{"doc_name": k, "doc_id": v["doc_id"], "count": v["count"]} for k, v in sorted(ranks["doc_aggs"].items(), key=lambda x: x[1]["count"] * -1)]
        ranks["chunks"] = ranks["chunks"][:page_size]

        return ranks

    def sql_retrieval(self, sql, fetch_size=128, format="json"):
        tbl = self.dataStore.sql(sql, fetch_size, format)
        return tbl

    def chunk_list(self, doc_id: str, tenant_id: str, kb_ids: list[str], max_count=1024, offset=0, fields=["docnm_kwd", "content_with_weight", "img_id"]):
        condition = {"doc_id": doc_id}
        res = []
        bs = 128
        for p in range(offset, max_count, bs):
            es_res = self.dataStore.search(fields, [], condition, [], OrderByExpr(), p, bs, index_name(tenant_id), kb_ids)
            dict_chunks = self.dataStore.getFields(es_res, fields)
            for id, doc in dict_chunks.items():
                doc["id"] = id
            if dict_chunks:
                res.extend(dict_chunks.values())
            if len(dict_chunks.values()) < bs:
                break
        return res

    def all_tags(self, tenant_id: str, kb_ids: list[str], S=1000):
        if not self.dataStore.indexExist(index_name(tenant_id), kb_ids[0]):
            return []
        res = self.dataStore.search([], [], {}, [], OrderByExpr(), 0, 0, index_name(tenant_id), kb_ids, ["tag_kwd"])
        return self.dataStore.getAggregation(res, "tag_kwd")

    def all_tags_in_portion(self, tenant_id: str, kb_ids: list[str], S=1000):
        res = self.dataStore.search([], [], {}, [], OrderByExpr(), 0, 0, index_name(tenant_id), kb_ids, ["tag_kwd"])
        res = self.dataStore.getAggregation(res, "tag_kwd")
        total = np.sum([c for _, c in res])
        return {t: (c + 1) / (total + S) for t, c in res}

    def tag_content(self, tenant_id: str, kb_ids: list[str], doc, all_tags, topn_tags=3, keywords_topn=30, S=1000):
        idx_nm = index_name(tenant_id)
        match_txt = self.qryr.paragraph(doc["title_tks"] + " " + doc["content_ltks"], doc.get("important_kwd", []), keywords_topn)
        res = self.dataStore.search([], [], {}, [match_txt], OrderByExpr(), 0, 0, idx_nm, kb_ids, ["tag_kwd"])
        aggs = self.dataStore.getAggregation(res, "tag_kwd")
        if not aggs:
            return False
        cnt = np.sum([c for _, c in aggs])
        tag_fea = sorted([(a, round(0.1 * (c + 1) / (cnt + S) / max(1e-6, all_tags.get(a, 0.0001)))) for a, c in aggs], key=lambda x: x[1] * -1)[:topn_tags]
        doc[TAG_FLD] = {a: c for a, c in tag_fea if c > 0}
        return True

    def tag_query(self, question: str, tenant_ids: str | list[str], kb_ids: list[str], all_tags, topn_tags=3, S=1000):
        if isinstance(tenant_ids, str):
            idx_nms = index_name(tenant_ids)
        else:
            idx_nms = [index_name(tid) for tid in tenant_ids]
        match_txt, _ = self.qryr.question(question, min_match=0.0)
        res = self.dataStore.search([], [], {}, [match_txt], OrderByExpr(), 0, 0, idx_nms, kb_ids, ["tag_kwd"])
        aggs = self.dataStore.getAggregation(res, "tag_kwd")
        if not aggs:
            return {}
        cnt = np.sum([c for _, c in aggs])
        tag_fea = sorted([(a, round(0.1 * (c + 1) / (cnt + S) / max(1e-6, all_tags.get(a, 0.0001)))) for a, c in aggs], key=lambda x: x[1] * -1)[:topn_tags]
        return {a.replace(".", "_"): max(1, c) for a, c in tag_fea}
