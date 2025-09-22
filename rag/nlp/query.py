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
import json
import re
from rag.utils.doc_store_conn import MatchTextExpr

from rag.nlp import rag_tokenizer, term_weight, synonym
from rag.nlp.korean_tokenizer import KoreanTokenizer

logger = logging.getLogger('ragflow.query')

class FulltextQueryer:
    def __init__(self):
        self.tw = term_weight.Dealer()
        self.syn = synonym.Dealer()
        self.korean_tokenizer = KoreanTokenizer()
        self.query_fields = [
            "title_tks^10",
            "title_sm_tks^5",
            "important_kwd^30",
            "important_tks^20",
            "question_tks^20",
            "content_ltks^2",
            "content_sm_ltks",
        ]

    @staticmethod
    def subSpecialChar(line):
        return re.sub(r"([:\{\}/\[\]\-\*\"\(\)\|\+~\^])", r"\\\1", line).strip()

    @staticmethod
#    def isChinese(line):
#        arr = re.split(r"[ \t]+", line)
#        if len(arr) <= 3:
#            return True
#        e = 0
#        for t in arr:
#            if not re.match(r"[a-zA-Z]+$", t):
#                e += 1
#        return e * 1.0 / len(arr) >= 0.7
    def isChinese(line):
        """
        중국어(간체/번체) 문자를 직접 체크하여 중국어 여부를 판별합니다.
        
        알고리즘: 공백으로 분리된 토큰 중 중국어 문자(한자)를 포함하는 토큰의 비율이 30% 이상이면 중국어로 판단합니다.
        중국어 유니코드 범위: 
        - \u4e00-\u9fff: CJK 통합 한자
        - \u3400-\u4dbf: CJK 확장 A
        - \uf900-\ufaff: CJK 호환 한자
        """
        arr = re.split(r"[ \t]+", line)
        if not arr:
            return False
        chinese_count = 0
        for token in arr:
            # 중국어 문자(한자) 포함 여부 체크
            if re.search(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]", token):
                chinese_count += 1
        return (chinese_count * 1.0 / len(arr)) >= 0.3

    @staticmethod
    def isKorean(line):
        """
        간단한 한국어 여부 판별기입니다.

        알고리즘: 공백으로 분리된 토큰 중 한글(가-힣)을 포함하는 토큰의 비율이 30% 이상이면 한국어로 판단합니다.
        이 방식은 혼합 텍스트에서도 한글 중심 여부를 빠르게 판별하는 데 적절합니다.
        """
        arr = re.split(r"[ \t]+", line)
        if not arr:
            return False
        k = 0
        for t in arr:
            if re.search(r"[\uac00-\ud7a3]", t):
                k += 1
        return (k * 1.0 / len(arr)) >= 0.3

    @staticmethod
    def rmWWW(txt):
        """
        텍스트에서 WWW(WHAT, WHO, WHERE 등 의문사)를 제거합니다.

        이 함수는 일련의 정규식 패턴을 통해 텍스트의 의문사를 식별하고 대체하여 텍스트를 단순화하거나 후속 처리를 준비합니다.
        매개변수:
        - txt: 처리할 텍스트 문자열.

        반환:
        - 처리된 텍스트 문자열. 모든 의문사가 제거되어 텍스트가 비어 있으면 원본 텍스트를 반환합니다.
        """
        patts = [
            (
                r"是*(什么样的|哪家|一下|那家|请问|啥样|咋样了|什么时候|何时|何地|何人|是否|是不是|多少|哪里|怎么|哪儿|怎么样|如何|哪些|是啥|啥是|啊|吗|呢|吧|咋|什么|有没有|呀|谁|哪位|哪个)是*",
                "",
            ),
            (r"(^| )(what|who|how|which|where|why)('re|'s)? ", " "),
            (
                r"(^| )('s|'re|is|are|were|was|do|does|did|don't|doesn't|didn't|has|have|be|there|you|me|your|my|mine|just|please|may|i|should|would|wouldn't|will|won't|done|go|for|with|so|the|a|an|by|i'm|it's|he's|she's|they|they're|you're|as|by|on|in|at|up|out|down|of|to|or|and|if) ",
                " ",
            ),
        ]
        otxt = txt
        for r, p in patts:
            txt = re.sub(r, p, txt, flags=re.IGNORECASE)
        if not txt:
            txt = otxt
        return txt

    @staticmethod
    def add_space_between_eng_zh(txt):
        """
        영어와 중국어 사이에 공백을 추가합니다.

        이 함수는 정규식을 통해 텍스트에서 영어와 중국어가 인접한 경우를 찾아 그 사이에 공백을 삽입합니다.
        이렇게 하면 특히 영어와 중국어가 혼용될 때 텍스트의 가독성을 향상시킬 수 있습니다.

        매개변수:
        txt (str): 처리할 텍스트 문자열.

        반환:
        str: 영어와 중국어 사이에 공백이 추가된 처리된 텍스트 문자열.
        """
        # (영문/영문+숫자) + 중문
        txt = re.sub(r"([A-Za-z]+[0-9]+)([\u4e00-\u9fa5]+)", r"\1 \2", txt)
        # 영문 + 중문
        txt = re.sub(r"([A-Za-z])([\u4e00-\u9fa5]+)", r"\1 \2", txt)
        # 중문 + (영문/영문+숫자)
        txt = re.sub(r"([\u4e00-\u9fa5]+)([A-Za-z]+[0-9]+)", r"\1 \2", txt)
        txt = re.sub(r"([\u4e00-\u9fa5]+)([A-Za-z])", r"\1 \2", txt)
        return txt

    def question(self, txt, tbl="qa", min_match: float = 0.6):
        """
        입력된 텍스트를 기반으로 데이터베이스에서 관련 질문을 매칭하기 위한 쿼리 표현식을 생성합니다.

        매개변수:
        - txt (str): 입력 텍스트.
        - tbl (str): 데이터 테이블 이름, 기본값 "qa".
        - min_match (float): 최소 매칭 유사도, 기본값 0.6.

        반환:
        - MatchTextExpr: 생성된 쿼리 표현식 객체.
        - keywords (list): 추출된 키워드 리스트.
        """
        if self.isChinese(txt):
            txt = FulltextQueryer.add_space_between_eng_zh(txt)  # 영어와 중국어 사이에 공백 추가
            # 정규식을 사용하여 특수 문자를 단일 공백으로 바꾸고, 텍스트를 간체 중국어와 소문자로 변환
            txt = re.sub(
                r"[ :|\r\n\t,，。？?/`!！&^%%()\[\]{}<>]+",
                " ",
                rag_tokenizer.tradi2simp(rag_tokenizer.strQ2B(txt.lower())),
            ).strip()
            otxt = txt
            txt = FulltextQueryer.rmWWW(txt)

        # 텍스트가 중국어가 아니면 영어/한국어 등 비중국어 처리
        if not self.isChinese(txt):
            txt = FulltextQueryer.rmWWW(txt)
            # 한글이 포함된 경우에는 한국어 전용 토크나이저를 사용
            # 한글이 포함된 경우에는 한국어 전용 토크나이저를 사용
            if self.isKorean(txt):
                tks = self.korean_tokenizer.tokenize(txt)
            else:
                tks = rag_tokenizer.tokenize(txt).split()
            keywords = [t for t in tks if t]
            tks_w = self.tw.weights(tks, preprocess=False)
            tks_w = [(re.sub(r"[ \\\"'^]", "", tk), w) for tk, w in tks_w]
            tks_w = [(re.sub(r"^[a-z0-9]$", "", tk), w) for tk, w in tks_w if tk]
            tks_w = [(re.sub(r"^[\+-]", "", tk), w) for tk, w in tks_w if tk]
            tks_w = [(tk.strip(), w) for tk, w in tks_w if tk.strip()]
            syns = []
            for tk, w in tks_w[:256]:
                syn = self.syn.lookup(tk)
                syn = rag_tokenizer.tokenize(" ".join(syn)).split()
                keywords.extend(syn)
                syn = ['"{}"^{:.4f}'.format(s, w / 4.0) for s in syn if s.strip()]
                syns.append(" ".join(syn))

            q = ["({}^{:.4f}".format(tk, w) + " {})".format(syn) for (tk, w), syn in zip(tks_w, syns) if tk and not re.match(r"[.^+\(\)-]", tk)]
            for i in range(1, len(tks_w)):
                left, right = tks_w[i - 1][0].strip(), tks_w[i][0].strip()
                if not left or not right:
                    continue
                q.append(
                    '"%s %s"^%.4f'
                    % (
                        tks_w[i - 1][0],
                        tks_w[i][0],
                        max(tks_w[i - 1][1], tks_w[i][1]) * 2,
                    )
                )
            # Elasticsearch 쿼리 문자열에서 특수 문자를 이스케이프 처리
            txt = FulltextQueryer.subSpecialChar(txt)
            if not q:
                q.append(txt)
            query = " OR ".join(q)
            query = f'("{txt}"^2.00) OR ({query})'
            logger.info(f'query: {query}')
            return MatchTextExpr(self.query_fields, query, 100), keywords

        def need_fine_grained_tokenize(tk):
            """
            단어를 세분화된 토큰으로 나눌 필요가 있는지 판단합니다.

            매개변수:
            - tk (str): 판단할 단어.

            반환:
            - bool: 세분화된 토큰화가 필요한지 여부.
            """
            # 길이가 3 미만인 단어는 처리하지 않음
            if len(tk) < 3:
                return False
            # 특정 패턴(예: 숫자, 영문자, 기호 조합)과 일치하는 단어는 처리하지 않음
            if re.match(r"[0-9a-z\.\+#_\*-]+$", tk):
                return False
            return True

        txt = FulltextQueryer.rmWWW(txt)
        qs, keywords = [], []
        # 텍스트 분할 후 앞 256개 조각을 순회 (너무 긴 텍스트 처리 방지)
        for tt in self.tw.split(txt)[:256]:  # 참고: 이 split은 영어용으로 설계된 것으로 보이며, 중국어에는 작동하지 않음
            if not tt:
                continue
            # 현재 조각을 키워드 목록에 추가
            keywords.append(tt)
            # 현재 조각의 가중치 가져오기
            twts = self.tw.weights([tt])
            # 동의어 찾기
            syns = self.syn.lookup(tt)
            # 동의어가 있고 키워드 수가 32개를 넘지 않으면 동의어를 키워드 목록에 추가
            if syns and len(keywords) < 32:
                keywords.extend(syns)
            # 디버그 로그: 가중치 정보 출력
            logging.debug(json.dumps(twts, ensure_ascii=False))
            # 쿼리 조건 목록 초기화
            tms = []
            # 각 토큰을 가중치 내림차순으로 정렬하여 처리
            for tk, w in sorted(twts, key=lambda x: x[1] * -1):
                # 세분화된 토큰화가 필요한 경우, 토큰화 진행
                sm = rag_tokenizer.fine_grained_tokenize(tk).split() if need_fine_grained_tokenize(tk) else []
                # 각 토큰화 결과 정제:
                # 1. 구두점 및 특수 문자 제거
                # 2. subSpecialChar를 사용하여 추가 처리
                # 3. 길이가 1 이하인 단어 필터링
                sm = [
                    re.sub(
                        r"[ ,\./;'\[\]\\`~!@#$%\^&\*\(\)=\+_<>\?:\"\{\}\|，。；‘’【】、！￥……（）——《》？：“”-]+",
                        "",
                        m,
                    )
                    for m in sm
                ]
                sm = [FulltextQueryer.subSpecialChar(m) for m in sm if len(m) > 1]
                sm = [m for m in sm if len(m) > 1]

                # 키워드 수가 상한에 도달하지 않은 경우, 처리된 토큰과 토큰화 결과 추가
                if len(keywords) < 32:
                    keywords.append(re.sub(r"[ \\\"']+", "", tk))  # 이스케이프 문자 제거
                    keywords.extend(sm)  # 토큰화 결과 추가
                # 현재 토큰의 동의어를 가져와 처리
                tk_syns = self.syn.lookup(tk)
                tk_syns = [FulltextQueryer.subSpecialChar(s) for s in tk_syns]
                # 유효한 동의어를 키워드 목록에 추가
                if len(keywords) < 32:
                    keywords.extend([s for s in tk_syns if s])
                # 동의어를 토큰화하고, 공백이 포함된 동의어에 따옴표 추가
                tk_syns = [rag_tokenizer.fine_grained_tokenize(s) for s in tk_syns if s]
                tk_syns = [f'"{s}"' if s.find(" ") > 0 else s for s in tk_syns]

                # 키워드 수가 상한에 도달하면 처리 중지
                if len(keywords) >= 32:
                    break

                # 쿼리 조건 구성을 위해 현재 토큰 처리:
                # 1. 특수 문자 처리
                # 2. 공백이 포함된 토큰에 따옴표 추가
                # 3. 동의어가 있으면 OR 조건을 구성하고 가중치 낮춤
                # 4. 토큰화 결과가 있으면 OR 조건 추가
                tk = FulltextQueryer.subSpecialChar(tk)
                if tk.find(" ") > 0:
                    tk = '"%s"' % tk
                if tk_syns:
                    tk = f"({tk} OR (%s)^0.2)" % " ".join(tk_syns)
                if sm:
                    tk = f'{tk} OR "%s" OR ("%s"~2)^0.5' % (" ".join(sm), " ".join(sm))
                if tk.strip():
                    tms.append((tk, w))

            # 처리된 쿼리 조건을 가중치에 따라 문자열로 조합
            tms = " ".join([f"({t})^{w}" for t, w in tms])

            # 가중치 항목이 여러 개일 경우, 구문 검색 조건 추가 (인접 단어 매칭 가중치 높임)
            if len(twts) > 1:
                tms += ' ("%s"~2)^1.5' % rag_tokenizer.tokenize(tt)

            # 동의어 쿼리 조건 처리
            syns = " OR ".join(['"%s"' % rag_tokenizer.tokenize(FulltextQueryer.subSpecialChar(s)) for s in syns])
            # 주 쿼리 조건과 동의어 조건 조합
            if syns and tms:
                tms = f"({tms})^5 OR ({syns})^0.7"
            # 최종 쿼리 조건을 목록에 추가
            qs.append(tms)

        # 모든 쿼리 조건 처리
        if qs:
            # 모든 쿼리 조건을 OR 관계로 조합
            query = " OR ".join([f"({t})" for t in qs if t])
            # 쿼리 조건이 비어 있으면 원본 텍스트 사용
            if not query:
                query = otxt
            # 텍스트 표현식과 키워드 매칭 반환
            return MatchTextExpr(self.query_fields, query, 100, {"minimum_should_match": min_match}), keywords
        # 생성된 쿼리 조건이 없으면 키워드만 반환
        return None, keywords

    def hybrid_similarity(self, avec, bvecs, atks, btkss, tkweight=0.3, vtweight=0.7):
        from sklearn.metrics.pairwise import cosine_similarity as CosineSimilarity
        import numpy as np

        sims = CosineSimilarity([avec], bvecs)
        tksim = self.token_similarity(atks, btkss)
        if np.sum(sims[0]) == 0:
            return np.array(tksim), tksim, sims[0]
        return np.array(sims[0]) * vtweight + np.array(tksim) * tkweight, tksim, sims[0]

    def token_similarity(self, atks, btkss):
        def toDict(tks):
            d = {}
            if isinstance(tks, str):
                tks = tks.split()
            for t, c in self.tw.weights(tks, preprocess=False):
                if t not in d:
                    d[t] = 0
                d[t] += c
            return d

        atks = toDict(atks)
        btkss = [toDict(tks) for tks in btkss]
        return [self.similarity(atks, btks) for btks in btkss]

    def similarity(self, qtwt, dtwt):
        if isinstance(dtwt, type("")):
            dtwt = {t: w for t, w in self.tw.weights(self.tw.split(dtwt), preprocess=False)}
        if isinstance(qtwt, type("")):
            qtwt = {t: w for t, w in self.tw.weights(self.tw.split(qtwt), preprocess=False)}
        s = 1e-9
        for k, v in qtwt.items():
            if k in dtwt:
                s += v  # * dtwt[k]
        q = 1e-9
        for k, v in qtwt.items():
            q += v
        return s / q

    def paragraph(self, content_tks: str, keywords: list = [], keywords_topn=30):
        if isinstance(content_tks, str):
            content_tks = [c.strip() for c in content_tks.strip() if c.strip()]
        tks_w = self.tw.weights(content_tks, preprocess=False)

        keywords = [f'"{k.strip()}"' for k in keywords]
        for tk, w in sorted(tks_w, key=lambda x: x[1] * -1)[:keywords_topn]:
            tk_syns = self.syn.lookup(tk)
            tk_syns = [FulltextQueryer.subSpecialChar(s) for s in tk_syns]
            tk_syns = [rag_tokenizer.fine_grained_tokenize(s) for s in tk_syns if s]
            tk_syns = [f'"{s}"' if s.find(" ") > 0 else s for s in tk_syns]
            tk = FulltextQueryer.subSpecialChar(tk)
            if tk.find(" ") > 0:
                tk = '"%s"' % tk
            if tk_syns:
                tk = f"({tk} OR (%s)^0.2)" % " ".join(tk_syns)
            if tk:
                keywords.append(f"{tk}^{w}")

        return MatchTextExpr(self.query_fields, " ".join(keywords), 100, {"minimum_should_match": min(3, len(keywords) / 10)})
