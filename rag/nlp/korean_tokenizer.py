"""
kiwipiepy를 사용한 한국어 토크나이저
범용 한국어 토큰 추출기
"""

from kiwipiepy import Kiwi
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from api.utils.file_utils import get_project_base_directory

class KoreanTokenizer:
    def __init__(self):
        self.kiwi = Kiwi()
    
    def extract_tokens_with_position(self, text):
        """텍스트에서 내용어(명사) 토큰과 위치 정보 추출

        동사 및 형용사 관련 태그는 더 이상 추출하지 않습니다.
        """
        try:
            analysis = self.kiwi.analyze(text)
            if analysis and analysis[0]:
                tokens_with_pos = []
                for token in analysis[0][0]:
                    # 명사류 + 영어 단어/약어 + 숫자 포함 (동사/형용사는 제외)
                    if token.tag in ('NNG', 'NNP', 'NNB', 'SL', 'SN'):
                        # NNG=일반명사, NNP=고유명사, NNB=의존명사
                        # VV=동사, VA=형용사, VX=보조용언
                        # SL=영어, SN=숫자
                        tokens_with_pos.append({
                            'form': token.form,
                            'start': token.start,
                            'len': token.len,
                            'end': token.start + token.len
                        })
                return tokens_with_pos
        except Exception as e:
            print(f"토큰 추출 오류: {e}")
        return []
    
    def extract_tokens(self, text):
        """텍스트에서 내용어 토큰 추출 (기존 호환성 유지)"""
        tokens_with_pos = self.extract_tokens_with_position(text)
        return [token['form'] for token in tokens_with_pos]
    
    def extract_keywords_by_spacing(self, text):
        """형태소 분석 후 원문 띄어쓰기 기준으로 복합어/단일어 구분"""
        try:
            analysis = self.kiwi.analyze(text)
            if not analysis or not analysis[0]:
                return []
            
            # 1. 형태소 분석으로 내용어와 위치 정보 추출 (명사, 영어, 숫자)
            tokens = []
            for token in analysis[0][0]:
                # 명사류 + 영어 단어/약어 + 숫자 포함 (동사/형용사는 제외)
                if token.tag in ('NNG', 'NNP', 'NNB', 'SL', 'SN'):
                    tokens.append({
                        'form': token.form,
                        'start': token.start,
                        'end': token.start + token.len,
                        'original_pos': token.start,  # 원문에서의 위치
                        'tag': token.tag  # 품사 태그 추가
                    })
            
            if not tokens:
                return []
            
            #print(f"추출된 토큰들: {[(t['form'], t['tag'], t['start'], t['end']) for t in tokens]}")
            
            # 2. 원문에서 띄어쓰기 기준으로 그룹화
            result_keywords = []
            i = 0
            
            while i < len(tokens):
                current_group = [tokens[i]]
                current_end = tokens[i]['end']
                
                # 다음 토큰들과 연속성 체크
                j = i + 1
                while j < len(tokens):
                    next_token = tokens[j]
                    
                    # 원문에서 띄어쓰기 확인: 토큰 사이에 공백이 있는지 체크
                    between_text = text[current_end:next_token['start']]
                    has_space = ' ' in between_text or '\t' in between_text or '\n' in between_text
                    
                    # 공백이 없고 바로 인접한 경우 (복합어)
                    if not has_space and (next_token['start'] - current_end <= 1):
                        current_group.append(next_token)
                        current_end = next_token['end']
                        j += 1
                    else:
                        break
                # 3. 그룹 처리
                group_tokens = [token['form'] for token in current_group]
                added = set()
                if len(current_group) > 1:
                    # 복합어 생성
                    compound_token = ''.join(group_tokens)
                    if self.filter_relevant_tokens([compound_token]):
                        result_keywords.append(compound_token)
                        added.add(compound_token)
                # 개별 명사도 복합어 뒤에 추가
                for t in group_tokens:
                    if t not in added and self.filter_relevant_tokens([t]):
                        result_keywords.append(t)
                        added.add(t)

                i = j if j > i else i + 1

            # 중복 제거(순서 유지)
            seen = set()
            final_keywords = []
            for kw in result_keywords:
                if kw not in seen:
                    final_keywords.append(kw)
                    seen.add(kw)
            return final_keywords                
        except Exception as e:
            print(f"토큰화 오류: {e}")
            return []
    
    
    def combine_compound_tokens(self, tokens):
        """기존 호환성을 위해 유지 (더 이상 사용하지 않음, 현재는 내용어 전체 처리)"""
        return [], tokens
    
    def filter_relevant_tokens(self, tokens):
        """유용한 내용어 토큰만 필터링 (명사, 영어 단어/약어/숫자 포함)

        동사/형용사 관련 불용어는 더 이상 포함하지 않습니다.
        """
        # 기본적인 불용어나 너무 일반적인 단어들 제외 (대명사, 지시어, 의문사 위주)
        basic_stopwords = {
            '것', '때', '경우', '후', '전', '중', '내', '외', '등',
            '어떻게', '무엇', '언제', '어디서', '누구', '왜', '어떤', '몇',
            '이것', '저것', '그것', '여기', '저기', '거기', '이렇게', '저렇게', '그렇게'
        }
        
        filtered_tokens = []
        for token in tokens:
            # 불용어 체크
            if token in basic_stopwords:
                continue
            
            # 영어 단어/약어 체크 (영어가 포함된 경우 더 관대하게)
            has_english = any(c.isascii() and c.isalpha() for c in token)
            # 숫자 체크
            has_digit = any(c.isdigit() for c in token)
            
            is_relevant = False
            
            if has_english:
                # 영어가 포함된 경우: 1글자 이상이면 허용 (약어 포함)
                if len(token) >= 1 and any(c.isalpha() for c in token):
                    is_relevant = True
            
            # 숫자가 포함된 경우 (순수 숫자 또는 숫자+단위)
            if has_digit:
                # 순수 숫자인 경우: 1자리 이상이면 허용
                if token.isdigit():
                    is_relevant = True
                # 숫자+한글 조합 (예: "20일", "30년", "100개")
                elif any(c.isdigit() for c in token) and any(c.isalpha() for c in token):
                    is_relevant = True
            
            # 순수 한글인 경우: 기존 규칙 적용
            if not has_english and not has_digit:
                # 너무 짧은 단어 제외 (1글자)
                if len(token) > 1:
                    # 알파벳이 하나라도 있는 경우 허용
                    if any(c.isalpha() for c in token):
                        is_relevant = True
            
            # 유용성 체크 통과시 추가
            if is_relevant:
                filtered_tokens.append(token)
                
        return filtered_tokens
    
    def tokenize(self, text):
        """텍스트 토큰화 메인 함수 (extract_keywords에서 변경)"""
        #print(f"입력 텍스트: {text}")
        
        # 새로운 방식: 형태소 분석 + 원문 띄어쓰기 기준
        extracted_tokens = self.extract_keywords_by_spacing(text)
        #print(f"추출된 토큰: {extracted_tokens}")
        
        # 관련성 필터링 (이미 extract_keywords_by_spacing에서 수행되지만 추가 필터링)
        final_tokens = self.filter_relevant_tokens(extracted_tokens)
        #print(f"필터링된 최종 토큰: {final_tokens}")
        
        # 중복 제거 및 순서 유지
        unique_tokens = []
        for token in final_tokens:
            if token not in unique_tokens:
                unique_tokens.append(token)
        
        return unique_tokens


def main():
    # 한국어 토크나이저 초기화
    tokenizer = KoreanTokenizer()
    
    # 테스트 질문들 (명사, 동사, 형용사, 영어 단어/약어/숫자+단위 포함)
    test_questions = [
        "주총 20일 전 공시는 어떤 부서에서 주관하며, 담당 부서는 무엇인가요?",
        "지체 없이 공시해야 하는 사항은 어떤 부서에서 주관하며, 관련 법규는 무엇인가요?",
        "IT 사업을 추진하는 과정에서 정보보호팀은 어떤 역할을 하나요?",
        "CCM 매뉴얼을 작성하는 주요 목적은 무엇인가요?",
        "30년 만기 상품을 개발할 때 고려해야 할 중요한 사항은?",
        "계약을 체결한 후 10일 이내에 해지할 수 있나요?",
        "신속하게 처리해야 하는 업무는 어떤 것들이 있나요?",
        "복잡한 법률 문제를 해결하는 방법은 무엇인가요?",
        "복잡한 법률 문제를 해결하는 먼저 움직여야 하는 팀은 어디인가요?",
    ]
    
    print("=" * 80)
    print("kiwipiepy를 사용한 한국어 토크나이저 테스트 (명사/동사/형용사/영어/숫자)")
    print("=" * 80)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n[테스트 {i}]")
        tokens = tokenizer.tokenize(question)
        print(f"최종 토큰: {', '.join(tokens)}")
        print("-" * 60)
    
    # 추가 테스트: 사용자 입력
    print("\n" + "=" * 80)
    print("대화형 테스트 (종료하려면 'quit' 입력)")
    print("=" * 80)
    
    while True:
        user_input = input("\n텍스트를 입력하세요: ").strip()
        if user_input.lower() in ['quit', 'exit', '종료']:
            break
        if user_input:
            tokens = tokenizer.tokenize(user_input)
            print(f"추출된 토큰: {', '.join(tokens)}")


if __name__ == "__main__":
    main()
