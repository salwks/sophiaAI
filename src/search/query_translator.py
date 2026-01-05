"""
Query Translator: Korean medical queries → English technical keywords
========================================================================
한글/한영 혼용 의학 질문을 영문 기술 키워드로 변환하여 검색 정확도 향상
"""

import re
import requests
import json
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class QueryTranslator:
    """LLM 기반 의학 쿼리 번역기"""

    def __init__(self, ollama_url: str = "http://localhost:11434", model: str = "qwen2.5:14b"):
        self.ollama_url = ollama_url
        self.model = model

    def needs_translation(self, query: str) -> bool:
        """한글이 포함되어 있으면 번역 필요"""
        return bool(re.search(r'[가-힣]', query))

    def translate(self, query: str) -> str:
        """
        한글 의학 질문을 영문 기술 키워드로 변환

        Args:
            query: 사용자 질문 (한글 또는 한영 혼용)

        Returns:
            영문 기술 키워드 (검색용)
        """
        if not self.needs_translation(query):
            # 이미 영어면 그대로 반환
            return query

        try:
            # 원본 쿼리에서 숫자 추출 (safety net)
            original_numbers = re.findall(r'\d+', query)

            # LLM으로 번역
            translated = self._translate_with_llm(query)

            if translated and len(translated.strip()) > 0:
                # 숫자 보존 검증 및 복구
                translated = self._preserve_numbers(translated, original_numbers)

                logger.info(f"Query translated: '{query}' → '{translated}'")
                return translated
            else:
                logger.warning(f"Translation empty, using original: {query}")
                return query

        except Exception as e:
            logger.error(f"Translation failed: {e}, using original query")
            return query

    def _translate_with_llm(self, query: str) -> Optional[str]:
        """LLM을 사용한 쿼리 번역"""

        system_prompt = """You are a medical terminology translator specializing in breast imaging.

Your task: Convert Korean medical questions into English technical search keywords.

Rules:
1. Extract medical concepts and translate to English medical terms
2. Include relevant technical keywords (kVp, mAs, BI-RADS, etc.)
3. Keep it concise - only essential search keywords
4. DO NOT include explanations, just keywords
5. Remove question words (what, how, when, etc.)
6. **CRITICAL: ALWAYS preserve numbers from the original query**
   - "4가지" → "4"
   - "3개" → "3"
   - "2가지 차이" → "2 differences"

Examples:
Input: "유방촬영 노출 기법을 설명해줘"
Output: mammography exposure technique kVp mAs radiation dose

Input: "BI-RADS 카테고리 5는 무엇인가요?"
Output: BI-RADS Category 5 definition malignancy

Input: "맘모그래피 포지셔닝 방법"
Output: mammography positioning technique procedure

Input: "DBT와 일반 맘모그래피 차이"
Output: DBT digital breast tomosynthesis mammography difference comparison

Input: "mass의 margin 분류 4가지는 무엇인가요?"
Output: mass margin 4 classifications

Input: "석회화의 3가지 형태는?"
Output: calcification 3 types morphology"""

        user_prompt = f"""Convert this Korean medical question to English search keywords:

Question: {query}

Search keywords:"""

        try:
            response = requests.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # 낮은 온도로 일관된 번역
                        "num_predict": 50,   # 짧은 키워드만 생성
                    }
                },
                timeout=15
            )

            response.raise_for_status()
            result = response.json()

            translated = result.get("message", {}).get("content", "").strip()

            # 후처리: 불필요한 문장 제거
            translated = self._clean_translation(translated)

            return translated

        except Exception as e:
            logger.error(f"LLM translation error: {e}")
            return None

    def _preserve_numbers(self, translated: str, original_numbers: list) -> str:
        """
        번역 결과에 원본 쿼리의 숫자가 보존되었는지 확인하고 필요시 추가

        Args:
            translated: 번역된 쿼리
            original_numbers: 원본 쿼리에서 추출한 숫자 리스트

        Returns:
            숫자가 보존된 번역 결과
        """
        if not original_numbers:
            return translated

        # 번역된 쿼리에 원본 숫자가 있는지 확인
        for num in original_numbers:
            if num in translated:
                # 이미 숫자가 있으면 그대로 반환
                return translated

        # 숫자가 누락된 경우 추가
        logger.warning(f"Number '{original_numbers[0]}' missing in translation, adding it")

        # classification, type, category 등의 단어 앞에 숫자 삽입
        for keyword in ['classification', 'type', 'category', 'morphology', 'descriptor', 'difference']:
            pattern = f'({keyword})'
            if re.search(pattern, translated, re.IGNORECASE):
                translated = re.sub(
                    pattern,
                    f'{original_numbers[0]} \\1',
                    translated,
                    count=1,
                    flags=re.IGNORECASE
                )
                logger.info(f"Inserted number '{original_numbers[0]}' before '{keyword}'")
                return translated

        # 키워드를 찾지 못한 경우 끝에 추가
        translated = f"{translated} {original_numbers[0]}"
        logger.info(f"Appended number '{original_numbers[0]}' to end of query")

        return translated

    def _clean_translation(self, text: str) -> str:
        """번역 결과 정리"""
        # 줄바꿈 제거
        text = text.replace('\n', ' ')

        # 여러 공백을 하나로
        text = re.sub(r'\s+', ' ', text)

        # "Output:", "Keywords:" 등 제거
        text = re.sub(r'^(Output|Keywords|Search keywords|Result):\s*', '', text, flags=re.IGNORECASE)

        # 따옴표 제거
        text = text.strip('"\'')

        # 문장 부호 정리
        text = re.sub(r'[.!?]+$', '', text)

        return text.strip()


# 싱글톤 인스턴스
_translator_instance = None

def get_translator(ollama_url: str = "http://localhost:11434", model: str = "qwen2.5:14b") -> QueryTranslator:
    """번역기 싱글톤 인스턴스"""
    global _translator_instance
    if _translator_instance is None:
        _translator_instance = QueryTranslator(ollama_url, model)
    return _translator_instance

def reset_translator():
    """싱글톤 인스턴스 리셋 (테스트/디버깅용)"""
    global _translator_instance
    _translator_instance = None
    logger.info("Translator instance reset")
