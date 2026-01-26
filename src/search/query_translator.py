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

    def __init__(self, ollama_url: str = "http://localhost:11434", model: str = "gpt-oss:20b"):
        """Phase 7.7: GPT-OSS 20B (번역 + 답변 모두)"""
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
        """LLM을 사용한 쿼리 번역 (DeepSeek-R1 3단계 사고 모드)"""

        # 3단계 사고 과정: 의도분석 → 개념확장 → 표준용어매핑
        system_prompt = """# Role
You are a Mammography & Medical Physics specialist librarian and search strategist.
Analyze user questions to generate English queries optimized for ACR BI-RADS guidelines and research papers.

# Mission
1. Distinguish if the question is 'Clinical guideline' or 'Physics principle'
2. Recognize that terms may be expressed differently in guidelines - derive synonyms and technical parent concepts
3. For physics topics like 'K-edge', expand to filter materials (Mo, Rh, Ag), Energy Spectrum, Dose Optimization

# Operational Logic (Inside <think>)
1. **Analyze Intent**: What is the core intent? (specific value, physics mechanism, patient management)
2. **Concept Mapping**:
   - If abstract (e.g., K-edge) → convert to technical terms (Filtration, Characteristic X-ray, Attenuation)
   - Consider both guideline keywords (standardized) AND paper keywords (technical/academic)
3. **Redundancy Check**: Build 3-5 keyword combinations for high recall

# Constraints
- Final output MUST be comma-separated English keywords ONLY
- NO explanations outside <think> block
- Do NOT force-match concepts that don't exist in guidelines (honest search)

# Examples
- "K-edge 필터링이 대조도에 미치는 영향"
  → K-edge filtration, Contrast-to-noise ratio CNR, X-ray energy spectrum, Molybdenum Rhodium filter, mammography image quality

- "7cm 두께 유방의 MGD 효율"
  → breast thickness 7cm, mean glandular dose MGD, Tungsten Silver target filter, dose efficiency, extremely dense breast"""

        user_prompt = f"Convert to search keywords: {query}"

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
                        "num_predict": 800,   # 사고 과정 충분히 + 간결한 출력
                        "temperature": 0.15,  # 창의적 보수성 (0.0은 경직, 0.3은 과다)
                        "top_p": 0.3,         # 동의어 후보군 적절히 확보
                    }
                },
                timeout=45
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
        """번역 결과 정리 (DeepSeek-R1 최적화)"""
        # 1. DeepSeek-R1 <think> 태그와 내용 완전 제거
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

        # 2. 마지막 줄만 추출 (DeepSeek-R1은 마지막에 결과 출력)
        lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
        if lines:
            # 마지막 비어있지 않은 줄 사용
            text = lines[-1]

        # 3. "Output:", "Keywords:", "Search keywords:" 등 레이블 제거
        text = re.sub(r'^(Output|Keywords|Search keywords?|Result|English keywords?|Translation):\s*', '', text, flags=re.IGNORECASE)

        # 4. 따옴표 제거
        text = text.strip('"\'`')

        # 5. 문장 부호 정리
        text = re.sub(r'[.!?]+$', '', text)

        # 6. 여러 공백을 하나로
        text = re.sub(r'\s+', ' ', text)

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
