"""
RelayLLM Router: SLM-LLM 협업 파이프라인
========================================
가벼운 SLM이 빠른 전처리(번역, 키워드 추출, 의도 분류)를 담당하고,
복잡한 추론만 LLM에게 전달하는 릴레이 방식의 효율적 추론 시스템.

Architecture:
    User Query
        ↓
    [SLM: Dispatcher] ─── 한→영 번역, 키워드 추출, 의도 분류
        ↓
    [Rule-based: Knowledge Fetcher] ─── KnowledgeManager 지식 로드
        ↓
    [Router Decision] ─── simple_lookup → SLM 직접 응답
        │                  complex_reasoning → LLM 전달
        ↓
    [LLM: Reasoning Engine] ─── 심층 추론, 수식 증명 (필요시만)
        ↓
    [SLM: Auditor] ─── 형식 검사, 수치 대조 (Phase 2 고도화 예정)
"""

import json
import re
import logging
import asyncio
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass, field
from enum import Enum
import requests
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    """질문 의도 분류"""
    SIMPLE_LOOKUP = "simple_lookup"          # 단순 정의/수치 조회 → SLM 처리 가능
    PHYSICS_CALCULATION = "physics_calculation"  # 물리 수식 계산/증명 → LLM 필요
    CLINICAL_GUIDELINE = "clinical_guideline"    # 임상 가이드라인 해석 → LLM 필요
    COMPLEX_REASONING = "complex_reasoning"      # 복합 추론 필요 → LLM 필요
    UNKNOWN = "unknown"                          # 분류 불가 → LLM 전달


@dataclass
class DispatchResult:
    """SLM Dispatcher 결과"""
    original_query: str
    translated_query: str
    keywords: List[str]
    intent: QueryIntent
    confidence: float
    can_slm_answer: bool
    slm_answer: Optional[str] = None
    reasoning: str = ""
    # KnowledgeManager 통합
    has_knowledge: bool = False
    knowledge_answer: Optional[str] = None
    knowledge_modules: List[str] = field(default_factory=list)


@dataclass
class RelayConfig:
    """RelayLLM 설정"""
    ollama_url: str = "http://localhost:11434"
    slm_model: str = "glm4:9b"           # 빠른 전처리용
    llm_model: str = "gpt-oss:20b"           # Phase 7.7: GPT-OSS 20B
    slm_timeout: int = 15                     # SLM 타임아웃 (초)
    llm_timeout: int = 90                     # LLM 타임아웃 (초)
    confidence_threshold: float = 0.8         # SLM 직접 응답 신뢰도 임계값
    enable_slm_direct_answer: bool = True     # SLM 직접 응답 활성화


class RelayRouter:
    """
    RelayLLM 라우터: SLM-LLM 협업 관리

    Usage:
        router = RelayRouter()
        result = router.dispatch(query)

        if result.can_slm_answer:
            # SLM이 직접 응답 가능
            answer = result.slm_answer
        else:
            # LLM에게 전달
            answer = router.reason_with_llm(result, context)
    """

    def __init__(self, config: Optional[RelayConfig] = None):
        self.config = config or RelayConfig()
        self._executor = ThreadPoolExecutor(max_workers=2)

    # =========================================================================
    # SLM Dispatcher: 빠른 전처리
    # =========================================================================

    def dispatch(self, query: str) -> DispatchResult:
        """
        SLM을 사용한 빠른 전처리

        1. 한→영 번역
        2. 의학 키워드 추출
        3. 의도 분류
        4. (가능하면) 직접 응답

        Args:
            query: 사용자 질문

        Returns:
            DispatchResult: 전처리 결과
        """
        try:
            # SLM에게 구조화된 JSON 응답 요청
            slm_response = self._call_slm_dispatcher(query)

            if slm_response:
                return self._parse_dispatch_response(query, slm_response)
            else:
                # SLM 실패 시 기본값 반환
                return DispatchResult(
                    original_query=query,
                    translated_query=query,
                    keywords=[query],
                    intent=QueryIntent.UNKNOWN,
                    confidence=0.0,
                    can_slm_answer=False,
                    reasoning="SLM dispatch failed"
                )

        except Exception as e:
            logger.error(f"Dispatch error: {e}")
            return DispatchResult(
                original_query=query,
                translated_query=query,
                keywords=[query],
                intent=QueryIntent.UNKNOWN,
                confidence=0.0,
                can_slm_answer=False,
                reasoning=f"Error: {str(e)}"
            )

    def _call_slm_dispatcher(self, query: str) -> Optional[Dict]:
        """SLM Dispatcher 호출"""

        system_prompt = """You are a medical query analyzer specializing in mammography and breast imaging.

Your task is to analyze the user's question and return a JSON object with the following structure:

{
  "translated_query": "English translation of the query for search",
  "keywords": ["keyword1", "keyword2", "keyword3"],
  "intent": "simple_lookup|physics_calculation|clinical_guideline|complex_reasoning",
  "confidence": 0.0-1.0,
  "can_answer_directly": true|false,
  "direct_answer": "Only if can_answer_directly is true and intent is simple_lookup"
}

Intent Classification Rules:
- simple_lookup: Simple definition or value lookup (e.g., "What is BI-RADS 3?", "SNR formula")
- physics_calculation: Physics formula derivation or calculation (e.g., "Prove MGD formula", "Calculate dose")
- clinical_guideline: Clinical management questions (e.g., "When to biopsy?", "Follow-up interval")
- complex_reasoning: Multi-step reasoning or comparison (e.g., "Compare DBT vs 2D", "Why does X cause Y?")

Confidence Rules:
- 0.9+: Exact match to known medical term/value
- 0.7-0.9: High confidence in classification
- 0.5-0.7: Moderate confidence
- <0.5: Uncertain

can_answer_directly = true ONLY if:
- intent is "simple_lookup"
- confidence >= 0.85
- Answer is a well-known medical fact (BI-RADS categories, standard formulas, etc.)

RESPOND ONLY WITH VALID JSON. NO EXPLANATION."""

        user_prompt = f"Analyze this medical query: {query}"

        try:
            response = requests.post(
                f"{self.config.ollama_url}/api/chat",
                json={
                    "model": self.config.slm_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "stream": False,
                    "format": "json",  # JSON 모드 강제
                    "options": {
                        "num_predict": 500,
                        "temperature": 0.1,  # 낮은 온도로 일관성 확보
                    }
                },
                timeout=self.config.slm_timeout
            )

            response.raise_for_status()
            content = response.json().get("message", {}).get("content", "")

            # JSON 파싱
            return json.loads(content)

        except json.JSONDecodeError as e:
            logger.warning(f"SLM JSON parse error: {e}")
            return None
        except Exception as e:
            logger.error(f"SLM call error: {e}")
            return None

    def _parse_dispatch_response(self, query: str, response: Dict) -> DispatchResult:
        """SLM 응답 파싱"""
        intent_map = {
            "simple_lookup": QueryIntent.SIMPLE_LOOKUP,
            "physics_calculation": QueryIntent.PHYSICS_CALCULATION,
            "clinical_guideline": QueryIntent.CLINICAL_GUIDELINE,
            "complex_reasoning": QueryIntent.COMPLEX_REASONING,
        }

        intent_str = response.get("intent", "unknown").lower()
        intent = intent_map.get(intent_str, QueryIntent.UNKNOWN)

        confidence = float(response.get("confidence", 0.5))
        can_answer = response.get("can_answer_directly", False)

        # SLM 직접 응답 조건 검증
        can_slm_answer = (
            self.config.enable_slm_direct_answer
            and can_answer
            and intent == QueryIntent.SIMPLE_LOOKUP
            and confidence >= self.config.confidence_threshold
        )

        return DispatchResult(
            original_query=query,
            translated_query=response.get("translated_query", query),
            keywords=response.get("keywords", [query]),
            intent=intent,
            confidence=confidence,
            can_slm_answer=can_slm_answer,
            slm_answer=response.get("direct_answer") if can_slm_answer else None,
            reasoning=f"Intent: {intent_str}, Confidence: {confidence:.2f}"
        )

    # =========================================================================
    # LLM Reasoning Engine: 심층 추론
    # =========================================================================

    def reason_with_llm(
        self,
        dispatch_result: DispatchResult,
        context: str,
        physics_knowledge: str = ""
    ) -> str:
        """
        LLM을 사용한 심층 추론

        Args:
            dispatch_result: SLM 전처리 결과
            context: 검색된 문서 컨텍스트
            physics_knowledge: KnowledgeManager에서 로드한 물리 지식

        Returns:
            LLM 응답
        """
        # 의도별 프롬프트 최적화
        system_prompt = self._build_llm_system_prompt(dispatch_result.intent)

        user_prompt = self._build_llm_user_prompt(
            query=dispatch_result.original_query,
            context=context,
            physics_knowledge=physics_knowledge,
            keywords=dispatch_result.keywords
        )

        try:
            response = requests.post(
                f"{self.config.ollama_url}/api/chat",
                json={
                    "model": self.config.llm_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "stream": False,
                    "options": {
                        "num_predict": 2000,
                        "temperature": 0.3,
                    }
                },
                timeout=self.config.llm_timeout
            )

            response.raise_for_status()
            content = response.json().get("message", {}).get("content", "")

            # DeepSeek-R1 <think> 태그 제거
            content = self._clean_llm_response(content)

            return content

        except Exception as e:
            logger.error(f"LLM reasoning error: {e}")
            return f"추론 중 오류가 발생했습니다: {str(e)}"

    def _build_llm_system_prompt(self, intent: QueryIntent) -> str:
        """의도별 LLM 시스템 프롬프트"""

        base_prompt = """당신은 유방영상의학 전문가입니다.
제공된 근거 자료(표준 참조 자료, 가이드라인, 논문)를 바탕으로 정확한 답변을 제공하세요.

중요 규칙:
1. 근거 자료에 명시된 수치와 공식을 정확히 인용하세요.
2. 추측하지 말고, 자료에 없는 내용은 "자료에서 확인되지 않음"이라고 답하세요.
3. 한국어로 답변하세요."""

        intent_additions = {
            QueryIntent.PHYSICS_CALCULATION: """

물리 수식 질문입니다. 다음을 포함하세요:
- 관련 공식의 정확한 표기
- 변수 정의
- 수치 대입 예시 (해당 시)
- 물리적 의미 설명""",

            QueryIntent.CLINICAL_GUIDELINE: """

임상 가이드라인 질문입니다. 다음을 포함하세요:
- 권고 등급 (있는 경우)
- 구체적인 수치 기준
- 예외 상황
- 출처 명시""",

            QueryIntent.COMPLEX_REASONING: """

복합 추론 질문입니다. 다음 형식으로 답변하세요:
1. 핵심 개념 정리
2. 단계별 논리 전개
3. 결론 및 임상적 의의"""
        }

        return base_prompt + intent_additions.get(intent, "")

    def _build_llm_user_prompt(
        self,
        query: str,
        context: str,
        physics_knowledge: str,
        keywords: List[str]
    ) -> str:
        """LLM 사용자 프롬프트 구성"""

        parts = []

        if physics_knowledge:
            parts.append(f"### 표준 참조 자료 (검증된 물리 지식)\n{physics_knowledge}")

        if context:
            parts.append(f"### 검색된 문서\n{context}")

        parts.append(f"### 질문\n{query}")

        if keywords:
            parts.append(f"### 핵심 키워드\n{', '.join(keywords)}")

        return "\n\n".join(parts)

    def _clean_llm_response(self, text: str) -> str:
        """LLM 응답 정리"""
        # DeepSeek-R1 <think> 태그 제거
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        return text.strip()

    # =========================================================================
    # Async Pipeline (비동기 파이프라인)
    # =========================================================================

    async def dispatch_async(self, query: str) -> DispatchResult:
        """비동기 dispatch"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.dispatch, query)

    async def reason_async(
        self,
        dispatch_result: DispatchResult,
        context: str,
        physics_knowledge: str = ""
    ) -> str:
        """비동기 LLM 추론"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self.reason_with_llm,
            dispatch_result,
            context,
            physics_knowledge
        )

    # =========================================================================
    # 편의 메서드
    # =========================================================================

    def should_use_llm(self, dispatch_result: DispatchResult) -> bool:
        """LLM 사용 여부 결정"""
        # SLM이 직접 답변 가능하면 LLM 불필요
        if dispatch_result.can_slm_answer:
            return False

        # 복잡한 의도는 LLM 필요
        complex_intents = {
            QueryIntent.PHYSICS_CALCULATION,
            QueryIntent.CLINICAL_GUIDELINE,
            QueryIntent.COMPLEX_REASONING,
            QueryIntent.UNKNOWN
        }

        return dispatch_result.intent in complex_intents

    def get_model_used(self, dispatch_result: DispatchResult) -> str:
        """사용된 모델 반환 (UI 표시용)"""
        if dispatch_result.has_knowledge:
            return "KnowledgeManager (verified)"
        elif dispatch_result.can_slm_answer:
            return f"SLM ({self.config.slm_model})"
        else:
            return f"LLM ({self.config.llm_model})"

    # =========================================================================
    # KnowledgeManager 통합
    # =========================================================================

    def enrich_with_knowledge(self, dispatch_result: DispatchResult) -> DispatchResult:
        """
        KnowledgeManager를 사용하여 dispatch 결과 보강

        simple_lookup 의도의 경우 KnowledgeManager에 정확한 답변이 있으면
        SLM의 부정확한 응답 대신 검증된 지식을 사용합니다.

        Args:
            dispatch_result: SLM dispatch 결과

        Returns:
            보강된 DispatchResult (knowledge_answer 포함)
        """
        try:
            from src.knowledge.manager import get_knowledge_manager

            km = get_knowledge_manager()
            query = dispatch_result.original_query

            # 관련 지식 모듈 검색 (복합 질문을 위해 3개까지 허용)
            matched_modules = km.get_relevant_knowledge(query, max_modules=3)

            if matched_modules:
                # 지식 모듈 발견
                dispatch_result.has_knowledge = True
                dispatch_result.knowledge_modules = [m.get("id", "unknown") for m in matched_modules]

                # common_questions에서 직접 답변 검색
                direct_answer = self._find_direct_answer_in_knowledge(query, matched_modules)

                if direct_answer:
                    dispatch_result.knowledge_answer = direct_answer
                    # KnowledgeManager에 답변이 있으면 SLM 직접 응답 비활성화
                    # (검증된 지식이 SLM보다 정확)
                    dispatch_result.can_slm_answer = False
                    dispatch_result.slm_answer = None
                    logger.info(f"Knowledge answer found for: {query[:30]}...")

            return dispatch_result

        except Exception as e:
            logger.error(f"Knowledge enrichment error: {e}")
            return dispatch_result

    def _find_direct_answer_in_knowledge(
        self,
        query: str,
        modules: List[Dict]
    ) -> Optional[str]:
        """
        지식 모듈의 common_questions에서 직접 답변 검색

        Args:
            query: 사용자 질문
            modules: 매칭된 지식 모듈들

        Returns:
            직접 답변 (없으면 None)
        """
        query_lower = query.lower()

        # 숫자 및 카테고리 추출 (4A, 4B, 4C, 3, 5 등)
        query_numbers = set(re.findall(r'\d+[abc]?', query_lower))

        best_match = None
        best_score = 0

        for module in modules:
            common_qa = module.get("common_questions", [])

            for qa in common_qa:
                question = qa.get("question", "").lower()
                answer = qa.get("answer", "")

                # 숫자/카테고리 매칭 (필수 조건)
                qa_numbers = set(re.findall(r'\d+[abc]?', question))

                # 숫자가 있는 질문인데 숫자가 일치하지 않으면 스킵
                if query_numbers and qa_numbers:
                    if not query_numbers & qa_numbers:
                        continue

                # 키워드 매칭
                query_keywords = set(re.findall(r'[a-z가-힣]+', query_lower))
                qa_keywords = set(re.findall(r'[a-z가-힣]+', question))

                # 핵심 키워드 교집합
                common_keywords = query_keywords & qa_keywords

                # 점수 계산: 공통 키워드 수 + 숫자 매칭 보너스
                score = len(common_keywords)
                if query_numbers & qa_numbers:
                    score += 5  # 숫자 매칭 보너스

                # 최소 3개 이상 공통 키워드 필요
                if score > best_score and len(common_keywords) >= 2:
                    best_score = score
                    best_match = answer

        return best_match

    def get_best_answer(self, dispatch_result: DispatchResult) -> Tuple[str, str]:
        """
        최적의 답변 반환 (우선순위: Knowledge > SLM)

        Args:
            dispatch_result: 보강된 dispatch 결과

        Returns:
            (answer, source) 튜플
        """
        if dispatch_result.knowledge_answer:
            return (dispatch_result.knowledge_answer, "KnowledgeManager")
        elif dispatch_result.slm_answer:
            return (dispatch_result.slm_answer, "SLM")
        else:
            return ("", "None")


# =============================================================================
# 싱글톤 및 편의 함수
# =============================================================================

_router_instance: Optional[RelayRouter] = None


def get_relay_router(config: Optional[RelayConfig] = None) -> RelayRouter:
    """RelayRouter 싱글톤 인스턴스"""
    global _router_instance
    if _router_instance is None:
        _router_instance = RelayRouter(config)
    return _router_instance


def reset_relay_router():
    """싱글톤 리셋 (테스트용)"""
    global _router_instance
    _router_instance = None


# =============================================================================
# 테스트
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    router = RelayRouter()

    test_queries = [
        "BI-RADS Category 3의 악성 확률은?",              # simple_lookup
        "SNR을 2배로 높이려면 선량을 얼마나 증가해야?",    # physics_calculation
        "DBT에서 T-factor가 0.94인 이유를 설명해",        # complex_reasoning
        "Category 4A 병변의 추적 관찰 주기는?",           # clinical_guideline
    ]

    print("=" * 60)
    print("RelayRouter Test")
    print("=" * 60)

    for query in test_queries:
        print(f"\n질문: {query}")
        print("-" * 40)

        result = router.dispatch(query)

        print(f"  번역: {result.translated_query}")
        print(f"  키워드: {result.keywords}")
        print(f"  의도: {result.intent.value}")
        print(f"  신뢰도: {result.confidence:.2f}")
        print(f"  SLM 직접 응답: {result.can_slm_answer}")
        print(f"  LLM 필요: {router.should_use_llm(result)}")

        if result.slm_answer:
            print(f"  SLM 응답: {result.slm_answer}")
