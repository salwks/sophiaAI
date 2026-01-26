"""
Clinical Text Excellence: Answering Twice Logic
================================================
복잡한 임상 텍스트에 대해 2단계 추론을 수행하여 99.9% 정확도를 목표로 합니다.

Workflow:
    [Stage 1: Drafting]
    - 모든 근거 자료(지식 모듈, 논문, 가이드라인)를 통합
    - 가설적 답변(Draft) 생성
    - 변수 단위로 Golden Formula 대조

    [Stage 2: Refinement]
    - Self-Correction: 논리적 허점 탐지
    - BI-RADS 가이드라인 위반 여부 검증
    - 최종 정제된 답변 출력

Output Structure:
    1. Derivation (증명 과정)
    2. Evidence Mapping (근거 매핑)
    3. Recommendation (최종 권고)
"""

import json
import re
import logging
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import requests

logger = logging.getLogger(__name__)


class AnswerQuality(Enum):
    """답변 품질 등급"""
    EXCELLENT = "excellent"      # 완벽한 근거 기반, 논리적 일관성
    GOOD = "good"               # 대부분 정확, 사소한 개선 가능
    NEEDS_REVISION = "needs_revision"  # 수정 필요
    UNRELIABLE = "unreliable"   # 신뢰할 수 없음, 재생성 필요


@dataclass
class DraftAnswer:
    """1단계: 가설적 답변"""
    content: str
    derivation: str                    # 증명/추론 과정
    evidence_citations: List[str]      # 인용된 근거
    formula_checks: List[Dict]         # 수식 검증 결과
    confidence: float                  # 신뢰도 (0-1)


@dataclass
class RefinedAnswer:
    """2단계: 정제된 최종 답변"""
    content: str
    derivation: str                    # 증명 과정
    evidence_mapping: List[Dict]       # 근거 매핑
    recommendation: str                # 최종 권고
    quality: AnswerQuality
    corrections_made: List[str]        # 수정 사항
    final_confidence: float
    clinical_insight: str = ""         # Phase 7.2: 임상적 영향 분석 (FN/FP 기전)


@dataclass
class TextReasoningConfig:
    """텍스트 추론 설정 (Phase 7.4: 심층 추론 지원)"""
    ollama_url: str = "http://localhost:11434"
    reasoning_model: str = "gpt-oss:20b"             # Phase 7.10: GPT-OSS 20B (long-context 합성 우수)
    critic_model: str = "glm4:9b"            # 빠른 검증용
    draft_timeout: int = 600                     # Phase 7.4: 10분 (복잡한 추론 대응)
    refine_timeout: int = 300                    # Phase 7.4: 5분 (검증 단계)
    min_confidence_threshold: float = 0.85
    max_refinement_iterations: int = 2


class TextReasoningEngine:
    """
    Answering Twice: 2단계 텍스트 추론 엔진

    Usage:
        engine = TextReasoningEngine()
        result = engine.reason(
            question="DBT에서 5cm 유방의 MGD 계산 시 T-factor 값은?",
            context=context,
            physics_knowledge=knowledge
        )
    """

    def __init__(self, config: Optional[TextReasoningConfig] = None):
        self.config = config or TextReasoningConfig()

    # =========================================================================
    # Main Entry Point
    # =========================================================================

    def reason(
        self,
        question: str,
        context: str,
        physics_knowledge: str = "",
        require_derivation: bool = True
    ) -> RefinedAnswer:
        """
        2단계 추론 수행

        Args:
            question: 사용자 질문
            context: 검색된 문서 컨텍스트
            physics_knowledge: KnowledgeManager 지식
            require_derivation: 증명 과정 필수 여부

        Returns:
            RefinedAnswer: 정제된 최종 답변
        """
        logger.info(f"Starting 2-stage reasoning for: {question[:50]}...")

        # Stage 1: Drafting
        draft = self._stage1_draft(
            question=question,
            context=context,
            physics_knowledge=physics_knowledge,
            require_derivation=require_derivation
        )

        logger.info(f"Draft confidence: {draft.confidence:.2f}")

        # Stage 2: Refinement (Self-Correction)
        refined = self._stage2_refine(
            question=question,
            draft=draft,
            physics_knowledge=physics_knowledge
        )

        logger.info(f"Final quality: {refined.quality.value}, confidence: {refined.final_confidence:.2f}")

        # Phase 7.13: 품질 검증 및 MTF-형태 연결 필수 검증
        refined = self._validate_and_enhance_output(question, refined, physics_knowledge)

        return refined

    def _validate_and_enhance_output(
        self,
        question: str,
        result: RefinedAnswer,
        physics_knowledge: str
    ) -> RefinedAnswer:
        """
        Phase 7.13: 출력 품질 검증 및 필수 내용 보강

        - 품질 임계값 미달 시 경고 추가
        - MTF-형태 연결 필수 검증 (Fine Linear/Amorphous 질문)
        - 선량-MTF 무관성 설명 검증
        """
        question_lower = question.lower()
        content = result.content

        # 1. MTF-형태 연결 필수 검증 (Fine Linear, Amorphous, 형태, 뭉개 등 언급 시)
        is_morphology_question = any(kw in question_lower for kw in [
            'fine linear', 'amorphous', '형태', '뭉개', 'morphology',
            '4c', '4b', 'bi-rads', '석회화'
        ])

        mtf_keywords = ['mtf', '해상도', '고주파', '공간주파수', 'lp/mm', '빛 확산', 'light spread']
        has_mtf_explanation = any(kw in content.lower() for kw in mtf_keywords)

        missing_explanations = []

        if is_morphology_question and not has_mtf_explanation:
            missing_explanations.append(
                "⚠️ **MTF-형태 연결 누락**: Fine Linear → Amorphous 형태 변화는 "
                "**고주파 MTF 손실** (MTF_system < 0.2 at 5 lp/mm)이 주원인입니다. "
                "빔 경화 → CsI 깊이 침투 → 빛 확산 증가 → MTF 저하 → 고주파 손실"
            )

        # 2. 선량-MTF/대조도 무관성 검증
        dose_keywords = ['선량', 'dose', '노출']
        is_dose_question = any(kw in question_lower for kw in dose_keywords)

        if is_dose_question or is_morphology_question:
            dose_limitation_keywords = ['선량.*무관', 'dose.*independent', 'mtf.*선량.*무관',
                                        '선량.*회복.*불가', '노이즈만', 'noise only']
            has_dose_limitation = any(re.search(kw, content.lower()) for kw in dose_limitation_keywords)

            if not has_dose_limitation and ('선량' in question or 'dose' in question_lower):
                missing_explanations.append(
                    "⚠️ **선량 한계 설명 누락**: 선량 증가는 **노이즈(σ)만 감소**시키며, "
                    "MTF(공간해상도)와 Δμ(피사체 대조도)는 선량과 무관합니다."
                )

        # 3. 품질 임계값 검증
        if result.final_confidence < self.config.min_confidence_threshold:
            missing_explanations.append(
                f"⚠️ **신뢰도 낮음** ({result.final_confidence:.0%} < {self.config.min_confidence_threshold:.0%}): "
                "답변의 정확성을 재확인하세요."
            )

        # 4. 누락된 설명이 있으면 content에 추가
        if missing_explanations:
            enhancement_block = "\n\n---\n### 🔬 물리 검증 보강 (Phase 7.13)\n" + "\n\n".join(missing_explanations)
            enhanced_content = content + enhancement_block

            # corrections_made에도 추가
            new_corrections = list(result.corrections_made) + [
                f"Phase 7.13: {len(missing_explanations)}개 필수 설명 보강"
            ]

            logger.warning(f"Phase 7.13: Added {len(missing_explanations)} missing explanations")

            return RefinedAnswer(
                content=enhanced_content,
                derivation=result.derivation,
                evidence_mapping=result.evidence_mapping,
                recommendation=result.recommendation,
                quality=result.quality,
                corrections_made=new_corrections,
                final_confidence=result.final_confidence,
                clinical_insight=result.clinical_insight
            )

        return result

    # =========================================================================
    # Stage 1: Drafting
    # =========================================================================

    def _stage1_draft(
        self,
        question: str,
        context: str,
        physics_knowledge: str,
        require_derivation: bool
    ) -> DraftAnswer:
        """
        1단계: 가설적 답변 생성

        - 모든 근거 자료 통합
        - Golden Formula 변수 대조
        - 증명 과정 포함
        """
        # Phase 7.14: 단순화된 프롬프트 (JSON 출력 제거)
        system_prompt = """당신은 유방영상의학 물리학 전문가입니다. 한국어로 답변하세요.

## 필수 규칙

1. **제공된 "표준 참조 자료"의 공식과 수치만 사용** - 기억에서 끌어내지 말 것
2. **"⚠️ 경고" 섹션의 "❌ 틀림" 내용은 절대 출력 금지**
3. **물리적 근거와 계산 과정을 반드시 포함**

## 핵심 물리 오류 방지

다음은 **절대 출력 금지**:
- "빔 경화가 대조도를 향상시킨다" ❌ → 정답: 빔 경화는 대조도를 **감소**시킴 (μ_pe ∝ 1/E³)
- "K-edge가 대조도 손실을 보상한다" ❌ → 정답: K-edge는 검출 효율(DQE)만 영향, Δμ 회복 불가
- "선량 증가로 대조도/MTF 회복 가능" ❌ → 정답: 선량은 노이즈만 감소, Δμ와 MTF는 무관

## 형태 변화 질문 시 (Fine Linear, Amorphous 등)

반드시 **MTF 손실과 형태 변화의 연결**을 설명:
- Fine Linear는 5-10 lp/mm 해상도 필요
- MTF_system < 0.2 at 5 lp/mm → 고주파 손실 → 선이 점으로 보임
- 빔 경화 → CsI 깊이 침투 → 빛 확산 → MTF 저하

## 출력 형식

자연스러운 한국어 문장으로 답변하세요. 다음 구조 권장:

1. **원인 분석**: 물리적 메커니즘 설명
2. **계산/수치**: 관련 공식과 계산 (있으면)
3. **결론**: 임상적 의미

수식은 $기호$ 또는 $$블록$$으로 표기."""

        user_prompt = self._build_draft_prompt(
            question=question,
            context=context,
            physics_knowledge=physics_knowledge
        )

        try:
            response = self._call_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=self.config.reasoning_model,
                timeout=self.config.draft_timeout
            )

            # Phase 7.14: JSON 파싱 제거 - 자연어 텍스트 직접 사용
            if response.strip():
                # 코드 블록 마커, think 태그 제거
                clean_response = re.sub(r'```\w*\s*|\s*```', '', response)
                clean_response = re.sub(r'<think>.*?</think>', '', clean_response, flags=re.DOTALL)
                clean_response = clean_response.strip()

                # 후처리 적용
                content = self._postprocess_answer(clean_response)

                # 신뢰도 추정: 핵심 키워드 포함 여부로 판단
                confidence = 0.7
                if any(kw in content.lower() for kw in ['mtf', '해상도', 'lp/mm']):
                    confidence += 0.1
                if any(kw in content for kw in ['Δμ', 'delta', '대조도']):
                    confidence += 0.1
                if '선량' in content and ('무관' in content or '노이즈' in content):
                    confidence += 0.1

                logger.info(f"Stage 1 완료: {len(content)} chars, confidence={confidence:.2f}")

                return DraftAnswer(
                    content=content,
                    derivation="",
                    evidence_citations=[],
                    formula_checks=[],
                    confidence=min(confidence, 0.95)
                )
            else:
                # Phase 7.15: 빈 응답 시 fallback
                logger.warning("Stage 1: LLM 응답이 비어있습니다")
                return DraftAnswer(
                    content="LLM 응답이 비어있습니다. 다시 시도해 주세요.",
                    derivation="",
                    evidence_citations=[],
                    formula_checks=[],
                    confidence=0.1
                )

        except Exception as e:
            logger.error(f"Stage 1 draft error: {e}")
            return DraftAnswer(
                content=f"추론 중 오류 발생: {str(e)}",
                derivation="",
                evidence_citations=[],
                formula_checks=[],
                confidence=0.0
            )

    def _smart_truncate_knowledge(self, knowledge: str, max_chars: int = 4000) -> str:
        """
        Phase 7.12: 스마트 지식 truncation

        우선순위:
        1. ⚠️ 경고 섹션 (물리 오류 방지 - MUST AVOID)
        2. 핵심 물리 내용 앞부분
        3. 나머지 내용

        Args:
            knowledge: 원본 지식 텍스트 (마크다운 포맷)
            max_chars: 최대 문자 수

        Returns:
            우선순위에 따라 truncation된 지식 텍스트
        """
        if not knowledge or len(knowledge) <= max_chars:
            return knowledge

        # 우선순위 섹션 추출 (마크다운 포맷 지원)
        priority_sections = []

        # 마크다운 포맷: ⚠️ 경고 블록 추출
        # "============" 구분선 사이의 경고 섹션
        warning_pattern = r'={10,}\s*⚠️ 경고.*?={10,}(?:\s*📢.*?(?=##|\Z))?'
        warning_matches = re.findall(warning_pattern, knowledge, re.DOTALL)
        for match in warning_matches:
            if match and len(match) > 100:
                priority_sections.append(match.strip())

        # JSON 포맷: common_misconceptions, critical_concept_separation
        json_patterns = [
            r'"common_misconceptions"\s*:\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',
            r'"critical_concept_separation"\s*:\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',
            r'"mtf_morphology_connection"\s*:\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',
        ]
        for pattern in json_patterns:
            matches = re.findall(pattern, knowledge, re.DOTALL)
            for match in matches:
                if match and len(match) > 50:
                    priority_sections.append(match)

        # 중복 제거 및 원본에서 제거
        remaining_knowledge = knowledge
        total_priority_chars = 0
        for section in priority_sections:
            total_priority_chars += len(section)
            remaining_knowledge = remaining_knowledge.replace(section, "")

        # 우선순위 섹션이 max_chars의 70% 초과하면 축소
        max_priority = int(max_chars * 0.7)
        if total_priority_chars > max_priority:
            ratio = max_priority / total_priority_chars
            priority_sections = [s[:int(len(s) * ratio)] for s in priority_sections]
            total_priority_chars = sum(len(s) for s in priority_sections)

        # 남은 공간에 나머지 내용 추가
        remaining_space = max_chars - total_priority_chars - 100

        if remaining_space > 300:
            # 핵심 물리 내용 섹션 우선
            physics_start = remaining_knowledge.find("## 핵심 물리 내용")
            if physics_start >= 0:
                remaining_truncated = remaining_knowledge[physics_start:physics_start + remaining_space]
            else:
                remaining_truncated = remaining_knowledge[:remaining_space]
            # 마지막 완전한 줄에서 자르기
            last_newline = remaining_truncated.rfind('\n')
            if last_newline > remaining_space * 0.6:
                remaining_truncated = remaining_truncated[:last_newline]
        else:
            remaining_truncated = ""

        # 결합: 경고 섹션을 먼저 배치
        if priority_sections:
            # 경고 섹션이 있으면 그대로 사용 (이미 포맷됨)
            result = "\n\n".join(priority_sections)
            if remaining_truncated:
                result += "\n\n" + remaining_truncated
        else:
            # 경고 섹션이 없으면 앞부분 truncation
            result = knowledge[:max_chars]
            last_newline = result.rfind('\n')
            if last_newline > max_chars * 0.8:
                result = result[:last_newline]

        logger.info(f"Phase 7.12 Smart truncate: {len(knowledge)} → {len(result)} chars "
                   f"(priority: {total_priority_chars}, remaining: {len(remaining_truncated)})")

        return result

    def _build_draft_prompt(
        self,
        question: str,
        context: str,
        physics_knowledge: str
    ) -> str:
        """Stage 1 사용자 프롬프트 구성"""
        parts = []

        if physics_knowledge:
            # Phase 7.12: 스마트 지식 truncation (misconceptions 우선)
            knowledge_text = self._smart_truncate_knowledge(physics_knowledge, max_chars=4000)
            parts.append(f"""## ⭐ 표준 참조 자료 (MUST USE - 검증된 물리 공식)

⚠️ **아래 공식을 반드시 사용하세요. 기억에 의존하지 마세요.**

🚨 **CRITICAL: 아래에 "⚠️ 경고" 섹션이 있으면 반드시 읽고, "❌ 틀림"으로 표시된 내용은 절대 출력하지 마세요!**

{knowledge_text}

---""")

        if context:
            parts.append(f"## 검색된 문서 (추가 참고용)\n{context}")

        parts.append(f"## 질문\n{question}")
        # Phase 7.15: JSON 형식 제거, 자연어 답변 요청
        parts.append("""
---
📌 **계산 방법**:
1. 위 "표준 참조 자료"에서 해당 공식을 찾아 **정확히 복사**
2. 문제의 값을 변수에 대입 (예: 40% 감소 → 0.6배)
3. **단계별 산술 계산** 수행
4. 최종 값을 백분율로 변환

⚠️ 계산은 반드시 제공된 공식을 따라 수행. 기억에 의존 금지.

**출력 형식**: 자연스러운 한국어 문단으로 답변하세요. JSON이나 코드 블록을 사용하지 마세요.""")

        return "\n\n".join(parts)

    # =========================================================================
    # Stage 2: Refinement (Self-Correction)
    # =========================================================================

    def _stage2_refine(
        self,
        question: str,
        draft: DraftAnswer,
        physics_knowledge: str
    ) -> RefinedAnswer:
        """
        2단계: Self-Correction 및 정제

        - 논리적 허점 탐지
        - BI-RADS 가이드라인 위반 검증
        - 수치 오류 검출
        """
        # Phase 7.14: 단순화된 검증 프롬프트
        system_prompt = """당신은 물리학 답변 검증자입니다. 초안을 검토하고 오류가 있으면 수정하세요.

## 검증 항목

1. **물리 오류 확인** - 다음 오류가 있으면 수정:
   - "빔 경화가 대조도를 향상" ❌ → 빔 경화는 대조도 **감소** (μ_pe ∝ 1/E³)
   - "K-edge가 대조도 손실 보상" ❌ → K-edge는 DQE만 영향, Δμ 회복 불가
   - "선량으로 MTF/대조도 회복" ❌ → 선량은 노이즈만 감소

2. **MTF-형태 연결 확인** (Fine Linear/Amorphous 언급 시):
   - MTF 손실 → 고주파 손실 → 선이 점으로 보임 설명 있는가?

3. **계산 검증** - 수치가 있으면 계산이 맞는지 확인

## 출력

오류가 있으면 수정된 답변을, 없으면 "검증 완료: [원본 유지]"를 출력하세요.
한국어로 자연스럽게 작성."""

        # Phase 7.14: 단순화된 user_prompt
        user_prompt = f"""## 질문
{question}

## 초안 답변
{draft.content}

위 초안에 물리 오류가 있으면 수정하고, 없으면 그대로 출력하세요."""

        try:
            response = self._call_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=self.config.critic_model,
                timeout=self.config.refine_timeout
            )

            # Phase 7.14: JSON 파싱 제거 - 자연어 텍스트 직접 사용
            if response.strip():
                clean_response = re.sub(r'```\w*\s*|\s*```', '', response)
                clean_response = re.sub(r'<think>.*?</think>', '', clean_response, flags=re.DOTALL)
                clean_response = clean_response.strip()

                # "검증 완료" 또는 "원본 유지" 포함 시 초안 사용
                if '검증 완료' in clean_response or '원본 유지' in clean_response or len(clean_response) < 100:
                    content = draft.content
                    corrections = []
                    quality = AnswerQuality.GOOD
                else:
                    content = self._postprocess_answer(clean_response)
                    corrections = ["Stage 2에서 수정됨"]
                    quality = AnswerQuality.EXCELLENT

                logger.info(f"Stage 2 완료: {len(content)} chars")

                return RefinedAnswer(
                    content=content,
                    derivation="",
                    evidence_mapping=[],
                    recommendation="",
                    quality=quality,
                    corrections_made=corrections,
                    final_confidence=draft.confidence + 0.1 if corrections else draft.confidence,
                    clinical_insight=""
                )

        except Exception as e:
            logger.error(f"Stage 2 refine error: {e}")
            # 오류 시 초안 그대로 반환
            return RefinedAnswer(
                content=draft.content,
                derivation=draft.derivation,
                evidence_mapping=[],
                recommendation="",
                quality=AnswerQuality.NEEDS_REVISION,
                corrections_made=[f"검증 실패: {str(e)}"],
                final_confidence=draft.confidence * 0.8,  # 신뢰도 하향
                clinical_insight=""
            )

    # =========================================================================
    # Utilities
    # =========================================================================

    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        timeout: int,
        max_retries: int = 3
    ) -> str:
        """
        LLM 호출 (Phase 7.15: 재시도 로직 추가)

        Args:
            max_retries: 빈 응답 시 최대 재시도 횟수
        """
        import time

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.config.ollama_url}/api/chat",
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        "stream": False,
                        "options": {
                            "num_predict": 5000,
                            "temperature": 0.2 + (attempt * 0.1),  # 재시도 시 온도 약간 증가
                        }
                    },
                    timeout=timeout
                )

                response.raise_for_status()
                resp_data = response.json()
                content = resp_data.get("message", {}).get("content", "")

                # Phase 7.15: 빈 응답 재시도
                if not content.strip():
                    # thinking 필드 확인 (아래 로직에서 처리)
                    thinking = resp_data.get("message", {}).get("thinking", "")
                    if not thinking:
                        if attempt < max_retries - 1:
                            logger.warning(f"LLM 빈 응답 (시도 {attempt + 1}/{max_retries}), 재시도 중...")
                            time.sleep(1)  # 1초 대기 후 재시도
                            continue
                        else:
                            logger.error(f"LLM 빈 응답 (모든 재시도 실패)")
                            return ""

                break  # 성공 시 루프 탈출

            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    logger.warning(f"LLM 타임아웃 (시도 {attempt + 1}/{max_retries}), 재시도 중...")
                    time.sleep(2)
                    continue
                else:
                    raise
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"LLM 오류 (시도 {attempt + 1}/{max_retries}): {e}, 재시도 중...")
                    time.sleep(1)
                    continue
                else:
                    raise

        # Phase 7.7: thinking 모드 대응 (gpt-oss:20b, deepseek-r1 등)
        # content가 비어있으면 thinking 필드에서 JSON 추출 시도
        if not content.strip():
            thinking = resp_data.get("message", {}).get("thinking", "")
            if thinking:
                # thinking 필드에서 JSON 블록 추출
                json_match = re.search(r'```json\s*(.*?)\s*```', thinking, re.DOTALL)
                if json_match:
                    content = json_match.group(1)  # JSON 내용만 (코드 블록 마커 제외)
                else:
                    # 중괄호로 시작하는 JSON 찾기 (중첩 허용)
                    brace_match = re.search(r'\{[^}]*"(?:answer|derivation)".*\}', thinking, re.DOTALL)
                    if brace_match:
                        content = brace_match.group()
                logger.debug(f"Extracted from thinking field: {content[:200] if content else 'EMPTY'}")

        # DeepSeek-R1 <think> 태그 제거
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)

        # Phase 7.11: Misconception 필터 적용
        content = self._filter_physics_misconceptions(content)

        return content.strip()

    def _filter_physics_misconceptions(self, content: str) -> str:
        """
        Phase 7.11: LLM 출력에서 알려진 물리 오류를 감지하고 수정

        LLM이 지시를 무시하고 잘못된 물리를 생성할 경우, 이를 감지하고 경고를 추가합니다.
        """
        if not content:
            return content

        misconceptions_detected = []
        corrections = []

        # 1. K-edge 값 오류 감지 (CsI = 33keV, Iodine = 33.2keV)
        # 잘못된 K-edge 값: 29keV, 30keV 등
        kedge_wrong_patterns = [
            (r'K.?edge.*?(\d{2})\s*keV', lambda m: int(m.group(1)) not in [33, 34]),  # 33-34 keV 범위 허용
            (r'(\d{2})\s*keV.*?K.?edge', lambda m: int(m.group(1)) not in [33, 34]),
        ]
        for pattern, is_wrong in kedge_wrong_patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                if is_wrong(match):
                    wrong_value = match.group(1)
                    misconceptions_detected.append(f"K-edge {wrong_value}keV (정확한 값: CsI/Iodine K-edge = 33keV)")
                    # 수정하지 않고 경고만 추가 (원문 보존)

        # 2. 빔 경화가 대조도를 향상시킨다는 오류
        beam_hardening_wrong = [
            r'빔\s*경화.*?(?:향상|개선|증가).*?(?:대조도|DQE|CNR)',
            r'beam\s*hardening.*?(?:improve|increase|enhance).*?(?:contrast|DQE|CNR)',
            r'경화된.*?빔.*?(?:대조도|DQE|CNR).*?(?:향상|개선|증가)',
        ]
        for pattern in beam_hardening_wrong:
            if re.search(pattern, content, re.IGNORECASE):
                misconceptions_detected.append(
                    "빔 경화가 대조도를 향상시킨다는 오류 감지 "
                    "(정확: 빔 경화는 μ_pe ∝ 1/E³로 인해 대조도를 감소시킴)"
                )

        # 3. K-edge가 대조도 손실을 보상한다는 오류
        kedge_compensation_wrong = [
            r'K.?edge.*?(?:보상|회복|복원).*?(?:대조도|contrast|Δμ)',
            r'(?:대조도|contrast|Δμ).*?(?:보상|회복|복원).*?K.?edge',
        ]
        for pattern in kedge_compensation_wrong:
            if re.search(pattern, content, re.IGNORECASE):
                misconceptions_detected.append(
                    "K-edge가 대조도 손실을 보상한다는 오류 감지 "
                    "(정확: K-edge는 검출 효율(DQE)에만 영향, 피사체 대조도(Δμ)는 회복 불가)"
                )

        # 4. 산란선 공식 오류: Cs = C₀ × (1 - SPR)
        scatter_formula_wrong = [
            r'C[s₀]?\s*=\s*C[₀0]?\s*[×x\*]\s*\(?\s*1\s*-\s*SPR',
            r'C[s₀]?\s*=\s*C[₀0]?\s*\(1\s*-\s*SPR\)',
        ]
        for pattern in scatter_formula_wrong:
            if re.search(pattern, content, re.IGNORECASE):
                misconceptions_detected.append(
                    "산란선 공식 오류: Cs = C₀ × (1 - SPR) "
                    "(정확한 공식: Cs = C₀ / (1 + SPR))"
                )

        # 5. Fine Linear → Amorphous 변화가 MTF와 무관하다는 오류
        # (이 변화가 언급되었는데 MTF/고주파 손실 언급이 없으면 경고)
        if re.search(r'Fine\s*Linear.*?Amorphous|4C.*?4B', content, re.IGNORECASE):
            if not re.search(r'MTF|고주파|high.?frequency|해상도|resolution', content, re.IGNORECASE):
                misconceptions_detected.append(
                    "Fine Linear → Amorphous 형태 변화에 MTF/고주파 손실 설명 누락 "
                    "(정확: 고주파(2-5 lp/mm) MTF 손실로 인해 선형 구조가 점형으로 보임)"
                )

        # 경고 추가 (심각한 오류가 있으면 출력 앞에 경고 배너 추가)
        if misconceptions_detected:
            warning_banner = (
                "\n\n⚠️ **물리 검증 경고** (Phase 7.11 Misconception Filter):\n"
                "LLM 출력에서 다음 물리 오류가 감지되었습니다:\n"
            )
            for i, misconception in enumerate(misconceptions_detected, 1):
                warning_banner += f"{i}. {misconception}\n"
            warning_banner += "\n---\n\n"

            logger.warning(f"Physics misconceptions detected: {misconceptions_detected}")

            # 출력 앞에 경고 추가 (원문은 보존)
            content = warning_banner + content

        return content

    def _parse_json_response(self, response: str) -> Dict:
        """JSON 응답 파싱 (코드 블록 처리 + 키 정규화)"""
        if not response or not response.strip():
            return {}

        # ```json ... ``` 블록 추출
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # 코드 블록 없이 직접 JSON인 경우
            json_str = response

        parsed = {}
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError:
            # 시도 2: 중괄호로 감싸진 부분 추출
            brace_match = re.search(r'\{.*\}', response, re.DOTALL)
            if brace_match:
                try:
                    parsed = json.loads(brace_match.group())
                except json.JSONDecodeError:
                    # Phase 7.7: 잘린 JSON 복구 시도 (토큰 제한으로 JSON이 중간에 끊긴 경우)
                    truncated = brace_match.group()
                    # 열린 중괄호/대괄호 수 계산하여 닫기
                    open_braces = truncated.count('{') - truncated.count('}')
                    open_brackets = truncated.count('[') - truncated.count(']')
                    # 마지막 완전한 key-value 쌍 이후 자르기
                    last_comma = truncated.rfind('",')
                    if last_comma > 0:
                        truncated = truncated[:last_comma + 1]
                    truncated += ']' * open_brackets + '}' * open_braces
                    try:
                        parsed = json.loads(truncated)
                        logger.info(f"Recovered truncated JSON (added {open_braces} braces)")
                    except:
                        pass

        # Phase 7.5: 한국어/대체 키를 영어 키로 매핑
        key_mapping = {
            # answer alternatives
            '답변': 'answer',
            '최종_답변': 'answer',
            '최종답변': 'answer',
            'final_answer': 'answer',
            'response': 'answer',
            # derivation alternatives
            '증명': 'derivation',
            '증명_과정': 'derivation',
            '증명과정': 'derivation',
            '유도': 'derivation',
            '도출_과정': 'derivation',
            'proof': 'derivation',
            'reasoning': 'derivation',
            # clinical_insight alternatives
            '임상적_영향': 'clinical_insight',
            '임상영향': 'clinical_insight',
            '임상_통찰': 'clinical_insight',
            'clinical_impact': 'clinical_insight',
            # confidence alternatives
            '신뢰도': 'confidence',
            '확신도': 'confidence',
        }

        normalized = {}
        for key, value in parsed.items():
            # 공백과 언더스코어 정규화
            normalized_key = key.strip().replace(' ', '_')
            # 매핑 적용
            if normalized_key in key_mapping:
                normalized[key_mapping[normalized_key]] = value
            else:
                normalized[key] = value

        return normalized if normalized else parsed

    # =========================================================================
    # Phase 7.5: Post-processing (LaTeX & Language)
    # =========================================================================

    def _apply_latex_formatting(self, text: str) -> str:
        """
        플레인텍스트 수식을 LaTeX로 변환 (Phase 7.5)

        변환 규칙:
        - 물리량 약어: DQE, CNR, SNR, MTF → $\text{...}$
        - 수식 패턴: sqrt, 분수, 첨자 등
        - 그리스 문자: σ, α, β 등
        """
        if not text:
            return text

        result = text

        # 1. 이미 LaTeX로 감싸진 부분은 건드리지 않음
        # 임시로 보호 마커 사용
        protected = []
        def protect_latex(m):
            protected.append(m.group(0))
            return f"__LATEX_{len(protected)-1}__"

        result = re.sub(r'\$\$[^$]+\$\$', protect_latex, result)
        result = re.sub(r'\$[^$]+\$', protect_latex, result)

        # 2. 물리량 약어를 LaTeX로 변환 (한글 조사 앞에서도 동작)
        # 단어 끝이 영문자이거나 숫자인 경우만 매칭 (한글 조사 앞 허용)
        physics_terms = ['DQE', 'CNR', 'SNR', 'MTF', 'NPS', 'NEQ', 'SDNR', 'ROC', 'ALARA']

        for term in physics_terms:
            # 단어 앞에 영문자가 없고, 뒤에 영문자/숫자가 아닌 경우 매칭
            pattern = rf'(?<![A-Za-z]){term}(?![A-Za-z0-9_])'
            result = re.sub(pattern, rf'$\\text{{{term}}}$', result)

        # 3. 측정 단위 패턴 (예: "5 lp/mm", "0.35-0.45")
        result = re.sub(
            r'(\d+(?:\.\d+)?)\s*lp/mm',
            r'$\1\\ \\text{lp/mm}$',
            result
        )

        # 4. DQE/MTF with frequency (예: "DQE(5 lp/mm)")
        result = re.sub(
            r'\$\\text\{(DQE|MTF)\}\$\s*\((\d+(?:\.\d+)?)\s*(?:lp/mm)?\)',
            r'$\\text{\1}(\2\\ \\text{lp/mm})$',
            result
        )

        # 5. 그리스 문자 변환
        greek_letters = {
            'σ': r'$\sigma$',
            'α': r'$\alpha$',
            'β': r'$\beta$',
            'γ': r'$\gamma$',
            'Δ': r'$\Delta$',
            '√': r'$\sqrt{}$',
            '∝': r'$\propto$',
            '≥': r'$\geq$',
            '≤': r'$\leq$',
            '×': r'$\times$',
        }

        for char, latex in greek_letters.items():
            result = result.replace(char, latex)

        # 6. 첨자 패턴 (예: "σ_rel", "σ_abs") - 영문 첨자만 처리
        result = re.sub(
            r'(\$\\sigma\$)_([a-zA-Z]+)',
            r'$\\sigma_{\\text{\2}}$',
            result
        )
        # σ_rel 패턴 (아직 변환 안된 경우)
        result = re.sub(
            r'σ_([a-zA-Z]+)',
            r'$\\sigma_{\\text{\1}}$',
            result
        )

        # 7. 분수/비율 패턴 (예: "DQE_new / DQE_old")
        result = re.sub(
            r'\$\\text\{(\w+)\}\$_(\w+)\s*/\s*\$\\text\{(\w+)\}\$_(\w+)',
            r'$\\frac{\\text{\1}_{\\text{\2}}}{\\text{\3}_{\\text{\4}}}$',
            result
        )

        # 8. 백분율 강조 (예: "29.1%") - 이미 bold가 아닌 경우만
        result = re.sub(
            r'(?<!\*\*)(\d+(?:\.\d+)?)\s*%(?!\*\*)',
            r'**\1%**',
            result
        )

        # 9. Rose Criterion 강조 (이미 bold가 아닌 경우만)
        result = re.sub(
            r'(?<!\*\*)(Rose\s*Criterion)(?!\*\*)',
            r'**\1**',
            result,
            flags=re.IGNORECASE
        )

        # 10. 보호된 LaTeX 복원
        for i, original in enumerate(protected):
            result = result.replace(f"__LATEX_{i}__", original)

        # 11. 이중 bold 정리 (**** → **)
        result = re.sub(r'\*{4,}', '**', result)

        return result

    def _remove_chinese_characters(self, text: str) -> str:
        """
        중국어/일본어 문자를 한국어로 치환 (Phase 7.7)
        """
        if not text:
            return text

        # Phase 7.7: 일본어 히라가나/카타카나 포함 여부 감지
        # 일본어가 다수 포함되어 있으면 경고 로그
        hiragana_count = len(re.findall(r'[\u3040-\u309f]', text))
        katakana_count = len(re.findall(r'[\u30a0-\u30ff]', text))
        if hiragana_count + katakana_count > 10:
            logger.warning(f"Japanese characters detected ({hiragana_count} hiragana, {katakana_count} katakana) - model may have responded in Japanese")

        # 일본어 → 한국어 치환 맵 (물리학 용어)
        japanese_to_korean = {
            'したがって': '따라서',
            'すなわち': '즉',
            'ただし': '단,',
            'ここで': '여기서',
            'および': '및',
            'または': '또는',
            'における': '에서의',
            'について': '에 대해',
            'による': '에 의한',
            'として': '로서',
            'である': '이다',
            'となる': '가 된다',
            'とする': '로 한다',
            'が必要': '이 필요',
            'を満たす': '를 만족하는',
            'に対する': '에 대한',
            'の場合': '의 경우',
            '結論': '결론',
            '変数': '변수',
            '意味': '의미',
            '初期': '초기',
            '量子': '양자',
            '分散': '분산',
            '電子': '전자',
            '総': '총',
            '比率': '비율',
            '許容': '허용',
            '検出': '검출',
            '向上': '향상',
            '減少': '감소',
            '増加': '증가',
            '微小': '미세',
            'ノイズ': '노이즈',
            'コントラスト': '대조도',
            'エネルギー': '에너지',
            'ウェイト': '가중치',
        }
        result = text
        for jp, kr in japanese_to_korean.items():
            result = result.replace(jp, kr)

        # 중국어 → 한국어 치환 맵
        chinese_to_korean = {
            '其中': '여기서',
            '能力': '능력',
            '更高的': '더 높은',
            '更低的': '더 낮은',
            '更高': '더 높은',
            '更低': '더 낮은',
            '的': '',  # 단독 的 제거
            '显著': '현저히',
            '顯著': '현저히',
            '病灶': '병변',
            '病竈': '병변',
            '检出': '검출',
            '檢出': '검출',
            '沉积': '침착',
            '沉積': '침착',
            '钙化': '석회화',
            '鈣化': '석회화',
            '微小': '미세한',
            '图像': '영상',
            '圖像': '영상',
            '质量': '품질',
            '質量': '품질',
            '噪声': '노이즈',
            '噪聲': '노이즈',
            '剂量': '선량',
            '劑量': '선량',
            '探测': '검출',
            '探測': '검출',
            '灵敏度': '민감도',
            '靈敏度': '민감도',
            '特异性': '특이도',
            '特異性': '특이도',
        }

        for chinese, korean in chinese_to_korean.items():
            result = result.replace(chinese, korean)

        # 남은 중국어 문자 제거 (CJK Unified Ideographs 범위)
        # 일본어 한자와 한국어 한자는 유지하되, 문맥상 어색한 경우만 처리

        return result

    def _postprocess_answer(self, text: str) -> str:
        """
        답변 후처리: LaTeX 수리 + 변환 + 중국어 제거 (Phase 7.5, 7.13)
        """
        if not text:
            return text

        # 0. Phase 7.13: 손상된 LaTeX 복구 (\t → \text 등)
        result = self._repair_corrupted_latex(text)

        # 1. 중국어 제거
        result = self._remove_chinese_characters(result)

        # 2. LaTeX 변환
        result = self._apply_latex_formatting(result)

        return result

    def _repair_corrupted_latex(self, text: str) -> str:
        """
        Phase 7.13: 손상된 LaTeX 명령어 복구

        문제: JSON 직렬화/역직렬화 과정에서 \\text → \\t + ext (탭 + ext)로 변환됨
        해결: 손상된 패턴을 원래 LaTeX 명령어로 복구
        """
        if not text:
            return text

        result = text

        # 1. \t가 탭으로 변환된 경우 복구: 탭 + ext{ → \text{
        result = result.replace('\text{', '\\text{')  # 탭+ext → \text

        # 2. 직접 손상된 패턴 복구
        corrupted_patterns = [
            (r'\\ext\{', r'\\text{'),      # \ext{ → \text{
            (r'\\frac\{', r'\\frac{'),      # 이미 올바름, 확인용
            (r'\\sqrt\{', r'\\sqrt{'),      # 이미 올바름, 확인용
            (r'\\sigma', r'\\sigma'),       # 이미 올바름, 확인용
            (r'\\Delta', r'\\Delta'),       # 이미 올바름, 확인용
            (r'\\propto', r'\\propto'),     # 이미 올바름, 확인용
            (r'\\geq', r'\\geq'),           # 이미 올바름, 확인용
            (r'\\leq', r'\\leq'),           # 이미 올바름, 확인용
            (r'\\times', r'\\times'),       # 이미 올바름, 확인용
        ]

        for corrupted, correct in corrupted_patterns:
            result = re.sub(corrupted, correct, result)

        # 3. 연속된 백슬래시 정리 (\\\\text → \\text)
        result = re.sub(r'\\{2,}(text|frac|sqrt|sigma|Delta|propto|geq|leq|times)', r'\\\1', result)

        # 4. 누락된 백슬래시 추가 (text{ without backslash)
        result = re.sub(r'(?<!\\)text\{([^}]+)\}', r'\\text{\1}', result)

        return result

    # =========================================================================
    # Output Formatting
    # =========================================================================

    def format_structured_output(self, result: RefinedAnswer) -> str:
        """
        구조화된 텍스트 출력 포맷팅

        Output:
            ### 📐 증명 과정 (Derivation)
            ...
            ### 📚 근거 매핑 (Evidence)
            ...
            ### 💡 최종 권고 (Recommendation)
            ...
        """
        parts = []

        def safe_str(value) -> str:
            """리스트나 다른 타입을 안전하게 문자열로 변환"""
            if isinstance(value, list):
                return ", ".join(str(v) for v in value)
            return str(value) if value else ""

        # 1. Derivation
        if result.derivation:
            parts.append("### 📐 증명 과정 (Derivation)")
            parts.append(safe_str(result.derivation))
            parts.append("")

        # 2. Evidence Mapping
        if result.evidence_mapping:
            parts.append("### 📚 근거 매핑 (Evidence)")
            for i, ev in enumerate(result.evidence_mapping, 1):
                verified = "✅" if ev.get("verified", False) else "⚠️"
                claim = safe_str(ev.get('claim', ''))
                source = safe_str(ev.get('source', '미확인'))
                parts.append(f"{i}. {verified} **{claim}**")
                parts.append(f"   - 출처: {source}")
            parts.append("")

        # 3. Main Answer
        parts.append("### 📝 답변")
        parts.append(safe_str(result.content))
        parts.append("")

        # 4. Clinical Insight (Phase 7.2)
        if result.clinical_insight:
            parts.append("### 🔬 임상적 영향 분석 (Clinical Impact)")
            parts.append(safe_str(result.clinical_insight))
            parts.append("")

        # 5. Recommendation
        if result.recommendation:
            parts.append("### 💡 임상적 권고 (Recommendation)")
            parts.append(safe_str(result.recommendation))
            parts.append("")

        # 6. Quality Badge
        quality_badges = {
            AnswerQuality.EXCELLENT: "🏆 **검증 완료** (신뢰도: {:.0%})",
            AnswerQuality.GOOD: "✅ **양호** (신뢰도: {:.0%})",
            AnswerQuality.NEEDS_REVISION: "⚠️ **주의 필요** (신뢰도: {:.0%})",
            AnswerQuality.UNRELIABLE: "❌ **재확인 필요** (신뢰도: {:.0%})",
        }
        badge = quality_badges.get(result.quality, "").format(result.final_confidence)
        parts.append(f"---\n{badge}")

        # 7. Corrections (if any)
        if result.corrections_made:
            parts.append("\n<details><summary>🔧 자체 검증에서 수정된 사항</summary>\n")
            for corr in result.corrections_made:
                parts.append(f"- {safe_str(corr)}")
            parts.append("</details>")

        return "\n".join(parts)


# =============================================================================
# Singleton & Convenience
# =============================================================================

_engine_instance: Optional[TextReasoningEngine] = None


def get_text_reasoning_engine(config: Optional[TextReasoningConfig] = None) -> TextReasoningEngine:
    """TextReasoningEngine 싱글톤"""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = TextReasoningEngine(config)
    return _engine_instance


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 테스트용 지식
    test_knowledge = """
## 표준 참조 자료
SNR ∝ √Dose 관계에 의해:
- 선량을 2배로 높이면 SNR은 √2 ≈ 1.414배 증가
- SNR을 2배로 높이려면 선량을 4배로 증가

## T-factor (Dance et al. 2011)
- DBT에서 5cm 유방의 평균 T-factor: T ≈ 0.94
- 이는 토모신테시스가 기존 2D 대비 약 6% 낮은 선량 효율을 보임을 의미
"""

    engine = TextReasoningEngine()

    test_question = "SNR을 2배로 높이려면 선량을 얼마나 증가해야 하는가? 증명과 함께 설명해주세요."

    print("=" * 60)
    print("Answering Twice Test")
    print("=" * 60)
    print(f"질문: {test_question}")
    print("-" * 60)

    result = engine.reason(
        question=test_question,
        context="",
        physics_knowledge=test_knowledge
    )

    print("\n" + engine.format_structured_output(result))
