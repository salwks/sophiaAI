"""
Dynamic Evidence Pipeline: 동적 근거 강화 시스템
================================================
Phase 7의 핵심: PMC 전문 실시간 인출 + 근거 추적 + Phase 6 추론을 통합

Workflow:
    1. [Search] 초록 기반 검색으로 후보 논문 선정
    2. [Fetch] PMC ID 보유 논문의 전문을 비동기로 인출
    3. [Chunk] 질문 관련 섹션만 스마트 추출
    4. [Reason] Phase 6 Answering Twice로 심층 추론
    5. [Map] Evidence Mapper로 근거 출처 추적
    6. [Output] 구조화된 답변 + 근거 인용

"제가 PMC에서 해당 논문 전문을 확인해본 결과..."
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field

from src.retrieval.pmc_fetcher import PMCFetcher, PMCArticle, get_pmc_fetcher
from src.retrieval.summarizer import MedicalTextSummarizer, get_medical_summarizer, SummaryResult
from src.reasoning.evidence_mapper import EvidenceMapper, EvidenceReport, get_evidence_mapper
from src.reasoning.text_logic import TextReasoningEngine, RefinedAnswer, get_text_reasoning_engine
from src.evaluation.agent_judge import AgentJudge, JudgeResult, JudgeVerdict, get_agent_judge
from src.knowledge.manager import KnowledgeManager, get_knowledge_manager
# Phase 7.6: Agentic Orchestrator
from src.reasoning.orchestrator import AgenticOrchestrator, get_agentic_orchestrator, OrchestrationResult

logger = logging.getLogger(__name__)


@dataclass
class EnrichedContext:
    """강화된 컨텍스트"""
    original_context: str           # 초록 기반 원본 컨텍스트
    enriched_context: str           # PMC 전문으로 강화된 컨텍스트
    pmc_articles: List[PMCArticle]  # 인출된 PMC 논문들
    fetched_count: int              # 인출된 논문 수
    total_chars: int                # 총 문자 수
    used_fulltext: bool = False     # 전문 사용 여부
    used_summarizer: bool = False   # SLM 요약 사용 여부 (Phase 7.1)


@dataclass
class DynamicEvidenceResult:
    """동적 근거 강화 결과"""
    answer: str                     # 최종 답변 (구조화)
    refined_answer: RefinedAnswer   # Phase 6 추론 결과
    judge_result: JudgeResult       # 품질 평가 결과
    evidence_report: EvidenceReport # 근거 매핑 보고서
    enriched_context: EnrichedContext  # 강화된 컨텍스트
    used_fulltext: bool             # 전문 사용 여부
    used_summarizer: bool = False   # SLM 요약 사용 여부 (Phase 7.1)


class DynamicEvidencePipeline:
    """
    동적 근거 강화 파이프라인

    Usage:
        pipeline = DynamicEvidencePipeline()

        result = await pipeline.process_async(
            question="DBT에서 5cm 유방의 MGD 계산 시 T-factor는?",
            papers=[{"pmid": "123", "pmc_id": "PMC456", "abstract": "..."}],
            physics_knowledge="..."
        )

        print(result.answer)
        print(f"전문 사용: {result.used_fulltext}")
    """

    def __init__(self, use_summarizer: bool = True, enable_decomposition: bool = True):
        self.pmc_fetcher = get_pmc_fetcher()
        self.summarizer = get_medical_summarizer() if use_summarizer else None
        self.evidence_mapper = get_evidence_mapper()
        self.reasoning_engine = get_text_reasoning_engine()
        self.judge = get_agent_judge()
        self.knowledge_manager = get_knowledge_manager()
        self.use_summarizer = use_summarizer
        # Phase 7.6: Agentic Orchestrator
        self.enable_decomposition = enable_decomposition
        self.orchestrator = get_agentic_orchestrator() if enable_decomposition else None

    async def process_async(
        self,
        question: str,
        papers: List[Dict[str, Any]],
        physics_knowledge: str = "",
        max_pmc_fetch: int = 3
    ) -> DynamicEvidenceResult:
        """
        비동기 동적 근거 강화 처리

        Args:
            question: 사용자 질문
            papers: 검색된 논문 목록 (pmid, pmc_id, abstract 포함)
            physics_knowledge: KnowledgeManager 지식
            max_pmc_fetch: 최대 PMC 인출 수

        Returns:
            DynamicEvidenceResult
        """
        logger.info(f"Processing with dynamic evidence: {question[:50]}...")

        # 1. PMC ID가 있는 논문 필터링 및 전문 인출
        enriched_context = await self._enrich_context_async(
            question, papers, max_pmc_fetch
        )

        # 2. Phase 6 Answering Twice 추론
        combined_context = self._build_combined_context(
            enriched_context, physics_knowledge
        )

        refined_answer = self.reasoning_engine.reason(
            question=question,
            context=combined_context,
            physics_knowledge=physics_knowledge,
            require_derivation=True
        )

        # Phase 7.16: 빈 응답 검증 및 재시도
        if not refined_answer.content.strip():
            logger.warning("LLM 응답이 비어있음 - 단순 모드로 재시도")
            # 재시도: 더 짧은 컨텍스트로
            retry_answer = self.reasoning_engine.reason(
                question=question,
                context="",  # 컨텍스트 제거
                physics_knowledge=physics_knowledge[:3000] if physics_knowledge else "",
                require_derivation=False
            )
            if retry_answer.content.strip():
                refined_answer = retry_answer
            else:
                # 최종 실패 시 기본 메시지
                from src.reasoning.text_logic import RefinedAnswer, AnswerQuality
                refined_answer = RefinedAnswer(
                    content="죄송합니다. 응답 생성에 실패했습니다. 질문을 다시 시도해 주세요.",
                    derivation="",
                    evidence_mapping=[],
                    recommendation="",
                    quality=AnswerQuality.UNRELIABLE,
                    corrections_made=["LLM 응답 생성 실패"],
                    final_confidence=0.0,
                    clinical_insight=""
                )

        # 3. Agent-as-a-Judge 평가
        judge_result = self.judge.evaluate(
            question=question,
            answer=refined_answer.content,
            derivation=refined_answer.derivation,
            reference_knowledge=physics_knowledge,
            context=combined_context[:3000]
        )

        # 4. Evidence Mapping
        all_sources = self._build_source_list(papers, enriched_context.pmc_articles)
        evidence_report = self.evidence_mapper.analyze_answer(
            refined_answer.content, all_sources
        )

        # 5. 최종 답변 포맷팅
        formatted_answer = self._format_final_answer(
            refined_answer, evidence_report, enriched_context.used_fulltext
        )

        return DynamicEvidenceResult(
            answer=formatted_answer,
            refined_answer=refined_answer,
            judge_result=judge_result,
            evidence_report=evidence_report,
            enriched_context=enriched_context,
            used_fulltext=enriched_context.fetched_count > 0,
            used_summarizer=enriched_context.used_summarizer
        )

    def process_sync(
        self,
        question: str,
        papers: List[Dict[str, Any]],
        physics_knowledge: str = "",
        max_pmc_fetch: int = 3
    ) -> DynamicEvidenceResult:
        """동기 방식 처리 (래퍼)"""
        return asyncio.run(self.process_async(
            question, papers, physics_knowledge, max_pmc_fetch
        ))

    # =========================================================================
    # Phase 7.17: Simplified Pipeline (복잡한 레이어 제거)
    # =========================================================================

    async def process_simple_async(
        self,
        question: str,
        papers: List[Dict[str, Any]],
        physics_knowledge: str = "",
        max_pmc_fetch: int = 2
    ) -> DynamicEvidenceResult:
        """
        단순화된 파이프라인 (Phase 7.17)

        복잡한 레이어들을 모두 제거하고 기본 RAG + LLM만 사용:
        - QueryPlanner 분해 ❌
        - PhysicsTriage 분류 ❌
        - MathVerifier 검증 ❌
        - ReflectiveEngine 힌트 ❌
        - ResponseSynthesizer 합성 ❌

        단순 흐름: Question → Knowledge → LLM → Answer
        """
        logger.info(f"[SIMPLE] Processing: {question[:50]}...")

        # 1. PMC 컨텍스트 (선택적, 최소화)
        enriched_context = await self._enrich_context_async(
            question, papers, max_pmc_fetch
        )

        # 2. CQO/QOCO 프롬프트 구성 (Phase 1: Lost in the Prompt Order)
        prompt_strategy = self._select_prompt_strategy(question)
        if prompt_strategy == "qoco":
            simple_prompt = self._build_qoco_prompt(
                question, physics_knowledge, enriched_context
            )
            logger.info(f"[SIMPLE] Using QOCO prompt strategy (calculation/comparison)")
        else:
            simple_prompt = self._build_simple_prompt(
                question, physics_knowledge, enriched_context
            )
            logger.info(f"[SIMPLE] Using CQO prompt strategy (concept/principle)")

        # 3. 직접 LLM 호출 (단일 단계)
        import requests
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "gpt-oss:20b",
                    "prompt": simple_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 3000,
                        "num_ctx": 8192
                    }
                },
                timeout=300
            )
            response.raise_for_status()
            answer_content = response.json().get("response", "").strip()

            if not answer_content:
                answer_content = "응답 생성에 실패했습니다."
                confidence = 0.0
            else:
                confidence = 0.85  # 단순 파이프라인 기본 신뢰도

        except Exception as e:
            logger.error(f"[SIMPLE] LLM 호출 실패: {e}")
            answer_content = f"오류 발생: {str(e)}"
            confidence = 0.0

        # 4. 최소 후처리 (LaTeX 등)
        answer_content = self._simple_postprocess(answer_content)

        # 5. 결과 반환 (간소화된 구조 + 전략 정보)
        from src.reasoning.text_logic import RefinedAnswer, AnswerQuality
        from src.evaluation.agent_judge import JudgeResult, JudgeVerdict, EvaluationCriteria
        from src.reasoning.evidence_mapper import EvidenceReport

        refined = RefinedAnswer(
            content=answer_content,
            derivation="",
            evidence_mapping=[],
            recommendation="",
            quality=AnswerQuality.GOOD if confidence > 0.5 else AnswerQuality.UNRELIABLE,
            corrections_made=[],
            final_confidence=confidence,
            clinical_insight=""
        )

        # 평가 생략 (속도 우선) - 기본값 사용
        default_criteria = EvaluationCriteria(
            factual_accuracy=confidence * 100,
            logical_consistency=confidence * 100,
            guideline_compliance=confidence * 100,
            completeness=confidence * 100,
            clinical_insight=confidence * 100
        )
        judge_result = JudgeResult(
            verdict=JudgeVerdict.APPROVED if confidence > 0.5 else JudgeVerdict.REVISION_REQUIRED,
            total_score=confidence * 100,
            criteria=default_criteria,
            issues_found=[],
            suggestions=[],
            reasoning="[SIMPLE MODE] 평가 생략"
        )

        evidence_report = EvidenceReport(
            mapped_claims=[],
            total_claims=0,
            verified_claims=0,
            gold_sources=0,
            high_sources=0,
            overall_credibility=confidence
        )

        # 포맷팅 (프롬프트 전략 표시)
        strategy_label = "CQO" if prompt_strategy == "cqo" else "QOCO"
        formatted = f"""## 답변

{answer_content}

---
*[Phase 1: {strategy_label} 프롬프트] 신뢰도: {confidence:.0%}*
"""

        return DynamicEvidenceResult(
            answer=formatted,
            refined_answer=refined,
            judge_result=judge_result,
            evidence_report=evidence_report,
            enriched_context=enriched_context,
            used_fulltext=enriched_context.fetched_count > 0,
            used_summarizer=False
        )

    def _build_simple_prompt(
        self,
        question: str,
        physics_knowledge: str,
        enriched_context: EnrichedContext
    ) -> str:
        """
        CQO 프롬프트 구성 (Phase 1: Lost in the Prompt Order 적용)

        CQO = Context → Question → Options
        - Context: 모든 참조 정보를 먼저 제시 (decoder-only LLM이 더 잘 활용)
        - Question: 컨텍스트 직후 명확히 제시
        - Options: 출력 형식 지시는 최소화하여 마지막에

        논문 근거: "Lost in the Prompt Order" (2026) - CQO 순서가
        QCO, OQC 대비 +14.7% 정확도 향상 (decoder-only LLM 기준)
        """
        prompt_parts = []

        # ═══════════════════════════════════════════════════════════════
        # [C] CONTEXT: 모든 참조 정보 (역할 + 지식 + 문헌)
        # ═══════════════════════════════════════════════════════════════

        # 역할 정의 (Context의 일부로 포함)
        prompt_parts.append(
            "═══ 맥락 정보 ═══\n\n"
            "역할: 유방영상의학 물리학 전문가"
        )

        # 물리학 지식 (핵심 컨텍스트)
        if physics_knowledge:
            knowledge_excerpt = physics_knowledge[:4000]
            prompt_parts.append(f"\n\n[표준 물리학 참조]\n{knowledge_excerpt}")

        # 논문/문헌 컨텍스트
        if enriched_context.enriched_context and len(enriched_context.enriched_context) > 100:
            context_excerpt = enriched_context.enriched_context[:2000]
            prompt_parts.append(f"\n\n[관련 문헌]\n{context_excerpt}")

        # ═══════════════════════════════════════════════════════════════
        # [Q] QUESTION: 컨텍스트 직후 명확히 제시
        # ═══════════════════════════════════════════════════════════════
        prompt_parts.append(f"\n\n═══ 질문 ═══\n\n{question}")

        # ═══════════════════════════════════════════════════════════════
        # [O] OPTIONS: 출력 지시 (최소화)
        # ═══════════════════════════════════════════════════════════════
        prompt_parts.append(
            "\n\n═══ 답변 형식 ═══\n"
            "- 물리 원리 기반 설명\n"
            "- 필요시 계산 과정 포함\n"
            "- 한국어, 자연스러운 문단"
        )

        return "\n".join(prompt_parts)

    def _build_qoco_prompt(
        self,
        question: str,
        physics_knowledge: str,
        enriched_context: EnrichedContext
    ) -> str:
        """
        QOCO 프롬프트 구성 (Phase 1 대안)

        QOCO = Question-framing → Options → Context → Output
        - 질문 의도를 먼저 명확히 → 어떤 정보가 필요한지 프레이밍
        - 옵션/제약조건 제시
        - 그 다음 컨텍스트 (질문 의도를 알고 읽음)
        - 마지막에 출력 형식

        사용 케이스: 계산/비교 질문 (명확한 답 형식이 필요한 경우)
        """
        prompt_parts = []

        # ═══════════════════════════════════════════════════════════════
        # [Q] QUESTION FRAMING: 질문 의도 명확화
        # ═══════════════════════════════════════════════════════════════
        prompt_parts.append(
            f"═══ 질문 ═══\n\n"
            f"다음 유방영상의학 물리학 질문에 답변하시오:\n\n{question}"
        )

        # ═══════════════════════════════════════════════════════════════
        # [O] OPTIONS: 답변 제약조건
        # ═══════════════════════════════════════════════════════════════
        prompt_parts.append(
            "\n\n═══ 요구사항 ═══\n"
            "- 물리 공식과 원리 기반으로 설명\n"
            "- 수치 계산 시 단계별 과정 제시\n"
            "- 참조 지식과 일관성 유지"
        )

        # ═══════════════════════════════════════════════════════════════
        # [C] CONTEXT: 참조 정보
        # ═══════════════════════════════════════════════════════════════
        if physics_knowledge:
            knowledge_excerpt = physics_knowledge[:4000]
            prompt_parts.append(f"\n\n═══ 참조 지식 ═══\n{knowledge_excerpt}")

        if enriched_context.enriched_context and len(enriched_context.enriched_context) > 100:
            context_excerpt = enriched_context.enriched_context[:2000]
            prompt_parts.append(f"\n\n═══ 관련 문헌 ═══\n{context_excerpt}")

        # ═══════════════════════════════════════════════════════════════
        # [O] OUTPUT: 출력 형식
        # ═══════════════════════════════════════════════════════════════
        prompt_parts.append(
            "\n\n═══ 답변 ═══\n"
            "한국어로 명확하게 답변:"
        )

        return "\n".join(prompt_parts)

    def _select_prompt_strategy(self, question: str) -> str:
        """
        질문 유형에 따른 프롬프트 전략 선택 (Phase 1)

        CQO: 개념 설명, 원리 질문 (컨텍스트 이해가 중요)
        QOCO: 계산, 비교, 수치 질문 (명확한 답 형식 필요)

        Returns: "cqo" or "qoco"
        """
        # 계산/수치 키워드
        calculation_keywords = [
            "계산", "구하", "얼마", "몇", "%", "배",
            "비교", "차이", "vs", "대비",
            "공식", "수치", "값"
        ]

        # 개념/원리 키워드
        concept_keywords = [
            "왜", "어떻게", "원리", "메커니즘", "이유",
            "설명", "의미", "영향", "관계"
        ]

        question_lower = question.lower()

        calc_score = sum(1 for kw in calculation_keywords if kw in question_lower)
        concept_score = sum(1 for kw in concept_keywords if kw in question_lower)

        # 계산 질문이면 QOCO, 아니면 CQO (기본)
        if calc_score > concept_score and calc_score >= 2:
            return "qoco"
        return "cqo"

    def _simple_postprocess(self, text: str) -> str:
        """최소 후처리"""
        import re

        # 중국어 제거
        chinese_map = {
            "其中": "여기서", "更高的": "더 높은", "因此": "따라서",
            "可以": "할 수 있다", "通过": "통해", "需要": "필요하다"
        }
        for ch, ko in chinese_map.items():
            text = text.replace(ch, ko)

        # <think> 태그 제거
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

        return text.strip()

    # =========================================================================
    # Phase 7.6: Agentic Decomposition (Legacy)
    # =========================================================================

    async def process_with_decomposition_async(
        self,
        question: str,
        papers: List[Dict[str, Any]],
        physics_knowledge: str = "",
        max_pmc_fetch: int = 3
    ) -> DynamicEvidenceResult:
        """
        복합 질문 분해 기반 처리 (Phase 7.6)

        복잡한 질문을 하위 작업으로 분해하여 각각 처리 후 합성합니다.
        단순 질문은 기존 process_async와 동일하게 처리됩니다.

        Args:
            question: 사용자 질문
            papers: 검색된 논문 목록
            physics_knowledge: KnowledgeManager 지식
            max_pmc_fetch: 최대 PMC 인출 수

        Returns:
            DynamicEvidenceResult
        """
        if not self.orchestrator:
            # 오케스트레이터 비활성화 시 기존 방식
            return await self.process_async(question, papers, physics_knowledge, max_pmc_fetch)

        logger.info(f"Processing with decomposition: {question[:50]}...")

        # 1. 질문 분해
        from src.reasoning.planner import DecompositionResult
        decomposition = self.orchestrator.planner.decompose(question)

        # 단순 질문이면 기존 방식으로
        if not decomposition.is_complex or len(decomposition.subtasks) <= 1:
            logger.info("Simple query - using standard pipeline")
            return await self.process_async(question, papers, physics_knowledge, max_pmc_fetch)

        logger.info(f"Complex query - decomposed into {len(decomposition.subtasks)} subtasks")

        # 2. PMC 컨텍스트 인출 (전체 질문 기준 - 한 번만)
        enriched_context = await self._enrich_context_async(question, papers, max_pmc_fetch)
        combined_context = self._build_combined_context(enriched_context, physics_knowledge)

        # 3. 각 subtask 개별 처리
        for task in decomposition.subtasks:
            logger.info(f"Processing subtask {task.id}: {task.type.value} - {task.query[:50]}...")

            # subtask별 지식 모듈 로드
            task_knowledge = self.orchestrator._get_task_knowledge(task)

            try:
                result = self.reasoning_engine.reason(
                    question=task.query,
                    context=combined_context,
                    physics_knowledge=task_knowledge or physics_knowledge,
                    require_derivation=(task.type.value == "MATH")
                )
                task.answer = result.content
                task.confidence = result.final_confidence
                logger.info(f"Subtask {task.id} done: confidence={result.final_confidence:.0%}")

            except Exception as e:
                logger.error(f"Subtask {task.id} failed: {e}")
                task.answer = f"분석 실패: {str(e)}"
                task.confidence = 0.0

        # 4. 결과 합성
        synthesis_report = self.orchestrator.synthesizer.synthesize(decomposition)

        # 5. 합성된 답변으로 평가 (첫 번째 subtask 기준)
        first_task = decomposition.subtasks[0]
        judge_result = self.judge.evaluate(
            question=question,
            answer=synthesis_report.content,
            derivation=first_task.answer,
            reference_knowledge=physics_knowledge,
            context=combined_context[:3000]
        )

        # 6. Evidence Mapping
        all_sources = self._build_source_list(papers, enriched_context.pmc_articles)
        evidence_report = self.evidence_mapper.analyze_answer(
            synthesis_report.content, all_sources
        )

        # 7. 최종 결과 반환 (합성된 답변 사용)
        # RefinedAnswer 객체 생성 (합성 결과 기반)
        from src.reasoning.text_logic import RefinedAnswer, AnswerQuality
        synthesized_refined = RefinedAnswer(
            content=synthesis_report.content,
            derivation=first_task.answer if first_task else "",
            evidence_mapping=[],
            recommendation="",
            quality=AnswerQuality.GOOD if synthesis_report.total_confidence >= 0.7 else AnswerQuality.NEEDS_REVISION,
            corrections_made=[],
            final_confidence=synthesis_report.total_confidence,
            clinical_insight=""
        )

        return DynamicEvidenceResult(
            answer=synthesis_report.content,
            refined_answer=synthesized_refined,
            judge_result=judge_result,
            evidence_report=evidence_report,
            enriched_context=enriched_context,
            used_fulltext=enriched_context.fetched_count > 0,
            used_summarizer=enriched_context.used_summarizer
        )

    # =========================================================================
    # Context Enrichment
    # =========================================================================

    async def _enrich_context_async(
        self,
        question: str,
        papers: List[Dict[str, Any]],
        max_fetch: int
    ) -> EnrichedContext:
        """PMC 전문으로 컨텍스트 강화"""

        # PMC ID가 있는 논문 선별
        pmc_papers = [
            p for p in papers
            if p.get("pmc_id") and not p.get("pmc_id", "").startswith("None")
        ][:max_fetch]

        pmc_ids = [p["pmc_id"] for p in pmc_papers]

        # 원본 컨텍스트 (초록)
        original_context = self._build_abstract_context(papers)

        if not pmc_ids:
            return EnrichedContext(
                original_context=original_context,
                enriched_context=original_context,
                pmc_articles=[],
                fetched_count=0,
                total_chars=len(original_context),
                used_fulltext=False,
                used_summarizer=False
            )

        # 비동기 PMC 인출
        logger.info(f"Fetching {len(pmc_ids)} PMC articles...")
        articles = await self.pmc_fetcher.fetch_multiple_async(pmc_ids)

        # 질문 관련 컨텍스트 추출 + SLM 요약 (Phase 7.1)
        enriched_parts = [original_context]
        summarizer_used = False

        for article in articles:
            # 1단계: 기존 방식으로 관련 섹션 추출
            relevant_context = self.pmc_fetcher.extract_relevant_context(
                article, question, max_chars=6000  # 요약 전이므로 더 큰 청크 허용
            )

            if relevant_context:
                # 2단계: SLM 요약 레이어 (Phase 7.1 핵심)
                if self.use_summarizer and self.summarizer and len(relevant_context) > 1500:
                    logger.info(f"Summarizing {article.pmc_id}: {len(relevant_context):,} chars")
                    summary_result = self.summarizer.summarize(
                        full_text=relevant_context,
                        question=question,
                        pmc_id=article.pmc_id
                    )
                    # 요약된 텍스트 사용
                    final_context = summary_result.summary
                    summarizer_used = True
                    logger.info(f"Summarized {article.pmc_id}: {len(relevant_context):,} → {len(final_context):,} chars ({summary_result.compression_ratio:.1%})")
                else:
                    # 짧거나 요약 비활성화 시 원본 사용
                    final_context = relevant_context

                enriched_parts.append(
                    f"\n\n[PMC 전문: {article.pmc_id}]\n{final_context}"
                )

        enriched_context = "\n".join(enriched_parts)

        return EnrichedContext(
            original_context=original_context,
            enriched_context=enriched_context,
            pmc_articles=articles,
            fetched_count=len(articles),
            total_chars=len(enriched_context),
            used_fulltext=len(articles) > 0,
            used_summarizer=summarizer_used
        )

    def _build_abstract_context(self, papers: List[Dict]) -> str:
        """초록 기반 컨텍스트 구성"""
        parts = []
        for i, paper in enumerate(papers[:5], 1):
            abstract = paper.get("abstract", "")[:1500]
            title = paper.get("title", "Unknown")
            parts.append(f"[{i}] {title}\n{abstract}")
        return "\n\n".join(parts)

    def _build_combined_context(
        self,
        enriched: EnrichedContext,
        physics_knowledge: str
    ) -> str:
        """통합 컨텍스트 구성"""
        parts = []

        if physics_knowledge:
            parts.append(f"### 표준 참조 자료\n{physics_knowledge}")

        parts.append(f"### 검색된 문서\n{enriched.enriched_context}")

        return "\n\n".join(parts)

    # =========================================================================
    # Source Building
    # =========================================================================

    def _build_source_list(
        self,
        papers: List[Dict],
        pmc_articles: List[PMCArticle]
    ) -> List[Dict]:
        """Evidence Mapper용 소스 목록 구성"""
        sources = []

        # 논문 (초록)
        for paper in papers:
            sources.append({
                "id": paper.get("pmid", ""),
                "pmc_id": paper.get("pmc_id"),
                "title": paper.get("title", ""),
                "text": paper.get("abstract", ""),
                "has_fulltext": paper.get("pmc_id") is not None
            })

        # PMC 전문
        for article in pmc_articles:
            sources.append({
                "id": article.pmc_id,
                "pmc_id": article.pmc_id,
                "title": article.title,
                "text": article.results + " " + article.discussion,
                "section": "Results/Discussion",
                "has_fulltext": True
            })

        return sources

    # =========================================================================
    # Output Formatting
    # =========================================================================

    def _format_final_answer(
        self,
        refined: RefinedAnswer,
        evidence_report: EvidenceReport,
        used_fulltext: bool
    ) -> str:
        """최종 답변 포맷팅"""
        parts = []

        # 전문 사용 알림
        if used_fulltext:
            parts.append("💡 **PMC 전문 분석 완료**: 초록에 없는 상세 정보가 포함되었습니다.\n")

        # Phase 6 구조화 출력
        parts.append(self.reasoning_engine.format_structured_output(refined))

        # 근거 인용 (검증된 것만)
        verified_claims = [mc for mc in evidence_report.mapped_claims if mc.is_verified]
        if verified_claims:
            parts.append("\n" + self.evidence_mapper.format_citations(verified_claims))

        return "\n".join(parts)


# =============================================================================
# Singleton
# =============================================================================

_pipeline_instance: Optional[DynamicEvidencePipeline] = None


def get_dynamic_evidence_pipeline() -> DynamicEvidencePipeline:
    """DynamicEvidencePipeline 싱글톤"""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = DynamicEvidencePipeline()
    return _pipeline_instance


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)

    async def test():
        pipeline = DynamicEvidencePipeline()

        # 테스트 논문 (실제 PMC ID)
        test_papers = [
            {
                "pmid": "32478901",
                "pmc_id": "PMC7533093",
                "title": "Digital Mammography Optimization",
                "abstract": "This study examines the optimization of digital mammography..."
            }
        ]

        test_knowledge = """
SNR ∝ √Dose 관계에 의해:
- SNR을 2배로 높이려면 선량을 4배로 증가
"""

        question = "디지털 유방촬영에서 화질과 선량의 관계는?"

        print("=" * 60)
        print("Dynamic Evidence Pipeline Test")
        print("=" * 60)
        print(f"질문: {question}")
        print("-" * 60)

        result = await pipeline.process_async(
            question=question,
            papers=test_papers,
            physics_knowledge=test_knowledge,
            max_pmc_fetch=1
        )

        print(f"\n전문 사용: {result.used_fulltext}")
        print(f"인출된 PMC: {result.enriched_context.fetched_count}개")
        print(f"품질 점수: {result.judge_result.total_score:.0f}/100")
        print(f"검증된 주장: {result.evidence_report.verified_claims}/{result.evidence_report.total_claims}")
        print()
        print(result.answer[:1500] + "..." if len(result.answer) > 1500 else result.answer)

    asyncio.run(test())
