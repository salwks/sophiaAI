"""
Sophia AI: Agentic Orchestrator (Phase 7.9)
=============================================
복합 질문 분해 및 병렬 처리 오케스트레이터

PhysicsTriageLayer → QueryPlanner → MammoPhysicsSolver → Executors → ResponseSynthesizer

Phase 7.9 (Physics Triage Layer):
- PhysicsClassifier: 질문 → Phase 1-5 매핑 (dual-path: 키워드 + 의미)
- SolverRouter: Phase → solver 메서드 호출 + 파라미터 자동 추출
- FrameworkInjector: solver 결과를 풀이 전략으로 변환 (정답 미포함)
- PostVerifier: LLM 답변 vs solver 정답 비교 (multi-phase)

Phase 7.8 (Layer 2 Deterministic Solver, legacy fallback):
- MammoPhysicsSolver: 결정론적 물리 계산 (Constitutional Axioms 기반)
- Hard-Gate: 오차 > 1% 시 Rejection
- Rejection Loop: 최대 2회 재생성

Reference: Numina-Lean-Agent (2026), 3-Layer Knowledge Internalization
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from src.reasoning.planner import (
    QueryPlanner, DecompositionResult, SubTask, TaskType,
    ExtractedConstraints, get_query_planner
)
from src.reasoning.synthesizer import (
    ResponseSynthesizer, SynthesizedReport,
    get_response_synthesizer
)
from src.reasoning.text_logic import TextReasoningEngine, RefinedAnswer
from src.reasoning.math_verifier import (
    MathVerifier, VerificationResult, VerificationStatus,
    PhysicsCalculation, get_math_verifier
)
from src.reasoning.reflective_engine import (
    ReflectivePromptGenerator, ReflectionHint,
    get_reflective_hint_generator
)
from src.knowledge.manager import KnowledgeManager
from src.reasoning.physics_triage import (
    PhysicsTriageLayer, PhysicsDomain,
    ClassificationResult, SolverResult, PostVerificationResult,
    get_physics_triage
)

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    """오케스트레이터 설정"""
    enable_decomposition: bool = True  # 분해 활성화 여부
    parallel_execution: bool = False   # 병렬 실행 (현재 비활성화)
    use_llm_synthesis: bool = False    # LLM 합성 사용 여부
    # Phase 7.9: Physics Triage Layer
    enable_physics_triage: bool = True     # 물리 Triage 활성화
    # Phase 7.8: Math Verification (Layer 2 Deterministic Solver)
    enable_math_verification: bool = True  # 수식 검증 활성화
    max_rejection_retries: int = 2         # 최대 재시도 횟수
    verification_tolerance: float = 1.0    # 허용 오차 (%) - Layer 2: 5% → 1%


@dataclass
class OrchestrationResult:
    """오케스트레이션 결과"""
    final_answer: str
    decomposition: DecompositionResult
    subtask_results: List[RefinedAnswer]
    synthesis_report: Optional[SynthesizedReport]
    is_decomposed: bool
    total_confidence: float
    # Phase 7.6.2: Verification info (legacy)
    verification_results: List[VerificationResult] = field(default_factory=list)
    physics_calculation: Optional[PhysicsCalculation] = None
    rejection_count: int = 0
    # Phase 7.9: Physics Triage info
    triage_domain: Optional[str] = None
    triage_confidence: float = 0.0
    triage_solver_result: Any = None  # SolverResult or Dict[PhysicsDomain, SolverResult]
    post_verification: Optional[PostVerificationResult] = None
    is_multi_domain: bool = False


class AgenticOrchestrator:
    """
    에이전틱 오케스트레이터

    복합 질문을 분해하여 각 하위 작업을 독립적으로 처리하고,
    결과를 합성하여 최종 답변을 생성합니다.
    """

    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
        planner: Optional[QueryPlanner] = None,
        synthesizer: Optional[ResponseSynthesizer] = None,
        reasoning_engine: Optional[TextReasoningEngine] = None,
        knowledge_manager: Optional[KnowledgeManager] = None,
        math_verifier: Optional[MathVerifier] = None,
        physics_triage: Optional[PhysicsTriageLayer] = None
    ):
        self.config = config or OrchestratorConfig()
        self.planner = planner or get_query_planner()
        self.synthesizer = synthesizer or get_response_synthesizer()
        self.reasoning_engine = reasoning_engine or TextReasoningEngine()
        self.knowledge_manager = knowledge_manager or KnowledgeManager()
        # Phase 7.6.2: Math Verifier (legacy fallback)
        self.math_verifier = math_verifier or get_math_verifier()
        # Phase 7.6.3: Reflective Hint Generator
        self.hint_generator = get_reflective_hint_generator()
        # Phase 7.9: Physics Triage Layer
        self.physics_triage = physics_triage or get_physics_triage()

    def process(
        self,
        question: str,
        context: str = "",
        physics_knowledge: str = ""
    ) -> OrchestrationResult:
        """
        질문 처리 (분해 포함)

        Args:
            question: 사용자 질문
            context: 검색된 문서 컨텍스트
            physics_knowledge: 물리 지식

        Returns:
            OrchestrationResult
        """
        logger.info(f"Orchestrating: {question[:50]}...")

        # 1. 질문 분해
        if self.config.enable_decomposition:
            decomposition = self.planner.decompose(question)
        else:
            decomposition = self._create_single_task(question)

        # 2. 복합 질문인 경우 각 subtask 개별 처리
        if decomposition.is_complex and len(decomposition.subtasks) > 1:
            logger.info(f"Complex query: processing {len(decomposition.subtasks)} subtasks")
            return self._process_complex(decomposition, context, physics_knowledge)

        # 3. 단순 질문은 기존 방식으로 처리
        logger.info("Simple query: direct processing")
        return self._process_simple(decomposition, context, physics_knowledge)

    def _create_single_task(self, question: str) -> DecompositionResult:
        """단일 작업 생성 (분해 비활성화 시)"""
        from src.reasoning.planner import SubTask, TaskType

        return DecompositionResult(
            original_query=question,
            is_complex=False,
            subtasks=[
                SubTask(
                    id=1,
                    type=TaskType.CONCEPT,
                    query=question,
                    required_modules=[]
                )
            ],
            reasoning="분해 비활성화"
        )

    def _process_simple(
        self,
        decomposition: DecompositionResult,
        context: str,
        physics_knowledge: str
    ) -> OrchestrationResult:
        """
        단순 질문 처리 (Dual-Track B→C)

        Flow:
        1. PhysicsTriage.pre_solve_explain(): 분류 + solver + 설명 유도 프롬프트
        2. LLM reasoning: 물리적 해석/설명 (수치 계산 ❌)
        3. PostVerifier: 품질 검증 (키워드 기반, 수치 비교 ❌)
        4. 최종 답변 = LLM 해석 + Solver 확정 수치 (Dual-Track C)
        """
        task = decomposition.subtasks[0]
        constraints = decomposition.constraints
        verification_results = []
        total_rejections = 0

        # Dual-Track: Physics Triage (Explain Mode)
        explain_prompt = None
        solver_result = None
        classification = None
        solver_summary = None
        post_verification = None

        if self.config.enable_physics_triage:
            explain_prompt, solver_result, classification, solver_summary = \
                self.physics_triage.pre_solve_explain(task.query)

            if explain_prompt:
                logger.info(
                    f"PhysicsTriage Dual-Track: domain={classification.primary_domain.value}, "
                    f"confidence={classification.confidence:.2f}, "
                    f"multi_domain={isinstance(solver_result, dict)}"
                )

        # 프롬프트 구성: 설명 유도 프롬프트 주입
        enhanced_knowledge = physics_knowledge
        if explain_prompt:
            # Dual-Track: Explain mode (Solver 결과 포함, 계산 요구 ❌)
            enhanced_knowledge = f"{explain_prompt}\n{physics_knowledge}"
        elif task.type == TaskType.MATH and constraints and constraints.dose_ratio:
            # Legacy fallback (Phase 7.8): 기존 Double-Anchor
            primary_anchor = f"""
╔══════════════════════════════════════════════════════════════╗
║  🔒 CRITICAL CONSTRAINTS - 반드시 이 조건을 적용하세요       ║
╠══════════════════════════════════════════════════════════════╣
║  • 선량 감소: {(1-constraints.dose_ratio)*100:.0f}%                                          ║
║  • 전자 노이즈: {(constraints.electronic_noise_fraction or 0)*100:.0f}%                                    ║
╚══════════════════════════════════════════════════════════════╝
"""
            enhanced_knowledge = f"{primary_anchor}\n{physics_knowledge}"

        # LLM 추론 (Dual-Track: 1회 실행, rejection loop 불필요)
        result = self.reasoning_engine.reason(
            question=task.query,
            context=context,
            physics_knowledge=enhanced_knowledge,
            require_derivation=(solver_result is not None or task.type == TaskType.MATH)
        )

        # Dual-Track 품질 검증 (수치 비교 ❌, 키워드 기반 ✅)
        if solver_result and classification and self.config.enable_math_verification:
            if isinstance(solver_result, dict):
                # Multi-domain: 대표 도메인으로 품질 검증
                representative_result = list(solver_result.values())[0]
                post_verification = self.physics_triage.post_verify_quality(
                    result.content, representative_result, classification
                )
            else:
                # 단일 도메인: 품질 검증
                verify_text = result.content
                if hasattr(result, 'derivation') and result.derivation:
                    verify_text = f"{result.content}\n{result.derivation}"
                post_verification = self.physics_triage.post_verify_quality(
                    verify_text, solver_result, classification
                )

            if post_verification.should_reject:
                # 품질 미달: 1회 재시도 (설명 보강 힌트)
                logger.warning(
                    f"Quality check failed: {post_verification.explanation}"
                )
                total_rejections += 1
                quality_hint = (
                    f"\n\n⚠️ [품질 보강 요청]\n"
                    f"이전 답변이 물리적 설명으로 불충분합니다.\n"
                    f"다음을 보강하세요:\n"
                    f"- 핵심 물리 원칙의 구체적 설명\n"
                    f"- EID vs PCD 비교 분석\n"
                    f"- 실무적/임상적 시사점\n"
                    f"- Solver 결과의 물리적 의미 해석\n"
                )
                result = self.reasoning_engine.reason(
                    question=task.query,
                    context=context,
                    physics_knowledge=f"{enhanced_knowledge}\n{quality_hint}",
                    require_derivation=True
                )
            else:
                logger.info(f"Quality check PASSED: {post_verification.explanation}")

        # Legacy fallback: Phase 7.8 MathVerifier (triage 없을 때)
        elif (not solver_result and
              task.type == TaskType.MATH and
              self.config.enable_math_verification and
              constraints and constraints.dose_ratio):

            verification = self.math_verifier.verify_snr_calculation(
                result.content,
                constraints.dose_ratio,
                constraints.electronic_noise_fraction or 0.0
            )
            verification_results.append(verification)

            if verification.should_reject:
                total_rejections += 1
                logger.warning(
                    f"Legacy MathVerifier FAILED: "
                    f"LLM={verification.llm_value}, Verified={verification.verified_value:.1f}%"
                )
                if self.config.max_rejection_retries > 0:
                    hint = self.hint_generator.generate_reflection_hint(
                        task_type="MATH",
                        llm_answer=result.content,
                        llm_value=verification.llm_value,
                        constraints={
                            "dose_ratio": constraints.dose_ratio,
                            "electronic_noise_fraction": constraints.electronic_noise_fraction or 0.0,
                            "raw_text": constraints.raw_text
                        },
                        error_description=verification.explanation
                    )
                    reflection_prompt = self.hint_generator.format_reflection_prompt(
                        result.content, hint, 1
                    )
                    result = self.reasoning_engine.reason(
                        question=task.query,
                        context=context,
                        physics_knowledge=f"{enhanced_knowledge}\n\n{reflection_prompt}",
                        require_derivation=True
                    )

        # Dual-Track C: 최종 답변 조합 (LLM 해석 + Solver 확정 수치)
        task.answer = result.content
        task.confidence = result.final_confidence

        # 포맷팅된 답변 생성
        formatted = self.reasoning_engine.format_structured_output(result)

        # Track C: Solver 수치 요약 첨부
        if solver_summary:
            formatted = f"{formatted}\n\n{solver_summary}"

        return OrchestrationResult(
            final_answer=formatted,
            decomposition=decomposition,
            subtask_results=[result],
            synthesis_report=None,
            is_decomposed=False,
            total_confidence=result.final_confidence,
            verification_results=verification_results,
            physics_calculation=None,
            rejection_count=total_rejections,
            triage_domain=classification.primary_domain.value if classification else None,
            triage_confidence=classification.confidence if classification else 0.0,
            triage_solver_result=solver_result,
            post_verification=post_verification,
            is_multi_domain=isinstance(solver_result, dict)
        )

    def _process_complex(
        self,
        decomposition: DecompositionResult,
        context: str,
        physics_knowledge: str
    ) -> OrchestrationResult:
        """
        복합 질문 처리 (Phase 7.9: Physics Triage per subtask)

        각 subtask에 대해 개별적으로 Triage를 수행하여
        적절한 풀이 전략을 주입하고 검증합니다.
        """
        subtask_results = []
        verification_results = []
        total_rejections = 0
        # 전체 질문에 대한 triage (첫 번째 MATH subtask 기준 보고용)
        overall_classification = None
        overall_solver_result = None
        overall_post_verification = None

        constraints = decomposition.constraints

        for task in decomposition.subtasks:
            logger.info(f"Processing subtask {task.id}: {task.type.value}")

            # 각 subtask에 맞는 지식 모듈 선택
            task_knowledge = self._get_task_knowledge(task)
            base_knowledge = task_knowledge or physics_knowledge

            # Dual-Track: 각 subtask에 대해 개별 Triage (Explain Mode)
            explain_prompt = None
            solver_result = None
            classification = None
            solver_summary = None
            post_verification = None

            if self.config.enable_physics_triage:
                explain_prompt, solver_result, classification, solver_summary = \
                    self.physics_triage.pre_solve_explain(task.query)

                if explain_prompt:
                    logger.info(
                        f"Subtask {task.id} triage (Dual-Track): "
                        f"domain={classification.primary_domain.value}, "
                        f"confidence={classification.confidence:.2f}"
                    )
                    if overall_classification is None:
                        overall_classification = classification
                        overall_solver_result = solver_result

            # 프롬프트 구성
            enhanced_knowledge = base_knowledge
            if explain_prompt:
                enhanced_knowledge = f"{explain_prompt}\n{base_knowledge}"
            elif task.type == TaskType.MATH and constraints and constraints.dose_ratio:
                primary_anchor, secondary_anchor = self.planner.format_double_anchor_prompt(
                    constraints, None
                )
                enhanced_knowledge = f"{primary_anchor}\n{base_knowledge}\n{secondary_anchor}"

            # LLM 추론 (Dual-Track: 1회 실행)
            result = None
            try:
                result = self.reasoning_engine.reason(
                    question=task.query,
                    context=context,
                    physics_knowledge=enhanced_knowledge,
                    require_derivation=(solver_result is not None or task.type == TaskType.MATH)
                )

                # Dual-Track 품질 검증
                if solver_result and classification and self.config.enable_math_verification:
                    target_result = (
                        list(solver_result.values())[0] if isinstance(solver_result, dict)
                        else solver_result
                    )
                    verify_text = result.content
                    if hasattr(result, 'derivation') and result.derivation:
                        verify_text = f"{result.content}\n{result.derivation}"
                    post_verification = self.physics_triage.post_verify_quality(
                        verify_text, target_result, classification
                    )
                    if post_verification.should_reject:
                        total_rejections += 1
                        logger.warning(
                            f"Subtask {task.id} quality check failed: "
                            f"{post_verification.explanation}"
                        )

                # Legacy fallback
                elif (not solver_result and
                      task.type == TaskType.MATH and
                      self.config.enable_math_verification and
                      constraints and constraints.dose_ratio):
                    verification = self.math_verifier.verify_snr_calculation(
                        result.content,
                        constraints.dose_ratio,
                        constraints.electronic_noise_fraction or 0.0
                    )
                    verification_results.append(verification)

            except Exception as e:
                logger.error(f"Subtask {task.id} failed: {e}")
                result = None

            # 결과 저장
            if result:
                task.answer = result.content
                # Track C: Solver 수치 요약 첨부
                if solver_summary:
                    task.answer = f"{result.content}\n\n{solver_summary}"

                task.confidence = result.final_confidence
                subtask_results.append(result)
                logger.info(f"Subtask {task.id} completed: confidence={result.final_confidence:.0%}")

                if post_verification and overall_post_verification is None:
                    overall_post_verification = post_verification
            else:
                task.answer = "분석 실패"
                task.confidence = 0.0

        # 결과 합성
        if self.config.use_llm_synthesis:
            synthesis_report = self.synthesizer.synthesize_with_llm(decomposition)
        else:
            synthesis_report = self.synthesizer.synthesize(decomposition)

        return OrchestrationResult(
            final_answer=synthesis_report.content,
            decomposition=decomposition,
            subtask_results=subtask_results,
            synthesis_report=synthesis_report,
            is_decomposed=True,
            total_confidence=synthesis_report.total_confidence,
            verification_results=verification_results,
            physics_calculation=None,
            rejection_count=total_rejections,
            triage_domain=overall_classification.primary_domain.value if overall_classification else None,
            triage_confidence=overall_classification.confidence if overall_classification else 0.0,
            triage_solver_result=overall_solver_result,
            post_verification=overall_post_verification
        )

    def _inject_verified_value(self, content: str, verified_value: float) -> str:
        """검증된 수치를 답변에 주입"""
        import re

        # 기존 퍼센트 값을 검증된 값으로 대체
        pattern = r'(\d+(?:\.\d+)?)\s*%\s*(감소|하락|저하)'
        replacement = f"{verified_value:.1f}% \\2 (MathVerifier 검증 완료)"

        modified = re.sub(pattern, replacement, content, count=1)

        # 변경이 없으면 끝에 추가
        if modified == content:
            modified += f"\n\n**📊 MathVerifier 검증 결과**: SNR {verified_value:.1f}% 감소"

        return modified

    def _get_task_knowledge(self, task: SubTask) -> str:
        """subtask에 맞는 지식 모듈 로드"""
        if not task.required_modules:
            return ""

        # required_modules에 지정된 모듈들 로드
        knowledge_parts = []
        for module_id in task.required_modules:
            if module_id == "PMC":
                continue  # PMC는 별도 처리

            module = self.knowledge_manager.get_knowledge_by_id(module_id)
            if module:
                formatted = self.knowledge_manager.format_for_context([module])
                knowledge_parts.append(formatted)

        return "\n\n".join(knowledge_parts)


# =============================================================================
# Singleton
# =============================================================================

_orchestrator_instance: Optional[AgenticOrchestrator] = None


def get_agentic_orchestrator() -> AgenticOrchestrator:
    """AgenticOrchestrator 싱글톤"""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = AgenticOrchestrator()
    return _orchestrator_instance


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    orchestrator = AgenticOrchestrator()

    # 복합 질문 테스트
    complex_query = """차세대 맘모그래피 시스템에 광자 계수 검출기(PCD) 도입을 검토 중이다.
기존 EID 시스템에서 MGD를 50% 감축했을 때, 전자 노이즈가 전체 노이즈의 30%를 차지하게 된다면
Rose Criterion(k=5)을 만족하기 위한 SNR의 하락폭을 수식으로 증명하시오.

동일한 50% 저선량 환경에서 PCD를 사용할 경우, 전자 노이즈 제거와 에너지 가중치 최적화가
d'(Detectability Index)를 어떻게 회복시키는지 기술하고, 이것이 미세 석회화의 대조도 향상에
미치는 영향을 최근 3년 내 PMC 논문을 근거로 논하시오."""

    result = orchestrator.process(
        question=complex_query,
        context="PCD provides energy resolution.",
        physics_knowledge=""
    )

    print(f"\n{'='*60}")
    print(f"Is Decomposed: {result.is_decomposed}")
    print(f"Total Confidence: {result.total_confidence:.0%}")
    print(f"Subtasks: {len(result.decomposition.subtasks)}")
    print(f"\n{'='*60}")
    print("FINAL ANSWER:")
    print(f"{'='*60}")
    print(result.final_answer[:2000])
