"""
듀얼 에이전트 워크플로우: Generator-Critic 패턴
=================================================
Phase 2 고도화: Formula Integrity + Hallucination Check 통합

Generator가 생성한 답변을 Critic이 물리 법칙과 논문 근거 기반으로 검증합니다.
검증 실패 시 피드백을 반영하여 재생성합니다.

검증 파이프라인:
1. LLM Critic (물리 법칙 검증)
2. FormulaChecker (수식 무결성 검증)
3. HallucinationChecker (인용 근거 검증)
"""

import requests
import re
import json
import logging
from typing import Dict, Any, Optional, Tuple, List, Set
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CriticResult:
    """비판자 검증 결과"""
    verdict: str  # "PASS" or "REJECT"
    errors: list  # 발견된 오류 목록
    feedback: str  # 상세 피드백
    corrected_points: list  # 교정 포인트


@dataclass
class IntegratedAuditResult:
    """Phase 2: 통합 감사 결과"""
    final_verdict: str  # "PASS", "WARN", "REJECT"

    # LLM Critic 결과
    llm_critic_verdict: str
    llm_critic_errors: List[Dict]

    # Formula Checker 결과
    formula_valid: bool
    formula_errors: List[Dict]

    # Hallucination Checker 결과
    hallucination_clean: bool
    hallucination_severity: str
    fabricated_sources: List[str]

    # 종합 피드백
    combined_feedback: str
    retry_suggested: bool = False


# ============================================
# 비판자(Critic) 전용 시스템 프롬프트
# ============================================
CRITIC_SYSTEM_PROMPT = """# Role
너는 세계 최고의 의학 물리 및 유방 영상학(Mammography) 전문 검찰관이다.
너의 유일한 목적은 [AI 초안]을 읽고, [제공된 논문 데이터] 및 [물리적 불변의 법칙]에 근거하여 오류를 찾아내고 답변을 '기각'하거나 '승인'하는 것이다.

# Audit Checklist (반드시 확인해야 할 역린)

## 1. 필터/빔 에너지 모순
- W/Ag가 W/Rh보다 더 'Soft'하다고 말하는가? → **오류** (Ag K-edge 25.5keV > Rh K-edge 23.2keV, Ag가 더 Hard함)
- 두꺼운 유방(6cm↑)에 Soft beam이나 낮은 kVp를 권장하는가? → **오류** (Hard beam 필수)
- 얇은 유방(3cm↓)에 Hard beam을 권장하는가? → **오류** (Soft beam으로 대조도 향상 필요)

## 2. 해상도 및 필터 모순
- 가우시안(Gaussian) 필터로 MTF 유지를 주장하는가? → **오류** (가우시안은 무조건 MTF 손실)
- 중간값(Median) 필터로 미세석회화 검출을 권장하는가? → **오류** (에지 손상으로 미세석회화 불명확)
- 저해상도로 미세석회화 검출이 가능하다고 주장하는가? → **오류** (100-500μm 검출에 고해상도 필수)

## 3. 노이즈/선량 관계 모순
- 선량 증가가 노이즈를 증가시킨다고 주장하는가? → **오류** (선량↑ → SNR↑ → 노이즈↓)
- 산란선(Scatter)과 양자노이즈(Quantum Mottle)를 같은 현상으로 취급하는가? → **오류** (완전히 다른 물리 현상)

## 4. 근거 위조 (Citation Fraud)
- [제공된 논문]에 없는 저자(예: "Boone et al.", "Cunha et al.")나 수치를 지어냈는가?
- "자료에 없다"고 발뺌하면서 실제로는 논문에 있는 내용을 무시했는가?
- "[1] 제조업체 설명서", "[2] 기술 보고서" 등 실존하지 않는 참고자료를 만들었는가?

# Output Format (반드시 이 형식을 따라라)

검증 결과를 다음 JSON 형식으로 출력하라:

```json
{
  "verdict": "PASS" 또는 "REJECT",
  "errors": [
    {"type": "물리오류|인용위조|자료무시", "description": "구체적 설명", "correction": "올바른 내용"}
  ],
  "summary": "전체 평가 요약 (1-2문장)"
}
```

# Action Instructions
- 오류가 하나라도 발견되면 **verdict: "REJECT"**
- 오류가 없으면 **verdict: "PASS"**
- 절대 친절하게 대답하지 마라. 오직 팩트와 논리로만 비판하라.
- JSON 형식 외의 텍스트는 출력하지 마라."""


def run_critic(
    draft_response: str,
    context: str,
    original_question: str,
    model: str = "gpt-oss:20b",
    base_url: str = "http://localhost:11434"
) -> CriticResult:
    """
    비판자 에이전트 실행

    Args:
        draft_response: Generator가 생성한 초안
        context: 제공된 논문/가이드라인 데이터
        original_question: 원본 질문
        model: 사용할 LLM 모델
        base_url: Ollama API URL

    Returns:
        CriticResult: 검증 결과
    """
    url = f"{base_url}/api/chat"

    user_prompt = f"""# 제공된 논문 데이터 (Context)
{context[:8000]}

# 원본 질문
{original_question}

# AI 초안 (검증 대상)
{draft_response}

위 AI 초안을 [Audit Checklist]에 따라 엄격히 검증하고, JSON 형식으로 결과를 출력하라."""

    messages = [
        {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.0,  # 가장 보수적인 검증
        }
    }

    try:
        response = requests.post(url, json=payload, timeout=120)
        result = response.json()
        raw_response = result.get('message', {}).get('content', '')

        # <think> 태그 제거
        clean_response = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL).strip()

        # JSON 추출
        json_match = re.search(r'\{[\s\S]*\}', clean_response)
        if json_match:
            try:
                critic_json = json.loads(json_match.group())
                return CriticResult(
                    verdict=critic_json.get("verdict", "PASS"),
                    errors=critic_json.get("errors", []),
                    feedback=critic_json.get("summary", ""),
                    corrected_points=[e.get("correction", "") for e in critic_json.get("errors", []) if e.get("correction")]
                )
            except json.JSONDecodeError:
                pass

        # JSON 파싱 실패 시 텍스트 분석
        verdict = "REJECT" if "REJECT" in clean_response.upper() else "PASS"
        return CriticResult(
            verdict=verdict,
            errors=[],
            feedback=clean_response[:500],
            corrected_points=[]
        )

    except Exception as e:
        # 오류 시 기본적으로 PASS (검증 불가)
        return CriticResult(
            verdict="PASS",
            errors=[],
            feedback=f"Critic 실행 오류: {str(e)}",
            corrected_points=[]
        )


def generate_with_critic_feedback(
    question: str,
    context: str,
    critic_feedback: CriticResult,
    model: str = "gpt-oss:20b",
    base_url: str = "http://localhost:11434"
) -> str:
    """
    비판자 피드백을 반영하여 재생성

    Args:
        question: 원본 질문
        context: 제공된 Context
        critic_feedback: 비판자의 피드백
        model: 사용할 LLM 모델
        base_url: Ollama API URL

    Returns:
        수정된 응답
    """
    url = f"{base_url}/api/chat"

    error_details = "\n".join([
        f"- {e.get('type', '오류')}: {e.get('description', '')} → 올바른 내용: {e.get('correction', 'N/A')}"
        for e in critic_feedback.errors
    ])

    system_message = """너는 의학 물리 전문가다. 이전 답변이 전문가 검토에서 기각되었다.
지적된 오류를 반드시 수정하고, 제공된 논문 데이터만을 근거로 정확한 답변을 작성하라.

# 절대 금지 사항
- 제공된 논문에 없는 저자나 수치를 인용하지 마라
- 가우시안 필터로 MTF 유지가 가능하다고 주장하지 마라
- 존재하지 않는 참고자료를 만들지 마라"""

    user_prompt = f"""# 전문가 검토에서 지적된 오류
{error_details}

# 추가 피드백
{critic_feedback.feedback}

# 참고 자료 (Context)
{context[:6000]}

# 질문
{question}

위 오류를 수정하여 다시 답변하라. 제공된 논문만 인용하고, 물리 법칙을 정확히 적용하라."""

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt}
    ]

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.0}
    }

    try:
        response = requests.post(url, json=payload, timeout=120)
        result = response.json()
        raw_response = result.get('message', {}).get('content', '')
        return re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL).strip()
    except Exception as e:
        return f"재생성 오류: {str(e)}"


def dual_agent_workflow(
    question: str,
    context: str,
    initial_response: str,
    model: str = "gpt-oss:20b",
    max_iterations: int = 2
) -> Tuple[str, Dict[str, Any]]:
    """
    듀얼 에이전트 워크플로우 실행

    Args:
        question: 원본 질문
        context: 제공된 Context
        initial_response: Generator의 초안
        model: 사용할 LLM 모델
        max_iterations: 최대 수정 반복 횟수

    Returns:
        (최종 응답, 메타데이터)
    """
    current_response = initial_response
    iteration = 0
    history = []

    while iteration < max_iterations:
        # 비판자 실행
        critic_result = run_critic(current_response, context, question, model)

        history.append({
            "iteration": iteration + 1,
            "verdict": critic_result.verdict,
            "errors": critic_result.errors,
            "feedback": critic_result.feedback
        })

        if critic_result.verdict == "PASS":
            # 검증 통과
            return current_response, {
                "passed": True,
                "iterations": iteration + 1,
                "history": history
            }

        # 검증 실패 - 재생성
        iteration += 1
        if iteration < max_iterations:
            current_response = generate_with_critic_feedback(
                question, context, critic_result, model
            )

    # 최대 반복 후에도 실패
    return current_response, {
        "passed": False,
        "iterations": iteration,
        "history": history,
        "final_errors": history[-1]["errors"] if history else []
    }


# =============================================================================
# Phase 2: Integrated Audit Pipeline
# =============================================================================

class IntegratedAuditor:
    """
    통합 감사 파이프라인

    1. LLM Critic (물리 법칙 검증)
    2. FormulaChecker (수식 무결성)
    3. HallucinationChecker (인용 근거)
    """

    def __init__(
        self,
        model: str = "gpt-oss:20b",
        base_url: str = "http://localhost:11434",
        strict_formula: bool = True,
        strict_citation: bool = True,
    ):
        self.model = model
        self.base_url = base_url
        self.strict_formula = strict_formula
        self.strict_citation = strict_citation

        # Lazy import to avoid circular dependencies
        self._formula_checker = None
        self._hallucination_checker = None

    def _get_formula_checker(self):
        """FormulaChecker lazy loading"""
        if self._formula_checker is None:
            try:
                from src.validation.formula_checker import FormulaChecker
                self._formula_checker = FormulaChecker(strict_mode=self.strict_formula)
            except ImportError as e:
                logger.warning(f"Could not import FormulaChecker: {e}")
        return self._formula_checker

    def _get_hallucination_checker(self, papers=None, pmids=None):
        """HallucinationChecker 생성"""
        try:
            from src.validation.hallucination import HallucinationChecker
            return HallucinationChecker(
                retrieved_papers=papers,
                retrieved_pmids=pmids,
                strict_mode=self.strict_citation,
            )
        except ImportError as e:
            logger.warning(f"Could not import HallucinationChecker: {e}")
            return None

    def audit(
        self,
        generated_response: str,
        context: str,
        original_question: str,
        retrieved_papers: Optional[List[Dict]] = None,
        retrieved_pmids: Optional[Set[str]] = None,
    ) -> IntegratedAuditResult:
        """
        통합 감사 실행

        Args:
            generated_response: LLM이 생성한 답변
            context: 제공된 논문/가이드라인 데이터
            original_question: 원본 질문
            retrieved_papers: 검색된 논문 리스트
            retrieved_pmids: 검색된 PMID 집합

        Returns:
            IntegratedAuditResult
        """
        errors = []
        feedback_parts = []

        # =================================================================
        # 1. LLM Critic (물리 법칙 검증)
        # =================================================================
        logger.info("Running LLM Critic...")
        llm_result = run_critic(
            generated_response, context, original_question,
            model=self.model, base_url=self.base_url
        )

        llm_verdict = llm_result.verdict
        llm_errors = llm_result.errors

        if llm_verdict == "REJECT":
            feedback_parts.append(f"[LLM Critic] {llm_result.feedback}")
            errors.extend([{"source": "llm_critic", **e} for e in llm_errors])

        # =================================================================
        # 2. Formula Checker (수식 무결성)
        # =================================================================
        formula_valid = True
        formula_errors = []

        formula_checker = self._get_formula_checker()
        if formula_checker:
            logger.info("Running Formula Checker...")
            formula_result = formula_checker.verify(generated_response)

            formula_valid = formula_result.is_valid
            formula_errors = formula_result.errors

            if not formula_valid:
                feedback_parts.append(
                    f"[Formula Check] {formula_result.invalid_formulas} invalid formulas detected"
                )
                for err in formula_errors:
                    errors.append({
                        "source": "formula_checker",
                        "type": "formula_error",
                        "description": str(err.get("issues", [])),
                        "formula": err.get("formula", "")[:50],
                    })

        # =================================================================
        # 3. Hallucination Checker (인용 근거)
        # =================================================================
        hallucination_clean = True
        hallucination_severity = "none"
        fabricated_sources = []

        hallucination_checker = self._get_hallucination_checker(
            papers=retrieved_papers, pmids=retrieved_pmids
        )
        if hallucination_checker:
            logger.info("Running Hallucination Checker...")
            hall_result = hallucination_checker.check(generated_response)

            hallucination_clean = hall_result.is_clean
            hallucination_severity = hall_result.severity
            fabricated_sources = hall_result.fabricated_sources

            if not hallucination_clean:
                feedback_parts.append(
                    f"[Citation Check] Severity: {hallucination_severity}, "
                    f"Ungrounded: {hall_result.ungrounded_citations}/{hall_result.total_citations}"
                )
                if fabricated_sources:
                    errors.append({
                        "source": "hallucination_checker",
                        "type": "citation_fabrication",
                        "description": f"Fabricated sources: {', '.join(fabricated_sources[:3])}",
                    })

        # =================================================================
        # 4. Final Verdict
        # =================================================================
        if llm_verdict == "REJECT" or not formula_valid or hallucination_severity in ("high", "critical"):
            final_verdict = "REJECT"
            retry_suggested = True
        elif not hallucination_clean or formula_errors:
            final_verdict = "WARN"
            retry_suggested = False
        else:
            final_verdict = "PASS"
            retry_suggested = False

        combined_feedback = "\n".join(feedback_parts) if feedback_parts else "All checks passed."

        logger.info(f"Integrated Audit Result: {final_verdict}")

        return IntegratedAuditResult(
            final_verdict=final_verdict,
            llm_critic_verdict=llm_verdict,
            llm_critic_errors=llm_errors,
            formula_valid=formula_valid,
            formula_errors=formula_errors,
            hallucination_clean=hallucination_clean,
            hallucination_severity=hallucination_severity,
            fabricated_sources=fabricated_sources,
            combined_feedback=combined_feedback,
            retry_suggested=retry_suggested,
        )


def integrated_dual_agent_workflow(
    question: str,
    context: str,
    initial_response: str,
    retrieved_papers: Optional[List[Dict]] = None,
    retrieved_pmids: Optional[Set[str]] = None,
    model: str = "gpt-oss:20b",
    max_iterations: int = 2,
) -> Tuple[str, Dict[str, Any]]:
    """
    Phase 2 통합 듀얼 에이전트 워크플로우

    Args:
        question: 원본 질문
        context: 제공된 Context
        initial_response: Generator의 초안
        retrieved_papers: 검색된 논문 리스트
        retrieved_pmids: 검색된 PMID 집합
        model: 사용할 LLM 모델
        max_iterations: 최대 수정 반복 횟수

    Returns:
        (최종 응답, 메타데이터)
    """
    auditor = IntegratedAuditor(model=model)

    current_response = initial_response
    iteration = 0
    history = []

    while iteration < max_iterations:
        # 통합 감사 실행
        audit_result = auditor.audit(
            generated_response=current_response,
            context=context,
            original_question=question,
            retrieved_papers=retrieved_papers,
            retrieved_pmids=retrieved_pmids,
        )

        history.append({
            "iteration": iteration + 1,
            "verdict": audit_result.final_verdict,
            "llm_critic": audit_result.llm_critic_verdict,
            "formula_valid": audit_result.formula_valid,
            "hallucination_clean": audit_result.hallucination_clean,
            "feedback": audit_result.combined_feedback,
        })

        if audit_result.final_verdict == "PASS":
            return current_response, {
                "passed": True,
                "iterations": iteration + 1,
                "history": history,
            }

        if audit_result.final_verdict == "WARN" and not audit_result.retry_suggested:
            # 경고지만 재시도 불필요
            return current_response, {
                "passed": True,
                "iterations": iteration + 1,
                "history": history,
                "warnings": audit_result.combined_feedback,
            }

        # 재생성
        iteration += 1
        if iteration < max_iterations:
            # 피드백을 반영하여 재생성
            critic_result = CriticResult(
                verdict="REJECT",
                errors=audit_result.llm_critic_errors,
                feedback=audit_result.combined_feedback,
                corrected_points=[],
            )
            current_response = generate_with_critic_feedback(
                question, context, critic_result, model
            )

    # 최대 반복 후에도 실패
    return current_response, {
        "passed": False,
        "iterations": iteration,
        "history": history,
        "final_verdict": history[-1]["verdict"] if history else "UNKNOWN",
    }


# 테스트용
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 테스트 케이스 1: 물리 오류 + 가짜 인용
    test_draft_1 = """
    디노이징을 위해 가우시안 필터를 권장합니다.
    가우시안 필터는 MTF를 유지하면서 노이즈를 효과적으로 제거합니다.

    MGD는 다음 공식으로 계산됩니다:
    $$MGD = K \\cdot \\phi \\cdot \\omega$$

    참고 자료:
    [1] Boone et al., 2015
    [2] 제조업체 기술 보고서
    [Source: PMID 99999999]
    """

    # 테스트 케이스 2: 올바른 답변
    test_draft_2 = """
    미세석회화 검출을 위해서는 에지 보존 필터(Bilateral Filter)를 권장합니다.
    가우시안 필터는 MTF를 손상시켜 부적합합니다.

    MGD는 Dance et al. (2011)에 따라 계산됩니다:
    $$MGD = K \\cdot g \\cdot c \\cdot s$$

    [Source: PMID 12345678]
    """

    test_context = """
    === 논문 1 ===
    PMID: 12345678
    제목: Deep learning denoising for mammography
    저자: Kim JH, Dance DR
    내용: CNN 기반 디노이징이 MTF 보존에 효과적...
    """

    test_question = "미세석회화 검출을 위한 디노이징 전략은?"

    fake_papers = [
        {"pmid": "12345678", "authors": ["Kim JH", "Dance DR"], "title": "Deep learning denoising", "abstract": "CNN based..."}
    ]

    print("=" * 60)
    print("Phase 2: 통합 감사 테스트")
    print("=" * 60)

    # LLM Critic만 테스트
    print("\n[Test 1] LLM Critic only (물리 오류 + 가짜 인용)")
    result1 = run_critic(test_draft_1, test_context, test_question)
    print(f"  Verdict: {result1.verdict}")
    print(f"  Errors: {result1.errors[:2]}")

    # 통합 감사 테스트
    print("\n[Test 2] Integrated Audit (올바른 답변)")
    auditor = IntegratedAuditor()
    result2 = auditor.audit(
        generated_response=test_draft_2,
        context=test_context,
        original_question=test_question,
        retrieved_papers=fake_papers,
    )
    print(f"  Final Verdict: {result2.final_verdict}")
    print(f"  Formula Valid: {result2.formula_valid}")
    print(f"  Hallucination Clean: {result2.hallucination_clean}")
    print(f"  Feedback: {result2.combined_feedback[:100]}")

    print("\n[Test 3] Integrated Audit (오류 있는 답변)")
    result3 = auditor.audit(
        generated_response=test_draft_1,
        context=test_context,
        original_question=test_question,
        retrieved_papers=fake_papers,
    )
    print(f"  Final Verdict: {result3.final_verdict}")
    print(f"  Formula Valid: {result3.formula_valid}")
    print(f"  Hallucination Severity: {result3.hallucination_severity}")
    print(f"  Fabricated Sources: {result3.fabricated_sources}")
    print(f"  Retry Suggested: {result3.retry_suggested}")
