"""
MathVerifier: Symbolic Verification Gate (Phase 7.8)
=====================================================
Layer 2 Deterministic Solver 기반 검증

MammoPhysicsSolver의 결정론적 계산을 기준으로 LLM 수치를 검증합니다.
오차 > 1% 시 Hard-Gate Rejection.

핵심 원리 (Constitutional Axioms):
- Signal ∝ Dose (절대값, 선형)
- σ_quantum² ∝ Dose (포아송 통계, 절대값)
- σ_electronic² = const (하드웨어 특성)
- Hard-Gate: LLM 수치가 Python 계산과 1% 이상 차이 시 Rejection

Reference: Numina-Lean-Agent (2026), 3-Layer Knowledge Internalization
"""

import re
import math
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum

# Sympy 임포트 (심볼릭 수학) - 레거시 호환용
try:
    import sympy as sp
    from sympy import Symbol, sqrt, Rational, simplify, N
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False

from src.reasoning.mammo_physics_solver import MammoPhysicsSolver, get_mammo_solver

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

class VerificationStatus(Enum):
    """검증 상태"""
    PASSED = "passed"           # 물리적으로 타당 (오차 < 5%)
    FAILED = "failed"           # 물리 법칙 위반 또는 오차 > 5%
    CORRECTED = "corrected"     # 자동 수정됨
    UNCERTAIN = "uncertain"     # 수치 추출 실패


@dataclass
class VerificationResult:
    """수식 검증 결과"""
    status: VerificationStatus
    llm_value: Optional[float]          # LLM이 계산한 값
    verified_value: float               # Python/Sympy가 계산한 정답
    error_percent: float                # 오차율 (%)
    physical_law: str                   # 적용된 물리 법칙
    explanation: str                    # 설명
    corrections: List[str] = field(default_factory=list)
    should_reject: bool = False         # Hard-Gate: 재생성 필요 여부


@dataclass
class PhysicsCalculation:
    """물리 계산 결과"""
    eid_snr_reduction: float            # EID SNR 감소율 (%)
    eid_snr_ratio: float                # EID SNR 비율 (0-1)
    pcd_snr_reduction: float            # PCD SNR 감소율 (%)
    pcd_snr_ratio: float                # PCD SNR 비율 (0-1)
    pcd_recovery_percent: float         # PCD의 EID 대비 회복율 (%)
    rose_criterion_eid: bool            # EID가 Rose Criterion 충족?
    rose_criterion_pcd: bool            # PCD가 Rose Criterion 충족?
    derivation_steps: List[str]         # 유도 과정 (LaTeX)


# =============================================================================
# Symbolic Physics Engine (Sympy 기반)
# =============================================================================

class SymbolicPhysicsEngine:
    """
    물리 수식 엔진 (Phase 7.8: MammoPhysicsSolver 위임)

    MammoPhysicsSolver의 결정론적 계산을 사용하며,
    PhysicsCalculation 형식으로 변환하여 반환합니다.

    Constitutional Axioms:
    - Signal ∝ Dose (절대값)
    - σ_quantum² ∝ Dose (포아송, 절대값)
    - σ_electronic² = const
    """

    def __init__(self):
        self._solver = get_mammo_solver()

    def calculate_snr_with_electronic_noise(
        self,
        dose_ratio: float,
        electronic_noise_fraction: float
    ) -> PhysicsCalculation:
        """
        전자 노이즈를 포함한 SNR 계산 (MammoPhysicsSolver 위임)

        핵심 공식:
            SNR_new/SNR_0 = √(D × (1 - f_e × (1-D)))
            (f_e = 선량 변화 후 전자노이즈 분산 비율)

        Args:
            dose_ratio: 새 선량 / 기존 선량 (예: 0.5 = 50% 감소)
            electronic_noise_fraction: 선량 변화 후 전자노이즈 분산 비율 (예: 0.3)

        Returns:
            PhysicsCalculation with all derived values
        """
        # MammoPhysicsSolver로 정확한 계산 수행
        solution = self._solver.solve_snr_with_electronic_noise(
            dose_ratio=dose_ratio,
            electronic_noise_fraction=electronic_noise_fraction
        )

        # PhysicsSolution → PhysicsCalculation 변환
        derivation = []
        for step in solution.derivation_steps:
            derivation.append(f"**Step {step.step_num}: {step.title}**")
            derivation.append(f"$${step.latex}$$")

        return PhysicsCalculation(
            eid_snr_reduction=solution.eid_snr_reduction_pct,
            eid_snr_ratio=solution.eid_snr_ratio,
            pcd_snr_reduction=solution.pcd_snr_reduction_pct,
            pcd_snr_ratio=solution.pcd_snr_ratio,
            pcd_recovery_percent=solution.pcd_recovery_pct,
            rose_criterion_eid=solution.rose_eid_satisfied,
            rose_criterion_pcd=solution.rose_pcd_satisfied,
            derivation_steps=derivation
        )


# =============================================================================
# MathVerifier: Main Class
# =============================================================================

class MathVerifier:
    """
    수식 검증기 (Hard-Gate)

    LLM 출력에서 수치를 추출하고 물리 계산과 대조합니다.
    오차가 5% 이상이면 Rejection하고 재생성을 요청합니다.
    """

    # 허용 오차 (%) - Phase 7.8: 5% → 1% (Layer 2 결정론적 검증)
    TOLERANCE = 1.0

    def __init__(self):
        self.physics = SymbolicPhysicsEngine()
        logger.info("MathVerifier initialized (Sympy available: %s)", SYMPY_AVAILABLE)

    def verify_snr_calculation(
        self,
        llm_answer: str,
        dose_ratio: float,
        electronic_noise_fraction: float
    ) -> VerificationResult:
        """
        SNR 계산 검증 (Hard-Gate)

        Args:
            llm_answer: LLM이 생성한 답변
            dose_ratio: 선량 비율 (예: 0.5)
            electronic_noise_fraction: 전자 노이즈 비율 (예: 0.3)

        Returns:
            VerificationResult with rejection flag if needed
        """
        # 1. 물리 엔진으로 정답 계산
        physics = self.physics.calculate_snr_with_electronic_noise(
            dose_ratio, electronic_noise_fraction
        )
        correct_value = physics.eid_snr_reduction

        # 2. LLM 답변에서 SNR 감소율 추출
        llm_value = self._extract_snr_reduction(llm_answer)

        # 3. 검증
        if llm_value is None:
            return VerificationResult(
                status=VerificationStatus.UNCERTAIN,
                llm_value=None,
                verified_value=correct_value,
                error_percent=100.0,
                physical_law="SNR ∝ Signal/σ_total, σ_total² = σ_quantum² + σ_elec²",
                explanation="LLM 답변에서 SNR 감소율 수치를 추출할 수 없음",
                corrections=[f"정답 제공: SNR {correct_value:.1f}% 감소"],
                should_reject=True  # 재생성 필요
            )

        # 4. 오차 계산
        error = abs(llm_value - correct_value)

        # 5. Hard-Gate: 5% 이상 오차 시 Rejection
        if error > self.TOLERANCE:
            # 물리 법칙 위반 체크 (선량 감소 시 SNR 증가?)
            if llm_value < 0:
                explanation = "❌ 물리 법칙 위반: 선량 감소 시 SNR이 증가할 수 없음"
            else:
                explanation = f"❌ 오차 {error:.1f}% > 허용치 {self.TOLERANCE}%"

            return VerificationResult(
                status=VerificationStatus.FAILED,
                llm_value=llm_value,
                verified_value=correct_value,
                error_percent=error,
                physical_law="σ_total² = σ_quantum² + σ_elec², σ_quantum ∝ 1/√Dose",
                explanation=explanation,
                corrections=[
                    f"LLM 계산: {llm_value:.1f}%",
                    f"정답: {correct_value:.1f}%",
                    f"오차: {error:.1f}%",
                    "⚠️ 답변 폐기 및 재생성 필요"
                ],
                should_reject=True
            )

        # 6. 통과
        return VerificationResult(
            status=VerificationStatus.PASSED,
            llm_value=llm_value,
            verified_value=correct_value,
            error_percent=error,
            physical_law="σ_total² = σ_quantum² + σ_elec²",
            explanation=f"✅ LLM 계산({llm_value:.1f}%)이 정답({correct_value:.1f}%)과 일치 (오차 {error:.1f}%)",
            corrections=[],
            should_reject=False
        )

    def calculate_verified_answer(
        self,
        dose_ratio: float,
        electronic_noise_fraction: float
    ) -> PhysicsCalculation:
        """
        검증된 정답 계산 (프롬프트 주입용)

        이 함수의 결과를 프롬프트에 주입하여 LLM이 반드시 이 수치를 사용하도록 강제합니다.
        """
        return self.physics.calculate_snr_with_electronic_noise(
            dose_ratio, electronic_noise_fraction
        )

    def format_constraint_prompt(
        self,
        dose_ratio: float,
        electronic_noise_fraction: float
    ) -> str:
        """
        Double-Anchor용 제약 조건 프롬프트 생성

        시스템 프롬프트 상단과 답변 직전에 배치합니다.
        """
        calc = self.calculate_verified_answer(dose_ratio, electronic_noise_fraction)

        prompt = f"""
╔══════════════════════════════════════════════════════════════╗
║  🔒 CRITICAL CONSTRAINTS (MathVerifier 검증 완료)            ║
╠══════════════════════════════════════════════════════════════╣
║  선량 감소: {(1-dose_ratio)*100:.0f}%                                          ║
║  전자 노이즈 비율: {electronic_noise_fraction*100:.0f}%                                    ║
╠══════════════════════════════════════════════════════════════╣
║  📊 검증된 정답 (반드시 이 수치 사용):                       ║
║                                                              ║
║  • EID SNR 감소율: {calc.eid_snr_reduction:.1f}%                              ║
║  • PCD SNR 감소율: {calc.pcd_snr_reduction:.1f}%                              ║
║  • PCD 회복율 (vs EID): +{calc.pcd_recovery_percent:.1f}%                       ║
║                                                              ║
║  ⚠️ 위 수치와 1% 이상 차이 시 답변 자동 거부                ║
╚══════════════════════════════════════════════════════════════╝
"""
        return prompt

    def format_derivation_latex(
        self,
        dose_ratio: float,
        electronic_noise_fraction: float
    ) -> str:
        """
        수식 유도 과정을 LaTeX align 환경으로 포맷팅

        Render-of-Thought (RoT) 시각화용
        """
        calc = self.calculate_verified_answer(dose_ratio, electronic_noise_fraction)

        output = ["### 📐 수식 유도 과정 (Symbolic Verification)\n"]
        for step in calc.derivation_steps:
            output.append(step)
            output.append("")

        return "\n".join(output)

    def _extract_snr_reduction(self, text: str) -> Optional[float]:
        """
        텍스트에서 SNR 감소율 추출

        다양한 패턴을 지원합니다.
        """
        patterns = [
            # 한국어 패턴 (SNR 문맥 우선)
            r'SNR[은이가를의]?\s*(?:약\s*)?(\d+(?:\.\d+)?)\s*%\s*(?:감소|하락|저하)',
            r'SNR[은이가를의]?\s*(?:약\s*)?(?:[\w\s]{0,10}?)(\d+(?:\.\d+)?)\s*%\s*(?:감소|하락|저하)',
            r'(\d+(?:\.\d+)?)\s*%\s*(?:의\s*)?SNR\s*(?:감소|하락)',
            r'SNR\s*(?:감소|하락|저하)[^\d]*(\d+(?:\.\d+)?)\s*%',
            # 영어 패턴
            r'SNR\s*(?:decreases?|reduction|drops?)\s*(?:by\s*)?(\d+(?:\.\d+)?)\s*%',
            r'(\d+(?:\.\d+)?)\s*%\s*(?:reduction|decrease|drop)',
            # 수식 패턴
            r'\\mathbf\{(\d+(?:\.\d+)?)\s*\\%\}',
            r'=\s*(\d+(?:\.\d+)?)\s*%',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                logger.debug(f"Extracted SNR reduction: {value}% (pattern: {pattern[:30]}...)")
                return value

        # 폴백: SNR 감소에 해당하는 퍼센트 값 추출 (선량 값 제외)
        # "선량 50% 감소" 같은 문맥에서 50%을 추출하지 않도록 주의
        all_pcts = re.findall(r'(\d+(?:\.\d+)?)\s*%', text)
        dose_pcts = set()
        for m in re.finditer(r'(?:선량|dose|MGD|mGy)[^%]*?(\d+(?:\.\d+)?)\s*%', text, re.IGNORECASE):
            dose_pcts.add(m.group(1))
        for pct_str in all_pcts:
            if pct_str in dose_pcts:
                continue
            value = float(pct_str)
            if 20 < value < 60:  # SNR 감소 합리적 범위
                return value

        return None

    def _extract_recovery_percent(self, text: str) -> Optional[float]:
        """텍스트에서 회복율 추출"""
        patterns = [
            r'(\d+(?:\.\d+)?)\s*%\s*(?:회복|개선|향상|recovery|improvement)',
            r'(?:회복|개선|향상)[^\d]*(\d+(?:\.\d+)?)\s*%',
            r'\+\s*(\d+(?:\.\d+)?)\s*%',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return float(match.group(1))

        return None


# =============================================================================
# Singleton
# =============================================================================

_verifier_instance: Optional[MathVerifier] = None


def get_math_verifier() -> MathVerifier:
    """MathVerifier 싱글톤"""
    global _verifier_instance
    if _verifier_instance is None:
        _verifier_instance = MathVerifier()
    return _verifier_instance


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    verifier = MathVerifier()

    print("=" * 70)
    print("MathVerifier Test: 50% 선량 감소, 30% 전자 노이즈")
    print("=" * 70)

    # 정답 계산
    calc = verifier.calculate_verified_answer(
        dose_ratio=0.5,
        electronic_noise_fraction=0.3
    )

    print("\n[검증된 정답 (Sympy 계산)]")
    print(f"  EID SNR 감소: {calc.eid_snr_reduction:.1f}%")
    print(f"  PCD SNR 감소: {calc.pcd_snr_reduction:.1f}%")
    print(f"  PCD 회복율: +{calc.pcd_recovery_percent:.1f}%")
    print(f"  Rose Criterion - EID: {'✅' if calc.rose_criterion_eid else '❌'}")
    print(f"  Rose Criterion - PCD: {'✅' if calc.rose_criterion_pcd else '❌'}")

    print("\n[수식 유도 과정]")
    print(verifier.format_derivation_latex(0.5, 0.3)[:1500])

    print("\n[LLM 답변 검증 테스트]")
    test_cases = [
        ("선량 50% 감소 시 SNR은 약 34.8% 하락합니다.", "정답 (Layer 2 검증)"),
        ("SNR이 34.5% 감소합니다.", "1% 이내 근사"),
        ("SNR이 29.3% 감소합니다.", "전자 노이즈 미고려 (PCD 값)"),
        ("SNR이 50% 감소합니다.", "단순 비례 오류"),
    ]

    for answer, desc in test_cases:
        result = verifier.verify_snr_calculation(
            answer,
            dose_ratio=0.5,
            electronic_noise_fraction=0.3
        )
        print(f"\n  [{desc}] \"{answer}\"")
        print(f"    Status: {result.status.value}")
        print(f"    LLM: {result.llm_value}, Verified: {result.verified_value:.1f}%")
        print(f"    Reject: {'⚠️ YES' if result.should_reject else '✅ NO'}")

    print("\n[제약 조건 프롬프트]")
    print(verifier.format_constraint_prompt(0.5, 0.3))
