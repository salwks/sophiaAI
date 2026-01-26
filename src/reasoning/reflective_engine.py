"""
Reflective Engine: 하이브리드 성찰 엔진 (Phase 7.6.3)
=====================================================
LLM이 틀렸을 때 정답을 주입하지 않고, 물리 법칙 힌트를 제공하여
스스로 수정하게 만드는 R⁴ 기반 성찰 시스템

핵심 원리:
- Reflect: 오류의 물리적 원인 분석
- Repair: 물리 법칙 힌트로 자가 수정 유도
- 정답 직접 제공 X → 물리 원리 환기 O

Reference: R⁴ (Route, Retrieve, Reflect, Repair) 2025
"""

import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from enum import Enum

logger = logging.getLogger(__name__)


class ReflectionType(Enum):
    """성찰 유형"""
    PHYSICS_VIOLATION = "physics_violation"     # 물리 법칙 위반
    CALCULATION_ERROR = "calculation_error"     # 산술 오류
    CONSTRAINT_IGNORED = "constraint_ignored"   # 제약 조건 무시
    LOGIC_INCONSISTENCY = "logic_inconsistency" # 논리적 모순


@dataclass
class ReflectionHint:
    """성찰 힌트"""
    hint_type: ReflectionType
    physics_principle: str      # 적용해야 할 물리 원리
    specific_guidance: str      # 구체적 지침
    constraints_reminder: str   # 제약 조건 상기
    expected_direction: str     # 예상 결과 방향 (정답 수치 아님)


class ReflectivePromptGenerator:
    """
    성찰 힌트 생성기

    LLM이 틀렸을 때 정답을 알려주지 않고,
    물리 법칙을 환기시켜 스스로 재계산하도록 유도합니다.
    """

    # 물리 원리 사전 (정답이 아닌 원리만 포함)
    PHYSICS_PRINCIPLES = {
        "poisson_statistics": {
            "name": "푸아송 통계",
            "formula": r"$\sigma_{\text{quantum}}^2 \propto \frac{1}{D}$ (분산은 선량에 반비례)",
            "implication": "선량이 절반이 되면 양자 노이즈 분산은 2배가 됩니다."
        },
        "noise_addition": {
            "name": "노이즈 합성 법칙",
            "formula": r"$\sigma_{\text{total}}^2 = \sigma_{\text{quantum}}^2 + \sigma_{\text{electronic}}^2$",
            "implication": "총 노이즈는 각 노이즈 분산의 합으로 계산합니다."
        },
        "snr_definition": {
            "name": "SNR 정의",
            "formula": r"$\text{SNR} = \frac{\text{Signal}}{\sigma_{\text{total}}}$",
            "implication": "신호 대 잡음비는 신호를 총 노이즈로 나눈 값입니다."
        },
        "signal_dose_relation": {
            "name": "신호-선량 관계",
            "formula": r"$\text{Signal} \propto D$",
            "implication": "신호는 선량에 정비례합니다."
        },
        "electronic_noise_constant": {
            "name": "전자 노이즈 불변",
            "formula": r"$\sigma_{\text{electronic}} = \text{const}$ (하드웨어 특성)",
            "implication": "전자 노이즈는 선량과 무관하게 일정합니다."
        },
        "pcd_threshold": {
            "name": "PCD 에너지 문턱치",
            "formula": r"PCD: $\sigma_{\text{electronic}} \approx 0$",
            "implication": "PCD는 에너지 문턱치로 전자 노이즈를 물리적으로 제거합니다."
        }
    }

    def generate_reflection_hint(
        self,
        task_type: str,
        llm_answer: str,
        llm_value: Optional[float],
        constraints: Dict[str, Any],
        error_description: str
    ) -> ReflectionHint:
        """
        물리 법칙 기반 성찰 힌트 생성

        Args:
            task_type: 작업 유형 (MATH, CONCEPT, SEARCH)
            llm_answer: LLM의 원본 답변
            llm_value: LLM이 계산한 수치 (추출된 경우)
            constraints: 추출된 제약 조건 (dose_ratio, noise_fraction 등)
            error_description: 오류 설명

        Returns:
            ReflectionHint: 정답 없이 물리 원리만 포함된 힌트
        """
        if task_type == "MATH":
            return self._generate_math_hint(llm_value, constraints, error_description)
        elif task_type == "CONCEPT":
            return self._generate_concept_hint(constraints, error_description)
        else:
            return self._generate_generic_hint(error_description)

    def _generate_math_hint(
        self,
        llm_value: Optional[float],
        constraints: Dict[str, Any],
        error_description: str
    ) -> ReflectionHint:
        """MATH 태스크용 물리 기반 힌트"""

        dose_ratio = constraints.get("dose_ratio", 0.5)
        noise_fraction = constraints.get("electronic_noise_fraction", 0.0)

        # 물리 원리 조합
        principles = [
            self.PHYSICS_PRINCIPLES["poisson_statistics"],
            self.PHYSICS_PRINCIPLES["noise_addition"],
            self.PHYSICS_PRINCIPLES["snr_definition"],
        ]

        if noise_fraction > 0:
            principles.append(self.PHYSICS_PRINCIPLES["electronic_noise_constant"])

        # 물리 원리 텍스트 생성
        physics_text = "\n".join([
            f"**{p['name']}**: {p['formula']}\n   → {p['implication']}"
            for p in principles
        ])

        # 구체적 지침 (정답 수치 없이)
        guidance_parts = [
            f"1. 선량이 {dose_ratio:.0%}로 변했을 때 양자 노이즈 분산이 어떻게 변하는지 계산하세요.",
            f"2. 신호(Signal)가 선량에 비례한다는 점을 반영하세요.",
        ]

        if noise_fraction > 0:
            guidance_parts.append(
                f"3. 전자 노이즈({noise_fraction:.0%})는 선량과 무관하게 일정하다는 점을 잊지 마세요."
            )
            guidance_parts.append(
                "4. 총 노이즈 = √(양자노이즈² + 전자노이즈²) 공식을 적용하세요."
            )

        guidance_parts.append(
            "5. 최종적으로 SNR = Signal / σ_total 공식으로 SNR 변화율을 도출하세요."
        )

        # 예상 방향 (정답 수치 아님)
        direction = "선량 감소 시 SNR은 반드시 감소해야 합니다 (물리적 필연)."
        if noise_fraction > 0:
            direction += " 전자 노이즈가 있으면 순수 양자 노이즈만 있을 때보다 SNR 감소폭이 더 큽니다."

        return ReflectionHint(
            hint_type=ReflectionType.PHYSICS_VIOLATION,
            physics_principle=physics_text,
            specific_guidance="\n".join(guidance_parts),
            constraints_reminder=f"제약 조건: 선량 {(1-dose_ratio)*100:.0f}% 감소, 전자 노이즈 {noise_fraction*100:.0f}%",
            expected_direction=direction
        )

    def _generate_concept_hint(
        self,
        constraints: Dict[str, Any],
        error_description: str
    ) -> ReflectionHint:
        """CONCEPT 태스크용 힌트"""

        pcd_principle = self.PHYSICS_PRINCIPLES["pcd_threshold"]

        return ReflectionHint(
            hint_type=ReflectionType.LOGIC_INCONSISTENCY,
            physics_principle=f"**{pcd_principle['name']}**: {pcd_principle['formula']}\n→ {pcd_principle['implication']}",
            specific_guidance=(
                "1. EID와 PCD의 노이즈 구성 차이를 명확히 설명하세요.\n"
                "2. 전자 노이즈 제거가 d'(Detectability Index)에 미치는 영향을 논하세요.\n"
                "3. 에너지 가중치(Energy Weighting)의 물리적 원리를 포함하세요."
            ),
            constraints_reminder=constraints.get("raw_text", ""),
            expected_direction="PCD는 저선량에서 EID보다 우수한 화질을 제공해야 합니다."
        )

    def _generate_generic_hint(self, error_description: str) -> ReflectionHint:
        """일반 힌트"""
        return ReflectionHint(
            hint_type=ReflectionType.LOGIC_INCONSISTENCY,
            physics_principle="제공된 지식 모듈의 물리 공식을 다시 확인하세요.",
            specific_guidance="논리적 인과관계를 단계별로 점검하세요.",
            constraints_reminder="",
            expected_direction="물리 법칙에 부합하는 결론을 도출하세요."
        )

    def format_reflection_prompt(
        self,
        original_answer: str,
        hint: ReflectionHint,
        attempt_number: int
    ) -> str:
        """
        성찰 프롬프트 포맷팅

        LLM에게 전달할 최종 성찰 프롬프트를 생성합니다.
        정답 수치는 포함하지 않습니다.
        """
        prompt = f"""
╔══════════════════════════════════════════════════════════════╗
║  🔬 물리적 성찰 피드백 (Attempt {attempt_number})                    ║
╚══════════════════════════════════════════════════════════════╝

### ❌ 이전 답변의 문제점
당신의 이전 계산에서 **물리적 모순**이 발견되었습니다.

### 📐 적용해야 할 물리 원리
{hint.physics_principle}

### 📋 수정 지침
{hint.specific_guidance}

### ⚠️ 제약 조건 확인
{hint.constraints_reminder}

### 🎯 예상 방향
{hint.expected_direction}

---

**이전 답변:**
{original_answer[:500]}{"..." if len(original_answer) > 500 else ""}

---

위의 물리 원리를 **단계별로 적용**하여 다시 계산하고,
**정확한 수치**와 함께 **유도 과정**을 제시하세요.
"""
        return prompt


# =============================================================================
# Reflective Reasoning Engine
# =============================================================================

class ReflectiveReasoningEngine:
    """
    성찰 기반 추론 엔진

    1차 추론 → 검증 → 실패 시 성찰 힌트 → 재추론
    """

    def __init__(
        self,
        reasoning_engine,
        math_verifier,
        max_reflections: int = 2
    ):
        self.reasoning_engine = reasoning_engine
        self.math_verifier = math_verifier
        self.hint_generator = ReflectivePromptGenerator()
        self.max_reflections = max_reflections

    def reason_with_reflection(
        self,
        question: str,
        context: str,
        physics_knowledge: str,
        constraints: Dict[str, Any],
        task_type: str = "MATH"
    ) -> Dict[str, Any]:
        """
        성찰 기반 추론 실행

        Args:
            question: 질문
            context: 컨텍스트
            physics_knowledge: 물리 지식
            constraints: 추출된 제약 조건
            task_type: 작업 유형

        Returns:
            Dict with answer, reflection_count, verification_status
        """
        reflection_history = []
        current_context = physics_knowledge

        for attempt in range(1, self.max_reflections + 2):
            logger.info(f"Reasoning attempt {attempt}...")

            # 추론 실행
            result = self.reasoning_engine.reason(
                question=question,
                context=context,
                physics_knowledge=current_context,
                require_derivation=(task_type == "MATH")
            )

            # MATH 태스크인 경우 검증
            if task_type == "MATH" and constraints.get("dose_ratio"):
                verification = self.math_verifier.verify_snr_calculation(
                    result.content,
                    constraints.get("dose_ratio", 0.5),
                    constraints.get("electronic_noise_fraction", 0.0)
                )

                if verification.should_reject:
                    logger.warning(
                        f"Attempt {attempt} failed verification: "
                        f"LLM={verification.llm_value}, error={verification.error_percent:.1f}%"
                    )

                    if attempt <= self.max_reflections:
                        # 성찰 힌트 생성 (정답 없이)
                        hint = self.hint_generator.generate_reflection_hint(
                            task_type=task_type,
                            llm_answer=result.content,
                            llm_value=verification.llm_value,
                            constraints=constraints,
                            error_description=verification.explanation
                        )

                        # 성찰 프롬프트로 컨텍스트 업데이트
                        reflection_prompt = self.hint_generator.format_reflection_prompt(
                            result.content, hint, attempt
                        )
                        current_context = f"{physics_knowledge}\n\n{reflection_prompt}"

                        reflection_history.append({
                            "attempt": attempt,
                            "llm_value": verification.llm_value,
                            "error_percent": verification.error_percent,
                            "hint_type": hint.hint_type.value
                        })
                        continue
                    else:
                        # 최대 재시도 초과
                        logger.warning(f"Max reflections reached. Final error: {verification.error_percent:.1f}%")

                else:
                    logger.info(f"Verification PASSED on attempt {attempt}")

                return {
                    "answer": result,
                    "reflection_count": attempt - 1,
                    "reflection_history": reflection_history,
                    "verification": verification,
                    "final_attempt": attempt
                }

            # MATH가 아닌 경우 바로 반환
            return {
                "answer": result,
                "reflection_count": 0,
                "reflection_history": [],
                "verification": None,
                "final_attempt": 1
            }

        # Fallback
        return {
            "answer": result,
            "reflection_count": self.max_reflections,
            "reflection_history": reflection_history,
            "verification": verification if 'verification' in dir() else None,
            "final_attempt": self.max_reflections + 1
        }


# =============================================================================
# Singleton
# =============================================================================

_hint_generator_instance: Optional[ReflectivePromptGenerator] = None


def get_reflective_hint_generator() -> ReflectivePromptGenerator:
    """ReflectivePromptGenerator 싱글톤"""
    global _hint_generator_instance
    if _hint_generator_instance is None:
        _hint_generator_instance = ReflectivePromptGenerator()
    return _hint_generator_instance


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    generator = ReflectivePromptGenerator()

    # 테스트: MATH 힌트 생성
    hint = generator.generate_reflection_hint(
        task_type="MATH",
        llm_answer="선량 50% 감소 시 SNR은 29.3% 하락합니다.",
        llm_value=29.3,
        constraints={
            "dose_ratio": 0.5,
            "electronic_noise_fraction": 0.3,
            "raw_text": "선량 50% 감소, 전자 노이즈 30%"
        },
        error_description="계산 오류: 전자 노이즈 미반영"
    )

    print("=" * 70)
    print("ReflectivePromptGenerator Test")
    print("=" * 70)
    print(f"\nHint Type: {hint.hint_type.value}")
    print(f"\n[Physics Principle]\n{hint.physics_principle}")
    print(f"\n[Specific Guidance]\n{hint.specific_guidance}")
    print(f"\n[Constraints Reminder]\n{hint.constraints_reminder}")
    print(f"\n[Expected Direction]\n{hint.expected_direction}")

    # 성찰 프롬프트 생성
    prompt = generator.format_reflection_prompt(
        original_answer="SNR은 29.3% 하락합니다.",
        hint=hint,
        attempt_number=1
    )
    print("\n" + "=" * 70)
    print("Generated Reflection Prompt:")
    print("=" * 70)
    print(prompt)
