"""
Sophia AI: Response Synthesizer (Phase 7.6)
===========================================
분해된 하위 답변들을 통합하여 최종 보고서 생성

Executor들이 생성한 개별 답변을 수집하여:
1. 논리적 일관성 검토
2. 중복 제거 및 정리
3. 구조화된 LaTeX 보고서로 합성
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import List, Optional
import requests

from src.reasoning.planner import SubTask, TaskType, DecompositionResult

logger = logging.getLogger(__name__)


@dataclass
class SynthesizerConfig:
    """합성기 설정"""
    ollama_url: str = "http://localhost:11434"
    synthesizer_model: str = "glm4:9b"  # 가벼운 모델로 합성
    timeout: int = 90
    max_answer_length: int = 3000


@dataclass
class SynthesizedReport:
    """합성된 최종 보고서"""
    content: str  # 전체 답변
    sections: List[dict]  # 섹션별 내용
    total_confidence: float
    synthesis_notes: str = ""


class ResponseSynthesizer:
    """
    응답 합성기

    분해된 하위 작업의 답변들을 모아 하나의 완성된 보고서로 통합합니다.
    """

    SYSTEM_PROMPT = r"""당신은 의학 물리 전문 보고서 작성자입니다.
여러 하위 분석 결과를 받아 하나의 완성된 전문가 보고서로 합성하십시오.

## 📋 합성 규칙:
1. **구조화**: 각 하위 답변을 논리적 순서로 배치
2. **연결성**: 섹션 간 자연스러운 전환 문구 추가
3. **중복 제거**: 동일한 내용이 반복되면 한 번만 기술
4. **수식 보존**: LaTeX 수식은 절대 수정하지 않고 그대로 유지

## 📤 출력 형식:

### 📐 1. [첫 번째 분석 제목]
[첫 번째 하위 답변 내용]

### 🔬 2. [두 번째 분석 제목]
[두 번째 하위 답변 내용]

### 📚 3. [세 번째 분석 제목]
[세 번째 하위 답변 내용]

### 💡 종합 결론
[전체 분석을 종합한 최종 결론]

---
🏆 **검증 완료** (종합 신뢰도: XX%)

## ⚠️ 주의사항:
- 수식($...$, $$...$$)은 절대 변경 금지
- 각 섹션의 핵심 수치와 결론은 반드시 포함
- 한국어로 작성"""

    SECTION_ICONS = {
        TaskType.MATH: "📐",
        TaskType.CONCEPT: "🔬",
        TaskType.SEARCH: "📚",
    }

    SECTION_TITLES = {
        TaskType.MATH: "수식 증명",
        TaskType.CONCEPT: "물리적 분석",
        TaskType.SEARCH: "문헌 근거",
    }

    def __init__(self, config: Optional[SynthesizerConfig] = None):
        self.config = config or SynthesizerConfig()

    def synthesize(self, decomposition: DecompositionResult) -> SynthesizedReport:
        """
        하위 답변들을 합성하여 최종 보고서 생성

        Args:
            decomposition: 하위 작업들 (답변 포함)

        Returns:
            SynthesizedReport: 합성된 보고서
        """
        subtasks = decomposition.subtasks

        # 답변이 있는 subtask만 필터링
        completed_tasks = [t for t in subtasks if t.answer]

        if not completed_tasks:
            logger.warning("No completed subtasks to synthesize")
            return SynthesizedReport(
                content="분석 결과가 없습니다.",
                sections=[],
                total_confidence=0.0,
                synthesis_notes="하위 작업 답변 없음"
            )

        logger.info(f"Synthesizing {len(completed_tasks)} subtask answers...")

        # 단일 답변이면 합성 없이 포맷팅만
        if len(completed_tasks) == 1:
            return self._format_single_answer(completed_tasks[0])

        # 복수 답변 합성
        return self._synthesize_multiple(completed_tasks, decomposition.original_query)

    def _format_single_answer(self, task: SubTask) -> SynthesizedReport:
        """단일 답변 포맷팅"""
        icon = self.SECTION_ICONS.get(task.type, "📝")
        title = self.SECTION_TITLES.get(task.type, "분석")

        content = f"### {icon} {title}\n\n{task.answer}\n\n"
        content += f"---\n✅ **완료** (신뢰도: {task.confidence:.0%})"

        return SynthesizedReport(
            content=content,
            sections=[{
                "type": task.type.value,
                "title": title,
                "content": task.answer,
                "confidence": task.confidence
            }],
            total_confidence=task.confidence,
            synthesis_notes="단일 분석"
        )

    def _synthesize_multiple(self, tasks: List[SubTask], original_query: str) -> SynthesizedReport:
        """복수 답변 합성"""

        # 1. 규칙 기반 합성 시도 (LLM 없이)
        sections = []
        content_parts = []

        for i, task in enumerate(tasks, 1):
            icon = self.SECTION_ICONS.get(task.type, "📝")
            title = self.SECTION_TITLES.get(task.type, "분석")

            section_content = f"### {icon} {i}. {title}\n\n{task.answer}"
            content_parts.append(section_content)

            sections.append({
                "type": task.type.value,
                "title": title,
                "content": task.answer,
                "confidence": task.confidence
            })

        # 2. 종합 결론 생성
        avg_confidence = sum(t.confidence for t in tasks) / len(tasks)
        conclusion = self._generate_conclusion(tasks, avg_confidence)

        content_parts.append(f"\n### 💡 종합 결론\n\n{conclusion}")

        # 3. 최종 신뢰도 배지
        if avg_confidence >= 0.9:
            badge = "🏆 **검증 완료**"
        elif avg_confidence >= 0.7:
            badge = "✅ **양호**"
        else:
            badge = "⚠️ **주의 필요**"

        content_parts.append(f"\n---\n{badge} (종합 신뢰도: {avg_confidence:.0%})")

        final_content = "\n\n".join(content_parts)

        return SynthesizedReport(
            content=final_content,
            sections=sections,
            total_confidence=avg_confidence,
            synthesis_notes=f"{len(tasks)}개 분석 통합"
        )

    def _generate_conclusion(self, tasks: List[SubTask], confidence: float) -> str:
        """종합 결론 생성"""
        conclusions = []

        for task in tasks:
            if task.type == TaskType.MATH:
                # 수학 결과에서 핵심 수치 추출
                numbers = re.findall(r'\*\*(\d+(?:\.\d+)?%?)\*\*', task.answer)
                if numbers:
                    conclusions.append(f"수식 분석 결과: {', '.join(numbers[:2])}")

            elif task.type == TaskType.CONCEPT:
                conclusions.append("물리적 기전이 분석되었습니다.")

            elif task.type == TaskType.SEARCH:
                conclusions.append("관련 문헌 근거가 확보되었습니다.")

        if not conclusions:
            return "모든 하위 분석이 완료되었습니다."

        return " ".join(conclusions)

    def synthesize_with_llm(self, decomposition: DecompositionResult) -> SynthesizedReport:
        """
        LLM을 사용한 고급 합성 (선택적)

        Args:
            decomposition: 하위 작업들 (답변 포함)

        Returns:
            SynthesizedReport: 합성된 보고서
        """
        subtasks = decomposition.subtasks
        completed_tasks = [t for t in subtasks if t.answer]

        if not completed_tasks:
            return SynthesizedReport(
                content="분석 결과가 없습니다.",
                sections=[],
                total_confidence=0.0
            )

        # LLM 프롬프트 구성
        task_summaries = []
        for task in completed_tasks:
            task_summaries.append(f"""
[하위 분석 {task.id}] 유형: {task.type.value}
질문: {task.query}
답변:
{task.answer}
신뢰도: {task.confidence:.0%}
""")

        user_prompt = f"""원본 질문: {decomposition.original_query}

다음 하위 분석 결과들을 하나의 완성된 보고서로 합성하세요:

{''.join(task_summaries)}

위 결과들을 통합하여 전문가 보고서를 작성하세요."""

        try:
            response = requests.post(
                f"{self.config.ollama_url}/api/chat",
                json={
                    "model": self.config.synthesizer_model,
                    "messages": [
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    "stream": False,
                    "options": {
                        "num_predict": 2000,
                        "temperature": 0.3,
                    }
                },
                timeout=self.config.timeout
            )

            response.raise_for_status()
            synthesized = response.json().get("message", {}).get("content", "")

            avg_confidence = sum(t.confidence for t in completed_tasks) / len(completed_tasks)

            return SynthesizedReport(
                content=synthesized,
                sections=[{"type": t.type.value, "content": t.answer} for t in completed_tasks],
                total_confidence=avg_confidence,
                synthesis_notes="LLM 합성 완료"
            )

        except Exception as e:
            logger.error(f"LLM synthesis failed: {e}")
            # 폴백: 규칙 기반 합성
            return self._synthesize_multiple(completed_tasks, decomposition.original_query)


# =============================================================================
# Singleton
# =============================================================================

_synthesizer_instance: Optional[ResponseSynthesizer] = None


def get_response_synthesizer() -> ResponseSynthesizer:
    """ResponseSynthesizer 싱글톤"""
    global _synthesizer_instance
    if _synthesizer_instance is None:
        _synthesizer_instance = ResponseSynthesizer()
    return _synthesizer_instance


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from src.reasoning.planner import DecompositionResult, SubTask, TaskType

    # 테스트 데이터
    test_result = DecompositionResult(
        original_query="PCD vs EID 복합 질문",
        is_complex=True,
        subtasks=[
            SubTask(
                id=1,
                type=TaskType.MATH,
                query="SNR 하락폭 계산",
                required_modules=["snr_cnr"],
                answer="선량 50% 감소 시 SNR은 $$\\sqrt{0.5} \\approx 0.707$$배로 감소합니다. 즉 **29.3%** 하락합니다.",
                confidence=0.95
            ),
            SubTask(
                id=2,
                type=TaskType.CONCEPT,
                query="PCD의 d' 회복 기전",
                required_modules=["detector_physics"],
                answer="PCD는 에너지 문턱치 설정으로 전자 노이즈를 물리적으로 제거합니다. 이로 인해 $d'$ 지수가 회복됩니다.",
                confidence=0.88
            ),
            SubTask(
                id=3,
                type=TaskType.SEARCH,
                query="최근 PCD 논문",
                required_modules=["PMC"],
                answer="Day et al. (2024)의 Monte-Carlo 연구에서 CdTe PCD가 대조도를 **25%** 향상시켰습니다.",
                confidence=0.75
            ),
        ],
        reasoning="수식 증명, 개념 설명, 문헌 검색이 모두 필요"
    )

    synthesizer = ResponseSynthesizer()
    report = synthesizer.synthesize(test_result)

    print(f"\n{'='*60}")
    print("SYNTHESIZED REPORT")
    print(f"{'='*60}")
    print(report.content)
    print(f"\nTotal Confidence: {report.total_confidence:.0%}")
    print(f"Notes: {report.synthesis_notes}")
