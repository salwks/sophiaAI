"""
Agent-as-a-Judge: 답변 품질 평가 시스템 (Phase 7.2)
===================================================
메인 에이전트가 작성한 텍스트 답변을 평가 에이전트가 가이드라인과 비교하여
점수화하고, 기준치 미달 시 자동으로 재추론을 명령합니다.

Evaluation Criteria (v2.9):
    1. Factual Accuracy (사실 정확성) - 35%
    2. Logical Consistency (논리적 일관성) - 20%
    3. Guideline Compliance (가이드라인 준수) - 15%
    4. Completeness (완전성) - 15%
    5. Clinical Insight (임상적 통찰) - 15%  ← NEW in Phase 7.2

Workflow:
    Answer → Judge Evaluation → Score < Threshold → Re-reason → Re-evaluate
"""

import json
import re
import logging
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum
import requests

logger = logging.getLogger(__name__)


class JudgeVerdict(Enum):
    """평가 판정"""
    APPROVED = "approved"           # 승인: 답변 품질 충분
    REVISION_REQUIRED = "revision_required"  # 수정 필요
    REJECTED = "rejected"           # 거부: 재생성 필요


@dataclass
class EvaluationCriteria:
    """평가 기준별 점수 (v2.9 - Phase 7.2)"""
    factual_accuracy: float      # 0-100: 사실 정확성
    logical_consistency: float   # 0-100: 논리적 일관성
    guideline_compliance: float  # 0-100: 가이드라인 준수
    completeness: float          # 0-100: 완전성
    clinical_insight: float      # 0-100: 임상적 통찰 (Phase 7.2 NEW)


@dataclass
class JudgeResult:
    """평가 결과"""
    verdict: JudgeVerdict
    total_score: float           # 0-100
    criteria: EvaluationCriteria
    issues_found: List[str]      # 발견된 문제점
    suggestions: List[str]       # 개선 제안
    reasoning: str               # 평가 근거


@dataclass
class JudgeConfig:
    """평가 시스템 설정 (Phase 7.4: 타임아웃 확장)"""
    ollama_url: str = "http://localhost:11434"
    judge_model: str = "glm4:9b"  # 빠른 평가용 SLM
    timeout: int = 120                 # Phase 7.4: 2분 (복잡한 답변 평가)
    approval_threshold: float = 80.0   # 이 점수 이상이면 승인
    revision_threshold: float = 60.0   # 이 점수 미만이면 거부
    weights: Dict[str, float] = None   # 평가 기준 가중치

    def __post_init__(self):
        if self.weights is None:
            # Phase 7.2: 임상적 통찰 추가 (v2.9)
            self.weights = {
                "factual_accuracy": 0.35,       # 35%
                "logical_consistency": 0.20,    # 20%
                "guideline_compliance": 0.15,   # 15%
                "completeness": 0.15,           # 15%
                "clinical_insight": 0.15        # 15% (Phase 7.2 NEW)
            }


class AgentJudge:
    """
    Agent-as-a-Judge: 답변 품질 평가자

    Usage:
        judge = AgentJudge()
        result = judge.evaluate(
            question="BI-RADS 4A의 악성 확률은?",
            answer="3-10%입니다.",
            reference_knowledge=knowledge
        )

        if result.verdict == JudgeVerdict.REJECTED:
            # 재추론 필요
            pass
    """

    def __init__(self, config: Optional[JudgeConfig] = None):
        self.config = config or JudgeConfig()

    def evaluate(
        self,
        question: str,
        answer: str,
        derivation: str = "",
        reference_knowledge: str = "",
        context: str = ""
    ) -> JudgeResult:
        """
        답변 평가 수행

        Args:
            question: 원본 질문
            answer: 평가할 답변
            derivation: 증명 과정 (있는 경우)
            reference_knowledge: 참조용 표준 지식
            context: 검색된 문서 컨텍스트

        Returns:
            JudgeResult: 평가 결과
        """
        logger.info(f"Evaluating answer for: {question[:50]}...")

        system_prompt = """당신은 유방영상의학 답변 품질 평가관입니다 (v2.9 - Phase 7.2).
주어진 답변을 표준 지식 및 가이드라인과 대조하여 점수를 매기세요.

## 평가 기준 (각 0-100점)

1. **Factual Accuracy (사실 정확성)** [35%]:
   - 100: 모든 수치와 용어가 표준과 정확히 일치
   - 80-99: 대부분 정확, 사소한 오류
   - 60-79: 핵심은 맞지만 일부 부정확
   - 40-59: 상당 부분 부정확
   - 0-39: 심각한 오류

2. **Logical Consistency (논리적 일관성)** [20%]:
   - 100: 추론 과정이 완벽하게 연결
   - 80-99: 논리적이나 약간의 비약
   - 60-79: 대체로 논리적이나 일부 허점
   - 40-59: 논리적 비약 다수
   - 0-39: 논리 구조 부재

3. **Guideline Compliance (가이드라인 준수)** [15%]:
   - 100: BI-RADS/ACR 용어와 기준 완벽 준수
   - 80-99: 대부분 준수, 경미한 불일치
   - 60-79: 기본 준수하나 일부 위반
   - 40-59: 상당 부분 위반
   - 0-39: 가이드라인 무시

4. **Completeness (완전성)** [15%]:
   - 100: 질문의 모든 측면에 답변
   - 80-99: 핵심 답변 완료, 부가 설명 부족
   - 60-79: 주요 부분 답변, 일부 누락
   - 40-59: 부분적 답변
   - 0-39: 불완전

5. **Clinical Insight (임상적 통찰)** [15%] ← Phase 7.2 NEW:
   - 100: FN/FP 발생 기전, 임상적 영향을 논리적으로 완벽 기술
   - 80-99: 임상적 맥락 제시, 경미한 누락
   - 60-79: 기본적 임상 연관성 언급
   - 40-59: 임상적 맥락 부족, 단순 수치만 제시
   - 0-39: 임상적 의미 전무

   ※ 임상적 통찰 평가 기준:
   - 노이즈 증가 → CNR 저하 → 미세석회화 검출 실패(FN) 기전 설명 여부
   - 노이즈 스펙클 → 위양성(FP) 발생 가능성 언급 여부
   - Rose Criterion (CNR ≥ 5), d' 지수 등 고급 메트릭 활용 여부
   - ALARA 원칙과 진단 품질 균형에 대한 고찰 여부

## 출력 형식 (JSON)

```json
{
  "factual_accuracy": 0-100,
  "logical_consistency": 0-100,
  "guideline_compliance": 0-100,
  "completeness": 0-100,
  "clinical_insight": 0-100,
  "issues": ["문제점 1", "문제점 2"],
  "suggestions": ["개선 제안 1"],
  "reasoning": "평가 근거 요약"
}
```

JSON만 출력하세요."""

        user_prompt = f"""## 질문
{question}

## 평가 대상 답변
{answer}

## 증명 과정 (있는 경우)
{derivation if derivation else "(없음)"}

## 표준 참조 지식
{reference_knowledge[:3000] if reference_knowledge else "(없음)"}

## 검색된 문서 컨텍스트
{context[:2000] if context else "(없음)"}

위 답변을 평가하고 JSON으로 점수를 매기세요."""

        try:
            response = self._call_llm(system_prompt, user_prompt)
            eval_data = self._parse_json_response(response)

            criteria = EvaluationCriteria(
                factual_accuracy=float(eval_data.get("factual_accuracy", 50)),
                logical_consistency=float(eval_data.get("logical_consistency", 50)),
                guideline_compliance=float(eval_data.get("guideline_compliance", 50)),
                completeness=float(eval_data.get("completeness", 50)),
                clinical_insight=float(eval_data.get("clinical_insight", 50))  # Phase 7.2
            )

            # 가중 평균 계산 (v2.9 - Phase 7.2)
            total_score = (
                criteria.factual_accuracy * self.config.weights["factual_accuracy"] +
                criteria.logical_consistency * self.config.weights["logical_consistency"] +
                criteria.guideline_compliance * self.config.weights["guideline_compliance"] +
                criteria.completeness * self.config.weights["completeness"] +
                criteria.clinical_insight * self.config.weights["clinical_insight"]  # Phase 7.2
            )

            # 판정
            if total_score >= self.config.approval_threshold:
                verdict = JudgeVerdict.APPROVED
            elif total_score >= self.config.revision_threshold:
                verdict = JudgeVerdict.REVISION_REQUIRED
            else:
                verdict = JudgeVerdict.REJECTED

            return JudgeResult(
                verdict=verdict,
                total_score=total_score,
                criteria=criteria,
                issues_found=eval_data.get("issues", []),
                suggestions=eval_data.get("suggestions", []),
                reasoning=eval_data.get("reasoning", "")
            )

        except TimeoutError as e:
            logger.warning(f"Judge timeout - using fallback scoring: {e}")
            # 타임아웃 시: 지식 모듈 기반 Fallback 평가
            return self._fallback_evaluate(answer, reference_knowledge)

        except ConnectionError as e:
            logger.error(f"Judge connection error - using fallback: {e}")
            # 연결 오류 시: 지식 모듈 기반 Fallback 평가
            return self._fallback_evaluate(answer, reference_knowledge)

        except Exception as e:
            logger.error(f"Judge evaluation error: {e}")
            # 기타 오류 시 보수적으로 수정 필요 판정
            return JudgeResult(
                verdict=JudgeVerdict.REVISION_REQUIRED,
                total_score=50.0,
                criteria=EvaluationCriteria(50, 50, 50, 50, 50),  # Phase 7.2: 5개 기준
                issues_found=[f"평가 실패: {str(e)}"],
                suggestions=["수동 검토 필요"],
                reasoning="평가 중 오류 발생"
            )

    def _fallback_evaluate(self, answer: str, reference_knowledge: str) -> JudgeResult:
        """
        Fallback 평가: LLM 불가 시 규칙 기반 간이 평가 (v2.9 - Phase 7.2)

        지식 모듈(Phase 4)의 핵심 수치/키워드가 답변에 포함되어 있는지 검사
        """
        import re

        # 기본 점수 (보수적)
        base_score = 65.0
        clinical_insight_score = 50.0  # Phase 7.2: 임상적 통찰 별도 채점

        # 1. 수치 포함 여부 (+10점)
        has_numbers = bool(re.search(r'\d+\.?\d*\s*(%|mGy|kV|mm|cm)', answer))
        if has_numbers:
            base_score += 10

        # 2. 참조 지식 키워드 매칭 (+15점)
        if reference_knowledge:
            ref_keywords = re.findall(r'\b[a-zA-Z가-힣]{3,}\b', reference_knowledge.lower())
            answer_lower = answer.lower()
            matched = sum(1 for kw in ref_keywords[:20] if kw in answer_lower)
            if matched >= 5:
                base_score += 15
            elif matched >= 2:
                base_score += 8

        # 3. 구조화 여부 (목록, 섹션 등) (+5점)
        has_structure = bool(re.search(r'(\n[-•*]\s|\n\d+\.|\n#{1,3}\s)', answer))
        if has_structure:
            base_score += 5

        # 4. Phase 7.2: 임상적 통찰 키워드 체크
        clinical_keywords = [
            'fn', 'fp', 'false negative', 'false positive', '위음성', '위양성',
            'cnr', 'snr', 'rose', 'd\'', 'd-prime', '검출', 'detection',
            '미세석회화', 'microcalcification', 'alara', '노이즈', 'noise',
            '임상', 'clinical', '진단', 'diagnostic'
        ]
        answer_lower = answer.lower()
        clinical_matched = sum(1 for kw in clinical_keywords if kw in answer_lower)
        if clinical_matched >= 5:
            clinical_insight_score = 75.0
        elif clinical_matched >= 3:
            clinical_insight_score = 65.0
        elif clinical_matched >= 1:
            clinical_insight_score = 55.0

        has_clinical = clinical_matched >= 1

        # 최대 90점 (LLM 평가 없이는 APPROVED까지 가지 않음)
        total_score = min(base_score, 75.0)

        # 판정 (Fallback은 최대 REVISION_REQUIRED)
        if total_score >= 70:
            verdict = JudgeVerdict.REVISION_REQUIRED  # Fallback은 APPROVED 안 줌
        else:
            verdict = JudgeVerdict.REVISION_REQUIRED

        return JudgeResult(
            verdict=verdict,
            total_score=total_score,
            criteria=EvaluationCriteria(
                factual_accuracy=total_score,
                logical_consistency=total_score,
                guideline_compliance=total_score,
                completeness=total_score,
                clinical_insight=clinical_insight_score  # Phase 7.2
            ),
            issues_found=["⚠️ Fallback 평가: LLM 평가 불가로 규칙 기반 간이 평가 수행"],
            suggestions=["네트워크 상태 확인 후 재시도 권장"],
            reasoning=f"Fallback 규칙 기반 평가 (수치:{has_numbers}, 키워드매칭, 구조화:{has_structure}, 임상통찰:{has_clinical})"
        )

    def _call_llm(self, system_prompt: str, user_prompt: str, max_retries: int = 2) -> str:
        """
        LLM 호출 (타임아웃 및 재시도 처리)

        Args:
            system_prompt: 시스템 프롬프트
            user_prompt: 사용자 프롬프트
            max_retries: 최대 재시도 횟수

        Returns:
            LLM 응답 텍스트

        Raises:
            TimeoutError: 모든 재시도 후에도 타임아웃
            ConnectionError: 서버 연결 실패
        """
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                response = requests.post(
                    f"{self.config.ollama_url}/api/chat",
                    json={
                        "model": self.config.judge_model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        "stream": False,
                        "format": "json",
                        "options": {
                            "num_predict": 500,
                            "temperature": 0.1,
                        }
                    },
                    timeout=self.config.timeout
                )

                response.raise_for_status()
                return response.json().get("message", {}).get("content", "")

            except requests.exceptions.Timeout as e:
                last_error = e
                logger.warning(f"Judge LLM timeout (attempt {attempt + 1}/{max_retries + 1})")
                if attempt < max_retries:
                    continue
                raise TimeoutError(f"Judge LLM timed out after {max_retries + 1} attempts")

            except requests.exceptions.ConnectionError as e:
                last_error = e
                logger.error(f"Judge LLM connection error: {e}")
                raise ConnectionError(f"Cannot connect to Ollama: {e}")

            except requests.exceptions.RequestException as e:
                last_error = e
                logger.error(f"Judge LLM request error: {e}")
                if attempt < max_retries:
                    continue
                raise

        raise last_error or Exception("Unknown error in _call_llm")

    def _parse_json_response(self, response: str) -> Dict:
        """JSON 응답 파싱"""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # 코드 블록 추출 시도
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            # 중괄호 추출 시도
            brace_match = re.search(r'\{.*\}', response, re.DOTALL)
            if brace_match:
                return json.loads(brace_match.group())
            return {}

    def format_evaluation_report(self, result: JudgeResult) -> str:
        """평가 결과 포맷팅 (v2.9 - Phase 7.2)"""
        verdict_badges = {
            JudgeVerdict.APPROVED: "✅ **승인됨**",
            JudgeVerdict.REVISION_REQUIRED: "⚠️ **수정 필요**",
            JudgeVerdict.REJECTED: "❌ **거부됨**",
        }

        lines = [
            f"## 📊 품질 평가 결과 (v2.9)",
            f"",
            f"**판정**: {verdict_badges[result.verdict]} (총점: {result.total_score:.1f}/100)",
            f"",
            f"| 평가 기준 | 점수 | 가중치 |",
            f"|----------|------|--------|",
            f"| 사실 정확성 | {result.criteria.factual_accuracy:.0f}/100 | 35% |",
            f"| 논리적 일관성 | {result.criteria.logical_consistency:.0f}/100 | 20% |",
            f"| 가이드라인 준수 | {result.criteria.guideline_compliance:.0f}/100 | 15% |",
            f"| 완전성 | {result.criteria.completeness:.0f}/100 | 15% |",
            f"| 🔬 임상적 통찰 | {result.criteria.clinical_insight:.0f}/100 | 15% |",
        ]

        if result.issues_found:
            lines.append("")
            lines.append("### 발견된 문제점")
            for issue in result.issues_found:
                lines.append(f"- {issue}")

        if result.suggestions:
            lines.append("")
            lines.append("### 개선 제안")
            for sug in result.suggestions:
                lines.append(f"- {sug}")

        return "\n".join(lines)


# =============================================================================
# Integrated Pipeline: Reasoning + Judging
# =============================================================================

class TextExcellencePipeline:
    """
    통합 파이프라인: Answering Twice + Agent-as-a-Judge

    1. 2단계 추론으로 답변 생성
    2. Agent-as-a-Judge로 품질 평가
    3. 기준 미달 시 재추론
    """

    def __init__(
        self,
        reasoning_config=None,
        judge_config=None,
        max_iterations: int = 2
    ):
        from src.reasoning.text_logic import TextReasoningEngine, TextReasoningConfig

        self.reasoning_engine = TextReasoningEngine(reasoning_config)
        self.judge = AgentJudge(judge_config)
        self.max_iterations = max_iterations

    def process(
        self,
        question: str,
        context: str = "",
        physics_knowledge: str = ""
    ) -> Tuple[str, JudgeResult]:
        """
        전체 파이프라인 실행

        Args:
            question: 사용자 질문
            context: 검색된 문서 컨텍스트
            physics_knowledge: 표준 물리 지식

        Returns:
            (formatted_answer, judge_result)
        """
        for iteration in range(self.max_iterations):
            logger.info(f"Pipeline iteration {iteration + 1}/{self.max_iterations}")

            # 1. Answering Twice 추론
            refined_answer = self.reasoning_engine.reason(
                question=question,
                context=context,
                physics_knowledge=physics_knowledge
            )

            # 2. Agent-as-a-Judge 평가
            judge_result = self.judge.evaluate(
                question=question,
                answer=refined_answer.content,
                derivation=refined_answer.derivation,
                reference_knowledge=physics_knowledge,
                context=context
            )

            logger.info(f"Judge verdict: {judge_result.verdict.value}, score: {judge_result.total_score:.1f}")

            # 3. 승인 또는 마지막 반복이면 종료
            if judge_result.verdict == JudgeVerdict.APPROVED:
                break
            if iteration == self.max_iterations - 1:
                logger.warning("Max iterations reached, returning best effort")
                break

            # 4. 거부/수정 필요 시 피드백 반영하여 재추론
            logger.info(f"Re-reasoning with feedback: {judge_result.issues_found}")

            # 피드백을 context에 추가
            feedback = f"\n\n[이전 답변의 문제점]\n" + "\n".join(judge_result.issues_found)
            context = context + feedback

        # 최종 출력 포맷팅
        formatted_answer = self.reasoning_engine.format_structured_output(refined_answer)

        return formatted_answer, judge_result


# =============================================================================
# Singleton
# =============================================================================

_judge_instance: Optional[AgentJudge] = None
_pipeline_instance: Optional[TextExcellencePipeline] = None


def get_agent_judge(config: Optional[JudgeConfig] = None) -> AgentJudge:
    """AgentJudge 싱글톤"""
    global _judge_instance
    if _judge_instance is None:
        _judge_instance = AgentJudge(config)
    return _judge_instance


def get_text_excellence_pipeline() -> TextExcellencePipeline:
    """TextExcellencePipeline 싱글톤"""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = TextExcellencePipeline()
    return _pipeline_instance


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    judge = AgentJudge()

    # 테스트 케이스: 정확한 답변
    result = judge.evaluate(
        question="BI-RADS Category 3의 악성 확률은?",
        answer="BI-RADS Category 3 (Probably Benign)의 악성 확률은 2% 이하입니다. 6개월 후 추적 검사를 권고합니다.",
        reference_knowledge="Category 3 (Probably Benign): 악성 확률 ≤2%, 6개월 추적 권고"
    )

    print("=" * 60)
    print("Agent-as-a-Judge Test")
    print("=" * 60)
    print(judge.format_evaluation_report(result))
