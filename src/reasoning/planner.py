"""
Sophia AI: Query Planner (Phase 7.6.2)
======================================
복합 질문을 하위 작업으로 분해하는 에이전틱 플래너

복잡한 전문가 질문을 분석하여:
1. 독립적인 하위 질문으로 분해
2. 각 질문의 유형 분류 (MATH, CONCEPT, SEARCH)
3. 필요한 지식 모듈 매핑
4. [NEW] 수치적 제약 조건 추출 (Constraint Extraction)
5. [NEW] Double-Anchor 프롬프트 생성 (Lost in Prompt Order 방지)

Reference: Lost in the Prompt Order (2026)
"""

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
import requests

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """하위 작업 유형"""
    MATH = "MATH"           # 수식 증명, 산술 계산
    CONCEPT = "CONCEPT"     # 물리적 기전, 임상적 영향 설명
    SEARCH = "SEARCH"       # 논문/가이드라인 검색


@dataclass
class SubTask:
    """분해된 하위 작업"""
    id: int
    type: TaskType
    query: str
    required_modules: List[str]
    answer: str = ""  # Executor가 채움
    confidence: float = 0.0


@dataclass
class ExtractedConstraints:
    """추출된 수치적 제약 조건 (Phase 7.6.2)"""
    dose_ratio: Optional[float] = None          # 선량 비율 (0.5 = 50% 감소)
    electronic_noise_fraction: Optional[float] = None  # 전자 노이즈 비율
    rose_k: float = 5.0                         # Rose Criterion 상수
    other_values: Dict[str, float] = field(default_factory=dict)
    raw_text: str = ""                          # 원본 제약 조건 텍스트


@dataclass
class DecompositionResult:
    """분해 결과"""
    original_query: str
    is_complex: bool  # 복합 질문 여부
    subtasks: List[SubTask] = field(default_factory=list)
    reasoning: str = ""  # 분해 근거
    constraints: Optional[ExtractedConstraints] = None  # Phase 7.6.2: 추출된 제약 조건


@dataclass
class PlannerConfig:
    """플래너 설정"""
    ollama_url: str = "http://localhost:11434"
    planner_model: str = "glm4:9b"  # 가벼운 모델로 분해
    timeout: int = 60
    complexity_threshold: int = 2  # 이 개수 이상이면 복합 질문


class QueryPlanner:
    """
    복합 질문 분해기

    사용자의 복잡한 질문을 분석하여 독립적인 하위 작업으로 분해합니다.
    각 하위 작업은 전용 Executor에 의해 처리됩니다.
    """

    SYSTEM_PROMPT = r"""당신은 의학 물리 및 영상 진단 시스템의 '수석 전략가'입니다.
사용자의 복잡한 질문을 분석하여, 논리적으로 완벽한 답변을 생성하기 위한 하위 작업(Sub-tasks)으로 분해하십시오.

## 📋 분해 규칙:
1. **독립성**: 각 하위 질문은 그 자체로 완벽한 질문이어야 합니다.
2. **유형 분류**: 질문을 다음 중 하나로 분류하십시오:
   - MATH: 수식 증명, 산술 계산, 수치 도출이 필요한 경우
   - CONCEPT: 물리적 기전, 임상적 영향, 원리 설명이 필요한 경우
   - SEARCH: 최신 논문이나 가이드라인 검색/인용이 필요한 경우
3. **지식 모듈 매핑**: 각 작업에 필요한 모듈을 지정하십시오:
   - snr_cnr: SNR, CNR, 노이즈, 선량 관련 수식
   - detector_physics: 검출기 물리학, DQE, MTF, PCD, EID
   - mgd_dance_2011: MGD, T-factor, DBT 선량
   - PMC: 논문 검색 필요시

## 📊 복잡도 판단:
- 질문에 "수식으로 증명", "계산", "도출" 등이 있으면 → MATH 포함
- 질문에 "기전", "원리", "영향", "설명" 등이 있으면 → CONCEPT 포함
- 질문에 "논문", "근거", "최근", "연구" 등이 있으면 → SEARCH 포함

## 📤 출력 형식 (JSON Only):
{
  "is_complex": true,
  "reasoning": "이 질문이 복합적인 이유 설명",
  "subtasks": [
    {"id": 1, "type": "MATH", "query": "분해된 질문 1", "required_modules": ["snr_cnr"]},
    {"id": 2, "type": "CONCEPT", "query": "분해된 질문 2", "required_modules": ["detector_physics"]},
    {"id": 3, "type": "SEARCH", "query": "분해된 질문 3", "required_modules": ["PMC"]}
  ]
}

단순 질문(하위 작업 1개)인 경우:
{
  "is_complex": false,
  "reasoning": "단일 질문으로 처리 가능",
  "subtasks": [
    {"id": 1, "type": "MATH", "query": "원본 질문", "required_modules": ["snr_cnr"]}
  ]
}

JSON만 출력하세요. 다른 텍스트는 금지입니다."""

    def __init__(self, config: Optional[PlannerConfig] = None):
        self.config = config or PlannerConfig()

    def decompose(self, query: str) -> DecompositionResult:
        """
        질문을 하위 작업으로 분해

        Args:
            query: 사용자 질문

        Returns:
            DecompositionResult: 분해 결과
        """
        logger.info(f"Decomposing query: {query[:50]}...")

        # Phase 7.6.2: 제약 조건 추출 (모든 질문에 대해)
        constraints = self._extract_constraints(query)
        if constraints.dose_ratio or constraints.electronic_noise_fraction:
            logger.info(f"Extracted constraints: dose_ratio={constraints.dose_ratio}, "
                       f"noise_fraction={constraints.electronic_noise_fraction}")

        # 1. 빠른 휴리스틱 체크 - 단순 질문이면 분해 없이 반환
        if self._is_simple_query(query):
            logger.info("Simple query detected, skipping decomposition")
            result = self._create_simple_result(query)
            result.constraints = constraints
            return result

        # 2. LLM을 통한 분해
        try:
            response = self._call_planner(query)
            result = self._parse_response(query, response)
            result.constraints = constraints  # 제약 조건 첨부

            logger.info(f"Decomposed into {len(result.subtasks)} subtasks: "
                       f"{[t.type.value for t in result.subtasks]}")

            return result

        except Exception as e:
            logger.error(f"Decomposition failed: {e}")
            result = self._create_simple_result(query)
            result.constraints = constraints
            return result

    def _extract_constraints(self, query: str) -> ExtractedConstraints:
        """
        질문에서 수치적 제약 조건 추출 (Phase 7.6.2)

        Lost in the Prompt Order 방지를 위해 핵심 수치를 명시적으로 추출합니다.
        """
        constraints = ExtractedConstraints()

        # 선량 감소율 추출 (예: "50% 감축", "MGD를 50%")
        dose_patterns = [
            r'(?:MGD|선량|dose)[를을]?\s*(\d+(?:\.\d+)?)\s*%\s*(?:감축|감소|줄)',
            r'(\d+(?:\.\d+)?)\s*%\s*(?:저선량|감축|감소)',
            r'선량[이가]?\s*(\d+(?:\.\d+)?)\s*%',
        ]
        for pattern in dose_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                percent = float(match.group(1))
                constraints.dose_ratio = (100 - percent) / 100  # 50% 감소 → 0.5
                break

        # 전자 노이즈 비율 추출 (예: "30%를 차지")
        noise_patterns = [
            r'전자\s*노이즈[가이]?\s*(?:전체\s*노이즈의\s*)?(\d+(?:\.\d+)?)\s*%',
            r'(\d+(?:\.\d+)?)\s*%[를을]?\s*차지',
            r'σ[_]?(?:elec|e)[가이]?\s*(\d+(?:\.\d+)?)\s*%',
        ]
        for pattern in noise_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                percent = float(match.group(1))
                constraints.electronic_noise_fraction = percent / 100  # 30% → 0.3
                break

        # Rose Criterion k값 추출
        rose_match = re.search(r'Rose\s*(?:Criterion)?\s*\(?k\s*[=:]\s*(\d+(?:\.\d+)?)\)?', query, re.IGNORECASE)
        if rose_match:
            constraints.rose_k = float(rose_match.group(1))

        # 원본 텍스트 저장 (Double-Anchor용)
        constraint_parts = []
        if constraints.dose_ratio:
            constraint_parts.append(f"선량 {(1-constraints.dose_ratio)*100:.0f}% 감소")
        if constraints.electronic_noise_fraction:
            constraint_parts.append(f"전자 노이즈 {constraints.electronic_noise_fraction*100:.0f}%")
        if constraints.rose_k != 5.0:
            constraint_parts.append(f"Rose k={constraints.rose_k}")

        constraints.raw_text = ", ".join(constraint_parts) if constraint_parts else ""

        return constraints

    def format_double_anchor_prompt(
        self,
        constraints: ExtractedConstraints,
        verified_values: Optional[Dict[str, float]] = None
    ) -> Tuple[str, str]:
        """
        Double-Anchor 전략 프롬프트 생성 (Phase 7.6.2)

        Lost in the Prompt Order 방지를 위해:
        - Primary Anchor: 시스템 프롬프트 최상단
        - Secondary Anchor: 컨텍스트 직후, 답변 직전

        Args:
            constraints: 추출된 제약 조건
            verified_values: MathVerifier가 계산한 검증된 수치

        Returns:
            (primary_anchor, secondary_anchor) 튜플
        """
        if not constraints.dose_ratio and not constraints.electronic_noise_fraction:
            return ("", "")

        # Primary Anchor (상단)
        primary = f"""
╔══════════════════════════════════════════════════════════════╗
║  🔒 CRITICAL CONSTRAINTS - 반드시 이 수치를 사용하세요       ║
╠══════════════════════════════════════════════════════════════╣"""

        if constraints.dose_ratio:
            primary += f"\n║  • 선량 감소: {(1-constraints.dose_ratio)*100:.0f}% (비율: {constraints.dose_ratio:.2f})          ║"
        if constraints.electronic_noise_fraction:
            primary += f"\n║  • 전자 노이즈: {constraints.electronic_noise_fraction*100:.0f}% (비율: {constraints.electronic_noise_fraction:.2f})       ║"

        if verified_values:
            primary += "\n╠══════════════════════════════════════════════════════════════╣"
            primary += "\n║  📊 MathVerifier 검증 완료 정답:                             ║"
            for key, value in verified_values.items():
                primary += f"\n║  • {key}: {value:.1f}%                                    ║"

        primary += "\n╚══════════════════════════════════════════════════════════════╝\n"

        # Secondary Anchor (하단)
        secondary = f"""
⚠️ [CONSTRAINT REMINDER] 답변 생성 전 확인:
"""
        if constraints.dose_ratio:
            secondary += f"- 선량 {(1-constraints.dose_ratio)*100:.0f}% 감소 조건이 계산에 반영되었는가?\n"
        if constraints.electronic_noise_fraction:
            secondary += f"- 전자 노이즈 {constraints.electronic_noise_fraction*100:.0f}% 조건이 반영되었는가?\n"

        if verified_values:
            secondary += f"- 계산 결과가 검증된 정답과 일치하는가?\n"

        secondary += "위 조건을 만족하지 않으면 답변이 거부됩니다.\n"

        return (primary, secondary)

    def _is_simple_query(self, query: str) -> bool:
        """단순 질문 여부 빠른 판단"""
        # 복합 질문 지표
        complex_indicators = [
            # 다중 요청
            ("수식으로 증명", "논하시오"),
            ("계산하시오", "설명하시오"),
            ("도출하시오", "분석하시오"),
            # 다중 주제
            ("첫째", "둘째"),
            ("1)", "2)"),
            ("①", "②"),
        ]

        query_lower = query.lower()

        for indicators in complex_indicators:
            if all(ind in query or ind in query_lower for ind in indicators):
                return False

        # 길이 기반 (300자 이상이면 복합 가능성)
        if len(query) > 300:
            return False

        return True

    def _create_simple_result(self, query: str) -> DecompositionResult:
        """단순 질문용 결과 생성"""
        task_type = self._infer_task_type(query)
        modules = self._infer_modules(query)

        return DecompositionResult(
            original_query=query,
            is_complex=False,
            subtasks=[
                SubTask(
                    id=1,
                    type=task_type,
                    query=query,
                    required_modules=modules
                )
            ],
            reasoning="단일 질문으로 처리"
        )

    def _infer_task_type(self, query: str) -> TaskType:
        """질문 유형 추론"""
        query_lower = query.lower()

        math_keywords = ['수식', '증명', '계산', '도출', '공식', '%', '비율', 'snr', 'cnr']
        search_keywords = ['논문', '연구', '근거', '최근', 'pmc', '문헌']

        if any(kw in query_lower for kw in math_keywords):
            return TaskType.MATH
        if any(kw in query_lower for kw in search_keywords):
            return TaskType.SEARCH
        return TaskType.CONCEPT

    def _infer_modules(self, query: str) -> List[str]:
        """필요 지식 모듈 추론"""
        query_lower = query.lower()
        modules = []

        module_keywords = {
            'snr_cnr': ['snr', 'cnr', '노이즈', '선량', 'dose', 'noise', 'rose'],
            'detector_physics': ['dqe', 'mtf', 'pcd', 'eid', '검출기', 'detector', '직접변환', '간접변환'],
            'mgd_dance_2011': ['mgd', 't-factor', 'dbt', '토모'],
        }

        for module, keywords in module_keywords.items():
            if any(kw in query_lower for kw in keywords):
                modules.append(module)

        return modules if modules else ['snr_cnr']  # 기본값

    def _call_planner(self, query: str) -> str:
        """Planner LLM 호출"""
        user_prompt = f"다음 질문을 분해하세요:\n\n{query}"

        response = requests.post(
            f"{self.config.ollama_url}/api/chat",
            json={
                "model": self.config.planner_model,
                "messages": [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                "stream": False,
                "options": {
                    "num_predict": 1000,
                    "temperature": 0.1,  # 낮은 온도로 일관성 확보
                }
            },
            timeout=self.config.timeout
        )

        response.raise_for_status()
        return response.json().get("message", {}).get("content", "")

    def _parse_response(self, original_query: str, response: str) -> DecompositionResult:
        """LLM 응답 파싱"""
        # JSON 추출
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if not json_match:
            logger.warning("No JSON found in planner response")
            return self._create_simple_result(original_query)

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")
            return self._create_simple_result(original_query)

        # SubTask 객체 생성
        subtasks = []
        for task_data in data.get("subtasks", []):
            try:
                task_type = TaskType(task_data.get("type", "CONCEPT"))
            except ValueError:
                task_type = TaskType.CONCEPT

            subtasks.append(SubTask(
                id=task_data.get("id", len(subtasks) + 1),
                type=task_type,
                query=task_data.get("query", ""),
                required_modules=task_data.get("required_modules", [])
            ))

        # 최소 1개 subtask 보장
        if not subtasks:
            return self._create_simple_result(original_query)

        is_complex = data.get("is_complex", len(subtasks) >= self.config.complexity_threshold)

        return DecompositionResult(
            original_query=original_query,
            is_complex=is_complex,
            subtasks=subtasks,
            reasoning=data.get("reasoning", "")
        )


# =============================================================================
# Singleton
# =============================================================================

_planner_instance: Optional[QueryPlanner] = None


def get_query_planner() -> QueryPlanner:
    """QueryPlanner 싱글톤"""
    global _planner_instance
    if _planner_instance is None:
        _planner_instance = QueryPlanner()
    return _planner_instance


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    planner = QueryPlanner()

    # 복합 질문 테스트
    complex_query = """차세대 맘모그래피 시스템에 광자 계수 검출기(PCD) 도입을 검토 중이다.
기존 EID 시스템에서 MGD를 50% 감축했을 때, 전자 노이즈가 전체 노이즈의 30%를 차지하게 된다면
Rose Criterion(k=5)을 만족하기 위한 SNR의 하락폭을 수식으로 증명하시오.

동일한 50% 저선량 환경에서 PCD를 사용할 경우, 전자 노이즈 제거와 에너지 가중치 최적화가
d'(Detectability Index)를 어떻게 회복시키는지 기술하고, 이것이 미세 석회화의 대조도 향상에
미치는 영향을 최근 3년 내 PMC 논문을 근거로 논하시오."""

    result = planner.decompose(complex_query)

    print(f"\n{'='*60}")
    print(f"Original Query: {result.original_query[:100]}...")
    print(f"Is Complex: {result.is_complex}")
    print(f"Reasoning: {result.reasoning}")
    print(f"\nSubtasks ({len(result.subtasks)}):")
    for task in result.subtasks:
        print(f"  [{task.id}] {task.type.value}: {task.query[:80]}...")
        print(f"      Modules: {task.required_modules}")
