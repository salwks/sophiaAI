"""
PhysicsCoT-RAG: Causal Chain-of-Thought RAG for Mammography Physics
====================================================================

MedCoT-RAG (arXiv:2508.15849)를 맘모그래피 물리학 도메인에 적용한 구현.

핵심 아이디어:
- 4단계 인과 추론 (현상→메커니즘→제약→통합)
- Causal-aware retrieval (의미 유사도 + 인과 관련도)
- 복합 질문의 인과 체인 명시적 모델링

Reference:
    MedCoT-RAG: Causal Chain-of-Thought RAG for Medical Question Answering
    arXiv:2508.15849 (2025)
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class PhysicsReasoningStage(Enum):
    """물리학 인과 추론 4단계 (MedCoT-RAG 적용)"""
    PHENOMENON_ANALYSIS = "phenomenon"      # 현상 분석 (MedCoT: Symptom Analysis)
    CAUSAL_MECHANISM = "mechanism"          # 물리적 메커니즘 (MedCoT: Causal Pathophysiology)
    PARAMETER_CONSTRAINTS = "constraints"   # 파라미터 제약 (MedCoT: Differential Diagnosis)
    EVIDENCE_SYNTHESIS = "synthesis"        # 근거 통합 (MedCoT: Evidence Synthesis)


@dataclass
class CausalChain:
    """인과 체인 표현"""
    cause: str                      # 원인
    mechanism: str                  # 중간 메커니즘
    effect: str                     # 결과
    formula: Optional[str] = None   # 관련 공식
    constraints: List[str] = field(default_factory=list)  # 제약 조건


@dataclass
class PhysicsCoTResult:
    """PhysicsCoT-RAG 결과"""
    answer: str                             # 최종 답변
    causal_chains: List[CausalChain]        # 식별된 인과 체인들
    reasoning_stages: Dict[str, str]        # 각 단계별 추론 결과
    retrieved_modules: List[str]            # 검색된 지식 모듈
    causal_scores: Dict[str, float]         # 모듈별 인과 점수
    confidence: float                       # 최종 신뢰도


class PhysicsCausalRetriever:
    """
    Physics Causal-Aware Retriever

    MedCoT-RAG의 인과 인식 검색을 물리학에 적용:
    s(d,q) = α·sim(q,d) + β·ψ_physics(d)
    """

    # 물리학 인과 연산자 (MedCoT의 causal operators 물리학 버전)
    CAUSAL_OPERATORS = {
        # 인과 관계
        "causes": ["때문에", "으로 인해", "원인", "유발", "causes", "leads to", "results in"],
        "effects": ["따라서", "결과", "영향", "effect", "consequently", "hence"],
        "proportional": ["비례", "∝", "proportional", "scales with", "증가하면", "감소하면"],
        "inverse": ["반비례", "역비례", "inverse", "decreases as"],
        "limits": ["제한", "한계", "제약", "최대", "최소", "limited by", "constrained"],
        "tradeoff": ["트레이드오프", "trade-off", "vs", "대비", "상충"],
    }

    # 물리 공식 패턴
    FORMULA_PATTERNS = [
        r'[A-Za-z_]+\s*[=∝∼]\s*[A-Za-z0-9_√×/\(\)\^]+',  # SNR = √Dose
        r'[A-Za-z]+\s*[↑↓]\s*→\s*[A-Za-z]+\s*[↑↓]',      # mA↓ → noise↑
        r'\d+\s*[±]\s*\d+',                                # 25 ± 10
    ]

    def __init__(self, alpha: float = 0.6, beta: float = 0.4):
        """
        Args:
            alpha: 의미 유사도 가중치
            beta: 인과 관련도 가중치
        """
        self.alpha = alpha
        self.beta = beta

    def compute_causal_score(self, text: str) -> float:
        """
        인과 관련도 점수 계산 (ψ_physics)

        MedCoT-RAG: "weighted keyword matching normalized by document length"
        """
        text_lower = text.lower()
        score = 0.0

        # 인과 연산자 감지
        for category, operators in self.CAUSAL_OPERATORS.items():
            for op in operators:
                count = text_lower.count(op.lower())
                if count > 0:
                    # 카테고리별 가중치
                    weight = {
                        "causes": 1.5,
                        "effects": 1.2,
                        "proportional": 1.8,  # 물리학에서 비례 관계가 중요
                        "inverse": 1.8,
                        "limits": 2.0,        # 제약 조건이 핵심
                        "tradeoff": 1.5,
                    }.get(category, 1.0)
                    score += count * weight

        # 공식 패턴 감지 (물리학 특화)
        for pattern in self.FORMULA_PATTERNS:
            matches = re.findall(pattern, text)
            score += len(matches) * 2.0  # 공식 포함 시 높은 가중치

        # 문서 길이로 정규화 (MedCoT-RAG 방식)
        doc_length = len(text.split())
        if doc_length > 0:
            score = score / (doc_length ** 0.5)  # sqrt 정규화

        return min(score, 1.0)  # 0~1 범위로 제한

    def rank_modules(
        self,
        query: str,
        modules: List[Dict[str, Any]],
        semantic_scores: Optional[Dict[str, float]] = None
    ) -> List[Tuple[str, float, float, float]]:
        """
        모듈 순위 결정 (의미 + 인과)

        Args:
            query: 사용자 질문
            modules: 후보 지식 모듈 리스트
            semantic_scores: 모듈별 의미 유사도 (없으면 키워드 매칭)

        Returns:
            [(module_id, total_score, semantic_score, causal_score), ...]
        """
        results = []

        for module in modules:
            module_id = module.get("id", "unknown")

            # 의미 유사도 (기본: 키워드 매칭)
            if semantic_scores and module_id in semantic_scores:
                sem_score = semantic_scores[module_id]
            else:
                sem_score = self._keyword_similarity(query, module)

            # 인과 관련도
            content_text = self._extract_text(module)
            causal_score = self.compute_causal_score(content_text)

            # Composite score: s(d,q) = α·sim(q,d) + β·ψ(d)
            total_score = self.alpha * sem_score + self.beta * causal_score

            results.append((module_id, total_score, sem_score, causal_score))

        # 점수순 정렬
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def _keyword_similarity(self, query: str, module: Dict) -> float:
        """키워드 기반 유사도"""
        query_lower = query.lower()
        keywords = module.get("keywords", [])

        if not keywords:
            return 0.0

        matched = sum(1 for kw in keywords if kw.lower() in query_lower)
        return matched / len(keywords)

    def _extract_text(self, module: Dict) -> str:
        """모듈에서 텍스트 추출"""
        parts = []

        # 키워드
        parts.extend(module.get("keywords", []))

        # 콘텐츠 (재귀적으로 추출)
        content = module.get("content", {})
        parts.append(self._flatten_content(content))

        # 공식
        formulas = module.get("formulas", {})
        if isinstance(formulas, dict):
            parts.extend(str(v) for v in formulas.values())

        # 오개념 경고
        misconceptions = module.get("common_misconceptions", {})
        if isinstance(misconceptions, dict):
            for key, value in misconceptions.items():
                if isinstance(value, dict):
                    parts.extend(str(v) for v in value.values())

        return " ".join(str(p) for p in parts)

    def _flatten_content(self, obj: Any, depth: int = 0) -> str:
        """중첩 딕셔너리를 문자열로 평탄화"""
        if depth > 5:
            return str(obj)

        if isinstance(obj, dict):
            parts = []
            for k, v in obj.items():
                parts.append(f"{k}: {self._flatten_content(v, depth + 1)}")
            return " ".join(parts)
        elif isinstance(obj, list):
            return " ".join(self._flatten_content(item, depth + 1) for item in obj)
        else:
            return str(obj)


class PhysicsCoTPromptBuilder:
    """
    4단계 인과 추론 프롬프트 생성기

    MedCoT-RAG의 4단계를 물리학에 맞게 재설계:
    1. 현상 분석 (Symptom Analysis → Phenomenon Analysis)
    2. 물리적 메커니즘 (Causal Pathophysiology → Causal Mechanism)
    3. 파라미터 제약 (Differential Diagnosis → Parameter Constraints)
    4. 근거 통합 (Evidence Synthesis)
    """

    STAGE_PROMPTS = {
        PhysicsReasoningStage.PHENOMENON_ANALYSIS: """
## 1단계: 현상 분석 (Phenomenon Analysis)

관찰된 현상을 분석하십시오:
- 무엇이 관찰되었는가? (예: "영상이 자글자글하다", "노이즈가 줄지 않는다")
- 어떤 조건에서 발생하는가? (예: "확대 촬영", "선량 증가 시")
- 기대와 다른 점은 무엇인가?
""",

        PhysicsReasoningStage.CAUSAL_MECHANISM: """
## 2단계: 물리적 메커니즘 (Causal Mechanism)

현상의 물리적 원인을 분석하십시오:
- 어떤 물리 법칙/공식이 관련되는가?
- 원인 → 중간 과정 → 결과의 인과 체인을 명시하시오
- 관련 공식: (예: SNR ∝ √mAs, HU = kVp × mA × s)
""",

        PhysicsReasoningStage.PARAMETER_CONSTRAINTS: """
## 3단계: 파라미터 제약 (Parameter Constraints)

시스템의 제약 조건을 분석하십시오:
- 어떤 파라미터가 제한되는가? (예: 소초점 mA ≤ 25±10)
- 왜 제한되는가? (물리적 이유)
- 이 제한이 관찰된 현상에 어떻게 영향을 미치는가?
""",

        PhysicsReasoningStage.EVIDENCE_SYNTHESIS: """
## 4단계: 근거 통합 (Evidence Synthesis)

모든 분석을 종합하여 결론을 도출하십시오:
- 인과 체인 요약: 원인 → 메커니즘 → 제약 → 결과
- 정량적 근거 (가능한 경우)
- 실무적 함의 또는 해결책
"""
    }

    def build_prompt(
        self,
        question: str,
        knowledge_context: str,
        include_all_stages: bool = True
    ) -> str:
        """
        PhysicsCoT 프롬프트 생성

        Args:
            question: 사용자 질문
            knowledge_context: 검색된 지식 컨텍스트
            include_all_stages: 모든 단계 포함 여부

        Returns:
            구조화된 프롬프트
        """
        prompt_parts = []

        # 역할 및 컨텍스트
        prompt_parts.append("""
═══ PhysicsCoT-RAG: 인과 추론 기반 물리학 분석 ═══

당신은 맘모그래피 물리학 전문가입니다.
아래 4단계 인과 추론 프레임워크를 따라 질문에 답변하십시오.

[참조 지식]
""")
        prompt_parts.append(knowledge_context)

        # 질문
        prompt_parts.append(f"""

═══ 질문 ═══
{question}

═══ 4단계 인과 추론 ═══
""")

        # 4단계 프롬프트
        if include_all_stages:
            for stage in PhysicsReasoningStage:
                prompt_parts.append(self.STAGE_PROMPTS[stage])

        # 출력 형식
        prompt_parts.append("""
═══ 답변 형식 ═══
각 단계를 순서대로 작성한 후, 마지막에 **최종 결론**을 한 문단으로 요약하십시오.
인과 체인은 "A → B → C" 형식으로 명시하십시오.
""")

        return "\n".join(prompt_parts)

    def build_cqo_physics_prompt(
        self,
        question: str,
        knowledge_context: str,
        causal_chain_hint: Optional[str] = None
    ) -> str:
        """
        CQO + PhysicsCoT 결합 프롬프트 (Phase 1과 통합)

        Args:
            question: 사용자 질문
            knowledge_context: 지식 컨텍스트
            causal_chain_hint: 사전 식별된 인과 체인 힌트
        """
        prompt_parts = []

        # [C] Context: 지식 + 인과 체인 힌트
        prompt_parts.append("═══ 맥락 정보 ═══\n")
        prompt_parts.append("역할: 맘모그래피 물리학 전문가 (인과 추론 기반)\n")
        prompt_parts.append(f"\n[물리학 참조]\n{knowledge_context}\n")

        if causal_chain_hint:
            prompt_parts.append(f"\n[인과 체인 힌트]\n{causal_chain_hint}\n")

        # [Q] Question
        prompt_parts.append(f"\n═══ 질문 ═══\n{question}\n")

        # [O] Options: 4단계 추론 요구
        prompt_parts.append("""
═══ 답변 요구사항 ═══
1. 현상 분석: 관찰된 문제가 무엇인가?
2. 물리적 메커니즘: 어떤 물리 법칙이 원인인가? (공식 포함)
3. 파라미터 제약: 어떤 제한이 있는가?
4. 결론: 인과 체인(A→B→C)으로 요약

한국어, 자연스러운 문단으로 작성.
""")

        return "\n".join(prompt_parts)


class PhysicsCausalChainDetector:
    """
    복합 질문에서 인과 체인 사전 감지

    질문 유형 템플릿과 예상 인과 체인을 매칭
    """

    # 복합 질문 유형 템플릿
    QUESTION_TEMPLATES = {
        "magnification_thermal": {
            "triggers": [
                ("확대", "열용량"), ("확대", "mA"), ("소초점", "노이즈"),
                ("확대", "자글자글"), ("magnification", "thermal"),
                ("small focal", "noise"), ("확대", "선량", "노이즈")
            ],
            "causal_chain": "소초점 사용 → mA 제한(25±10mA) → mAs 부족 → 양자 노이즈 증가",
            "required_modules": [
                "xray_tube_thermal_capacity",
                "magnification_geometry",
                "noise_analysis"
            ],
            "formulas": ["HU = kVp × mA × s", "Noise ∝ 1/√mAs"]
        },
        "mtf_beam_hardening": {
            "triggers": [
                ("MTF", "빔 경화"), ("MTF", "CsI"), ("MTF", "에너지"),
                ("해상도", "빔 경화"), ("Fine Linear", "Amorphous"),
                ("형태", "MTF"), ("blurring", "beam")
            ],
            "causal_chain": "빔 경화 → 고에너지 광자 → CsI 깊은 침투 → 빛 확산 증가 → MTF 저하 → 형태 왜곡",
            "required_modules": [
                "system_mtf_chain",
                "scintillator_light_spread",
                "csi_detector_physics",
                "calcification_contrast_physics"
            ],
            "formulas": ["MTF_system = MTF_focal × MTF_detector", "MTF_det(5 lp/mm) ≈ 0.25"]
        },
        "dose_snr_optimization": {
            "triggers": [
                ("선량", "SNR"), ("dose", "noise"), ("mAs", "화질"),
                ("선량", "노이즈"), ("DQE", "선량")
            ],
            "causal_chain": "선량 증가 → 광자 수 증가 → 양자 노이즈 감소 → SNR 향상",
            "required_modules": [
                "snr_cnr",
                "mgd_dosimetry",
                "noise_analysis"
            ],
            "formulas": ["SNR ∝ √Dose", "SNR ∝ √mAs"]
        },
        "scatter_contrast": {
            "triggers": [
                ("산란", "대조도"), ("scatter", "contrast"), ("두꺼운", "CNR"),
                ("6cm", "대조도"), ("SPR", "CNR")
            ],
            "causal_chain": "유방 두께 증가 → 산란선 증가(SPR↑) → 대조도 저하(CNR↓)",
            "required_modules": [
                "scatter_radiation",
                "antiscatter_grid",
                "snr_cnr"
            ],
            "formulas": ["CNR = (S₁-S₂) / √(σ₁²+σ₂²)", "SPR ∝ thickness"]
        },
        "pcd_vs_eid": {
            "triggers": [
                ("PCD", "EID"), ("photon counting", "에너지 적분"),
                ("전자 노이즈", "PCD"), ("저선량", "PCD")
            ],
            "causal_chain": "PCD 에너지 문턱치 → 전자 노이즈 제거 → 저선량 SNR 우위",
            "required_modules": [
                "pcd_low_dose_snr",
                "pcd_dqe_nps",
                "detector_physics"
            ],
            "formulas": ["SNR_PCD > SNR_EID (at low dose)", "DQE_PCD(0) ≈ 1"]
        }
    }

    def detect_question_type(self, question: str) -> Optional[Dict[str, Any]]:
        """
        질문 유형 감지 및 인과 체인 반환

        Args:
            question: 사용자 질문

        Returns:
            매칭된 템플릿 또는 None
        """
        question_lower = question.lower()

        best_match = None
        best_score = 0

        for template_name, template in self.QUESTION_TEMPLATES.items():
            score = 0
            for trigger_tuple in template["triggers"]:
                # 모든 트리거 키워드가 질문에 있는지 확인
                if all(kw.lower() in question_lower for kw in trigger_tuple):
                    score += len(trigger_tuple)  # 더 많은 키워드 매칭 = 높은 점수

            if score > best_score:
                best_score = score
                best_match = {
                    "type": template_name,
                    "causal_chain": template["causal_chain"],
                    "required_modules": template["required_modules"],
                    "formulas": template["formulas"],
                    "match_score": score
                }

        if best_match and best_match["match_score"] >= 2:
            logger.info(f"Detected question type: {best_match['type']} (score: {best_match['match_score']})")
            return best_match

        return None


class PhysicsCoTRAG:
    """
    PhysicsCoT-RAG: 물리학 인과 추론 RAG 시스템

    MedCoT-RAG를 맘모그래피 물리학에 적용한 통합 시스템.
    """

    def __init__(self):
        self.retriever = PhysicsCausalRetriever()
        self.prompt_builder = PhysicsCoTPromptBuilder()
        self.chain_detector = PhysicsCausalChainDetector()

    def process(
        self,
        question: str,
        knowledge_modules: List[Dict[str, Any]],
        use_cqo_integration: bool = True
    ) -> Tuple[str, PhysicsCoTResult]:
        """
        PhysicsCoT-RAG 처리

        Args:
            question: 사용자 질문
            knowledge_modules: 후보 지식 모듈 리스트
            use_cqo_integration: CQO 프롬프트와 통합 여부

        Returns:
            (prompt, result)
        """
        # 1. 질문 유형 감지 및 인과 체인 사전 식별
        detected_type = self.chain_detector.detect_question_type(question)

        # 2. Causal-aware retrieval
        ranked_modules = self.retriever.rank_modules(question, knowledge_modules)

        # 감지된 유형이 있으면 필수 모듈 우선
        if detected_type:
            required = detected_type["required_modules"]
            # 필수 모듈을 상위로 이동
            ranked_ids = [r[0] for r in ranked_modules]
            for req_mod in reversed(required):
                if req_mod in ranked_ids:
                    idx = ranked_ids.index(req_mod)
                    ranked_modules.insert(0, ranked_modules.pop(idx))
                    ranked_ids = [r[0] for r in ranked_modules]

        # 상위 모듈 선택
        top_modules = []
        causal_scores = {}
        for module_id, total, sem, causal in ranked_modules[:5]:
            for m in knowledge_modules:
                if m.get("id") == module_id:
                    top_modules.append(m)
                    causal_scores[module_id] = causal
                    break

        # 3. 지식 컨텍스트 구성
        knowledge_context = self._format_knowledge(top_modules)

        # 4. 프롬프트 생성
        causal_hint = detected_type["causal_chain"] if detected_type else None

        if use_cqo_integration:
            prompt = self.prompt_builder.build_cqo_physics_prompt(
                question, knowledge_context, causal_hint
            )
        else:
            prompt = self.prompt_builder.build_prompt(
                question, knowledge_context
            )

        # 5. 결과 구성
        causal_chains = []
        if detected_type:
            chain = CausalChain(
                cause=detected_type["causal_chain"].split("→")[0].strip(),
                mechanism=" → ".join(detected_type["causal_chain"].split("→")[1:-1]).strip(),
                effect=detected_type["causal_chain"].split("→")[-1].strip(),
                formula=detected_type["formulas"][0] if detected_type["formulas"] else None,
                constraints=[]
            )
            causal_chains.append(chain)

        result = PhysicsCoTResult(
            answer="",  # LLM 호출 후 채워짐
            causal_chains=causal_chains,
            reasoning_stages={},
            retrieved_modules=[m.get("id", "") for m in top_modules],
            causal_scores=causal_scores,
            confidence=0.0
        )

        return prompt, result

    def _format_knowledge(self, modules: List[Dict]) -> str:
        """지식 모듈을 컨텍스트 문자열로 포맷"""
        parts = []

        for module in modules:
            module_id = module.get("id", "unknown")
            title = module.get("title", module_id)

            parts.append(f"\n### [{module_id}] {title}\n")

            # 핵심 내용 추출
            content = module.get("content", {})
            if isinstance(content, dict):
                for key, value in list(content.items())[:3]:  # 상위 3개 섹션만
                    if isinstance(value, dict):
                        parts.append(f"**{key}**: {self._summarize_dict(value)}\n")
                    else:
                        parts.append(f"**{key}**: {value}\n")

            # 공식
            formulas = module.get("formulas", {})
            if formulas:
                parts.append("**공식**:\n")
                for name, formula in list(formulas.items())[:3]:
                    parts.append(f"  - {name}: {formula}\n")

            # 오개념 경고
            misconceptions = module.get("common_misconceptions", {})
            if misconceptions and "WARNING" in misconceptions:
                parts.append(f"**경고**: {misconceptions.get('WARNING', '')}\n")

        return "".join(parts)

    def _summarize_dict(self, d: Dict, max_items: int = 3) -> str:
        """딕셔너리 요약"""
        if not isinstance(d, dict):
            return str(d)

        items = []
        for k, v in list(d.items())[:max_items]:
            if isinstance(v, dict):
                items.append(f"{k}={list(v.values())[0] if v else ''}")
            else:
                items.append(f"{k}={v}")

        return ", ".join(items)


# =============================================================================
# Singleton
# =============================================================================

_physics_cot_instance: Optional[PhysicsCoTRAG] = None


def get_physics_cot_rag() -> PhysicsCoTRAG:
    """PhysicsCoT-RAG 싱글톤"""
    global _physics_cot_instance
    if _physics_cot_instance is None:
        _physics_cot_instance = PhysicsCoTRAG()
    return _physics_cot_instance


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 테스트 모듈
    test_modules = [
        {
            "id": "xray_tube_thermal_capacity",
            "title": "X-ray Tube Thermal Capacity",
            "keywords": ["열용량", "thermal", "mA 제한", "소초점"],
            "content": {"focal_spot": {"small": "25±10 mA"}},
            "formulas": {"heat_unit": "HU = kVp × mA × s"}
        },
        {
            "id": "magnification_geometry",
            "title": "Magnification Geometry",
            "keywords": ["확대", "소초점", "penumbra"],
            "content": {"penumbra": "P = f × (M-1)"},
            "formulas": {"penumbra": "P = f × (M-1)"}
        },
        {
            "id": "noise_analysis",
            "title": "Noise Analysis",
            "keywords": ["노이즈", "SNR", "quantum noise"],
            "content": {"snr": "SNR ∝ √mAs"},
            "formulas": {"noise": "Noise ∝ 1/√mAs"}
        }
    ]

    # 테스트 질문
    question = """확대 촬영에서 선량을 높여도 노이즈가 안 줄어요.
    소초점의 열용량 한계와 관련하여 설명해주세요."""

    # 처리
    cot_rag = get_physics_cot_rag()
    prompt, result = cot_rag.process(question, test_modules)

    print("=" * 60)
    print("PhysicsCoT-RAG Test")
    print("=" * 60)
    print(f"\n질문: {question[:50]}...")
    print(f"\n감지된 인과 체인: {result.causal_chains[0].cause if result.causal_chains else 'None'}")
    print(f"검색된 모듈: {result.retrieved_modules}")
    print(f"인과 점수: {result.causal_scores}")
    print("\n" + "=" * 60)
    print("생성된 프롬프트 (처음 1500자):")
    print("=" * 60)
    print(prompt[:1500])
