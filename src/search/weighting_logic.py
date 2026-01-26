"""
Sophia AI: Weighted Medical Reranking (WMR)
=============================================
Phase 3: 의학적 신뢰도 기반 가중치 리랭킹

최종 점수 = base_score × type_weight × recency_boost × citation_boost

- type_weight: 문서 유형 가중치 (가이드라인 > 메타분석 > 원저)
- recency_boost: 최신성 가중치 (2024+ 논문 우선)
- citation_boost: 피인용 수 가중치 (학계 검증)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

from src.models import Paper

logger = logging.getLogger(__name__)


# =============================================================================
# Weight Configuration (튜닝 가능한 설정)
# =============================================================================

@dataclass
class WMRConfig:
    """
    Weighted Medical Reranking 설정

    실무 테스트 후 값 조정 가능
    """

    # =========================================================================
    # 1. 문서 유형 가중치 (Type Weight)
    # =========================================================================
    # PMID 접두사 기반 (내부 가이드라인 문서)
    prefix_weights: Dict[str, float] = field(default_factory=lambda: {
        "BIRADS_": 1.50,      # BI-RADS Atlas (최우선)
        "ACR_": 1.40,         # ACR Practice Parameters
        "PHYSICS_": 1.35,     # 물리학 참조 문서
        "CLINICAL_": 1.30,    # 임상 가이드라인
        "DANCE_": 1.25,       # Dance et al. 물리 참조
    })

    # 연구 유형 가중치 (study_type 필드 기반)
    study_type_weights: Dict[str, float] = field(default_factory=lambda: {
        "meta-analysis": 1.20,    # 메타분석 (최고 근거 수준)
        "systematic review": 1.18,
        "review": 1.15,           # 종설
        "rct": 1.12,              # 무작위대조시험
        "prospective": 1.08,      # 전향적 연구
        "cohort": 1.05,           # 코호트 연구
        "retrospective": 1.00,    # 후향적 연구 (기준)
        "cross-sectional": 0.95,  # 단면 연구
        "case-control": 0.90,     # 환자-대조군
        "case series": 0.70,      # 증례 시리즈
        "case report": 0.60,      # 증례 보고 (최저)
    })

    # 출판 유형 기반 페널티 (publication_types 필드)
    publication_type_penalties: Dict[str, float] = field(default_factory=lambda: {
        "case reports": 0.60,
        "letter": 0.50,
        "comment": 0.55,
        "editorial": 0.65,
        "erratum": 0.30,
        "retracted publication": 0.10,  # 철회 논문 거의 제외
    })

    # =========================================================================
    # 2. 최신성 가중치 (Recency Boost)
    # =========================================================================
    # 연도별 가중치 (현재 연도 기준)
    recency_weights: Dict[str, float] = field(default_factory=lambda: {
        "current": 1.15,      # 올해 (2026)
        "last_year": 1.12,    # 작년 (2025)
        "2_years": 1.08,      # 2년 전 (2024)
        "3_years": 1.05,      # 3년 전 (2023)
        "4_5_years": 1.02,    # 4-5년 전 (2021-2022)
        "6_10_years": 1.00,   # 6-10년 전 (기준)
        "11_15_years": 0.95,  # 11-15년 전
        "old": 0.90,          # 15년 이상
    })

    # 가이드라인은 최신성 패널티 면제
    guideline_recency_exempt: bool = True

    # =========================================================================
    # 3. 인용 수 가중치 (Citation Boost)
    # =========================================================================
    citation_thresholds: Dict[str, Tuple[int, float]] = field(default_factory=lambda: {
        "highly_cited": (100, 1.12),   # 100회 이상
        "well_cited": (50, 1.08),      # 50-99회
        "moderately_cited": (20, 1.04), # 20-49회
        "cited": (5, 1.00),            # 5-19회 (기준)
        "rarely_cited": (0, 0.98),     # 0-4회
    })

    # 최신 논문(3년 이내)은 인용 수 페널티 면제
    new_paper_citation_exempt_years: int = 3

    # =========================================================================
    # 4. 저널 Impact Factor 가중치 (선택적)
    # =========================================================================
    journal_if_weights: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "top_tier": (10.0, 1.10),     # IF >= 10
        "high": (5.0, 1.05),          # IF 5-10
        "medium": (2.0, 1.02),        # IF 2-5
        "low": (0.0, 1.00),           # IF < 2 (기준)
    })

    # =========================================================================
    # 5. 쿼리 의도 기반 동적 가중치 (Query Intent Boost)
    # =========================================================================
    intent_boosts: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        # 가이드라인 질문일 때
        "guideline": {
            "BIRADS_": 1.8,
            "ACR_": 1.6,
            "review": 1.3,
        },
        # 진단 성능 질문일 때
        "diagnostic_performance": {
            "meta-analysis": 1.4,
            "prospective": 1.2,
        },
        # 선량/물리 질문일 때
        "physics": {
            "PHYSICS_": 1.5,
            "DANCE_": 1.4,
        },
    })


# 기본 설정 인스턴스
DEFAULT_WMR_CONFIG = WMRConfig()


# =============================================================================
# Weight Calculator
# =============================================================================

class WeightCalculator:
    """
    개별 가중치 계산기
    """

    def __init__(self, config: WMRConfig = None):
        self.config = config or DEFAULT_WMR_CONFIG
        self.current_year = datetime.now().year

    def calculate_type_weight(self, paper: Paper) -> float:
        """
        문서 유형 가중치 계산

        Args:
            paper: Paper 객체

        Returns:
            type_weight (float)
        """
        weight = 1.0

        # 1. PMID 접두사 기반 가중치 (가이드라인 문서)
        for prefix, w in self.config.prefix_weights.items():
            if paper.pmid and paper.pmid.startswith(prefix):
                weight = max(weight, w)
                logger.debug(f"[{paper.pmid}] Prefix boost: {prefix} → {w}")
                break

        # 2. 연구 유형 가중치 (일반 논문)
        if paper.study_type:
            study_type = paper.study_type.lower()
            for st, w in self.config.study_type_weights.items():
                if st in study_type:
                    weight = max(weight, w)
                    logger.debug(f"[{paper.pmid}] Study type: {study_type} → {w}")
                    break

        # 3. 출판 유형 페널티
        if paper.publication_types:
            for pub_type in paper.publication_types:
                pub_lower = pub_type.lower()
                for penalty_type, penalty in self.config.publication_type_penalties.items():
                    if penalty_type in pub_lower:
                        weight *= penalty
                        logger.debug(f"[{paper.pmid}] Pub type penalty: {pub_type} → ×{penalty}")
                        break

        return weight

    def calculate_recency_boost(self, paper: Paper) -> float:
        """
        최신성 가중치 계산

        Args:
            paper: Paper 객체

        Returns:
            recency_boost (float)
        """
        # 가이드라인은 최신성 패널티 면제
        if self.config.guideline_recency_exempt:
            for prefix in self.config.prefix_weights.keys():
                if paper.pmid and paper.pmid.startswith(prefix):
                    return 1.0  # 면제

        year = paper.year
        if not year:
            return 1.0

        years_old = self.current_year - year
        weights = self.config.recency_weights

        if years_old <= 0:
            boost = weights["current"]
        elif years_old == 1:
            boost = weights["last_year"]
        elif years_old == 2:
            boost = weights["2_years"]
        elif years_old == 3:
            boost = weights["3_years"]
        elif years_old <= 5:
            boost = weights["4_5_years"]
        elif years_old <= 10:
            boost = weights["6_10_years"]
        elif years_old <= 15:
            boost = weights["11_15_years"]
        else:
            boost = weights["old"]

        logger.debug(f"[{paper.pmid}] Year {year} ({years_old}y old) → recency {boost}")
        return boost

    def calculate_citation_boost(self, paper: Paper) -> float:
        """
        인용 수 가중치 계산

        Args:
            paper: Paper 객체

        Returns:
            citation_boost (float)
        """
        # 최신 논문은 인용 수 페널티 면제
        if paper.year:
            years_old = self.current_year - paper.year
            if years_old <= self.config.new_paper_citation_exempt_years:
                return 1.0

        citations = paper.citation_count or 0
        thresholds = self.config.citation_thresholds

        if citations >= thresholds["highly_cited"][0]:
            boost = thresholds["highly_cited"][1]
        elif citations >= thresholds["well_cited"][0]:
            boost = thresholds["well_cited"][1]
        elif citations >= thresholds["moderately_cited"][0]:
            boost = thresholds["moderately_cited"][1]
        elif citations >= thresholds["cited"][0]:
            boost = thresholds["cited"][1]
        else:
            boost = thresholds["rarely_cited"][1]

        logger.debug(f"[{paper.pmid}] Citations {citations} → boost {boost}")
        return boost

    def calculate_journal_boost(self, paper: Paper) -> float:
        """
        저널 IF 가중치 계산 (선택적)

        Args:
            paper: Paper 객체

        Returns:
            journal_boost (float)
        """
        if not paper.journal_if:
            return 1.0

        if_value = paper.journal_if
        weights = self.config.journal_if_weights

        if if_value >= weights["top_tier"][0]:
            boost = weights["top_tier"][1]
        elif if_value >= weights["high"][0]:
            boost = weights["high"][1]
        elif if_value >= weights["medium"][0]:
            boost = weights["medium"][1]
        else:
            boost = weights["low"][1]

        logger.debug(f"[{paper.pmid}] Journal IF {if_value} → boost {boost}")
        return boost

    def calculate_intent_boost(self, paper: Paper, intent: str) -> float:
        """
        쿼리 의도 기반 동적 가중치

        Args:
            paper: Paper 객체
            intent: 검색 의도 문자열

        Returns:
            intent_boost (float)
        """
        if not intent:
            return 1.0

        intent_lower = intent.lower()
        boost = 1.0

        # 가이드라인 의도
        if any(kw in intent_lower for kw in ["guideline", "definition", "criteria", "분류", "기준"]):
            intent_config = self.config.intent_boosts.get("guideline", {})
            for key, b in intent_config.items():
                if paper.pmid and paper.pmid.startswith(key):
                    boost = max(boost, b)
                elif paper.study_type and key in paper.study_type.lower():
                    boost = max(boost, b)

        # 진단 성능 의도
        if any(kw in intent_lower for kw in ["sensitivity", "specificity", "accuracy", "performance", "diagnostic"]):
            intent_config = self.config.intent_boosts.get("diagnostic_performance", {})
            for key, b in intent_config.items():
                if paper.study_type and key in paper.study_type.lower():
                    boost = max(boost, b)

        # 물리/선량 의도
        if any(kw in intent_lower for kw in ["dose", "mgd", "physics", "radiation", "선량"]):
            intent_config = self.config.intent_boosts.get("physics", {})
            for key, b in intent_config.items():
                if paper.pmid and paper.pmid.startswith(key):
                    boost = max(boost, b)

        if boost > 1.0:
            logger.debug(f"[{paper.pmid}] Intent '{intent[:30]}' → boost {boost}")

        return boost


# =============================================================================
# Weighted Medical Reranker
# =============================================================================

class WeightedMedicalReranker:
    """
    Weighted Medical Reranking (WMR)

    최종 점수 = base_score × type_weight × recency_boost × citation_boost × intent_boost
    """

    def __init__(
        self,
        config: WMRConfig = None,
        use_cross_encoder: bool = True,
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        """
        Args:
            config: WMR 설정
            use_cross_encoder: Cross-Encoder 사용 여부
            cross_encoder_model: Cross-Encoder 모델 이름
        """
        self.config = config or DEFAULT_WMR_CONFIG
        self.calculator = WeightCalculator(self.config)
        self.use_cross_encoder = use_cross_encoder

        # Cross-Encoder (lazy loading)
        self._cross_encoder = None
        self._cross_encoder_model = cross_encoder_model

    def _get_cross_encoder(self):
        """Cross-Encoder lazy loading"""
        if self._cross_encoder is None and self.use_cross_encoder:
            try:
                from sentence_transformers import CrossEncoder
                logger.info(f"Loading Cross-Encoder: {self._cross_encoder_model}")
                self._cross_encoder = CrossEncoder(self._cross_encoder_model)
            except ImportError:
                logger.warning("sentence-transformers not installed, using base scores only")
                self.use_cross_encoder = False
        return self._cross_encoder

    def _prepare_text(self, paper: Paper) -> str:
        """리랭킹용 텍스트 준비"""
        # 가이드라인 문서는 full_content 사용
        if paper.pmid and any(paper.pmid.startswith(p) for p in self.config.prefix_weights):
            text = f"{paper.title}\n\n{paper.full_content if paper.full_content else paper.abstract}"
            max_len = 2000
        else:
            text = f"{paper.title}\n\n{paper.abstract}"
            max_len = 1000

        return text[:max_len] if len(text) > max_len else text

    def calculate_final_score(
        self,
        paper: Paper,
        base_score: float,
        intent: str = "",
    ) -> Tuple[float, Dict[str, float]]:
        """
        최종 가중치 점수 계산

        Args:
            paper: Paper 객체
            base_score: 기본 점수 (Cross-Encoder 또는 BM25/Vector)
            intent: 검색 의도

        Returns:
            (final_score, weight_breakdown)
        """
        # 개별 가중치 계산
        type_weight = self.calculator.calculate_type_weight(paper)
        recency_boost = self.calculator.calculate_recency_boost(paper)
        citation_boost = self.calculator.calculate_citation_boost(paper)
        journal_boost = self.calculator.calculate_journal_boost(paper)
        intent_boost = self.calculator.calculate_intent_boost(paper, intent)

        # 최종 점수
        final_score = (
            base_score
            * type_weight
            * recency_boost
            * citation_boost
            * journal_boost
            * intent_boost
        )

        breakdown = {
            "base_score": base_score,
            "type_weight": type_weight,
            "recency_boost": recency_boost,
            "citation_boost": citation_boost,
            "journal_boost": journal_boost,
            "intent_boost": intent_boost,
            "final_score": final_score,
        }

        return final_score, breakdown

    def rerank(
        self,
        query: str,
        papers: List[Paper],
        base_scores: Optional[List[float]] = None,
        intent: str = "",
        top_k: int = 20,
    ) -> List[Tuple[Paper, float, Dict[str, float]]]:
        """
        WMR 리랭킹 실행

        Args:
            query: 검색 쿼리
            papers: 논문 리스트
            base_scores: 기존 점수 (없으면 Cross-Encoder 사용)
            intent: 검색 의도
            top_k: 반환할 상위 개수

        Returns:
            (Paper, final_score, breakdown) 튜플 리스트
        """
        if not papers:
            return []

        # 1. 기본 점수 계산
        if base_scores is None:
            cross_encoder = self._get_cross_encoder()
            if cross_encoder:
                pairs = [(query, self._prepare_text(p)) for p in papers]
                base_scores = cross_encoder.predict(pairs, show_progress_bar=False)
                base_scores = list(base_scores)
            else:
                # Cross-Encoder 없으면 균등 점수
                base_scores = [1.0] * len(papers)

        # 2. 가중치 적용
        results = []
        for paper, base_score in zip(papers, base_scores):
            final_score, breakdown = self.calculate_final_score(
                paper, float(base_score), intent
            )
            results.append((paper, final_score, breakdown))

        # 3. 최종 점수 기준 정렬
        results.sort(key=lambda x: x[1], reverse=True)

        logger.info(f"WMR reranked {len(papers)} papers → top {top_k}")

        # 상위 결과 로깅
        for i, (paper, score, breakdown) in enumerate(results[:5]):
            logger.debug(
                f"  [{i+1}] {paper.pmid}: {score:.4f} "
                f"(base={breakdown['base_score']:.3f}, "
                f"type={breakdown['type_weight']:.2f}, "
                f"recency={breakdown['recency_boost']:.2f})"
            )

        return results[:top_k]

    def rerank_simple(
        self,
        query: str,
        papers: List[Paper],
        intent: str = "",
        top_k: int = 20,
    ) -> List[Tuple[Paper, float]]:
        """
        간단한 리랭킹 (breakdown 없이)

        Returns:
            (Paper, final_score) 튜플 리스트
        """
        results = self.rerank(query, papers, intent=intent, top_k=top_k)
        return [(paper, score) for paper, score, _ in results]

    def normalize_scores(
        self,
        results: List[Tuple[Paper, float, Any]],
    ) -> List[Tuple[Paper, float, Any]]:
        """점수 0-1 정규화"""
        if not results:
            return results

        scores = [score for _, score, _ in results]
        min_s, max_s = min(scores), max(scores)

        if max_s == min_s:
            return [(p, 1.0, b) for p, _, b in results]

        return [
            (paper, (score - min_s) / (max_s - min_s), breakdown)
            for paper, score, breakdown in results
        ]


# =============================================================================
# Convenience Functions
# =============================================================================

def apply_wmr(
    query: str,
    papers: List[Paper],
    intent: str = "",
    top_k: int = 20,
) -> List[Tuple[Paper, float]]:
    """
    WMR 적용 (편의 함수)

    Args:
        query: 검색 쿼리
        papers: 논문 리스트
        intent: 검색 의도
        top_k: 반환할 상위 개수

    Returns:
        (Paper, score) 튜플 리스트
    """
    reranker = WeightedMedicalReranker()
    return reranker.rerank_simple(query, papers, intent=intent, top_k=top_k)


def get_paper_weight_breakdown(paper: Paper, intent: str = "") -> Dict[str, float]:
    """
    단일 논문의 가중치 breakdown

    Args:
        paper: Paper 객체
        intent: 검색 의도

    Returns:
        가중치 breakdown 딕셔너리
    """
    calculator = WeightCalculator()
    return {
        "type_weight": calculator.calculate_type_weight(paper),
        "recency_boost": calculator.calculate_recency_boost(paper),
        "citation_boost": calculator.calculate_citation_boost(paper),
        "journal_boost": calculator.calculate_journal_boost(paper),
        "intent_boost": calculator.calculate_intent_boost(paper, intent),
    }


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # 테스트용 가짜 논문
    test_papers = [
        Paper(
            pmid="BIRADS_2025_SECTION_V",
            title="BI-RADS Assessment Categories",
            journal="ACR",
            year=2025,
            abstract="BI-RADS category definitions...",
            study_type=None,
            citation_count=0,
        ),
        Paper(
            pmid="12345678",
            title="Meta-analysis of DBT vs FFDM",
            journal="Radiology",
            year=2024,
            abstract="Systematic review of 50 studies...",
            study_type="meta-analysis",
            citation_count=150,
            journal_if=12.5,
        ),
        Paper(
            pmid="87654321",
            title="Case Report: Rare Breast Lesion",
            journal="Case Reports in Radiology",
            year=2023,
            abstract="We present a case of...",
            study_type="case report",
            publication_types=["Case Reports"],
            citation_count=2,
            journal_if=0.5,
        ),
        Paper(
            pmid="11111111",
            title="Retrospective Study of Screening",
            journal="European Radiology",
            year=2015,
            abstract="We retrospectively analyzed...",
            study_type="retrospective",
            citation_count=45,
            journal_if=5.2,
        ),
    ]

    reranker = WeightedMedicalReranker(use_cross_encoder=False)

    print("=" * 70)
    print("WMR Test: Guideline Query")
    print("=" * 70)

    results = reranker.rerank(
        query="BI-RADS category 4A definition",
        papers=test_papers,
        base_scores=[0.8, 0.85, 0.6, 0.7],  # 가짜 base scores
        intent="BI-RADS guideline definition criteria",
        top_k=10,
    )

    for i, (paper, score, breakdown) in enumerate(results, 1):
        print(f"\n[{i}] {paper.pmid}")
        print(f"    Title: {paper.title[:50]}...")
        print(f"    Final Score: {score:.4f}")
        print(f"    Breakdown: {breakdown}")

    print("\n" + "=" * 70)
    print("WMR Test: Diagnostic Performance Query")
    print("=" * 70)

    results2 = reranker.rerank(
        query="DBT sensitivity specificity meta-analysis",
        papers=test_papers,
        base_scores=[0.5, 0.9, 0.4, 0.75],
        intent="diagnostic performance comparison",
        top_k=10,
    )

    for i, (paper, score, breakdown) in enumerate(results2, 1):
        print(f"\n[{i}] {paper.pmid}")
        print(f"    Final Score: {score:.4f}")
        print(f"    Type: {breakdown['type_weight']:.2f}, Recency: {breakdown['recency_boost']:.2f}")
