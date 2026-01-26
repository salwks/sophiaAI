"""
Evidence Mapper: 근거 추적 시스템
=================================
답변 내 모든 주장에 대해 출처를 정확히 트래킹하여
"이 수치는 어느 논문의 어느 섹션에서 왔는가"를 명시합니다.

Features:
    - 주장-근거 매핑 (Claim-Evidence Mapping)
    - 출처 신뢰도 평가 (Source Credibility)
    - 인용 포맷팅 (Citation Formatting)
"""

import re
import logging
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class SourceType(Enum):
    """근거 출처 유형"""
    KNOWLEDGE_MODULE = "knowledge_module"  # 내부 지식 모듈 (Phase 4)
    BIRADS_GUIDELINE = "birads_guideline"  # BI-RADS 가이드라인
    PMC_FULLTEXT = "pmc_fulltext"          # PMC 전문
    PUBMED_ABSTRACT = "pubmed_abstract"    # PubMed 초록
    DERIVED = "derived"                     # 추론으로 도출


class CredibilityLevel(Enum):
    """출처 신뢰도"""
    GOLD = "gold"           # 가이드라인, 검증된 지식 모듈
    HIGH = "high"           # PMC 전문 (peer-reviewed)
    MEDIUM = "medium"       # PubMed 초록
    LOW = "low"             # 추론, 간접 인용


@dataclass
class EvidenceSource:
    """단일 근거 출처"""
    source_type: SourceType
    source_id: str                  # PMC ID, PMID, knowledge module ID
    title: str = ""
    section: str = ""               # Results, Methods, etc.
    page_or_location: str = ""      # 페이지 또는 위치 정보
    original_text: str = ""         # 원문 텍스트
    credibility: CredibilityLevel = CredibilityLevel.MEDIUM


@dataclass
class MappedClaim:
    """매핑된 주장"""
    claim: str                      # 주장 내용
    evidence_sources: List[EvidenceSource] = field(default_factory=list)
    confidence: float = 0.0         # 근거 기반 신뢰도
    is_verified: bool = False       # 검증 여부


@dataclass
class EvidenceReport:
    """근거 매핑 보고서"""
    mapped_claims: List[MappedClaim]
    total_claims: int
    verified_claims: int
    gold_sources: int
    high_sources: int
    overall_credibility: float      # 전체 신뢰도 (0-1)


class EvidenceMapper:
    """
    근거 매핑 엔진

    Usage:
        mapper = EvidenceMapper()

        # 주장과 근거 매핑
        claim = "BI-RADS Category 3의 악성 확률은 2% 이하입니다."
        sources = [EvidenceSource(...)]
        mapped = mapper.map_claim(claim, sources)

        # 전체 답변 분석
        report = mapper.analyze_answer(answer_text, all_sources)
    """

    def __init__(self):
        # 수치 패턴 (악성 확률, 퍼센트, 배수 등)
        self.numeric_patterns = [
            r'(\d+(?:\.\d+)?)\s*%',           # 퍼센트
            r'(\d+(?:\.\d+)?)\s*배',           # 배수
            r'(\d+(?:\.\d+)?)\s*(?:mm|cm|mGy)', # 단위 포함
            r'[<>≤≥약]\s*(\d+(?:\.\d+)?)',     # 비교 연산자
            r'(\d+(?:\.\d+)?)\s*[-~]\s*(\d+(?:\.\d+)?)',  # 범위
        ]

    def map_claim(
        self,
        claim: str,
        available_sources: List[Dict[str, Any]]
    ) -> MappedClaim:
        """
        단일 주장에 대한 근거 매핑

        Args:
            claim: 주장 텍스트
            available_sources: 사용 가능한 근거 소스들

        Returns:
            MappedClaim: 매핑된 주장
        """
        evidence_sources = []
        claim_lower = claim.lower()

        # 주장에서 핵심 수치/용어 추출
        claim_numbers = self._extract_numbers(claim)
        claim_keywords = self._extract_keywords(claim)

        for source in available_sources:
            source_text = source.get("text", source.get("content", "")).lower()
            source_type = self._determine_source_type(source)

            # 수치 매칭
            number_match = any(num in source_text for num in claim_numbers)

            # 키워드 매칭
            keyword_match = sum(1 for kw in claim_keywords if kw in source_text)

            if number_match or keyword_match >= 2:
                evidence_sources.append(EvidenceSource(
                    source_type=source_type,
                    source_id=source.get("id", source.get("pmid", "unknown")),
                    title=source.get("title", ""),
                    section=source.get("section", ""),
                    original_text=self._find_matching_sentence(source_text, claim_keywords),
                    credibility=self._get_credibility(source_type)
                ))

        # 신뢰도 계산
        confidence = self._calculate_confidence(evidence_sources)
        is_verified = confidence >= 0.7

        return MappedClaim(
            claim=claim,
            evidence_sources=evidence_sources,
            confidence=confidence,
            is_verified=is_verified
        )

    def analyze_answer(
        self,
        answer_text: str,
        available_sources: List[Dict[str, Any]]
    ) -> EvidenceReport:
        """
        전체 답변에 대한 근거 분석

        Args:
            answer_text: 답변 텍스트
            available_sources: 사용 가능한 근거 소스들

        Returns:
            EvidenceReport: 근거 분석 보고서
        """
        # 답변을 문장 단위로 분리
        sentences = re.split(r'(?<=[.!?])\s+', answer_text)

        # 주장이 될 수 있는 문장 필터링 (수치나 판단이 포함된 문장)
        claims = [s for s in sentences if self._is_claim(s)]

        mapped_claims = []
        for claim in claims:
            mapped = self.map_claim(claim, available_sources)
            mapped_claims.append(mapped)

        # 통계 계산
        verified_claims = sum(1 for mc in mapped_claims if mc.is_verified)
        gold_sources = sum(
            1 for mc in mapped_claims
            for es in mc.evidence_sources
            if es.credibility == CredibilityLevel.GOLD
        )
        high_sources = sum(
            1 for mc in mapped_claims
            for es in mc.evidence_sources
            if es.credibility == CredibilityLevel.HIGH
        )

        overall_credibility = (
            sum(mc.confidence for mc in mapped_claims) / len(mapped_claims)
            if mapped_claims else 0.0
        )

        return EvidenceReport(
            mapped_claims=mapped_claims,
            total_claims=len(claims),
            verified_claims=verified_claims,
            gold_sources=gold_sources,
            high_sources=high_sources,
            overall_credibility=overall_credibility
        )

    def format_citations(self, mapped_claims: List[MappedClaim]) -> str:
        """
        매핑된 주장들을 인용 형식으로 포맷팅

        Returns:
            마크다운 형식의 인용 목록
        """
        if not mapped_claims:
            return ""

        lines = ["### 📖 근거 출처 (Evidence Sources)"]
        lines.append("")

        for i, mc in enumerate(mapped_claims, 1):
            if not mc.evidence_sources:
                continue

            verified_icon = "✅" if mc.is_verified else "⚠️"
            lines.append(f"**{i}. {verified_icon} \"{mc.claim[:50]}...\"**")

            for es in mc.evidence_sources:
                cred_icons = {
                    CredibilityLevel.GOLD: "🏆",
                    CredibilityLevel.HIGH: "📗",
                    CredibilityLevel.MEDIUM: "📄",
                    CredibilityLevel.LOW: "📝"
                }
                icon = cred_icons.get(es.credibility, "📄")

                source_info = f"{icon} {es.source_type.value}: {es.source_id}"
                if es.section:
                    source_info += f" ({es.section})"
                if es.title:
                    source_info += f" - {es.title[:40]}..."

                lines.append(f"   - {source_info}")

            lines.append("")

        return "\n".join(lines)

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _extract_numbers(self, text: str) -> List[str]:
        """텍스트에서 수치 추출"""
        numbers = []
        for pattern in self.numeric_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    numbers.extend(match)
                else:
                    numbers.append(match)
        return [str(n) for n in numbers if n]

    def _extract_keywords(self, text: str) -> List[str]:
        """핵심 키워드 추출"""
        text_lower = text.lower()
        # 의학 용어 패턴
        medical_terms = re.findall(
            r'\b(?:bi-rads|category|snr|cnr|mgd|dose|mgy|sensitivity|specificity|'
            r'malignancy|benign|calcification|mass|density|t-factor|dbt|'
            r'mammography|tomosynthesis|악성|양성|선량|확률)\b',
            text_lower
        )
        # 일반 단어
        words = re.findall(r'\b[a-z가-힣]{3,}\b', text_lower)

        return list(set(medical_terms + words[:10]))

    def _is_claim(self, sentence: str) -> bool:
        """문장이 주장(검증 가능한 진술)인지 판단"""
        # 수치, 퍼센트, 비교 표현이 있으면 주장
        has_number = bool(re.search(r'\d+(?:\.\d+)?', sentence))
        has_comparison = bool(re.search(r'[<>≤≥]|이상|이하|미만|초과', sentence))
        has_judgment = bool(re.search(r'권고|필요|해야|입니다|됩니다', sentence))

        return has_number or has_comparison or has_judgment

    def _determine_source_type(self, source: Dict) -> SourceType:
        """소스 유형 결정"""
        source_id = source.get("id", source.get("pmid", ""))

        if source_id.startswith("BIRADS_") or source_id.startswith("ACR_"):
            return SourceType.BIRADS_GUIDELINE
        elif source.get("source") == "knowledge_module":
            return SourceType.KNOWLEDGE_MODULE
        elif source.get("pmc_id") or source.get("has_fulltext"):
            return SourceType.PMC_FULLTEXT
        else:
            return SourceType.PUBMED_ABSTRACT

    def _get_credibility(self, source_type: SourceType) -> CredibilityLevel:
        """소스 유형에 따른 신뢰도"""
        credibility_map = {
            SourceType.KNOWLEDGE_MODULE: CredibilityLevel.GOLD,
            SourceType.BIRADS_GUIDELINE: CredibilityLevel.GOLD,
            SourceType.PMC_FULLTEXT: CredibilityLevel.HIGH,
            SourceType.PUBMED_ABSTRACT: CredibilityLevel.MEDIUM,
            SourceType.DERIVED: CredibilityLevel.LOW,
        }
        return credibility_map.get(source_type, CredibilityLevel.MEDIUM)

    def _calculate_confidence(self, sources: List[EvidenceSource]) -> float:
        """근거 기반 신뢰도 계산"""
        if not sources:
            return 0.0

        # 신뢰도별 가중치
        weights = {
            CredibilityLevel.GOLD: 1.0,
            CredibilityLevel.HIGH: 0.85,
            CredibilityLevel.MEDIUM: 0.6,
            CredibilityLevel.LOW: 0.3,
        }

        total_weight = sum(weights.get(s.credibility, 0.5) for s in sources)
        return min(total_weight / len(sources) * (1 + len(sources) * 0.1), 1.0)

    def _find_matching_sentence(self, text: str, keywords: List[str]) -> str:
        """키워드가 포함된 문장 찾기"""
        sentences = re.split(r'(?<=[.!?])\s+', text)

        for sent in sentences:
            if any(kw in sent.lower() for kw in keywords):
                return sent[:200]  # 최대 200자

        return ""


# =============================================================================
# Singleton
# =============================================================================

_mapper_instance: Optional[EvidenceMapper] = None


def get_evidence_mapper() -> EvidenceMapper:
    """EvidenceMapper 싱글톤"""
    global _mapper_instance
    if _mapper_instance is None:
        _mapper_instance = EvidenceMapper()
    return _mapper_instance


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    mapper = EvidenceMapper()

    # 테스트 답변
    test_answer = """
    BI-RADS Category 3의 악성 확률은 2% 이하입니다.
    6개월 후 추적 검사를 권고합니다.
    Category 4A의 경우 악성 확률은 3-10%로 조직검사가 필요합니다.
    """

    # 테스트 소스
    test_sources = [
        {
            "id": "birads_categories",
            "source": "knowledge_module",
            "title": "BI-RADS Categories",
            "text": "Category 3 (Probably Benign)의 악성 확률은 2% 이하입니다. Category 4A: 3-10% 악성 확률"
        },
        {
            "pmid": "12345678",
            "title": "Breast Cancer Screening",
            "text": "The malignancy rate for BI-RADS 3 lesions is less than 2 percent."
        }
    ]

    print("=" * 60)
    print("Evidence Mapper Test")
    print("=" * 60)

    report = mapper.analyze_answer(test_answer, test_sources)

    print(f"총 주장: {report.total_claims}")
    print(f"검증된 주장: {report.verified_claims}")
    print(f"전체 신뢰도: {report.overall_credibility:.2f}")
    print()
    print(mapper.format_citations(report.mapped_claims))
