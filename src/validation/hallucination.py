"""
Sophia AI: Hallucination Checker
=================================
Phase 2: 근거 교차 검증 (Citation Grounding)

LLM이 생성한 답변의 인용(PMID, 저자, 수치)이
실제 검색된 논문 데이터와 일치하는지 검증합니다.
"""

import re
import logging
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CitationMatch:
    """인용 매칭 결과"""
    cited_text: str  # 답변에서 인용된 텍스트
    citation_type: str  # "pmid", "author", "statistic", "claim"
    claimed_source: Optional[str]  # 주장된 출처
    matched_pmid: Optional[str]  # 매칭된 실제 PMID
    is_grounded: bool  # 근거 확인됨
    confidence: float  # 신뢰도 (0-1)
    evidence: str  # 근거 (매칭된 텍스트)
    issues: List[str] = field(default_factory=list)


@dataclass
class HallucinationCheckResult:
    """할루시네이션 검증 결과"""
    is_clean: bool  # 할루시네이션 없음
    total_citations: int  # 총 인용 수
    grounded_citations: int  # 근거 확인된 인용
    ungrounded_citations: int  # 근거 없는 인용
    fabricated_sources: List[str]  # 위조된 출처
    matches: List[CitationMatch]  # 개별 매칭 결과
    severity: str  # "none", "low", "medium", "high", "critical"
    summary: str  # 요약


# =============================================================================
# Citation Extractor
# =============================================================================

class CitationExtractor:
    """
    텍스트에서 인용 추출
    """

    # PMID 패턴
    PMID_PATTERNS = [
        r"PMID[:\s]*(\d{7,8})",
        r"\[PMID[:\s]*(\d{7,8})\]",
        r"PubMed[:\s]*(\d{7,8})",
        r"Source[:\s]*(?:PMID[:\s]*)?(\d{7,8})",
    ]

    # 저자 인용 패턴
    AUTHOR_PATTERNS = [
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+et\s+al\.?,?\s*(?:\(?(\d{4})\)?)?",
        r"([A-Z][a-z]+)\s+and\s+([A-Z][a-z]+)\s*(?:\(?(\d{4})\)?)?",
        r"\(([A-Z][a-z]+)\s+et\s+al\.?,?\s*(\d{4})\)",
    ]

    # 수치/통계 인용 패턴
    STATISTIC_PATTERNS = [
        r"(\d+(?:\.\d+)?)\s*%",  # 퍼센트
        r"sensitivity\s*(?:of\s*)?([\d.]+%?)",
        r"specificity\s*(?:of\s*)?([\d.]+%?)",
        r"PPV\s*(?:of\s*)?([\d.]+%?)",
        r"AUC\s*(?:of\s*)?([\d.]+)",
        r"p\s*[<>=]\s*([\d.]+)",
        r"OR\s*(?:of\s*)?([\d.]+)",
        r"HR\s*(?:of\s*)?([\d.]+)",
        r"(\d+(?:\.\d+)?)\s*(?:mGy|mSv|mm|cm|keV|kVp|mAs)",
    ]

    @classmethod
    def extract_pmids(cls, text: str) -> List[Tuple[str, str]]:
        """PMID 추출"""
        pmids = []
        for pattern in cls.PMID_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                pmid = match if isinstance(match, str) else match[0]
                pmids.append((pmid, "pmid"))
        return list(set(pmids))

    @classmethod
    def extract_author_citations(cls, text: str) -> List[Tuple[str, str]]:
        """저자 인용 추출"""
        citations = []
        for pattern in cls.AUTHOR_PATTERNS:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    author = match[0]
                    year = match[-1] if match[-1].isdigit() else None
                    citation = f"{author} et al." + (f" ({year})" if year else "")
                else:
                    citation = match
                citations.append((citation, "author"))
        return citations

    @classmethod
    def extract_statistics(cls, text: str) -> List[Tuple[str, str, str]]:
        """수치/통계 추출 (값, 컨텍스트)"""
        stats = []
        for pattern in cls.STATISTIC_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                # 컨텍스트 추출 (전후 50자)
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end]
                value = match.group(1) if match.groups() else match.group(0)
                stats.append((value, "statistic", context))
        return stats

    @classmethod
    def extract_all(cls, text: str) -> Dict[str, List]:
        """모든 인용 추출"""
        return {
            "pmids": cls.extract_pmids(text),
            "authors": cls.extract_author_citations(text),
            "statistics": cls.extract_statistics(text),
        }


# =============================================================================
# Hallucination Checker
# =============================================================================

class HallucinationChecker:
    """
    할루시네이션 검증기

    LLM 생성 답변의 인용이 실제 검색된 논문과 일치하는지 검증
    """

    # 알려진 가짜 인용 패턴 (흔히 환각되는 저자/논문)
    KNOWN_FABRICATIONS = [
        "Boone et al",  # 실존하지만 맘모그래피에서 자주 환각됨
        "Cunha et al",
        "Smith et al",
        "Kim et al",  # 너무 일반적인 이름
        "Lee et al",
        "manufacturer",
        "technical report",
        "unpublished",
    ]

    def __init__(
        self,
        retrieved_papers: Optional[List[Dict]] = None,
        retrieved_pmids: Optional[Set[str]] = None,
        strict_mode: bool = True,
    ):
        """
        Args:
            retrieved_papers: 검색된 논문 리스트 (Paper 객체 또는 딕셔너리)
            retrieved_pmids: 검색된 PMID 집합
            strict_mode: 엄격 모드 (검색되지 않은 인용 차단)
        """
        self.strict_mode = strict_mode
        self.extractor = CitationExtractor()

        # 검색된 논문 정보 구축
        self.retrieved_pmids: Set[str] = retrieved_pmids or set()
        self.retrieved_authors: Dict[str, Set[str]] = {}  # author -> {pmid}
        self.retrieved_content: Dict[str, str] = {}  # pmid -> full_text
        self.retrieved_stats: Dict[str, List[str]] = {}  # pmid -> [stats]

        if retrieved_papers:
            self._index_papers(retrieved_papers)

    def _index_papers(self, papers: List[Dict]):
        """검색된 논문 인덱싱"""
        for paper in papers:
            # Paper 객체 또는 딕셔너리 모두 처리
            if hasattr(paper, 'pmid'):
                pmid = paper.pmid
                authors = paper.authors if hasattr(paper, 'authors') else []
                abstract = paper.abstract if hasattr(paper, 'abstract') else ""
                title = paper.title if hasattr(paper, 'title') else ""
            else:
                pmid = paper.get('pmid', '')
                authors = paper.get('authors', [])
                abstract = paper.get('abstract', '')
                title = paper.get('title', '')

            self.retrieved_pmids.add(pmid)

            # 저자 인덱싱
            for author in authors:
                # 성(last name)만 추출
                last_name = author.split()[-1] if author else ""
                if last_name:
                    if last_name not in self.retrieved_authors:
                        self.retrieved_authors[last_name] = set()
                    self.retrieved_authors[last_name].add(pmid)

            # 전문 인덱싱
            full_text = f"{title} {abstract}"
            self.retrieved_content[pmid] = full_text

            # 통계 추출 및 인덱싱
            stats = self.extractor.extract_statistics(full_text)
            self.retrieved_stats[pmid] = [s[0] for s in stats]

    def check(self, generated_text: str) -> HallucinationCheckResult:
        """
        생성된 텍스트의 할루시네이션 검증

        Args:
            generated_text: LLM이 생성한 답변

        Returns:
            HallucinationCheckResult
        """
        # 1. 인용 추출
        citations = self.extractor.extract_all(generated_text)
        matches = []
        fabricated = []

        # 2. PMID 검증
        for pmid, _ in citations["pmids"]:
            match = self._verify_pmid(pmid)
            matches.append(match)
            if not match.is_grounded:
                fabricated.append(f"PMID:{pmid}")

        # 3. 저자 인용 검증
        for author_cite, _ in citations["authors"]:
            match = self._verify_author(author_cite)
            matches.append(match)
            if not match.is_grounded and self.strict_mode:
                fabricated.append(author_cite)

        # 4. 통계 검증
        for value, _, context in citations["statistics"]:
            match = self._verify_statistic(value, context)
            matches.append(match)
            if not match.is_grounded and self.strict_mode:
                # 통계는 경고만 (완전 차단하지 않음)
                pass

        # 5. 결과 집계
        grounded = sum(1 for m in matches if m.is_grounded)
        ungrounded = len(matches) - grounded

        # 심각도 결정
        if len(matches) == 0:
            severity = "none"
        elif ungrounded == 0:
            severity = "none"
        elif ungrounded <= 1:
            severity = "low"
        elif ungrounded <= 3:
            severity = "medium"
        elif fabricated:
            severity = "high"
        else:
            severity = "critical" if len(fabricated) > 2 else "high"

        is_clean = severity in ("none", "low")

        # 요약 생성
        summary = self._generate_summary(matches, fabricated, severity)

        return HallucinationCheckResult(
            is_clean=is_clean,
            total_citations=len(matches),
            grounded_citations=grounded,
            ungrounded_citations=ungrounded,
            fabricated_sources=fabricated,
            matches=matches,
            severity=severity,
            summary=summary,
        )

    def _verify_pmid(self, pmid: str) -> CitationMatch:
        """PMID 검증"""
        is_grounded = pmid in self.retrieved_pmids
        issues = []

        if not is_grounded:
            issues.append(f"PMID {pmid} not in retrieved papers")

        return CitationMatch(
            cited_text=f"PMID:{pmid}",
            citation_type="pmid",
            claimed_source=pmid,
            matched_pmid=pmid if is_grounded else None,
            is_grounded=is_grounded,
            confidence=1.0 if is_grounded else 0.0,
            evidence="Exact PMID match" if is_grounded else "",
            issues=issues,
        )

    def _verify_author(self, author_citation: str) -> CitationMatch:
        """저자 인용 검증"""
        # 저자 성 추출
        author_match = re.search(r"([A-Z][a-z]+)", author_citation)
        if not author_match:
            return CitationMatch(
                cited_text=author_citation,
                citation_type="author",
                claimed_source=author_citation,
                matched_pmid=None,
                is_grounded=False,
                confidence=0.0,
                evidence="",
                issues=["Could not parse author name"],
            )

        last_name = author_match.group(1)
        issues = []

        # 알려진 가짜 패턴 체크
        for fabrication in self.KNOWN_FABRICATIONS:
            if fabrication.lower() in author_citation.lower():
                issues.append(f"Potential fabrication pattern: {fabrication}")

        # 검색된 저자와 매칭
        matched_pmids = self.retrieved_authors.get(last_name, set())

        if matched_pmids:
            return CitationMatch(
                cited_text=author_citation,
                citation_type="author",
                claimed_source=author_citation,
                matched_pmid=list(matched_pmids)[0],
                is_grounded=True,
                confidence=0.8,  # 저자명만으로는 완벽하지 않음
                evidence=f"Author {last_name} found in retrieved papers",
                issues=issues,
            )

        issues.append(f"Author '{last_name}' not found in retrieved papers")

        return CitationMatch(
            cited_text=author_citation,
            citation_type="author",
            claimed_source=author_citation,
            matched_pmid=None,
            is_grounded=False,
            confidence=0.0,
            evidence="",
            issues=issues,
        )

    def _verify_statistic(self, value: str, context: str) -> CitationMatch:
        """통계/수치 검증"""
        # 검색된 논문에서 해당 수치 찾기
        for pmid, stats in self.retrieved_stats.items():
            if value in stats:
                return CitationMatch(
                    cited_text=f"{value} in: {context[:50]}",
                    citation_type="statistic",
                    claimed_source=None,
                    matched_pmid=pmid,
                    is_grounded=True,
                    confidence=0.9,
                    evidence=f"Value {value} found in PMID:{pmid}",
                    issues=[],
                )

        # 컨텍스트 매칭 시도
        for pmid, content in self.retrieved_content.items():
            if value in content:
                return CitationMatch(
                    cited_text=f"{value} in: {context[:50]}",
                    citation_type="statistic",
                    claimed_source=None,
                    matched_pmid=pmid,
                    is_grounded=True,
                    confidence=0.7,
                    evidence=f"Value {value} found in content of PMID:{pmid}",
                    issues=[],
                )

        # 일반적인 수치는 검증 불가로 처리 (할루시네이션으로 단정 X)
        return CitationMatch(
            cited_text=f"{value} in: {context[:50]}",
            citation_type="statistic",
            claimed_source=None,
            matched_pmid=None,
            is_grounded=False,
            confidence=0.3,  # 낮은 신뢰도 (검증 불가)
            evidence="",
            issues=["Statistic not found in retrieved papers (may be from other source)"],
        )

    def _generate_summary(
        self, matches: List[CitationMatch], fabricated: List[str], severity: str
    ) -> str:
        """검증 결과 요약 생성"""
        if severity == "none":
            return "All citations are properly grounded in retrieved sources."

        if severity == "low":
            return "Minor citation issues detected. Most citations are grounded."

        if fabricated:
            fab_list = ", ".join(fabricated[:3])
            if len(fabricated) > 3:
                fab_list += f" and {len(fabricated) - 3} more"
            return f"Potential fabricated sources detected: {fab_list}"

        ungrounded = [m for m in matches if not m.is_grounded]
        return f"{len(ungrounded)} citations could not be verified against retrieved sources."


# =============================================================================
# Convenience Functions
# =============================================================================

def check_hallucination(
    generated_text: str,
    retrieved_papers: List[Dict] = None,
    retrieved_pmids: Set[str] = None,
    strict: bool = True,
) -> HallucinationCheckResult:
    """
    할루시네이션 검증 (편의 함수)

    Args:
        generated_text: 검사할 텍스트
        retrieved_papers: 검색된 논문 리스트
        retrieved_pmids: 검색된 PMID 집합
        strict: 엄격 모드

    Returns:
        검증 결과
    """
    checker = HallucinationChecker(
        retrieved_papers=retrieved_papers,
        retrieved_pmids=retrieved_pmids,
        strict_mode=strict,
    )
    return checker.check(generated_text)


def extract_citations(text: str) -> Dict[str, List]:
    """
    텍스트에서 인용 추출 (편의 함수)
    """
    return CitationExtractor.extract_all(text)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 가상의 검색된 논문
    fake_retrieved_papers = [
        {
            "pmid": "12345678",
            "title": "Digital Breast Tomosynthesis Performance Study",
            "authors": ["Dance DR", "Young KC"],
            "abstract": "This study found sensitivity of 85% and specificity of 90%.",
        },
        {
            "pmid": "87654321",
            "title": "MGD Calculation in Mammography",
            "authors": ["Kim JH", "Lee SY"],
            "abstract": "Mean glandular dose was 1.5 mGy with AUC of 0.92.",
        },
    ]

    # 테스트 케이스
    test_responses = [
        # 올바른 인용
        """
        Dance et al. (2011) demonstrated that DBT sensitivity is 85%.
        [Source: PMID 12345678]
        """,

        # 위조된 인용
        """
        According to Boone et al. (2015), the optimal kVp is 28.
        Smith et al. reported similar findings in their technical report.
        [Source: PMID 99999999]
        """,

        # 검증 불가 (출처 없음)
        """
        Studies have shown that DBT has 95% sensitivity.
        The recommended dose is 2.0 mGy.
        """,
    ]

    checker = HallucinationChecker(retrieved_papers=fake_retrieved_papers)

    for i, response in enumerate(test_responses, 1):
        print(f"\n{'=' * 60}")
        print(f"Test Case {i}")
        print("=" * 60)

        result = checker.check(response)

        print(f"Is Clean: {result.is_clean}")
        print(f"Severity: {result.severity}")
        print(f"Total: {result.total_citations}, Grounded: {result.grounded_citations}")
        print(f"Summary: {result.summary}")

        if result.fabricated_sources:
            print(f"Fabricated: {result.fabricated_sources}")

        print("\nMatches:")
        for match in result.matches:
            status = "✓" if match.is_grounded else "✗"
            print(f"  {status} [{match.citation_type}] {match.cited_text[:50]}")
            if match.issues:
                print(f"    Issues: {match.issues}")
