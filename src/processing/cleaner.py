"""
MARIA-Mammo: Data Cleaner
=========================
수집된 논문 데이터 정제
"""

import logging
import re
from typing import Dict, List, Set, Tuple

from src.models import Paper

logger = logging.getLogger(__name__)


class DataCleaner:
    """데이터 정제 클래스"""

    # 제외할 출판 유형
    EXCLUDE_TYPES = {
        "Letter",
        "Comment",
        "Editorial",
        "Erratum",
        "Published Erratum",
        "Retracted Publication",
        "Retraction of Publication",
        "News",
        "Newspaper Article",
        "Biography",
        "Historical Article",
        "Portrait",
        "Video-Audio Media",
    }

    # Retracted 패턴
    RETRACTED_PATTERNS = [
        r"\[retracted\]",
        r"\[withdrawn\]",
        r"retraction notice",
        r"has been retracted",
    ]

    def __init__(
        self,
        min_abstract_length: int = 50,
        exclude_no_abstract: bool = False,
        exclude_certain_types: bool = True,
    ):
        """
        Args:
            min_abstract_length: 최소 초록 길이
            exclude_no_abstract: 초록 없는 논문 제외 여부
            exclude_certain_types: 특정 출판 유형 제외 여부
        """
        self.min_abstract_length = min_abstract_length
        self.exclude_no_abstract = exclude_no_abstract
        self.exclude_certain_types = exclude_certain_types

        self.stats = {
            "input": 0,
            "output": 0,
            "duplicate_pmid": 0,
            "duplicate_doi": 0,
            "no_abstract": 0,
            "short_abstract": 0,
            "retracted": 0,
            "excluded_type": 0,
        }

    def clean_abstract(self, text: str) -> str:
        """초록 텍스트 정제"""
        if not text:
            return ""

        # HTML 태그 제거
        text = re.sub(r"<[^>]+>", "", text)

        # HTML 엔티티 변환
        text = text.replace("&amp;", "&")
        text = text.replace("&lt;", "<")
        text = text.replace("&gt;", ">")
        text = text.replace("&quot;", '"')
        text = text.replace("&#39;", "'")

        # 연속 공백 제거
        text = re.sub(r"\s+", " ", text)

        # 앞뒤 공백 제거
        text = text.strip()

        return text

    def clean_title(self, text: str) -> str:
        """제목 정제"""
        if not text:
            return ""

        # HTML 태그 제거
        text = re.sub(r"<[^>]+>", "", text)

        # 앞뒤 공백 제거
        text = text.strip()

        # 마지막 마침표 제거 (있는 경우)
        if text.endswith("."):
            text = text[:-1]

        return text

    def normalize_authors(self, authors: List[str]) -> List[str]:
        """저자 목록 정규화"""
        normalized = []

        for author in authors:
            if not author:
                continue

            # 공백 정리
            author = re.sub(r"\s+", " ", author).strip()

            # 특수문자 제거 (하이픈, 아포스트로피 제외)
            author = re.sub(r"[^\w\s\-']", "", author)

            if author:
                normalized.append(author)

        return normalized

    def is_retracted(self, paper: Paper) -> bool:
        """Retracted 논문인지 확인"""
        # 출판 유형 확인
        for pub_type in paper.publication_types:
            if "retract" in pub_type.lower():
                return True

        # 제목에서 패턴 확인
        title_lower = paper.title.lower()
        for pattern in self.RETRACTED_PATTERNS:
            if re.search(pattern, title_lower, re.IGNORECASE):
                return True

        return False

    def should_exclude_by_type(self, paper: Paper) -> bool:
        """출판 유형으로 제외 여부 결정"""
        for pub_type in paper.publication_types:
            if pub_type in self.EXCLUDE_TYPES:
                return True
        return False

    def clean_paper(self, paper: Paper) -> Paper:
        """단일 논문 정제"""
        paper.title = self.clean_title(paper.title)
        paper.abstract = self.clean_abstract(paper.abstract)
        paper.authors = self.normalize_authors(paper.authors)
        return paper

    def deduplicate(self, papers: List[Paper]) -> Tuple[List[Paper], List[str]]:
        """
        중복 제거

        Returns:
            (중복 제거된 논문 리스트, 중복 PMID 리스트)
        """
        seen_pmids: Set[str] = set()
        seen_dois: Set[str] = set()
        unique_papers = []
        duplicate_pmids = []

        for paper in papers:
            # PMID 중복 체크
            if paper.pmid in seen_pmids:
                self.stats["duplicate_pmid"] += 1
                duplicate_pmids.append(paper.pmid)
                continue

            # DOI 중복 체크
            if paper.doi:
                if paper.doi in seen_dois:
                    self.stats["duplicate_doi"] += 1
                    duplicate_pmids.append(paper.pmid)
                    continue
                seen_dois.add(paper.doi)

            seen_pmids.add(paper.pmid)
            unique_papers.append(paper)

        return unique_papers, duplicate_pmids

    def filter_quality(self, papers: List[Paper]) -> List[Paper]:
        """품질 기준 필터링"""
        filtered = []

        for paper in papers:
            # Retracted 체크
            if self.is_retracted(paper):
                self.stats["retracted"] += 1
                continue

            # 출판 유형 체크
            if self.exclude_certain_types and self.should_exclude_by_type(paper):
                self.stats["excluded_type"] += 1
                continue

            # 초록 체크
            if not paper.abstract:
                self.stats["no_abstract"] += 1
                if self.exclude_no_abstract:
                    continue
            elif len(paper.abstract) < self.min_abstract_length:
                self.stats["short_abstract"] += 1
                if self.exclude_no_abstract:
                    continue

            filtered.append(paper)

        return filtered

    def clean_all(self, papers: List[Paper]) -> List[Paper]:
        """
        전체 정제 파이프라인

        Args:
            papers: 원본 논문 리스트

        Returns:
            정제된 논문 리스트
        """
        self.stats["input"] = len(papers)

        # 1. 텍스트 정제
        cleaned = [self.clean_paper(paper) for paper in papers]

        # 2. 중복 제거
        unique, _ = self.deduplicate(cleaned)

        # 3. 품질 필터
        filtered = self.filter_quality(unique)

        self.stats["output"] = len(filtered)

        logger.info(self._format_stats())

        return filtered

    def _format_stats(self) -> str:
        """통계 포맷팅"""
        return (
            f"Cleaning stats: {self.stats['input']} -> {self.stats['output']} papers | "
            f"Duplicates: PMID={self.stats['duplicate_pmid']}, DOI={self.stats['duplicate_doi']} | "
            f"Filtered: retracted={self.stats['retracted']}, "
            f"excluded_type={self.stats['excluded_type']}, "
            f"no_abstract={self.stats['no_abstract']}, "
            f"short_abstract={self.stats['short_abstract']}"
        )

    def get_stats(self) -> Dict:
        """통계 반환"""
        return self.stats.copy()
