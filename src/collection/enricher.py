"""
Sophia AI: Metadata Enricher
==============================
논문 메타데이터 보강 (인용수, Journal IF)
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.models import Paper

logger = logging.getLogger(__name__)

# Journal Impact Factor 데이터 (수동 관리)
JOURNAL_IF_FILE = Path(__file__).parent.parent / "data" / "journal_if.json"

DEFAULT_JOURNAL_IF = {
    # Radiology & Imaging (High Impact)
    "Radiology": 19.7,
    "European Radiology": 7.0,
    "Journal of Clinical Oncology": 45.3,
    "JAMA Oncology": 28.4,
    "The Lancet Oncology": 41.3,
    "Annals of Oncology": 32.0,

    # Breast Imaging Specialty
    "The Breast": 3.8,
    "Breast Cancer Research and Treatment": 4.4,
    "Breast Cancer Research": 7.4,
    "Breast Journal": 2.5,
    "Clinical Breast Cancer": 2.9,

    # Radiology Specialty
    "American Journal of Roentgenology": 4.7,
    "Korean Journal of Radiology": 4.9,
    "Academic Radiology": 4.3,
    "British Journal of Radiology": 2.6,
    "Radiographics": 4.1,
    "Clinical Radiology": 2.8,
    "Japanese Journal of Radiology": 2.5,

    # Medical Physics
    "Medical Physics": 4.5,
    "Physics in Medicine and Biology": 3.5,

    # Cancer Research
    "Cancer": 6.2,
    "Cancer Research": 11.2,
    "International Journal of Cancer": 5.1,
}


class MetadataEnricher:
    """메타데이터 보강 클래스"""

    def __init__(self, email: Optional[str] = None, s2_api_key: Optional[str] = None):
        """
        Args:
            email: CrossRef polite pool 이메일
            s2_api_key: Semantic Scholar API 키
        """
        self.email = email
        self.s2_api_key = s2_api_key

        self.client = httpx.Client(timeout=30.0)

        # Journal IF 로드
        self.journal_if = self._load_journal_if()

        # Rate limiting
        self.crossref_last_request = 0.0
        self.s2_last_request = 0.0

    def _load_journal_if(self) -> Dict[str, float]:
        """Journal IF 데이터 로드"""
        if JOURNAL_IF_FILE.exists():
            return json.loads(JOURNAL_IF_FILE.read_text())
        return DEFAULT_JOURNAL_IF

    def _rate_limit_crossref(self):
        """CrossRef rate limit (50 req/sec for polite pool)"""
        elapsed = time.time() - self.crossref_last_request
        if elapsed < 0.05:
            time.sleep(0.05 - elapsed)
        self.crossref_last_request = time.time()

    def _rate_limit_s2(self):
        """Semantic Scholar rate limit (100 req/5min = 0.33 req/sec)"""
        elapsed = time.time() - self.s2_last_request
        if elapsed < 3.0:  # 보수적으로 3초
            time.sleep(3.0 - elapsed)
        self.s2_last_request = time.time()

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5))
    def get_crossref_metadata(self, doi: str) -> Dict:
        """
        CrossRef에서 메타데이터 가져오기

        Args:
            doi: DOI

        Returns:
            메타데이터 딕셔너리
        """
        if not doi:
            return {}

        self._rate_limit_crossref()

        url = f"https://api.crossref.org/works/{doi}"
        headers = {}
        if self.email:
            headers["User-Agent"] = f"Sophia AI/1.0 (mailto:{self.email})"

        try:
            response = self.client.get(url, headers=headers)
            if response.status_code == 404:
                return {}
            response.raise_for_status()

            data = response.json()
            message = data.get("message", {})

            return {
                "citation_count": message.get("is-referenced-by-count", 0),
                "references_count": message.get("references-count", 0),
            }
        except Exception as e:
            logger.debug(f"CrossRef lookup failed for {doi}: {e}")
            return {}

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5))
    def get_s2_citation_count(self, doi: str) -> Optional[int]:
        """
        Semantic Scholar에서 인용수 가져오기

        Args:
            doi: DOI

        Returns:
            인용수 또는 None
        """
        if not doi:
            return None

        self._rate_limit_s2()

        url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}"
        params = {"fields": "citationCount"}
        headers = {}
        if self.s2_api_key:
            headers["x-api-key"] = self.s2_api_key

        try:
            response = self.client.get(url, params=params, headers=headers)
            if response.status_code == 404:
                return None
            response.raise_for_status()

            data = response.json()
            return data.get("citationCount")
        except Exception as e:
            logger.debug(f"S2 lookup failed for {doi}: {e}")
            return None

    def get_journal_if(self, journal_name: str) -> Optional[float]:
        """
        저널 Impact Factor 조회

        Args:
            journal_name: 저널 이름

        Returns:
            IF 값 또는 None
        """
        # 정확히 일치
        if journal_name in self.journal_if:
            return self.journal_if[journal_name]

        # 대소문자 무시하고 검색
        lower_name = journal_name.lower()
        for jname, jif in self.journal_if.items():
            if jname.lower() == lower_name:
                return jif

        # 부분 일치
        for jname, jif in self.journal_if.items():
            if jname.lower() in lower_name or lower_name in jname.lower():
                return jif

        return None

    def enrich_paper(self, paper: Paper, use_apis: bool = False) -> Paper:
        """
        단일 논문 메타데이터 보강

        Args:
            paper: 원본 Paper
            use_apis: 외부 API 호출 여부 (느림)

        Returns:
            보강된 Paper
        """
        # Journal IF
        if paper.journal_if is None:
            paper.journal_if = self.get_journal_if(paper.journal)

        # Citation count (API 호출)
        if use_apis and paper.citation_count == 0 and paper.doi:
            # CrossRef 먼저 시도
            crossref_data = self.get_crossref_metadata(paper.doi)
            if crossref_data.get("citation_count"):
                paper.citation_count = crossref_data["citation_count"]
            else:
                # Semantic Scholar fallback
                s2_count = self.get_s2_citation_count(paper.doi)
                if s2_count is not None:
                    paper.citation_count = s2_count

        return paper

    def enrich_papers(
        self,
        papers: List[Paper],
        use_apis: bool = False,
        progress: bool = True,
    ) -> List[Paper]:
        """
        여러 논문 메타데이터 보강

        Args:
            papers: 논문 리스트
            use_apis: 외부 API 호출 여부
            progress: 진행률 표시

        Returns:
            보강된 논문 리스트
        """
        from tqdm import tqdm

        enriched = []
        iterator = tqdm(papers, desc="Enriching") if progress else papers

        for paper in iterator:
            try:
                enriched_paper = self.enrich_paper(paper, use_apis=use_apis)
                enriched.append(enriched_paper)
            except Exception as e:
                logger.warning(f"Failed to enrich {paper.pmid}: {e}")
                enriched.append(paper)

        # 통계
        with_if = sum(1 for p in enriched if p.journal_if is not None)
        with_citations = sum(1 for p in enriched if p.citation_count > 0)

        logger.info(
            f"Enriched {len(enriched)} papers: "
            f"{with_if} with IF, {with_citations} with citations"
        )

        return enriched

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
