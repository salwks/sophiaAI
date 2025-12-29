"""
Sophia AI: Semantic Scholar Client
=====================================
Semantic Scholar API를 통한 논문 수집 및 인용수 보강
"""

import logging
import time
from typing import Dict, List, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.models import Paper

logger = logging.getLogger(__name__)

# Semantic Scholar API
S2_API_BASE = "https://api.semanticscholar.org/graph/v1"
S2_SEARCH_URL = f"{S2_API_BASE}/paper/search"
S2_PAPER_URL = f"{S2_API_BASE}/paper"
S2_BATCH_URL = f"{S2_API_BASE}/paper/batch"


class SemanticScholarClient:
    """Semantic Scholar API 클라이언트"""

    # 검색 필드
    PAPER_FIELDS = [
        "paperId",
        "externalIds",
        "title",
        "abstract",
        "venue",
        "year",
        "authors",
        "citationCount",
        "influentialCitationCount",
        "fieldsOfStudy",
        "publicationTypes",
        "publicationDate",
    ]

    def __init__(self, api_key: Optional[str] = None):
        """
        Args:
            api_key: Semantic Scholar API 키 (선택, 있으면 rate limit 완화)
        """
        self.api_key = api_key
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["x-api-key"] = api_key

        self.client = httpx.Client(timeout=60.0, headers=headers)

        # Rate limiting: 100 req/sec with key, 10 req/sec without
        self.requests_per_second = 100 if api_key else 10
        self.min_interval = 1.0 / self.requests_per_second
        self.last_request_time = 0.0

        logger.info(
            f"SemanticScholarClient initialized (rate: {self.requests_per_second} req/sec)"
        )

    def _rate_limit(self):
        """Rate limiting"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request_time = time.time()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def search(
        self,
        query: str,
        year_range: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict:
        """
        논문 검색

        Args:
            query: 검색어
            year_range: 연도 범위 (예: "2020-2025")
            limit: 페이지당 결과 수 (최대 100)
            offset: 시작 위치

        Returns:
            검색 결과 딕셔너리
        """
        self._rate_limit()

        params = {
            "query": query,
            "fields": ",".join(self.PAPER_FIELDS),
            "limit": min(limit, 100),
            "offset": offset,
        }

        if year_range:
            params["year"] = year_range

        logger.info(f"Searching S2: {query} (offset={offset})")

        response = self.client.get(S2_SEARCH_URL, params=params)
        response.raise_for_status()

        return response.json()

    def search_all(
        self,
        query: str,
        year_range: Optional[str] = None,
        max_results: int = 1000,
    ) -> List[dict]:
        """
        모든 결과 검색 (페이지네이션)

        Args:
            query: 검색어
            year_range: 연도 범위
            max_results: 최대 결과 수

        Returns:
            논문 리스트
        """
        all_papers = []
        offset = 0
        limit = 100

        while offset < max_results:
            try:
                result = self.search(query, year_range, limit, offset)
                papers = result.get("data", [])

                if not papers:
                    break

                all_papers.extend(papers)
                offset += len(papers)

                total = result.get("total", 0)
                logger.info(f"Retrieved {len(all_papers)}/{min(total, max_results)} papers")

                if len(papers) < limit or len(all_papers) >= max_results:
                    break

            except Exception as e:
                logger.error(f"Search failed at offset {offset}: {e}")
                break

        return all_papers[:max_results]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def get_paper_by_pmid(self, pmid: str) -> Optional[dict]:
        """
        PMID로 논문 조회

        Args:
            pmid: PubMed ID

        Returns:
            논문 정보 또는 None
        """
        self._rate_limit()

        url = f"{S2_PAPER_URL}/PMID:{pmid}"
        params = {"fields": ",".join(self.PAPER_FIELDS)}

        try:
            response = self.client.get(url, params=params)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError:
            return None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def get_citations_batch(self, pmids: List[str]) -> Dict[str, int]:
        """
        배치로 인용수 조회

        Args:
            pmids: PMID 리스트

        Returns:
            {pmid: citation_count} 딕셔너리
        """
        self._rate_limit()

        # PMID를 S2 ID 형식으로 변환
        ids = [f"PMID:{pmid}" for pmid in pmids]

        data = {"ids": ids}
        params = {"fields": "externalIds,citationCount,influentialCitationCount"}

        try:
            response = self.client.post(S2_BATCH_URL, json=data, params=params)
            response.raise_for_status()

            results = {}
            for paper in response.json():
                if paper:
                    pmid = paper.get("externalIds", {}).get("PubMed")
                    if pmid:
                        results[pmid] = paper.get("citationCount", 0)

            return results
        except Exception as e:
            logger.warning(f"Batch citation fetch failed: {e}")
            return {}

    def collect_mammography_papers(
        self,
        year_range: str = "2010-2025",
        max_per_query: int = 500,
    ) -> List[Paper]:
        """
        맘모그래피 관련 논문 수집

        Args:
            year_range: 연도 범위
            max_per_query: 쿼리당 최대 결과 수

        Returns:
            Paper 리스트
        """
        queries = [
            "mammography breast cancer screening",
            "digital breast tomosynthesis DBT",
            "breast imaging diagnosis",
            "mammogram detection",
            "breast density mammography",
            "contrast enhanced mammography",
        ]

        all_papers = []
        seen_ids = set()

        for query in queries:
            logger.info(f"Collecting: {query}")
            papers = self.search_all(query, year_range, max_per_query)

            for p in papers:
                paper_id = p.get("paperId")
                if paper_id and paper_id not in seen_ids:
                    seen_ids.add(paper_id)
                    all_papers.append(p)

        logger.info(f"Collected {len(all_papers)} unique papers from Semantic Scholar")

        # Paper 객체로 변환
        return [self._to_paper(p) for p in all_papers if self._is_relevant(p)]

    def _is_relevant(self, paper: dict) -> bool:
        """맘모그래피 관련 논문인지 확인"""
        title = (paper.get("title") or "").lower()
        abstract = (paper.get("abstract") or "").lower()
        text = f"{title} {abstract}"

        keywords = [
            "mammograph", "breast imag", "tomosynthesis", "dbt",
            "ffdm", "breast cancer screen", "mammogram",
        ]

        return any(kw in text for kw in keywords)

    def _to_paper(self, data: dict) -> Paper:
        """S2 데이터를 Paper로 변환"""
        external_ids = data.get("externalIds", {})

        authors = []
        for author in data.get("authors", []):
            name = author.get("name", "")
            if name:
                authors.append(name)

        return Paper(
            pmid=external_ids.get("PubMed") or f"S2_{data.get('paperId', '')[:8]}",
            doi=external_ids.get("DOI"),
            title=data.get("title", ""),
            authors=authors,
            journal=data.get("venue", ""),
            year=data.get("year") or 0,
            abstract=data.get("abstract", ""),
            citation_count=data.get("citationCount", 0),
            source="semantic_scholar",
        )

    def enrich_citations(self, papers: List[Paper], batch_size: int = 500) -> List[Paper]:
        """
        기존 논문에 인용수 추가

        Args:
            papers: Paper 리스트
            batch_size: 배치 크기

        Returns:
            인용수가 추가된 Paper 리스트
        """
        # PMID가 있는 논문만 필터
        papers_with_pmid = [(i, p) for i, p in enumerate(papers) if p.pmid and not p.pmid.startswith(("S2_", "KM_"))]

        logger.info(f"Enriching citations for {len(papers_with_pmid)} papers")

        enriched_count = 0
        for i in range(0, len(papers_with_pmid), batch_size):
            batch = papers_with_pmid[i:i + batch_size]
            pmids = [p.pmid for _, p in batch]

            citations = self.get_citations_batch(pmids)

            for idx, paper in batch:
                if paper.pmid in citations:
                    papers[idx].citation_count = citations[paper.pmid]
                    enriched_count += 1

            logger.info(f"Enriched {i + len(batch)}/{len(papers_with_pmid)} papers")

        logger.info(f"Added citations to {enriched_count} papers")
        return papers

    def close(self):
        """클라이언트 종료"""
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    api_key = os.getenv("S2_API_KEY")

    with SemanticScholarClient(api_key=api_key) as client:
        # 검색 테스트
        papers = client.collect_mammography_papers(
            year_range="2023-2025",
            max_per_query=50,
        )

        print(f"\nCollected {len(papers)} papers")
        for paper in papers[:5]:
            print(f"\n- {paper.title[:60]}...")
            print(f"  Citations: {paper.citation_count}")
            print(f"  Year: {paper.year}")
