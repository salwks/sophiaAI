"""
Sophia AI: RSNA Abstracts Scraper
====================================
RSNA 연례 학회 초록 스크래핑
"""

import logging
import re
import time
from typing import Dict, List, Optional

import httpx
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

from src.models import Paper

logger = logging.getLogger(__name__)

# RSNA Abstract Archive
RSNA_BASE_URL = "https://archive.rsna.org"
RSNA_SEARCH_URL = f"{RSNA_BASE_URL}/search"


class RSNAScraper:
    """RSNA 초록 스크래퍼"""

    # 맘모그래피 관련 검색어
    SEARCH_TERMS = [
        "mammography",
        "breast tomosynthesis",
        "breast imaging",
        "breast cancer screening",
        "breast density",
        "contrast enhanced mammography",
        "DBT",
        "FFDM",
    ]

    # 학회 연도 (최근 5년)
    MEETING_YEARS = [2024, 2023, 2022, 2021, 2020]

    def __init__(self):
        """초기화"""
        self.client = httpx.Client(
            timeout=60.0,
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
            follow_redirects=True,
        )
        self.last_request_time = 0.0
        self.min_interval = 2.0  # 2초 간격 (예의바른 스크래핑)

        logger.info("RSNAScraper initialized")

    def _rate_limit(self):
        """Rate limiting"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request_time = time.time()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=20))
    def search_abstracts(
        self,
        query: str,
        year: int,
        max_results: int = 100,
    ) -> List[dict]:
        """
        RSNA 초록 검색

        Args:
            query: 검색어
            year: 학회 연도
            max_results: 최대 결과 수

        Returns:
            초록 리스트
        """
        self._rate_limit()

        params = {
            "q": query,
            "year": year,
            "type": "abstract",
            "page": 1,
        }

        logger.info(f"Searching RSNA {year}: {query}")

        try:
            response = self.client.get(RSNA_SEARCH_URL, params=params)
            response.raise_for_status()
            return self._parse_search_results(response.text, year)
        except Exception as e:
            logger.warning(f"RSNA search failed: {e}")
            return []

    def _parse_search_results(self, html: str, year: int) -> List[dict]:
        """검색 결과 HTML 파싱"""
        results = []
        soup = BeautifulSoup(html, "lxml")

        # 검색 결과 항목 찾기
        items = soup.find_all("div", class_="search-result-item")

        for item in items:
            try:
                result = self._parse_result_item(item, year)
                if result:
                    results.append(result)
            except Exception as e:
                logger.warning(f"Failed to parse item: {e}")

        return results

    def _parse_result_item(self, item, year: int) -> Optional[dict]:
        """개별 검색 결과 파싱"""
        # 제목
        title_elem = item.find("h3", class_="title") or item.find("a", class_="title")
        if not title_elem:
            return None

        title = title_elem.get_text(strip=True)

        # 초록 ID 및 URL
        link = title_elem.find("a") if title_elem.name != "a" else title_elem
        abstract_url = link.get("href", "") if link else ""
        abstract_id = self._extract_abstract_id(abstract_url)

        # 저자
        authors_elem = item.find("div", class_="authors") or item.find("p", class_="authors")
        authors = []
        if authors_elem:
            authors_text = authors_elem.get_text(strip=True)
            authors = [a.strip() for a in authors_text.split(",") if a.strip()]

        # 초록 내용 (있으면)
        abstract_elem = item.find("div", class_="abstract") or item.find("p", class_="abstract")
        abstract = abstract_elem.get_text(strip=True) if abstract_elem else ""

        # 세션/카테고리
        session_elem = item.find("div", class_="session") or item.find("span", class_="category")
        session = session_elem.get_text(strip=True) if session_elem else ""

        return {
            "rsna_id": abstract_id,
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "year": year,
            "session": session,
            "url": f"{RSNA_BASE_URL}{abstract_url}" if abstract_url else "",
        }

    def _extract_abstract_id(self, url: str) -> str:
        """URL에서 초록 ID 추출"""
        match = re.search(r"/(\d+)/?$", url)
        return match.group(1) if match else ""

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=20))
    def get_abstract_detail(self, url: str) -> Optional[dict]:
        """
        초록 상세 페이지에서 전체 내용 가져오기

        Args:
            url: 초록 페이지 URL

        Returns:
            상세 정보 또는 None
        """
        if not url:
            return None

        self._rate_limit()

        try:
            response = self.client.get(url)
            response.raise_for_status()
            return self._parse_abstract_page(response.text)
        except Exception as e:
            logger.warning(f"Failed to get abstract detail: {e}")
            return None

    def _parse_abstract_page(self, html: str) -> Optional[dict]:
        """초록 상세 페이지 파싱"""
        soup = BeautifulSoup(html, "lxml")

        # 제목
        title_elem = soup.find("h1", class_="abstract-title") or soup.find("h1")
        title = title_elem.get_text(strip=True) if title_elem else ""

        # 초록 본문
        abstract_elem = soup.find("div", class_="abstract-body") or soup.find("div", class_="abstract-content")
        abstract = abstract_elem.get_text(strip=True) if abstract_elem else ""

        # 저자
        authors_elem = soup.find("div", class_="authors")
        authors = []
        if authors_elem:
            for author in authors_elem.find_all("span", class_="author"):
                name = author.get_text(strip=True)
                if name:
                    authors.append(name)

        # 소속
        affiliation_elem = soup.find("div", class_="affiliations")
        affiliation = affiliation_elem.get_text(strip=True) if affiliation_elem else ""

        return {
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "affiliation": affiliation,
        }

    def collect_mammography_abstracts(
        self,
        years: Optional[List[int]] = None,
        max_per_search: int = 50,
    ) -> List[Paper]:
        """
        맘모그래피 관련 RSNA 초록 수집

        Args:
            years: 수집할 연도 리스트
            max_per_search: 검색당 최대 결과 수

        Returns:
            Paper 리스트
        """
        if years is None:
            years = self.MEETING_YEARS

        all_abstracts = []
        seen_ids = set()

        for year in years:
            for term in self.SEARCH_TERMS:
                results = self.search_abstracts(term, year, max_per_search)

                for r in results:
                    key = r.get("rsna_id") or r.get("title")
                    if key and key not in seen_ids:
                        seen_ids.add(key)
                        all_abstracts.append(r)

                logger.info(f"Year {year}, '{term}': +{len(results)} abstracts")

        logger.info(f"Collected {len(all_abstracts)} unique RSNA abstracts")

        # Paper 객체로 변환
        papers = []
        for r in all_abstracts:
            try:
                paper = Paper(
                    pmid=f"RSNA_{r.get('year')}_{r.get('rsna_id', '')}",
                    title=r.get("title", ""),
                    authors=r.get("authors", []),
                    journal=f"RSNA {r.get('year')} Annual Meeting",
                    year=r.get("year", 0),
                    abstract=r.get("abstract", ""),
                    publication_types=["Abstract", "Conference"],
                    source="rsna",
                )
                papers.append(paper)
            except Exception as e:
                logger.warning(f"Failed to create Paper: {e}")

        return papers

    def close(self):
        """클라이언트 종료"""
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    with RSNAScraper() as scraper:
        # 2023-2024 초록 수집 테스트
        papers = scraper.collect_mammography_abstracts(
            years=[2024, 2023],
            max_per_search=20,
        )

        print(f"\nCollected {len(papers)} RSNA abstracts")
        for paper in papers[:5]:
            print(f"\n- {paper.title[:60]}...")
            print(f"  Year: {paper.year}")
            print(f"  Authors: {', '.join(paper.authors[:3])}...")
