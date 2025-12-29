"""
Sophia AI: KoreaMed Client
============================
KoreaMed API를 통한 국내 유방영상 논문 수집
"""

import logging
import time
from typing import List, Optional
from xml.etree import ElementTree as ET

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.models import Paper

logger = logging.getLogger(__name__)

# KoreaMed API Base URL
KOREAMED_SEARCH_URL = "https://koreamed.org/SearchBasic.php"
KOREAMED_API_URL = "https://koreamed.org/API"


class KoreMedClient:
    """KoreaMed API 클라이언트"""

    # 맘모그래피 관련 검색어
    SEARCH_TERMS = [
        "mammography",
        "breast imaging",
        "tomosynthesis",
        "breast cancer screening",
        "breast ultrasound",
        "breast MRI",
        "유방촬영",
        "유방암",
        "유방 초음파",
    ]

    def __init__(self):
        """초기화"""
        self.client = httpx.Client(timeout=60.0)
        self.last_request_time = 0.0
        self.min_interval = 1.0  # 1 request per second

        logger.info("KoreMedClient initialized")

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
        max_results: int = 1000,
    ) -> List[dict]:
        """
        KoreaMed 검색

        Args:
            query: 검색어
            max_results: 최대 결과 수

        Returns:
            검색 결과 리스트
        """
        self._rate_limit()

        params = {
            "SearchWord": query,
            "SearchField": "ALL",
            "SortBy": "Year",
            "SortOrder": "DESC",
            "PageSize": min(max_results, 100),
            "output": "xml",
        }

        logger.info(f"Searching KoreaMed: {query}")

        try:
            response = self.client.get(KOREAMED_API_URL, params=params)
            response.raise_for_status()
            return self._parse_search_results(response.text)
        except Exception as e:
            logger.warning(f"KoreaMed search failed: {e}")
            return []

    def _parse_search_results(self, xml_text: str) -> List[dict]:
        """검색 결과 XML 파싱"""
        results = []

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            logger.error(f"XML parsing error: {e}")
            return results

        for article in root.findall(".//Article"):
            try:
                result = {
                    "koreamed_id": article.findtext("ArticleId", ""),
                    "pmid": article.findtext("PMID", ""),
                    "title": article.findtext("Title", ""),
                    "authors": self._parse_authors(article),
                    "journal": article.findtext("Journal", ""),
                    "year": self._parse_year(article.findtext("Year", "")),
                    "abstract": article.findtext("Abstract", ""),
                    "doi": article.findtext("DOI", ""),
                    "keywords": self._parse_keywords(article),
                }
                if result["title"]:
                    results.append(result)
            except Exception as e:
                logger.warning(f"Failed to parse article: {e}")

        return results

    def _parse_authors(self, article: ET.Element) -> List[str]:
        """저자 파싱"""
        authors = []
        author_list = article.find("Authors")
        if author_list is not None:
            for author in author_list.findall("Author"):
                name = author.text
                if name:
                    authors.append(name.strip())
        return authors

    def _parse_keywords(self, article: ET.Element) -> List[str]:
        """키워드 파싱"""
        keywords = []
        keyword_list = article.find("Keywords")
        if keyword_list is not None:
            for kw in keyword_list.findall("Keyword"):
                if kw.text:
                    keywords.append(kw.text.strip())
        return keywords

    def _parse_year(self, year_text: str) -> int:
        """연도 파싱"""
        try:
            return int(year_text[:4]) if year_text else 0
        except ValueError:
            return 0

    def collect_mammography_papers(
        self,
        min_year: int = 2010,
    ) -> List[Paper]:
        """
        맘모그래피 관련 논문 수집

        Args:
            min_year: 최소 연도

        Returns:
            Paper 리스트
        """
        all_results = []
        seen_ids = set()

        for term in self.SEARCH_TERMS:
            results = self.search(term)
            for r in results:
                # 중복 제거
                key = r.get("koreamed_id") or r.get("pmid") or r.get("title")
                if key and key not in seen_ids:
                    seen_ids.add(key)
                    if r.get("year", 0) >= min_year:
                        all_results.append(r)

        logger.info(f"Collected {len(all_results)} unique papers from KoreaMed")

        # Paper 객체로 변환
        papers = []
        for r in all_results:
            try:
                paper = Paper(
                    pmid=r.get("pmid") or f"KM_{r.get('koreamed_id', '')}",
                    doi=r.get("doi"),
                    title=r.get("title", ""),
                    authors=r.get("authors", []),
                    journal=r.get("journal", ""),
                    year=r.get("year", 0),
                    abstract=r.get("abstract", ""),
                    keywords=r.get("keywords", []),
                    source="koreamed",
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

    with KoreMedClient() as client:
        papers = client.collect_mammography_papers(min_year=2020)
        print(f"\nCollected {len(papers)} papers from KoreaMed")
        for paper in papers[:5]:
            print(f"\n- {paper.title[:60]}...")
            print(f"  Journal: {paper.journal}")
            print(f"  Year: {paper.year}")
