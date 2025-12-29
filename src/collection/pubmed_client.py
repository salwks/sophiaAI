"""
Sophia AI: PubMed E-utilities Client
======================================
PubMed API를 통한 논문 검색 및 상세 정보 수집
"""

import logging
import time
from typing import List, Optional
from xml.etree import ElementTree as ET

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.models import Paper

logger = logging.getLogger(__name__)

# PubMed E-utilities Base URLs
EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
ESEARCH_URL = f"{EUTILS_BASE}/esearch.fcgi"
EFETCH_URL = f"{EUTILS_BASE}/efetch.fcgi"


class PubMedClient:
    """PubMed E-utilities API 클라이언트"""

    def __init__(self, api_key: Optional[str] = None, email: Optional[str] = None):
        """
        Args:
            api_key: NCBI API 키 (선택, 있으면 10 req/sec, 없으면 3 req/sec)
            email: 연락처 이메일 (NCBI 권장)
        """
        self.api_key = api_key
        self.email = email

        # Rate limiting
        self.requests_per_second = 10 if api_key else 3
        self.min_interval = 1.0 / self.requests_per_second
        self.last_request_time = 0.0

        # HTTP 클라이언트
        self.client = httpx.Client(timeout=60.0)

        logger.info(
            f"PubMedClient initialized (rate limit: {self.requests_per_second} req/sec)"
        )

    def _rate_limit(self):
        """Rate limiting 적용"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request_time = time.time()

    def _build_params(self, **kwargs) -> dict:
        """공통 파라미터 구성"""
        params = dict(kwargs)
        if self.api_key:
            params["api_key"] = self.api_key
        if self.email:
            params["email"] = self.email
        return params

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def search(
        self,
        query: str,
        min_date: Optional[str] = None,
        max_date: Optional[str] = None,
        retmax: int = 10000,
    ) -> List[str]:
        """
        PubMed 검색 실행

        Args:
            query: 검색 쿼리
            min_date: 시작 날짜 (YYYY/MM/DD 또는 YYYY)
            max_date: 종료 날짜
            retmax: 최대 반환 개수

        Returns:
            PMID 리스트
        """
        self._rate_limit()

        params = self._build_params(
            db="pubmed",
            term=query,
            retmax=retmax,
            retmode="json",
            usehistory="y",
        )

        if min_date:
            params["mindate"] = min_date
            params["datetype"] = "pdat"  # Publication date
        if max_date:
            params["maxdate"] = max_date

        logger.info(f"Searching PubMed: {query[:100]}...")

        response = self.client.get(ESEARCH_URL, params=params)
        response.raise_for_status()

        data = response.json()
        result = data.get("esearchresult", {})

        pmids = result.get("idlist", [])
        total_count = int(result.get("count", 0))

        logger.info(f"Found {total_count} papers, retrieved {len(pmids)} PMIDs")

        return pmids

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def fetch_details(
        self, pmids: List[str], batch_size: int = 200
    ) -> List[Paper]:
        """
        PMID 목록으로 논문 상세 정보 가져오기

        Args:
            pmids: PMID 리스트
            batch_size: 배치 크기 (최대 200)

        Returns:
            Paper 객체 리스트
        """
        papers = []
        total = len(pmids)

        for i in range(0, total, batch_size):
            batch = pmids[i : i + batch_size]
            batch_papers = self._fetch_batch(batch)
            papers.extend(batch_papers)

            progress = min(i + batch_size, total)
            logger.info(f"Fetched {progress}/{total} papers")

        return papers

    def _fetch_batch(self, pmids: List[str]) -> List[Paper]:
        """단일 배치 fetch"""
        self._rate_limit()

        params = self._build_params(
            db="pubmed",
            id=",".join(pmids),
            retmode="xml",
            rettype="abstract",
        )

        response = self.client.get(EFETCH_URL, params=params)
        response.raise_for_status()

        return self._parse_xml_response(response.text)

    def _parse_xml_response(self, xml_text: str) -> List[Paper]:
        """XML 응답 파싱"""
        papers = []

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            logger.error(f"XML parsing error: {e}")
            return papers

        for article in root.findall(".//PubmedArticle"):
            try:
                paper = self._parse_article(article)
                if paper:
                    papers.append(paper)
            except Exception as e:
                pmid = article.findtext(".//PMID", "unknown")
                logger.warning(f"Failed to parse article {pmid}: {e}")

        return papers

    def _parse_article(self, article: ET.Element) -> Optional[Paper]:
        """단일 article XML 파싱"""
        medline = article.find("MedlineCitation")
        if medline is None:
            return None

        article_elem = medline.find("Article")
        if article_elem is None:
            return None

        # PMID
        pmid = medline.findtext("PMID", "")
        if not pmid:
            return None

        # Title
        title = article_elem.findtext("ArticleTitle", "")

        # Abstract
        abstract_elem = article_elem.find("Abstract")
        abstract = ""
        if abstract_elem is not None:
            abstract_texts = []
            for text_elem in abstract_elem.findall("AbstractText"):
                label = text_elem.get("Label", "")
                text = text_elem.text or ""
                if label:
                    abstract_texts.append(f"{label}: {text}")
                else:
                    abstract_texts.append(text)
            abstract = " ".join(abstract_texts)

        # Authors
        authors = []
        author_list = article_elem.find("AuthorList")
        if author_list is not None:
            for author in author_list.findall("Author"):
                lastname = author.findtext("LastName", "")
                forename = author.findtext("ForeName", "")
                initials = author.findtext("Initials", "")
                if lastname:
                    if forename:
                        authors.append(f"{lastname} {forename}")
                    elif initials:
                        authors.append(f"{lastname} {initials}")
                    else:
                        authors.append(lastname)

        # Journal
        journal_elem = article_elem.find("Journal")
        journal = ""
        journal_abbrev = ""
        year = 0
        month = None

        if journal_elem is not None:
            journal = journal_elem.findtext("Title", "")
            journal_abbrev = journal_elem.findtext("ISOAbbreviation", "")

            # Publication Date
            pub_date = journal_elem.find("JournalIssue/PubDate")
            if pub_date is not None:
                year_text = pub_date.findtext("Year", "")
                if year_text:
                    try:
                        year = int(year_text)
                    except ValueError:
                        pass

                month_text = pub_date.findtext("Month", "")
                if month_text:
                    month = self._parse_month(month_text)

        # MeSH Terms
        mesh_terms = []
        mesh_list = medline.find("MeshHeadingList")
        if mesh_list is not None:
            for mesh in mesh_list.findall("MeshHeading"):
                descriptor = mesh.findtext("DescriptorName", "")
                if descriptor:
                    mesh_terms.append(descriptor)

        # Keywords
        keywords = []
        keyword_list = medline.find("KeywordList")
        if keyword_list is not None:
            for kw in keyword_list.findall("Keyword"):
                if kw.text:
                    keywords.append(kw.text)

        # Publication Types
        pub_types = []
        pub_type_list = article_elem.find("PublicationTypeList")
        if pub_type_list is not None:
            for pt in pub_type_list.findall("PublicationType"):
                if pt.text:
                    pub_types.append(pt.text)

        # Article IDs (DOI, PMC)
        doi = None
        pmc_id = None
        article_ids = article.find("PubmedData/ArticleIdList")
        if article_ids is not None:
            for id_elem in article_ids.findall("ArticleId"):
                id_type = id_elem.get("IdType", "")
                if id_type == "doi" and id_elem.text:
                    doi = id_elem.text
                elif id_type == "pmc" and id_elem.text:
                    pmc_id = id_elem.text

        return Paper(
            pmid=pmid,
            doi=doi,
            title=title,
            authors=authors,
            journal=journal,
            journal_abbrev=journal_abbrev,
            year=year,
            month=month,
            abstract=abstract,
            mesh_terms=mesh_terms,
            keywords=keywords,
            publication_types=pub_types,
            pmc_id=pmc_id,
        )

    def _parse_month(self, month_text: str) -> Optional[int]:
        """월 텍스트를 숫자로 변환"""
        month_map = {
            "jan": 1, "feb": 2, "mar": 3, "apr": 4,
            "may": 5, "jun": 6, "jul": 7, "aug": 8,
            "sep": 9, "oct": 10, "nov": 11, "dec": 12,
        }

        try:
            return int(month_text)
        except ValueError:
            return month_map.get(month_text.lower()[:3])

    def close(self):
        """클라이언트 종료"""
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# 테스트용 실행
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    api_key = os.getenv("NCBI_API_KEY")
    email = os.getenv("CROSSREF_EMAIL", "test@example.com")

    with PubMedClient(api_key=api_key, email=email) as client:
        # 2024년 mammography 논문 100개 검색
        pmids = client.search(
            query="mammography AND breast cancer",
            min_date="2024/01/01",
            max_date="2024/12/31",
            retmax=100,
        )

        print(f"\nFound {len(pmids)} PMIDs")

        if pmids:
            # 상세 정보 가져오기
            papers = client.fetch_details(pmids[:10])

            print(f"\nFetched {len(papers)} papers:")
            for paper in papers[:3]:
                print(f"\n- {paper.title[:80]}...")
                print(f"  Authors: {paper.author_string}")
                print(f"  Journal: {paper.journal} ({paper.year})")
                print(f"  PubMed: {paper.pubmed_url}")
