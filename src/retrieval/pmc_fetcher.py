"""
Dynamic PMC Fetcher: 실시간 논문 전문 인출 시스템
=================================================
PMC ID를 보유한 논문의 전문을 실시간으로 가져와서
초록만으로는 알 수 없는 세부 정보를 추출합니다.

Features:
    - 비동기(Async) 다중 논문 fetch
    - 로컬 캐싱으로 중복 요청 방지
    - 스마트 섹션 추출 (Results, Methods, Discussion)
    - 질문 관련 문장만 선별 (Context Chunking)
"""

import re
import json
import hashlib
import asyncio
import aiohttp
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import requests

logger = logging.getLogger(__name__)


@dataclass
class PMCArticle:
    """PMC 논문 전문 데이터"""
    pmc_id: str
    title: str = ""
    abstract: str = ""
    full_text: str = ""
    sections: Dict[str, str] = field(default_factory=dict)  # section_name -> content
    methods: str = ""
    results: str = ""
    discussion: str = ""
    conclusions: str = ""
    figures: List[str] = field(default_factory=list)
    tables: List[str] = field(default_factory=list)
    references_count: int = 0
    fetch_time: datetime = field(default_factory=datetime.now)
    source_url: str = ""


@dataclass
class FetchConfig:
    """PMC Fetcher 설정"""
    cache_dir: Path = Path("data/cache/pmc")
    cache_ttl_hours: int = 24 * 7  # 1주일 캐시
    max_concurrent: int = 5        # 동시 요청 수
    timeout: int = 20              # 요청 타임아웃 (초)
    max_chars_per_section: int = 3000  # 섹션당 최대 문자
    user_agent: str = "Mozilla/5.0 (compatible; SophiaAI/1.0; Medical Research)"


class PMCFetcher:
    """
    비동기 PMC 전문 Fetcher

    Usage:
        fetcher = PMCFetcher()

        # 단일 논문
        article = await fetcher.fetch_async("PMC7533093")

        # 다중 논문 (병렬)
        articles = await fetcher.fetch_multiple_async(["PMC123", "PMC456"])

        # 질문 관련 컨텍스트만 추출
        context = fetcher.extract_relevant_context(article, "SNR과 선량의 관계")
    """

    def __init__(self, config: Optional[FetchConfig] = None):
        self.config = config or FetchConfig()
        self._ensure_cache_dir()
        self._executor = ThreadPoolExecutor(max_workers=self.config.max_concurrent)

    def _ensure_cache_dir(self):
        """캐시 디렉토리 생성"""
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Caching
    # =========================================================================

    def _get_cache_path(self, pmc_id: str) -> Path:
        """캐시 파일 경로"""
        return self.config.cache_dir / f"{pmc_id}.json"

    def _load_from_cache(self, pmc_id: str) -> Optional[PMCArticle]:
        """캐시에서 논문 로드"""
        cache_path = self._get_cache_path(pmc_id)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # TTL 체크
            fetch_time = datetime.fromisoformat(data.get("fetch_time", "2000-01-01"))
            if datetime.now() - fetch_time > timedelta(hours=self.config.cache_ttl_hours):
                logger.info(f"Cache expired for {pmc_id}")
                return None

            return PMCArticle(
                pmc_id=data["pmc_id"],
                title=data.get("title", ""),
                abstract=data.get("abstract", ""),
                full_text=data.get("full_text", ""),
                sections=data.get("sections", {}),
                methods=data.get("methods", ""),
                results=data.get("results", ""),
                discussion=data.get("discussion", ""),
                conclusions=data.get("conclusions", ""),
                figures=data.get("figures", []),
                tables=data.get("tables", []),
                references_count=data.get("references_count", 0),
                fetch_time=fetch_time,
                source_url=data.get("source_url", "")
            )
        except Exception as e:
            logger.warning(f"Cache load error for {pmc_id}: {e}")
            return None

    def _save_to_cache(self, article: PMCArticle):
        """캐시에 논문 저장"""
        cache_path = self._get_cache_path(article.pmc_id)

        try:
            data = {
                "pmc_id": article.pmc_id,
                "title": article.title,
                "abstract": article.abstract,
                "full_text": article.full_text,
                "sections": article.sections,
                "methods": article.methods,
                "results": article.results,
                "discussion": article.discussion,
                "conclusions": article.conclusions,
                "figures": article.figures,
                "tables": article.tables,
                "references_count": article.references_count,
                "fetch_time": article.fetch_time.isoformat(),
                "source_url": article.source_url
            }

            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            logger.debug(f"Cached {article.pmc_id}")
        except Exception as e:
            logger.warning(f"Cache save error for {article.pmc_id}: {e}")

    # =========================================================================
    # Fetching
    # =========================================================================

    def fetch_sync(self, pmc_id: str) -> Optional[PMCArticle]:
        """동기 방식 PMC 전문 fetch"""
        # 캐시 확인
        cached = self._load_from_cache(pmc_id)
        if cached:
            logger.info(f"Cache hit for {pmc_id}")
            return cached

        # PMC ID 정규화
        pmc_id = pmc_id.replace("PMC", "").strip()
        pmc_id = f"PMC{pmc_id}"

        url = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmc_id}/"

        try:
            response = requests.get(
                url,
                timeout=self.config.timeout,
                headers={"User-Agent": self.config.user_agent}
            )
            response.raise_for_status()

            article = self._parse_pmc_html(pmc_id, response.text, url)

            # 캐시 저장
            self._save_to_cache(article)

            logger.info(f"Fetched {pmc_id}: {len(article.full_text)} chars")
            return article

        except Exception as e:
            logger.error(f"Fetch error for {pmc_id}: {e}")
            return None

    async def fetch_async(self, pmc_id: str) -> Optional[PMCArticle]:
        """비동기 방식 PMC 전문 fetch"""
        # 캐시 확인 (동기)
        cached = self._load_from_cache(pmc_id)
        if cached:
            logger.info(f"Cache hit for {pmc_id}")
            return cached

        # PMC ID 정규화
        pmc_id_clean = pmc_id.replace("PMC", "").strip()
        pmc_id = f"PMC{pmc_id_clean}"

        url = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmc_id}/"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout),
                    headers={"User-Agent": self.config.user_agent}
                ) as response:
                    if response.status != 200:
                        logger.warning(f"HTTP {response.status} for {pmc_id}")
                        return None

                    html = await response.text()

            article = self._parse_pmc_html(pmc_id, html, url)

            # 캐시 저장
            self._save_to_cache(article)

            logger.info(f"Fetched {pmc_id}: {len(article.full_text)} chars")
            return article

        except Exception as e:
            logger.error(f"Async fetch error for {pmc_id}: {e}")
            return None

    async def fetch_multiple_async(self, pmc_ids: List[str]) -> List[PMCArticle]:
        """다중 논문 병렬 fetch"""
        tasks = [self.fetch_async(pmc_id) for pmc_id in pmc_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        articles = []
        for result in results:
            if isinstance(result, PMCArticle):
                articles.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Fetch error: {result}")

        return articles

    # =========================================================================
    # HTML Parsing
    # =========================================================================

    def _parse_pmc_html(self, pmc_id: str, html: str, url: str) -> PMCArticle:
        """PMC HTML 파싱하여 구조화된 데이터 추출"""

        # 기본 정리
        html_clean = self._clean_html(html)

        # 제목 추출
        title = self._extract_title(html)

        # 섹션별 추출
        sections = self._extract_sections(html_clean)

        # 전문 텍스트
        full_text = self._extract_full_text(html_clean)

        return PMCArticle(
            pmc_id=pmc_id,
            title=title,
            abstract=sections.get("abstract", ""),
            full_text=full_text,
            sections=sections,
            methods=sections.get("methods", sections.get("materials and methods", "")),
            results=sections.get("results", ""),
            discussion=sections.get("discussion", ""),
            conclusions=sections.get("conclusions", sections.get("conclusion", "")),
            source_url=url
        )

    def _clean_html(self, html: str) -> str:
        """HTML 정리"""
        # script, style 제거
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<nav[^>]*>.*?</nav>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<footer[^>]*>.*?</footer>', '', html, flags=re.DOTALL | re.IGNORECASE)
        return html

    def _extract_title(self, html: str) -> str:
        """제목 추출"""
        # <h1 class="content-title"> 또는 <title> 태그
        match = re.search(r'<h1[^>]*class="[^"]*content-title[^"]*"[^>]*>(.*?)</h1>', html, re.DOTALL | re.IGNORECASE)
        if match:
            return re.sub(r'<[^>]+>', '', match.group(1)).strip()

        match = re.search(r'<title>(.*?)</title>', html, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        return ""

    def _extract_sections(self, html: str) -> Dict[str, str]:
        """섹션별 내용 추출"""
        sections = {}

        # PMC 섹션 패턴: <div class="tsec sec" id="s1"> ... <h2>Abstract</h2> ...
        section_patterns = [
            (r'abstract', r'(?:abstract|summary)'),
            (r'introduction', r'(?:introduction|background)'),
            (r'methods', r'(?:methods?|materials?\s*(?:and\s*)?methods?)'),
            (r'results', r'results?'),
            (r'discussion', r'discussion'),
            (r'conclusions', r'(?:conclusions?|summary)'),
        ]

        for section_key, pattern in section_patterns:
            # <h2>Section Name</h2> ... 다음 <h2> 전까지
            regex = rf'<h2[^>]*>.*?{pattern}.*?</h2>(.*?)(?=<h2|$)'
            match = re.search(regex, html, re.DOTALL | re.IGNORECASE)
            if match:
                text = self._html_to_text(match.group(1))
                sections[section_key] = text[:self.config.max_chars_per_section]

        return sections

    def _extract_full_text(self, html: str) -> str:
        """전문 텍스트 추출"""
        # article 태그 내용 추출
        article_match = re.search(r'<article[^>]*>(.*?)</article>', html, re.DOTALL | re.IGNORECASE)
        if article_match:
            html = article_match.group(1)

        return self._html_to_text(html)

    def _html_to_text(self, html: str) -> str:
        """HTML을 텍스트로 변환"""
        # 태그 제거
        text = re.sub(r'<[^>]+>', ' ', html)

        # 특수문자 변환
        text = text.replace('&nbsp;', ' ')
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&quot;', '"')

        # 여러 공백 정리
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    # =========================================================================
    # Context Extraction (Smart Chunking)
    # =========================================================================

    def extract_relevant_context(
        self,
        article: PMCArticle,
        question: str,
        max_chars: int = 4000
    ) -> str:
        """
        질문과 관련된 컨텍스트만 추출 (Smart Chunking)

        Args:
            article: PMC 논문 데이터
            question: 사용자 질문
            max_chars: 최대 문자 수

        Returns:
            질문과 관련된 핵심 내용
        """
        # 질문에서 키워드 추출
        keywords = self._extract_keywords(question)

        # 우선순위: Results > Methods > Discussion > Abstract
        priority_sections = ["results", "methods", "discussion", "abstract", "conclusions"]

        context_parts = []
        remaining_chars = max_chars

        for section_name in priority_sections:
            section_content = article.sections.get(section_name, "")
            if not section_content:
                continue

            # 키워드가 포함된 문장만 추출
            relevant_sentences = self._extract_relevant_sentences(
                section_content, keywords, max_sentences=5
            )

            if relevant_sentences:
                section_text = f"[{section_name.upper()}]\n{relevant_sentences}"

                if len(section_text) <= remaining_chars:
                    context_parts.append(section_text)
                    remaining_chars -= len(section_text)
                else:
                    # 남은 공간만큼만 추가
                    context_parts.append(section_text[:remaining_chars])
                    break

        if not context_parts:
            # 키워드 매칭 실패 시 전문에서 앞부분 사용
            return article.full_text[:max_chars]

        return "\n\n".join(context_parts)

    def _extract_keywords(self, question: str) -> List[str]:
        """질문에서 검색 키워드 추출"""
        # 의학 용어 및 숫자 보존
        question_lower = question.lower()

        # 불용어 제거
        stopwords = {'의', '는', '은', '이', '가', '을', '를', '에', '와', '과', '로', '으로',
                     'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'what', 'how', 'why', 'when', 'where', 'which'}

        # 단어 추출
        words = re.findall(r'[a-z가-힣]+|\d+(?:\.\d+)?', question_lower)
        keywords = [w for w in words if w not in stopwords and len(w) > 1]

        return keywords

    def _extract_relevant_sentences(
        self,
        text: str,
        keywords: List[str],
        max_sentences: int = 5
    ) -> str:
        """키워드가 포함된 문장 추출"""
        # 문장 분리
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # 키워드 매칭 점수 계산
        scored_sentences = []
        for sent in sentences:
            sent_lower = sent.lower()
            score = sum(1 for kw in keywords if kw in sent_lower)
            if score > 0:
                scored_sentences.append((score, sent))

        # 점수순 정렬 후 상위 N개
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        top_sentences = [sent for _, sent in scored_sentences[:max_sentences]]

        return " ".join(top_sentences)


# =============================================================================
# Singleton & Convenience
# =============================================================================

_fetcher_instance: Optional[PMCFetcher] = None


def get_pmc_fetcher(config: Optional[FetchConfig] = None) -> PMCFetcher:
    """PMCFetcher 싱글톤"""
    global _fetcher_instance
    if _fetcher_instance is None:
        _fetcher_instance = PMCFetcher(config)
    return _fetcher_instance


def fetch_pmc_sync(pmc_id: str) -> Optional[PMCArticle]:
    """동기 방식 PMC fetch (편의 함수)"""
    fetcher = get_pmc_fetcher()
    return fetcher.fetch_sync(pmc_id)


async def fetch_pmc_async(pmc_id: str) -> Optional[PMCArticle]:
    """비동기 방식 PMC fetch (편의 함수)"""
    fetcher = get_pmc_fetcher()
    return await fetcher.fetch_async(pmc_id)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)

    async def test():
        fetcher = PMCFetcher()

        # 테스트 PMC ID (유방암 관련 논문)
        test_pmc = "PMC7533093"

        print("=" * 60)
        print(f"PMC Fetcher Test: {test_pmc}")
        print("=" * 60)

        article = await fetcher.fetch_async(test_pmc)

        if article:
            print(f"제목: {article.title}")
            print(f"전문 길이: {len(article.full_text)} chars")
            print(f"섹션: {list(article.sections.keys())}")
            print()

            # 질문 관련 컨텍스트 추출
            question = "breast density and cancer risk"
            context = fetcher.extract_relevant_context(article, question)
            print(f"질문: {question}")
            print(f"관련 컨텍스트 ({len(context)} chars):")
            print(context[:500] + "...")
        else:
            print("Fetch failed")

    asyncio.run(test())
