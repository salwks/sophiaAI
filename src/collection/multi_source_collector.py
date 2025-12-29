"""
MARIA-Mammo: Multi-Source Collector
====================================
PubMed, KoreaMed, Semantic Scholar, RSNA 통합 수집
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from tqdm import tqdm

from src.collection.koreamed_client import KoreMedClient
from src.collection.pubmed_client import PubMedClient
from src.collection.rsna_scraper import RSNAScraper
from src.collection.semantic_scholar_client import SemanticScholarClient
from src.models import Paper

load_dotenv()
logger = logging.getLogger(__name__)


class MultiSourceCollector:
    """멀티 소스 논문 수집기"""

    def __init__(
        self,
        output_dir: Path = Path("data/raw"),
        ncbi_api_key: Optional[str] = None,
        s2_api_key: Optional[str] = None,
    ):
        """
        Args:
            output_dir: 출력 디렉토리
            ncbi_api_key: NCBI API 키
            s2_api_key: Semantic Scholar API 키
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.ncbi_api_key = ncbi_api_key or os.getenv("NCBI_API_KEY")
        self.s2_api_key = s2_api_key or os.getenv("S2_API_KEY")
        self.email = os.getenv("CROSSREF_EMAIL", "maria-mammo@example.com")

        logger.info(f"MultiSourceCollector initialized. Output: {self.output_dir}")

    def collect_all(
        self,
        start_year: int = 2010,
        end_year: int = 2025,
        sources: Optional[List[str]] = None,
    ) -> Dict[str, int]:
        """
        모든 소스에서 논문 수집

        Args:
            start_year: 시작 연도
            end_year: 종료 연도
            sources: 수집할 소스 리스트 (기본: 모두)

        Returns:
            소스별 수집 개수
        """
        if sources is None:
            sources = ["pubmed", "koreamed", "semantic_scholar", "rsna"]

        results = {}

        # 1. PubMed (기본)
        if "pubmed" in sources:
            logger.info("\n" + "=" * 60)
            logger.info("[1/4] Collecting from PubMed...")
            logger.info("=" * 60)
            count = self._collect_pubmed(start_year, end_year)
            results["pubmed"] = count

        # 2. KoreaMed
        if "koreamed" in sources:
            logger.info("\n" + "=" * 60)
            logger.info("[2/4] Collecting from KoreaMed...")
            logger.info("=" * 60)
            count = self._collect_koreamed(start_year)
            results["koreamed"] = count

        # 3. Semantic Scholar
        if "semantic_scholar" in sources:
            logger.info("\n" + "=" * 60)
            logger.info("[3/4] Collecting from Semantic Scholar...")
            logger.info("=" * 60)
            count = self._collect_semantic_scholar(start_year, end_year)
            results["semantic_scholar"] = count

        # 4. RSNA
        if "rsna" in sources:
            logger.info("\n" + "=" * 60)
            logger.info("[4/4] Collecting from RSNA...")
            logger.info("=" * 60)
            count = self._collect_rsna(start_year, end_year)
            results["rsna"] = count

        # 통계 저장
        self._save_collection_stats(results, start_year, end_year)

        total = sum(results.values())
        logger.info(f"\n=== Collection Complete ===")
        logger.info(f"Total: {total:,} papers from {len(results)} sources")
        for source, count in results.items():
            logger.info(f"  - {source}: {count:,}")

        return results

    def _collect_pubmed(self, start_year: int, end_year: int) -> int:
        """PubMed 수집"""
        from src.collection.collector import MammographyCollector

        with PubMedClient(api_key=self.ncbi_api_key, email=self.email) as client:
            collector = MammographyCollector(
                client=client,
                output_dir=self.output_dir,
            )
            results = collector.collect_all(
                start_year=start_year,
                end_year=end_year,
            )
            return sum(results.values())

    def _collect_koreamed(self, min_year: int) -> int:
        """KoreaMed 수집"""
        output_file = self.output_dir / "koreamed.json"

        try:
            with KoreMedClient() as client:
                papers = client.collect_mammography_papers(min_year=min_year)

                if papers:
                    self._save_papers(output_file, papers, "koreamed")
                    logger.info(f"Saved {len(papers)} KoreaMed papers")
                    return len(papers)
                else:
                    logger.warning("No papers collected from KoreaMed")
                    return 0
        except Exception as e:
            logger.error(f"KoreaMed collection failed: {e}")
            return 0

    def _collect_semantic_scholar(self, start_year: int, end_year: int) -> int:
        """Semantic Scholar 수집"""
        output_file = self.output_dir / "semantic_scholar.json"

        try:
            with SemanticScholarClient(api_key=self.s2_api_key) as client:
                papers = client.collect_mammography_papers(
                    year_range=f"{start_year}-{end_year}",
                    max_per_query=500,
                )

                if papers:
                    self._save_papers(output_file, papers, "semantic_scholar")
                    logger.info(f"Saved {len(papers)} Semantic Scholar papers")
                    return len(papers)
                else:
                    logger.warning("No papers collected from Semantic Scholar")
                    return 0
        except Exception as e:
            logger.error(f"Semantic Scholar collection failed: {e}")
            return 0

    def _collect_rsna(self, start_year: int, end_year: int) -> int:
        """RSNA 초록 수집"""
        output_file = self.output_dir / "rsna.json"

        # RSNA 연도 범위
        rsna_years = [y for y in range(start_year, end_year + 1) if y >= 2019]  # RSNA archive 제한

        if not rsna_years:
            logger.info("No RSNA years in range")
            return 0

        try:
            with RSNAScraper() as scraper:
                papers = scraper.collect_mammography_abstracts(
                    years=rsna_years,
                    max_per_search=50,
                )

                if papers:
                    self._save_papers(output_file, papers, "rsna")
                    logger.info(f"Saved {len(papers)} RSNA abstracts")
                    return len(papers)
                else:
                    logger.warning("No abstracts collected from RSNA")
                    return 0
        except Exception as e:
            logger.error(f"RSNA collection failed: {e}")
            return 0

    def _save_papers(self, output_file: Path, papers: List[Paper], source: str):
        """논문 저장"""
        data = {
            "source": source,
            "collected_at": datetime.now().isoformat(),
            "count": len(papers),
            "papers": [paper.model_dump(exclude={"embedding"}) for paper in papers],
        }

        output_file.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, default=str)
        )

    def _save_collection_stats(
        self, results: Dict[str, int], start_year: int, end_year: int
    ):
        """수집 통계 저장"""
        stats = {
            "collected_at": datetime.now().isoformat(),
            "year_range": f"{start_year}-{end_year}",
            "sources": results,
            "total": sum(results.values()),
        }

        stats_file = self.output_dir / "collection_stats.json"
        stats_file.write_text(json.dumps(stats, indent=2))
        logger.info(f"Saved collection stats to {stats_file}")

    def enrich_with_citations(self, papers_file: Path) -> int:
        """
        기존 논문에 Semantic Scholar 인용수 추가

        Args:
            papers_file: papers.json 경로

        Returns:
            enriched 논문 수
        """
        if not papers_file.exists():
            logger.error(f"File not found: {papers_file}")
            return 0

        # 논문 로드
        data = json.loads(papers_file.read_text())
        papers = [Paper(**p) for p in data.get("papers", [])]

        logger.info(f"Enriching citations for {len(papers)} papers...")

        with SemanticScholarClient(api_key=self.s2_api_key) as client:
            papers = client.enrich_citations(papers)

        # 저장
        data["papers"] = [p.model_dump(exclude={"embedding"}) for p in papers]
        data["enriched_at"] = datetime.now().isoformat()

        papers_file.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, default=str)
        )

        enriched = sum(1 for p in papers if p.citation_count > 0)
        logger.info(f"Enriched {enriched} papers with citations")
        return enriched

    def load_all_papers(self) -> List[Paper]:
        """모든 소스에서 수집된 논문 로드 및 중복 제거"""
        all_papers = []
        seen_keys = set()

        # PubMed 연도별 파일
        for file in sorted(self.output_dir.glob("*.json")):
            if file.stem.isdigit():  # 연도 파일
                data = json.loads(file.read_text())
                for p in data.get("papers", []):
                    paper = Paper(**p)
                    key = paper.doi or paper.pmid or paper.title
                    if key not in seen_keys:
                        seen_keys.add(key)
                        all_papers.append(paper)

        # 기타 소스 파일
        for source in ["koreamed", "semantic_scholar", "rsna"]:
            file = self.output_dir / f"{source}.json"
            if file.exists():
                data = json.loads(file.read_text())
                for p in data.get("papers", []):
                    paper = Paper(**p)
                    # 중복 체크 (DOI 또는 제목)
                    key = paper.doi or paper.title
                    if key not in seen_keys:
                        seen_keys.add(key)
                        all_papers.append(paper)
                    else:
                        logger.debug(f"Duplicate skipped: {paper.title[:50]}...")

        logger.info(f"Loaded {len(all_papers)} unique papers from all sources")
        return all_papers


def main():
    """CLI 실행"""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Collect papers from multiple sources")
    parser.add_argument("--start-year", type=int, default=2010, help="Start year")
    parser.add_argument("--end-year", type=int, default=2025, help="End year")
    parser.add_argument("--output-dir", type=str, default="data/raw", help="Output directory")
    parser.add_argument(
        "--sources",
        type=str,
        default="pubmed,koreamed,semantic_scholar,rsna",
        help="Comma-separated list of sources",
    )
    parser.add_argument("--enrich-citations", type=str, help="Enrich citations for given papers.json")

    args = parser.parse_args()

    collector = MultiSourceCollector(output_dir=Path(args.output_dir))

    if args.enrich_citations:
        collector.enrich_with_citations(Path(args.enrich_citations))
    else:
        sources = [s.strip() for s in args.sources.split(",")]
        results = collector.collect_all(
            start_year=args.start_year,
            end_year=args.end_year,
            sources=sources,
        )

        print("\n=== Collection Summary ===")
        for source, count in results.items():
            print(f"  {source}: {count:,} papers")
        print(f"  Total: {sum(results.values()):,} papers")


if __name__ == "__main__":
    main()
