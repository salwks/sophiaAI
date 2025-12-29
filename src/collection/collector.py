"""
Sophia AI: Collection Orchestrator
====================================
맘모그래피 관련 PubMed 논문 수집 관리
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from src.collection.pubmed_client import PubMedClient
from src.models import Paper

logger = logging.getLogger(__name__)


class MammographyCollector:
    """맘모그래피 논문 수집기"""

    # 맘모그래피 관련 검색 쿼리
    SEARCH_QUERY = """
    (mammography[MeSH] OR "digital breast tomosynthesis"[tiab] OR DBT[tiab]
     OR "breast imaging"[tiab] OR mammogram[tiab] OR "digital mammography"[tiab]
     OR FFDM[tiab] OR "contrast enhanced mammography"[tiab] OR CEM[tiab])
    AND (breast[tiab] OR mammary[tiab])
    AND (cancer OR carcinoma OR neoplasm OR tumor OR lesion OR mass
         OR calcification OR microcalcification OR screening OR detection
         OR density OR "BI-RADS" OR diagnosis OR diagnostic)
    AND English[Language]
    NOT (animal[MeSH] NOT human[MeSH])
    """

    def __init__(
        self,
        client: PubMedClient,
        output_dir: Path = Path("data/raw"),
    ):
        """
        Args:
            client: PubMed API 클라이언트
            output_dir: 출력 디렉토리
        """
        self.client = client
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Collector initialized. Output: {self.output_dir}")

    def collect_year(self, year: int) -> List[Paper]:
        """
        특정 연도의 논문 수집

        Args:
            year: 수집할 연도

        Returns:
            수집된 Paper 리스트
        """
        output_file = self.output_dir / f"{year}.json"

        # 이미 수집된 경우 스킵
        if output_file.exists():
            logger.info(f"Year {year} already collected, loading from file...")
            return self._load_year(year)

        logger.info(f"Collecting papers from {year}...")

        # 검색
        pmids = self.client.search(
            query=self.SEARCH_QUERY.strip(),
            min_date=f"{year}/01/01",
            max_date=f"{year}/12/31",
            retmax=10000,
        )

        if not pmids:
            logger.warning(f"No papers found for {year}")
            return []

        # 상세 정보 수집
        papers = self.client.fetch_details(pmids)

        # 저장
        self._save_year(year, papers)

        logger.info(f"Collected {len(papers)} papers for {year}")
        return papers

    def collect_all(
        self,
        start_year: int = 2005,
        end_year: int = 2025,
        skip_existing: bool = True,
    ) -> Dict[int, int]:
        """
        전체 연도 범위 수집

        Args:
            start_year: 시작 연도
            end_year: 종료 연도
            skip_existing: 기존 파일 스킵 여부

        Returns:
            연도별 수집 개수 딕셔너리
        """
        results = {}
        years = range(start_year, end_year + 1)

        logger.info(f"Collecting papers from {start_year} to {end_year}")

        for year in tqdm(years, desc="Collecting years"):
            output_file = self.output_dir / f"{year}.json"

            if skip_existing and output_file.exists():
                # 기존 파일에서 개수만 로드
                data = json.loads(output_file.read_text())
                results[year] = data.get("count", 0)
                continue

            try:
                papers = self.collect_year(year)
                results[year] = len(papers)
            except Exception as e:
                logger.error(f"Failed to collect {year}: {e}")
                results[year] = 0

        # 통계 출력
        total = sum(results.values())
        logger.info(f"\nCollection complete: {total} papers from {len(results)} years")

        return results

    def _save_year(self, year: int, papers: List[Paper]):
        """연도별 데이터 저장"""
        output_file = self.output_dir / f"{year}.json"

        data = {
            "year": year,
            "collected_at": datetime.now().isoformat(),
            "count": len(papers),
            "papers": [paper.model_dump(exclude={"embedding"}) for paper in papers],
        }

        output_file.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, default=str)
        )
        logger.info(f"Saved {len(papers)} papers to {output_file}")

    def _load_year(self, year: int) -> List[Paper]:
        """연도별 데이터 로드"""
        output_file = self.output_dir / f"{year}.json"

        if not output_file.exists():
            return []

        data = json.loads(output_file.read_text())
        papers = [Paper(**p) for p in data.get("papers", [])]

        return papers

    def load_all_papers(self) -> List[Paper]:
        """모든 수집된 논문 로드"""
        all_papers = []

        for file in sorted(self.output_dir.glob("*.json")):
            if file.stem.isdigit():  # 연도 파일만
                papers = self._load_year(int(file.stem))
                all_papers.extend(papers)

        logger.info(f"Loaded {len(all_papers)} papers from {self.output_dir}")
        return all_papers

    def get_stats(self) -> Dict:
        """수집 현황 통계"""
        stats = {
            "by_year": {},
            "total": 0,
            "collected_years": [],
        }

        for file in sorted(self.output_dir.glob("*.json")):
            if file.stem.isdigit():
                year = int(file.stem)
                data = json.loads(file.read_text())
                count = data.get("count", 0)

                stats["by_year"][year] = count
                stats["total"] += count
                stats["collected_years"].append(year)

        return stats


# 실행 스크립트용
def main():
    """CLI 실행"""
    import argparse
    import os

    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Collect mammography papers from PubMed")
    parser.add_argument("--start-year", type=int, default=2005, help="Start year")
    parser.add_argument("--end-year", type=int, default=2025, help="End year")
    parser.add_argument("--output-dir", type=str, default="data/raw", help="Output directory")
    parser.add_argument("--force", action="store_true", help="Force re-collection")
    parser.add_argument("--stats", action="store_true", help="Show stats only")

    args = parser.parse_args()

    api_key = os.getenv("NCBI_API_KEY")
    email = os.getenv("CROSSREF_EMAIL", "sophia-ai@example.com")

    with PubMedClient(api_key=api_key, email=email) as client:
        collector = MammographyCollector(
            client=client,
            output_dir=Path(args.output_dir),
        )

        if args.stats:
            stats = collector.get_stats()
            print("\n=== Collection Statistics ===")
            print(f"Total papers: {stats['total']}")
            print(f"Years collected: {len(stats['collected_years'])}")
            print("\nBy year:")
            for year, count in sorted(stats["by_year"].items()):
                print(f"  {year}: {count:,}")
        else:
            results = collector.collect_all(
                start_year=args.start_year,
                end_year=args.end_year,
                skip_existing=not args.force,
            )

            print("\n=== Collection Complete ===")
            total = sum(results.values())
            print(f"Total: {total:,} papers")


if __name__ == "__main__":
    main()
