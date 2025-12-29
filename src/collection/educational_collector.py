"""
Sophia AI: Educational Content Collector
===========================================
맘모그래피 기초 및 교육 자료 수집
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


class MammographyEducationalCollector:
    """맘모그래피 교육/기초 자료 수집기"""

    # 맘모그래피 기초/교육 검색 쿼리
    EDUCATIONAL_QUERIES = {
        "basics": """
            (mammography[MeSH] OR mammogram[tiab] OR "breast imaging"[tiab])
            AND (
                "review"[Publication Type] OR "guideline"[Publication Type]
                OR "practice guideline"[Publication Type]
                OR basics[tiab] OR fundamentals[tiab] OR principles[tiab]
                OR technique[tiab] OR techniques[tiab] OR methods[tiab]
                OR education[tiab] OR educational[tiab] OR training[tiab]
                OR tutorial[tiab] OR "how to"[tiab]
            )
            AND English[Language]
            NOT (animal[MeSH] NOT human[MeSH])
        """,

        "imaging_technique": """
            (mammography[MeSH] OR mammogram[tiab])
            AND (
                positioning[tiab] OR "image quality"[tiab]
                OR "quality assurance"[tiab] OR "quality control"[tiab]
                OR technique[tiab] OR protocol[tiab] OR protocols[tiab]
                OR "standard views"[tiab] OR "craniocaudal"[tiab] OR "mediolateral"[tiab]
                OR compression[tiab] OR exposure[tiab]
            )
            AND English[Language]
            NOT (animal[MeSH] NOT human[MeSH])
        """,

        "interpretation": """
            (mammography[MeSH] OR mammogram[tiab])
            AND (
                interpretation[tiab] OR reading[tiab] OR "image analysis"[tiab]
                OR "breast density"[tiab] OR "tissue density"[tiab]
                OR calcification[tiab] OR microcalcification[tiab]
                OR mass[tiab] OR masses[tiab] OR lesion[tiab]
                OR "BI-RADS"[tiab] OR "breast imaging reporting"[tiab]
            )
            AND (
                "review"[Publication Type] OR guideline[tiab] OR atlas[tiab]
            )
            AND English[Language]
            NOT (animal[MeSH] NOT human[MeSH])
        """,

        "guidelines": """
            (mammography[MeSH] OR mammogram[tiab] OR "breast imaging"[tiab])
            AND (
                "practice guideline"[Publication Type]
                OR guideline[tiab] OR guidelines[tiab]
                OR recommendation[tiab] OR recommendations[tiab]
                OR consensus[tiab] OR "best practice"[tiab]
                OR standard[tiab] OR standards[tiab]
            )
            AND (ACR[tiab] OR RSNA[tiab] OR "American College of Radiology"[tiab]
                 OR "society of breast imaging"[tiab] OR SBI[tiab]
                 OR screening[tiab] OR diagnosis[tiab])
            AND English[Language]
            NOT (animal[MeSH] NOT human[MeSH])
        """,

        "technology": """
            (mammography[MeSH] OR mammogram[tiab])
            AND (
                "digital mammography"[tiab] OR FFDM[tiab]
                OR "digital breast tomosynthesis"[tiab] OR DBT[tiab]
                OR "contrast enhanced"[tiab] OR CEM[tiab]
                OR technology[tiab] OR advances[tiab]
                OR "computer aided detection"[tiab] OR CAD[tiab]
                OR "artificial intelligence"[tiab] OR "machine learning"[tiab]
            )
            AND ("review"[Publication Type] OR overview[tiab])
            AND English[Language]
            NOT (animal[MeSH] NOT human[MeSH])
        """,
    }

    def __init__(
        self,
        client: PubMedClient,
        output_dir: Path = Path("data/raw/educational"),
    ):
        """
        Args:
            client: PubMed API 클라이언트
            output_dir: 출력 디렉토리
        """
        self.client = client
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Educational Collector initialized. Output: {self.output_dir}")

    def collect_by_category(
        self,
        category: str,
        max_results: int = 500,
    ) -> List[Paper]:
        """
        카테고리별 교육 자료 수집

        Args:
            category: 카테고리 (basics, imaging_technique, interpretation, guidelines, technology)
            max_results: 최대 수집 개수

        Returns:
            수집된 Paper 리스트
        """
        if category not in self.EDUCATIONAL_QUERIES:
            logger.error(f"Unknown category: {category}")
            return []

        output_file = self.output_dir / f"{category}.json"

        # 이미 수집된 경우 로드
        if output_file.exists():
            logger.info(f"Category '{category}' already collected, loading from file...")
            data = json.loads(output_file.read_text())
            return [Paper(**p) for p in data.get("papers", [])]

        logger.info(f"Collecting '{category}' materials...")

        query = self.EDUCATIONAL_QUERIES[category].strip()

        # 검색
        pmids = self.client.search(
            query=query,
            retmax=max_results,
        )

        if not pmids:
            logger.warning(f"No papers found for category: {category}")
            return []

        # 상세 정보 수집
        papers = self.client.fetch_details(pmids)

        # 저장
        self._save_category(category, papers)

        logger.info(f"Collected {len(papers)} papers for '{category}'")
        return papers

    def collect_all_categories(
        self,
        max_per_category: int = 500,
    ) -> Dict[str, int]:
        """
        모든 카테고리 수집

        Args:
            max_per_category: 카테고리당 최대 수집 개수

        Returns:
            카테고리별 수집 개수
        """
        results = {}

        categories = list(self.EDUCATIONAL_QUERIES.keys())
        logger.info(f"Collecting educational materials from {len(categories)} categories")

        for category in tqdm(categories, desc="Educational categories"):
            try:
                papers = self.collect_by_category(category, max_per_category)
                results[category] = len(papers)
            except Exception as e:
                logger.error(f"Failed to collect '{category}': {e}")
                results[category] = 0

        # 통계 출력
        total = sum(results.values())
        logger.info(f"\nEducational collection complete: {total} papers from {len(results)} categories")
        for cat, count in results.items():
            logger.info(f"  - {cat}: {count}")

        return results

    def _save_category(self, category: str, papers: List[Paper]):
        """카테고리별 데이터 저장"""
        output_file = self.output_dir / f"{category}.json"

        data = {
            "category": category,
            "collected_at": datetime.now().isoformat(),
            "count": len(papers),
            "papers": [paper.model_dump(exclude={"embedding"}) for paper in papers],
        }

        output_file.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, default=str)
        )
        logger.info(f"Saved {len(papers)} papers to {output_file}")


def main():
    """테스트 실행"""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Collect mammography educational materials")
    parser.add_argument(
        "--category",
        type=str,
        choices=["basics", "imaging_technique", "interpretation", "guidelines", "technology", "all"],
        default="all",
        help="Category to collect",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=500,
        help="Maximum results per category",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw/educational",
        help="Output directory",
    )

    args = parser.parse_args()

    # PubMed 클라이언트 초기화
    import os
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("NCBI_API_KEY")
    email = os.getenv("CROSSREF_EMAIL", "sophia-ai@example.com")

    with PubMedClient(api_key=api_key, email=email) as client:
        collector = MammographyEducationalCollector(
            client=client,
            output_dir=Path(args.output_dir),
        )

        if args.category == "all":
            results = collector.collect_all_categories(max_per_category=args.max_results)
        else:
            papers = collector.collect_by_category(args.category, args.max_results)
            results = {args.category: len(papers)}

        print("\n=== Collection Summary ===")
        for cat, count in results.items():
            print(f"{cat}: {count} papers")


if __name__ == "__main__":
    main()
