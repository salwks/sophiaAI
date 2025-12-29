"""
Sophia AI: Processing Pipeline
================================
전체 데이터 처리 파이프라인
"""

import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from src.collection.enricher import MetadataEnricher
from src.models import CollectionStats, Paper
from src.processing.classifier import PaperClassifier
from src.processing.cleaner import DataCleaner

logger = logging.getLogger(__name__)


class ProcessingPipeline:
    """전체 처리 파이프라인"""

    def __init__(
        self,
        input_dir: Path = Path("data/raw"),
        output_dir: Path = Path("data/processed"),
    ):
        """
        Args:
            input_dir: 원본 데이터 디렉토리
            output_dir: 처리된 데이터 출력 디렉토리
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.cleaner = DataCleaner(
            min_abstract_length=50,
            exclude_no_abstract=False,
            exclude_certain_types=True,
        )
        self.classifier = PaperClassifier()
        self.enricher = MetadataEnricher()

    def load_raw_papers(self) -> List[Paper]:
        """원본 데이터 로드"""
        all_papers = []

        for file in sorted(self.input_dir.glob("*.json")):
            if not file.stem.isdigit():
                continue

            try:
                data = json.loads(file.read_text())
                papers = [Paper(**p) for p in data.get("papers", [])]
                all_papers.extend(papers)
                logger.info(f"Loaded {len(papers)} papers from {file.name}")
            except Exception as e:
                logger.error(f"Failed to load {file}: {e}")

        logger.info(f"Total loaded: {len(all_papers)} papers")
        return all_papers

    def run(self, enrich_apis: bool = False) -> Dict:
        """
        전체 파이프라인 실행

        Args:
            enrich_apis: 외부 API로 메타데이터 보강 여부

        Returns:
            처리 결과 통계
        """
        logger.info("=" * 60)
        logger.info("Starting processing pipeline")
        logger.info("=" * 60)

        # 1. 데이터 로드
        logger.info("\n[1/4] Loading raw data...")
        papers = self.load_raw_papers()

        if not papers:
            logger.error("No papers found!")
            return {"error": "No papers found"}

        # 2. 데이터 정제
        logger.info("\n[2/4] Cleaning data...")
        papers = self.cleaner.clean_all(papers)

        # 3. 분류
        logger.info("\n[3/4] Classifying papers...")
        papers = self.classifier.classify_batch(papers)

        # 4. 메타데이터 보강
        logger.info("\n[4/4] Enriching metadata...")
        papers = self.enricher.enrich_papers(papers, use_apis=enrich_apis)

        # 결과 저장
        self._save_papers(papers)

        # 통계 생성 및 저장
        stats = self._generate_stats(papers)
        self._save_stats(stats)

        logger.info("\n" + "=" * 60)
        logger.info(f"Processing complete: {len(papers)} papers")
        logger.info("=" * 60)

        return {
            "total_papers": len(papers),
            "output_file": str(self.output_dir / "papers.json"),
            "stats_file": str(self.output_dir / "stats.json"),
        }

    def _save_papers(self, papers: List[Paper]):
        """처리된 논문 저장"""
        output_file = self.output_dir / "papers.json"

        data = {
            "processed_at": datetime.now().isoformat(),
            "total_count": len(papers),
            "papers": [paper.model_dump(exclude={"embedding"}) for paper in papers],
        }

        output_file.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, default=str)
        )
        logger.info(f"Saved {len(papers)} papers to {output_file}")

    def _generate_stats(self, papers: List[Paper]) -> Dict:
        """통계 생성"""
        stats = {
            "generated_at": datetime.now().isoformat(),
            "total_papers": len(papers),
            "by_year": {},
            "by_modality": {},
            "by_pathology": {},
            "by_study_type": {},
            "by_population": {},
            "by_journal": {},
            "top_journals": [],
            "with_abstract": 0,
            "with_doi": 0,
            "with_pmc": 0,
        }

        journal_counter = Counter()

        for paper in papers:
            # 연도별
            year = paper.year
            stats["by_year"][year] = stats["by_year"].get(year, 0) + 1

            # Modality별
            for m in paper.modality:
                stats["by_modality"][m] = stats["by_modality"].get(m, 0) + 1

            # Pathology별
            for p in paper.pathology:
                stats["by_pathology"][p] = stats["by_pathology"].get(p, 0) + 1

            # Study Type별
            if paper.study_type:
                stats["by_study_type"][paper.study_type] = (
                    stats["by_study_type"].get(paper.study_type, 0) + 1
                )

            # Population별
            if paper.population:
                stats["by_population"][paper.population] = (
                    stats["by_population"].get(paper.population, 0) + 1
                )

            # 저널별
            journal_counter[paper.journal] += 1

            # 기타 통계
            if paper.abstract:
                stats["with_abstract"] += 1
            if paper.doi:
                stats["with_doi"] += 1
            if paper.pmc_id:
                stats["with_pmc"] += 1

        # 상위 20개 저널
        stats["top_journals"] = journal_counter.most_common(20)
        stats["by_journal"] = dict(journal_counter.most_common(100))

        return stats

    def _save_stats(self, stats: Dict):
        """통계 저장"""
        stats_file = self.output_dir / "stats.json"
        stats_file.write_text(json.dumps(stats, ensure_ascii=False, indent=2))
        logger.info(f"Saved stats to {stats_file}")

    def load_processed_papers(self) -> List[Paper]:
        """처리된 논문 로드"""
        papers_file = self.output_dir / "papers.json"

        if not papers_file.exists():
            logger.warning("No processed papers found. Run pipeline first.")
            return []

        data = json.loads(papers_file.read_text())
        papers = [Paper(**p) for p in data.get("papers", [])]

        logger.info(f"Loaded {len(papers)} processed papers")
        return papers


def main():
    """CLI 실행"""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Process collected papers")
    parser.add_argument(
        "--input-dir", type=str, default="data/raw", help="Input directory"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/processed", help="Output directory"
    )
    parser.add_argument(
        "--enrich", action="store_true", help="Enrich with external APIs (slow)"
    )
    parser.add_argument("--stats-only", action="store_true", help="Show stats only")

    args = parser.parse_args()

    pipeline = ProcessingPipeline(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
    )

    if args.stats_only:
        stats_file = Path(args.output_dir) / "stats.json"
        if stats_file.exists():
            stats = json.loads(stats_file.read_text())
            print("\n=== Processing Statistics ===")
            print(f"Total papers: {stats['total_papers']:,}")
            print(f"\nBy year:")
            for year, count in sorted(stats["by_year"].items()):
                print(f"  {year}: {count:,}")
            print(f"\nBy modality:")
            for m, count in stats["by_modality"].items():
                print(f"  {m}: {count:,}")
            print(f"\nTop journals:")
            for journal, count in stats["top_journals"][:10]:
                print(f"  {journal}: {count:,}")
        else:
            print("No stats file found. Run pipeline first.")
    else:
        result = pipeline.run(enrich_apis=args.enrich)
        print("\n=== Processing Complete ===")
        print(f"Total papers: {result.get('total_papers', 0):,}")
        print(f"Output: {result.get('output_file')}")
        print(f"Stats: {result.get('stats_file')}")


if __name__ == "__main__":
    main()
