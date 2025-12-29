"""
MARIA-Mammo: Indexing Pipeline
==============================
전체 인덱싱 파이프라인 (임베딩 + DB 저장)
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Set

from tqdm import tqdm

from src.indexing.database import DatabaseManager
from src.indexing.embedder import PaperEmbedder
from src.models import Paper

logger = logging.getLogger(__name__)


class IndexingPipeline:
    """인덱싱 파이프라인"""

    def __init__(
        self,
        embedder: Optional[PaperEmbedder] = None,
        db: Optional[DatabaseManager] = None,
        db_path: Path = Path("data/index"),
    ):
        """
        Args:
            embedder: 임베딩 생성기 (없으면 자동 생성)
            db: 데이터베이스 매니저 (없으면 자동 생성)
            db_path: 데이터베이스 경로
        """
        self.embedder = embedder or PaperEmbedder()
        self.db = db or DatabaseManager(db_path)

        logger.info("Indexing pipeline initialized")

    def index_papers(
        self,
        papers: List[Paper],
        batch_size: int = 32,
        checkpoint_interval: int = 1000,
    ):
        """
        논문 인덱싱

        Args:
            papers: 논문 리스트
            batch_size: 임베딩 배치 크기
            checkpoint_interval: 체크포인트 저장 간격
        """
        logger.info(f"Indexing {len(papers)} papers...")

        # 이미 인덱싱된 PMID 확인
        existing_pmids = set(self.db.get_all_pmids())
        new_papers = [p for p in papers if p.pmid not in existing_pmids]

        if not new_papers:
            logger.info("All papers already indexed")
            return

        logger.info(f"New papers to index: {len(new_papers)}")

        # 배치 처리
        total_batches = (len(new_papers) + batch_size - 1) // batch_size

        for i in tqdm(range(0, len(new_papers), batch_size), total=total_batches, desc="Indexing"):
            batch_papers = new_papers[i : i + batch_size]

            # 임베딩 생성
            embeddings = self.embedder.embed_batch(batch_papers, show_progress=False)

            # DB 저장
            self.db.insert_papers(batch_papers, embeddings)

            # 체크포인트
            if (i + batch_size) % checkpoint_interval == 0:
                logger.info(f"Checkpoint: {i + batch_size}/{len(new_papers)} papers indexed")

        logger.info(f"Indexing complete: {len(new_papers)} papers")

    def index_from_file(self, file_path: Path):
        """
        파일에서 논문 로드 및 인덱싱

        Args:
            file_path: 처리된 논문 파일 경로
        """
        file_path = Path(file_path)

        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return

        logger.info(f"Loading papers from {file_path}...")
        data = json.loads(file_path.read_text())
        papers = [Paper(**p) for p in data.get("papers", [])]

        logger.info(f"Loaded {len(papers)} papers")
        self.index_papers(papers)

    def index_incremental(self, processed_dir: Path = Path("data/processed")):
        """
        증분 인덱싱 (새 논문만)

        Args:
            processed_dir: 처리된 데이터 디렉토리
        """
        papers_file = Path(processed_dir) / "papers.json"
        self.index_from_file(papers_file)

    def reindex_all(self, processed_dir: Path = Path("data/processed")):
        """
        전체 재인덱싱

        Args:
            processed_dir: 처리된 데이터 디렉토리
        """
        logger.info("Clearing existing index...")
        self.db.clear()

        papers_file = Path(processed_dir) / "papers.json"
        self.index_from_file(papers_file)

    def get_stats(self):
        """인덱싱 통계"""
        return self.db.get_stats()


def main():
    """CLI 실행"""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Index processed papers")
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="data/processed",
        help="Processed data directory",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default="data/index",
        help="Index output directory",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Full reindexing (clear existing)",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Incremental indexing (new papers only)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show index statistics",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding batch size",
    )

    args = parser.parse_args()

    db = DatabaseManager(Path(args.index_dir))

    if args.stats:
        stats = db.get_stats()
        print("\n=== Index Statistics ===")
        print(f"Total papers: {stats['total_papers']:,}")
        print(f"Total vectors: {stats['total_vectors']:,}")
        print("\nBy year:")
        for year, count in sorted(stats["by_year"].items()):
            print(f"  {year}: {count:,}")
        return

    embedder = PaperEmbedder()
    pipeline = IndexingPipeline(embedder=embedder, db=db)

    if args.full:
        pipeline.reindex_all(Path(args.processed_dir))
    else:
        pipeline.index_incremental(Path(args.processed_dir))

    # 최종 통계
    stats = pipeline.get_stats()
    print("\n=== Indexing Complete ===")
    print(f"Total papers: {stats['total_papers']:,}")
    print(f"Total vectors: {stats['total_vectors']:,}")


if __name__ == "__main__":
    main()
