"""
BI-RADS Mammography Indexer
============================
추출된 Mammography 챕터를 데이터베이스에 인덱싱
"""

import json
import logging
import re
import sqlite3
from pathlib import Path
from typing import List, Dict
import sys

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import Paper
from src.indexing.database import DatabaseManager
from src.indexing.embedder import PaperEmbedder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def clean_section_heading(heading: str) -> str:
    """
    섹션 헤딩 정리 (TOC 페이지 번호 제거 등)

    Args:
        heading: 원본 헤딩

    Returns:
        정리된 헤딩
    """
    # 페이지 번호 패턴 제거 (숫자만 있거나 점이 많은 경우)
    heading = re.sub(r'\s+\.\s*\d+$', '', heading)  # "... 123" 제거
    heading = re.sub(r'\.{3,}', '', heading)  # "..." 제거
    heading = re.sub(r'\s{2,}', ' ', heading)  # 다중 공백 제거

    return heading.strip()


def is_meaningful_section(section: Dict) -> bool:
    """
    의미 있는 섹션인지 판단

    Args:
        section: 섹션 데이터

    Returns:
        의미 있는 섹션이면 True
    """
    heading = section["heading"]
    content = section["content"]

    # 너무 짧은 섹션 제외
    if len(content) < 100:
        return False

    # TOC만 있는 섹션 제외
    if "TABLE OF CONTENTS" in heading.upper():
        return False

    # 주로 페이지 번호만 있는 섹션 제외
    lines = content.split('\n')
    non_empty_lines = [l for l in lines if l.strip()]
    if len(non_empty_lines) < 3:
        return False

    return True


def create_birads_papers(sections: List[Dict], metadata: Dict, images_dir: Path) -> List[Paper]:
    """
    섹션을 Paper 객체로 변환

    Args:
        sections: 섹션 리스트
        metadata: 메타데이터
        images_dir: 이미지 디렉토리

    Returns:
        Paper 리스트
    """
    papers = []

    # 주요 섹션 패턴
    category_pattern = re.compile(r'Category\s+(\d[A-C]?)', re.IGNORECASE)

    for idx, section in enumerate(sections):
        if not is_meaningful_section(section):
            continue

        heading = clean_section_heading(section["heading"])
        content = section["content"]

        # Category 섹션 감지
        category_match = category_pattern.search(heading)

        # PMID 생성
        if category_match:
            category_num = category_match.group(1)
            pmid = f"BIRADS_MAMMO_CATEGORY_{category_num.upper()}"
            title = f"BI-RADS Mammography - Category {category_num}"
        elif "APPENDIX A" in heading.upper() and "VIEW" in heading.upper():
            pmid = f"BIRADS_MAMMO_APPENDIX_A_VIEWS"
            title = "BI-RADS Mammography - Appendix A: Mammographic Views"
        elif "APPENDIX" in heading.upper():
            appendix_match = re.search(r'APPENDIX\s+([A-Z])', heading.upper())
            if appendix_match:
                letter = appendix_match.group(1)
                pmid = f"BIRADS_MAMMO_APPENDIX_{letter}"
                title = f"BI-RADS Mammography - Appendix {letter}"
            else:
                pmid = f"BIRADS_MAMMO_SECTION_{idx:03d}"
                title = f"BI-RADS Mammography - {heading}"
        elif heading.startswith("I.") or heading.startswith("II.") or heading.startswith("III.") or heading.startswith("IV."):
            # 로마 숫자 섹션
            roman_match = re.match(r'^(I{1,3}V?X?)\.\s+(.+)$', heading)
            if roman_match:
                roman = roman_match.group(1)
                section_title = roman_match.group(2)
                pmid = f"BIRADS_MAMMO_SECTION_{roman}"
                title = f"BI-RADS Mammography - {roman}. {section_title}"
            else:
                pmid = f"BIRADS_MAMMO_SECTION_{idx:03d}"
                title = f"BI-RADS Mammography - {heading}"
        else:
            # 일반 섹션
            pmid = f"BIRADS_MAMMO_SECTION_{idx:03d}"
            title = f"BI-RADS Mammography - {heading}"

        # 중복 PMID 방지
        if any(p.pmid == pmid for p in papers):
            pmid = f"{pmid}_{idx:03d}"

        # 이미지 경로 추출 (해당 섹션의 페이지에서)
        # metadata에서 섹션에 해당하는 페이지 찾기
        section_images = []

        # Paper 생성
        paper = Paper(
            pmid=pmid,
            doi=None,
            title=title,
            authors=["American College of Radiology"],
            journal="ACR BI-RADS Atlas v2025",
            year=2025,
            month="01",
            abstract=content[:500] + "..." if len(content) > 500 else content,
            full_content=content,
            citation_count=0,
            journal_if=None,
            modality=["Mammography"],
            pathology=[],
            study_type="Guideline",
        )

        papers.append(paper)
        logger.info(f"Created paper: {pmid} - {title[:60]}...")

    return papers


def index_birads_mammography(
    sections_file: Path = Path("data/raw/birads_2025/mammography_sections.json"),
    metadata_file: Path = Path("data/raw/birads_2025/mammography_metadata.json"),
    images_dir: Path = Path("data/raw/birads_2025/images"),
    db_path: Path = Path("data/index"),
):
    """
    BI-RADS Mammography 챕터를 데이터베이스에 인덱싱

    Args:
        sections_file: 섹션 JSON 파일
        metadata_file: 메타데이터 JSON 파일
        images_dir: 이미지 디렉토리
        db_path: 데이터베이스 경로
    """
    logger.info("Loading sections and metadata...")

    sections = json.loads(sections_file.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))

    logger.info(f"Total sections: {len(sections)}")

    # Paper 객체 생성
    logger.info("Creating Paper objects from sections...")
    papers = create_birads_papers(sections, metadata, images_dir)
    logger.info(f"Created {len(papers)} papers from {len(sections)} sections")

    # 데이터베이스에 저장
    logger.info("Saving to database...")
    db = DatabaseManager(db_path)

    # 임베딩 생성
    logger.info("Generating embeddings...")
    embedder = PaperEmbedder()

    for paper in papers:
        # 임베딩 생성
        paper.embedding = embedder.embed_paper(paper)

    # Papers 테이블과 벡터 인덱스에 동시 저장
    db.insert_papers(papers)
    logger.info(f"✅ Saved {len(papers)} BI-RADS papers to database and vector index")

    # 통계 출력
    stats = db.get_stats()
    logger.info(f"\n=== Database Statistics ===")
    logger.info(f"Total papers: {stats['total_papers']}")
    logger.info(f"Total vectors: {stats['total_vectors']}")
    logger.info(f"BI-RADS papers: {len(papers)}")

    return papers


def main():
    """메인 실행"""
    logger.info("=== BI-RADS Mammography Indexing ===\n")

    papers = index_birads_mammography()

    logger.info(f"\n✅ Indexing complete!")
    logger.info(f"   Total BI-RADS papers: {len(papers)}")

    # 샘플 출력
    logger.info("\n=== Sample Papers ===")
    for i, paper in enumerate(papers[:10]):
        logger.info(f"  [{i+1}] {paper.pmid}")
        logger.info(f"      {paper.title}")
        logger.info(f"      Content length: {len(paper.full_content):,} chars")


if __name__ == "__main__":
    main()
