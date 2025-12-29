#!/usr/bin/env python3
"""
BI-RADS 2025 가이드라인 인덱싱
==============================
섹션별 청킹 후 RAG 인덱스에 추가
"""

import json
import re
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 경로 설정
BIRADS_MD = Path("/Users/sinjaeho/Documents/works/sophyAI/MARIA_KNOWLEDGE/DOCS/GUIDELINES/converted/acr_bi_rads_2025_FIXED.md")
IMAGES_DIR = Path("/Users/sinjaeho/Documents/works/sophyAI/MARIA_KNOWLEDGE/DOCS/GUIDELINES/images")
OUTPUT_DIR = Path("data/processed")


def chunk_by_headers(content: str, min_chunk_size: int = 200) -> List[Dict]:
    """
    헤더 기준으로 문서를 청킹

    Args:
        content: 마크다운 콘텐츠
        min_chunk_size: 최소 청크 크기 (문자)

    Returns:
        청크 리스트
    """
    chunks = []

    # 헤더 패턴 (##, ###, ####)
    header_pattern = re.compile(r'^(#{2,4})\s+(.+)$', re.MULTILINE)

    # 모든 헤더 위치 찾기
    headers = list(header_pattern.finditer(content))

    if not headers:
        # 헤더 없으면 전체를 하나의 청크로
        return [{"title": "BI-RADS 2025", "content": content, "images": []}]

    for i, match in enumerate(headers):
        level = len(match.group(1))
        title = match.group(2).strip()
        start = match.end()

        # 다음 헤더까지 또는 문서 끝까지
        if i + 1 < len(headers):
            end = headers[i + 1].start()
        else:
            end = len(content)

        section_content = content[start:end].strip()

        # 너무 짧은 섹션은 건너뛰기
        if len(section_content) < min_chunk_size:
            continue

        # 이미지 참조 추출
        images = re.findall(r'!\[\]\((images/[^)]+)\)', section_content)

        # 청크 생성
        chunk = {
            "title": title,
            "content": section_content,
            "level": level,
            "images": images,
        }
        chunks.append(chunk)

    return chunks


def create_birads_papers(chunks: List[Dict]) -> List[Dict]:
    """
    청크를 Paper 형식으로 변환
    """
    papers = []

    for i, chunk in enumerate(chunks):
        # 고유 ID 생성
        chunk_id = hashlib.md5(chunk["title"].encode()).hexdigest()[:8]
        pmid = f"BIRADS_2025_{chunk_id}"

        # 이미지 경로를 절대 경로로
        image_paths = [str(IMAGES_DIR / img.replace("images/", "")) for img in chunk["images"]]

        paper = {
            "pmid": pmid,
            "doi": None,
            "title": f"BI-RADS 2025: {chunk['title']}",
            "authors": ["ACR Committee on BI-RADS"],
            "journal": "ACR BI-RADS Atlas",
            "journal_abbrev": "BI-RADS",
            "year": 2025,
            "month": 12,
            "abstract": chunk["content"][:2000],  # 처음 2000자
            "mesh_terms": ["BI-RADS", "Breast Imaging", "Radiology Guidelines"],
            "keywords": extract_keywords(chunk["title"]),
            "publication_types": ["Guideline", "Practice Guideline"],
            "modality": detect_modality(chunk["content"]),
            "pathology": [],
            "study_type": "guideline",
            "population": None,
            "citation_count": 0,
            "journal_if": None,
            "pmc_id": None,
            "source": "birads",
            "images": image_paths,
            "full_content": chunk["content"],
            "section_level": chunk.get("level", 2),
            "created_at": datetime.now().isoformat(),
            "updated_at": None,
        }
        papers.append(paper)

    return papers


def extract_keywords(title: str) -> List[str]:
    """제목에서 키워드 추출"""
    keywords = ["BI-RADS", "Breast Imaging"]

    # 카테고리 감지
    if re.search(r'category\s*[0-6]', title, re.I):
        match = re.search(r'category\s*([0-6])', title, re.I)
        if match:
            keywords.append(f"BI-RADS Category {match.group(1)}")

    # 모달리티 감지
    modalities = {
        "mammography": "Mammography",
        "ultrasound": "Ultrasound",
        "mri": "MRI",
        "cem": "Contrast Enhanced Mammography",
    }
    for key, val in modalities.items():
        if key in title.lower():
            keywords.append(val)

    # 소견 감지
    findings = ["mass", "calcification", "density", "asymmetry", "distortion"]
    for finding in findings:
        if finding in title.lower():
            keywords.append(finding.capitalize())

    return keywords


def detect_modality(content: str) -> List[str]:
    """콘텐츠에서 모달리티 감지"""
    modalities = []
    content_lower = content.lower()

    if "mammograph" in content_lower or "ffdm" in content_lower:
        modalities.append("FFDM")
    if "tomosynthe" in content_lower or "dbt" in content_lower:
        modalities.append("DBT")
    if "ultrasound" in content_lower:
        modalities.append("US")
    if "mri" in content_lower or "magnetic resonance" in content_lower:
        modalities.append("MRI")
    if "contrast enhanced" in content_lower or "cem" in content_lower:
        modalities.append("CEM")

    return modalities if modalities else ["FFDM"]


def main():
    """메인 실행"""
    logger.info("Loading BI-RADS markdown...")

    if not BIRADS_MD.exists():
        logger.error(f"File not found: {BIRADS_MD}")
        return

    content = BIRADS_MD.read_text(encoding="utf-8")
    logger.info(f"Loaded {len(content):,} characters")

    # 청킹
    logger.info("Chunking by headers...")
    chunks = chunk_by_headers(content, min_chunk_size=300)
    logger.info(f"Created {len(chunks)} chunks")

    # Paper 형식으로 변환
    logger.info("Converting to Paper format...")
    birads_papers = create_birads_papers(chunks)

    # 기존 papers.json 로드
    papers_file = OUTPUT_DIR / "papers.json"
    if papers_file.exists():
        data = json.loads(papers_file.read_text())
        existing_papers = data.get("papers", [])

        # 기존 BIRADS 제거 (업데이트 시)
        existing_papers = [p for p in existing_papers if not p.get("pmid", "").startswith("BIRADS_")]

        # 새 BIRADS 추가
        existing_papers.extend(birads_papers)

        data["papers"] = existing_papers
        data["updated_at"] = datetime.now().isoformat()
        data["sources"] = list(set(data.get("sources", []) + ["birads"]))
    else:
        data = {
            "papers": birads_papers,
            "created_at": datetime.now().isoformat(),
            "sources": ["birads"],
        }

    # 저장
    papers_file.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str))
    logger.info(f"Saved {len(birads_papers)} BI-RADS chunks to {papers_file}")
    logger.info(f"Total papers: {len(data['papers'])}")

    # 통계
    with_images = sum(1 for p in birads_papers if p.get("images"))
    total_images = sum(len(p.get("images", [])) for p in birads_papers)
    logger.info(f"Chunks with images: {with_images}")
    logger.info(f"Total image references: {total_images}")

    print(f"\n=== BI-RADS Indexing Complete ===")
    print(f"Chunks created: {len(birads_papers)}")
    print(f"With images: {with_images}")
    print(f"Total papers now: {len(data['papers'])}")
    print(f"\nRun 'uv run python scripts/index.py' to update vector index")


if __name__ == "__main__":
    main()
