"""
BI-RADS Mammography Chapter Extractor
========================================
ACR BI-RADS 2025 PDF에서 Mammography 챕터 추출 (텍스트, 이미지, 테이블)
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional
import fitz  # PyMuPDF
from PIL import Image
import io

def extract_images_from_page(page: fitz.Page, page_num: int, output_dir: Path) -> List[Dict]:
    """
    페이지에서 이미지 추출

    Args:
        page: PDF 페이지
        page_num: 페이지 번호
        output_dir: 이미지 저장 디렉토리

    Returns:
        이미지 정보 리스트
    """
    images = []
    image_list = page.get_images(full=True)

    for img_index, img in enumerate(image_list):
        xref = img[0]

        try:
            base_image = page.parent.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            # 이미지 저장
            image_filename = f"page_{page_num:04d}_img_{img_index:02d}.{image_ext}"
            image_path = output_dir / "images" / image_filename
            image_path.parent.mkdir(parents=True, exist_ok=True)

            with open(image_path, "wb") as f:
                f.write(image_bytes)

            # 이미지 크기 확인
            img_pil = Image.open(io.BytesIO(image_bytes))
            width, height = img_pil.size

            images.append({
                "filename": image_filename,
                "path": str(image_path.relative_to(output_dir)),
                "width": width,
                "height": height,
                "format": image_ext,
            })

        except Exception as e:
            print(f"  Warning: Failed to extract image {img_index} from page {page_num}: {e}")

    return images


def extract_tables_from_page(page: fitz.Page) -> List[Dict]:
    """
    페이지에서 테이블 추출 (간단한 구조 기반)

    Args:
        page: PDF 페이지

    Returns:
        테이블 정보 리스트
    """
    tables = []

    # PyMuPDF의 get_text("dict")로 블록 단위 분석
    blocks = page.get_text("dict")["blocks"]

    # 테이블은 보통 여러 줄의 구조화된 텍스트로 표현됨
    # 간단히 연속된 라인들을 테이블로 간주
    table_candidates = []
    current_table = []

    for block in blocks:
        if block["type"] == 0:  # Text block
            lines = block.get("lines", [])
            if len(lines) > 0:
                # 여러 span이 일정 간격으로 배열되면 테이블로 간주
                for line in lines:
                    spans = line.get("spans", [])
                    if len(spans) >= 2:  # 2개 이상의 컬럼
                        current_table.append(" | ".join([span["text"] for span in spans]))
                    elif current_table:
                        # 테이블 종료
                        if len(current_table) >= 3:  # 최소 3줄 이상
                            table_candidates.append(current_table[:])
                        current_table = []

    # 마지막 테이블 추가
    if len(current_table) >= 3:
        table_candidates.append(current_table)

    # 테이블 정보 생성
    for idx, table in enumerate(table_candidates):
        tables.append({
            "index": idx,
            "rows": len(table),
            "content": "\n".join(table),
        })

    return tables


def extract_mammography_chapter(
    pdf_path: str,
    start_page: int = 18,
    end_page: int = 189,
    output_dir: Path = Path("data/raw/birads_2025"),
) -> Dict:
    """
    Mammography 챕터 추출 (텍스트, 이미지, 테이블)

    Args:
        pdf_path: PDF 파일 경로
        start_page: 시작 페이지 (1-based)
        end_page: 종료 페이지 (1-based)
        output_dir: 출력 디렉토리

    Returns:
        추출 결과 메타데이터
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)

    # 0-based 인덱스로 변환
    start_idx = start_page - 1
    end_idx = end_page

    print(f"Extracting Mammography chapter: Pages {start_page}-{end_page} ({end_idx - start_idx} pages)")

    chapter_data = {
        "title": "ACR BI-RADS v2025 - Mammography",
        "source": pdf_path,
        "pages": {
            "start": start_page,
            "end": end_page,
            "total": end_idx - start_idx,
        },
        "sections": [],
        "total_images": 0,
        "total_tables": 0,
    }

    full_text = []

    for page_num in range(start_idx, end_idx):
        actual_page_num = page_num + 1
        page = doc[page_num]

        print(f"Processing page {actual_page_num}/{end_page}...", end="\r")

        # 텍스트 추출
        text = page.get_text()

        # 이미지 추출
        images = extract_images_from_page(page, actual_page_num, output_dir)

        # 테이블 추출
        tables = extract_tables_from_page(page)

        # 섹션 정보 저장
        section = {
            "page": actual_page_num,
            "text": text,
            "text_length": len(text),
            "images": images,
            "tables": tables,
        }

        chapter_data["sections"].append(section)
        chapter_data["total_images"] += len(images)
        chapter_data["total_tables"] += len(tables)

        full_text.append(text)

    print(f"\n✅ Extraction complete!")

    # 전체 텍스트 저장
    full_text_content = "\n\n".join(full_text)
    text_file = output_dir / "mammography_full_text.txt"
    text_file.write_text(full_text_content, encoding="utf-8")

    # 메타데이터 저장
    metadata_file = output_dir / "mammography_metadata.json"
    metadata_file.write_text(
        json.dumps(chapter_data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"\n📄 Full text: {text_file}")
    print(f"📊 Metadata: {metadata_file}")
    print(f"🖼️  Total images: {chapter_data['total_images']}")
    print(f"📋 Total tables: {chapter_data['total_tables']}")
    print(f"📝 Total text length: {len(full_text_content):,} characters")

    doc.close()

    return chapter_data


def segment_by_headings(text: str) -> List[Dict]:
    """
    텍스트를 제목 기준으로 섹션 분할

    Args:
        text: 전체 텍스트

    Returns:
        섹션 리스트
    """
    sections = []

    # BI-RADS 문서의 주요 헤딩 패턴
    # I., II., III., A., B., 1., 2., Category 0-6 등
    heading_patterns = [
        r'^(I{1,3}V?X?\.)\s+(.+)$',  # I., II., III., IV.
        r'^([A-Z]\.)\s+(.+)$',        # A., B., C.
        r'^(\d+\.)\s+(.+)$',          # 1., 2., 3.
        r'^(Category\s+\d[A-C]?)[:\.]?\s*(.*)$',  # Category 0-6
        r'^(APPENDIX\s+[A-Z])[:\-]?\s*(.+)$',     # APPENDIX A
    ]

    lines = text.split('\n')
    current_section = {
        "heading": "Introduction",
        "level": 0,
        "content": [],
    }

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 헤딩 매칭 확인
        is_heading = False
        for pattern in heading_patterns:
            match = re.match(pattern, line)
            if match:
                # 이전 섹션 저장
                if current_section["content"]:
                    current_section["content"] = "\n".join(current_section["content"])
                    sections.append(current_section)

                # 새 섹션 시작
                heading_marker = match.group(1)
                heading_text = match.group(2) if len(match.groups()) > 1 else ""

                current_section = {
                    "heading": f"{heading_marker} {heading_text}".strip(),
                    "level": 1,
                    "content": [],
                }
                is_heading = True
                break

        if not is_heading:
            current_section["content"].append(line)

    # 마지막 섹션 추가
    if current_section["content"]:
        current_section["content"] = "\n".join(current_section["content"])
        sections.append(current_section)

    return sections


def main():
    """메인 실행"""
    pdf_path = "/Users/sinjaeho/Documents/works/sophyAI/MARIA_KNOWLEDGE/DOCS/GUIDELINES/acr_bi_rads_2025_with_guides.pdf"
    output_dir = Path("data/raw/birads_2025")

    # Mammography 챕터 추출
    chapter_data = extract_mammography_chapter(
        pdf_path=pdf_path,
        start_page=18,
        end_page=189,
        output_dir=output_dir,
    )

    # 전체 텍스트 로드
    full_text = (output_dir / "mammography_full_text.txt").read_text(encoding="utf-8")

    # 섹션 분할
    print("\n\n=== Segmenting by headings ===")
    sections = segment_by_headings(full_text)
    print(f"✅ Found {len(sections)} sections")

    # 섹션 정보 저장
    sections_file = output_dir / "mammography_sections.json"
    sections_file.write_text(
        json.dumps(sections, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"📋 Sections: {sections_file}")

    # 주요 섹션 미리보기
    print("\n=== Major Sections ===")
    for i, section in enumerate(sections[:20]):
        heading = section["heading"]
        content_preview = section["content"][:100].replace('\n', ' ')
        print(f"  [{i+1}] {heading}")
        print(f"      {content_preview}...")

    print(f"\n✅ Extraction complete! Total sections: {len(sections)}")


if __name__ == "__main__":
    main()
