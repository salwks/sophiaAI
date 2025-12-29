"""
Sophia AI: BI-RADS Category Matcher
=====================================
BI-RADS 카테고리 직접 매칭 (의미 검색 대신 구조화 검색)
"""

import re
import sqlite3
from pathlib import Path
from typing import List, Optional, Tuple

from src.models import Paper


class BiradsCategoryMatcher:
    """BI-RADS 카테고리 구조화 매칭"""

    def __init__(self, db_path: Path = Path("data/index")):
        """
        Args:
            db_path: 데이터베이스 경로
        """
        self.db_path = Path(db_path)
        self.sqlite_path = self.db_path / "metadata.db"
        self.conn = sqlite3.connect(str(self.sqlite_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

    def parse_category(self, query: str) -> Optional[str]:
        """
        쿼리에서 BI-RADS 카테고리 추출

        Examples:
            "BI-RADS 5" → "5"
            "Category 5" → "5"
            "카테고리 5" → "5"
            "BI-RADS 5는 무엇인가요?" → "5"
            "Category 4A" → "4A"
            "BI-RADS category 0" → "0"

        Returns:
            카테고리 번호/문자 (없으면 None)
        """
        query = query.upper()

        # 패턴들
        patterns = [
            r'BI-?RADS[:\s]+CATEGORY[:\s]+([0-6][A-C]?)',  # BI-RADS Category 5
            r'BI-?RADS[:\s]+([0-6][A-C]?)',                # BI-RADS 5
            r'CATEGORY[:\s]+([0-6][A-C]?)',                # Category 5
            r'카테고리[:\s]+([0-6][A-C]?)',                # 카테고리 5
            r'CAT[\.\s]+([0-6][A-C]?)',                    # Cat. 5
        ]

        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                return match.group(1)

        return None

    def get_category_document(self, category: str) -> Optional[Paper]:
        """
        카테고리 번호로 BI-RADS 문서 직접 조회

        Args:
            category: 카테고리 (e.g., "5", "4A", "0")

        Returns:
            BI-RADS 카테고리 문서
        """
        cursor = self.conn.cursor()

        # Category X 형태로 검색
        category_pattern = f"%Category {category}%"

        cursor.execute("""
            SELECT pmid, doi, title, authors, journal, year, month,
                   abstract, full_content, citation_count, journal_if
            FROM papers
            WHERE pmid LIKE 'BIRADS_%'
              AND (title LIKE ? OR title LIKE ?)
            LIMIT 1
        """, (category_pattern, f"%Category {category}:%"))

        row = cursor.fetchone()
        if not row:
            return None

        # Paper 객체 생성
        paper = Paper(
            pmid=row['pmid'],
            doi=row['doi'],
            title=row['title'],
            authors=eval(row['authors']) if row['authors'] else [],
            journal=row['journal'] or "",
            year=row['year'] or 0,
            month=row['month'],
            abstract=row['abstract'] or "",
            full_content=row['full_content'],
            citation_count=row['citation_count'] or 0,
            journal_if=row['journal_if'],
        )

        return paper

    def get_all_categories(self) -> List[Paper]:
        """모든 BI-RADS 카테고리 문서 반환"""
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT pmid, doi, title, authors, journal, year, month,
                   abstract, full_content, citation_count, journal_if
            FROM papers
            WHERE pmid LIKE 'BIRADS_%'
              AND title LIKE '%Category%'
            ORDER BY title
        """)

        papers = []
        for row in cursor.fetchall():
            paper = Paper(
                pmid=row['pmid'],
                doi=row['doi'],
                title=row['title'],
                authors=eval(row['authors']) if row['authors'] else [],
                journal=row['journal'] or "",
                year=row['year'] or 0,
                month=row['month'],
                abstract=row['abstract'] or "",
                full_content=row['full_content'],
                citation_count=row['citation_count'] or 0,
                journal_if=row['journal_if'],
            )
            papers.append(paper)

        return papers

    def search(self, query: str) -> Optional[Paper]:
        """
        쿼리에서 BI-RADS 카테고리를 파싱하여 직접 조회

        Args:
            query: 사용자 쿼리

        Returns:
            매칭된 BI-RADS 문서 (없으면 None)
        """
        category = self.parse_category(query)
        if not category:
            return None

        return self.get_category_document(category)

    def is_birads_query(self, query: str) -> bool:
        """쿼리가 BI-RADS 카테고리 질문인지 판단"""
        query_upper = query.upper()

        # BI-RADS 키워드 체크
        birads_keywords = ['BI-RADS', 'BIRADS', 'CATEGORY', '카테고리']
        has_keyword = any(kw in query_upper for kw in birads_keywords)

        # 카테고리 번호 체크
        has_category = self.parse_category(query) is not None

        return has_keyword and has_category


def test():
    """테스트"""
    matcher = BiradsCategoryMatcher()

    test_queries = [
        "BI-RADS 5는 무엇인가요?",
        "Category 5",
        "BI-RADS Category 5",
        "카테고리 5",
        "What is BI-RADS 5?",
        "Category 4A",
        "BI-RADS 0",
        "breast cancer screening",  # 일반 질문
    ]

    print("=== Category Parsing Test ===")
    for query in test_queries:
        category = matcher.parse_category(query)
        is_birads = matcher.is_birads_query(query)
        print(f"Query: {query:40} → Category: {category}, IsBIRADS: {is_birads}")

    print("\n=== Direct Retrieval Test ===")
    query = "BI-RADS 5는 무엇인가요?"
    paper = matcher.search(query)
    if paper:
        print(f"✅ Found: {paper.title}")
        print(f"   PMID: {paper.pmid}")
        print(f"   Content length: {len(paper.full_content or '')}")
    else:
        print("❌ Not found")

    print("\n=== All Categories ===")
    categories = matcher.get_all_categories()
    print(f"Total BI-RADS category documents: {len(categories)}")
    for cat in categories[:10]:
        print(f"  - {cat.title[:60]}")


if __name__ == "__main__":
    test()
