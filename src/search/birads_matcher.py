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

    # 단독 사용시에만 매칭해야 하는 짧은/일반적인 키워드들
    # (다른 단어의 일부로 나타날 수 있어 False positive 유발)
    STRICT_MATCH_KEYWORDS = {
        '점',       # '차이점', '시점' 등에서 매칭 방지
        '암',       # '알고리즘' 등에서 매칭 방지
        '형태',     # 단독 사용 시에만
        'cc',       # C++ 등에서 매칭 방지
        'shape',    # 매우 일반적인 단어
        'view',     # 일반적인 단어
        'views',    # 일반적인 단어
        'density',  # 물리학에서도 흔히 사용
        'round',    # 일반적인 단어
        'linear',   # 수학/물리에서 흔히 사용
        'focal',    # 일반적인 단어
        'global',   # 프로그래밍에서 흔히 사용
        'location', # 일반적인 단어
    }

    def is_birads_general_query(self, query: str) -> bool:
        """
        BI-RADS/ACR 가이드라인 쿼리인지 확인
        KEYWORD_SECTION_MAP에 있는 키워드가 포함되면 True

        단, 짧거나 일반적인 키워드는 단독 사용 시에만 매칭
        """
        query_lower = query.lower()

        # 1. BI-RADS 또는 ACR 직접 언급
        if 'bi-rads' in query_lower or 'birads' in query_lower:
            return True
        if 'acr' in query_lower:
            return True

        # 2. KEYWORD_SECTION_MAP의 키워드 중 하나라도 매칭되면 True
        # 단, STRICT_MATCH_KEYWORDS는 단어 경계 확인 필요
        for keyword in self.KEYWORD_SECTION_MAP.keys():
            if keyword in query_lower:
                # 엄격한 매칭이 필요한 키워드인지 확인
                if keyword in self.STRICT_MATCH_KEYWORDS:
                    # 단어 경계 확인: 키워드 앞뒤가 공백/구두점/문자열 끝이어야 함
                    import re
                    # 한글/영문 모두 고려한 단어 경계 패턴
                    pattern = rf'(?<![a-zA-Z가-힣]){re.escape(keyword)}(?![a-zA-Z가-힣])'
                    if re.search(pattern, query_lower):
                        return True
                else:
                    return True

        return False

    # BI-RADS 키워드 → 섹션 매핑 (확장판)
    KEYWORD_SECTION_MAP = {
        # ============================================================
        # === Mass (종괴) - SECTION_IV_A ===
        # ============================================================
        'mass': ['BIRADS_2025_SECTION_IV_A%'],
        'masses': ['BIRADS_2025_SECTION_IV_A%'],
        '종괴': ['BIRADS_2025_SECTION_IV_A%'],
        '종양': ['BIRADS_2025_SECTION_IV_A%'],
        '덩어리': ['BIRADS_2025_SECTION_IV_A%'],
        '결절': ['BIRADS_2025_SECTION_IV_A%'],
        'nodule': ['BIRADS_2025_SECTION_IV_A%'],
        'lesion': ['BIRADS_2025_SECTION_IV_A%'],
        '병변': ['BIRADS_2025_SECTION_IV_A%'],

        # Mass Shape
        'shape': ['BIRADS_2025_SECTION_IV_A_CHUNK_SHAPE%'],
        '모양': ['BIRADS_2025_SECTION_IV_A_CHUNK_SHAPE%'],
        'oval': ['BIRADS_2025_SECTION_IV_A_CHUNK_SHAPE%'],
        '타원형': ['BIRADS_2025_SECTION_IV_A_CHUNK_SHAPE%'],
        'round': ['BIRADS_2025_SECTION_IV_A_CHUNK_SHAPE%', 'BIRADS_2025_SECTION_IV_B1_CHUNK_ROUND%'],
        '원형': ['BIRADS_2025_SECTION_IV_A_CHUNK_SHAPE%'],
        'irregular': ['BIRADS_2025_SECTION_IV_A_CHUNK_SHAPE%'],
        '불규칙': ['BIRADS_2025_SECTION_IV_A_CHUNK_SHAPE%'],

        # Mass Margin
        'margin': ['BIRADS_2025_SECTION_IV_A_CHUNK_MARGIN%'],
        '경계': ['BIRADS_2025_SECTION_IV_A_CHUNK_MARGIN%'],
        '변연': ['BIRADS_2025_SECTION_IV_A_CHUNK_MARGIN%'],
        'circumscribed': ['BIRADS_2025_SECTION_IV_A_CHUNK_MARGIN%'],
        '명확': ['BIRADS_2025_SECTION_IV_A_CHUNK_MARGIN%'],
        'obscured': ['BIRADS_2025_SECTION_IV_A_CHUNK_MARGIN%'],
        '가려진': ['BIRADS_2025_SECTION_IV_A_CHUNK_MARGIN%'],
        'indistinct': ['BIRADS_2025_SECTION_IV_A_CHUNK_MARGIN%'],
        '불분명': ['BIRADS_2025_SECTION_IV_A_CHUNK_MARGIN%'],
        'microlobulated': ['BIRADS_2025_SECTION_IV_A_CHUNK_MARGIN%'],
        '미세분엽': ['BIRADS_2025_SECTION_IV_A_CHUNK_MARGIN%'],
        'spiculated': ['BIRADS_2025_SECTION_IV_A_CHUNK_MARGIN%'],
        '침상': ['BIRADS_2025_SECTION_IV_A_CHUNK_MARGIN%'],
        '방사상': ['BIRADS_2025_SECTION_IV_A_CHUNK_MARGIN%'],

        # Mass Density
        'high density': ['BIRADS_2025_SECTION_IV_A_CHUNK_DENSITY%'],
        '고밀도': ['BIRADS_2025_SECTION_IV_A_CHUNK_DENSITY%'],
        'equal density': ['BIRADS_2025_SECTION_IV_A_CHUNK_DENSITY%'],
        '등밀도': ['BIRADS_2025_SECTION_IV_A_CHUNK_DENSITY%'],
        'low density': ['BIRADS_2025_SECTION_IV_A_CHUNK_DENSITY%'],
        '저밀도': ['BIRADS_2025_SECTION_IV_A_CHUNK_DENSITY%'],
        'fat-containing': ['BIRADS_2025_SECTION_IV_A_CHUNK_DENSITY%'],
        '지방포함': ['BIRADS_2025_SECTION_IV_A_CHUNK_DENSITY%'],

        # ============================================================
        # === Calcifications (석회화) - SECTION_IV_B ===
        # ============================================================
        'calcification': ['BIRADS_2025_SECTION_IV_B%'],
        'calcifications': ['BIRADS_2025_SECTION_IV_B%'],
        '석회화': ['BIRADS_2025_SECTION_IV_B%'],
        '석회': ['BIRADS_2025_SECTION_IV_B%'],
        'microcalcification': ['BIRADS_2025_SECTION_IV_B%'],
        'microcalcifications': ['BIRADS_2025_SECTION_IV_B%'],
        '미세석회화': ['BIRADS_2025_SECTION_IV_B%'],
        '미세석회': ['BIRADS_2025_SECTION_IV_B%'],

        # Calcification Morphology
        'morphology': ['BIRADS_2025_SECTION_IV_B2%'],
        '형태': ['BIRADS_2025_SECTION_IV_B2%'],

        # Typically Benign Calcifications
        'benign': ['BIRADS_2025_SECTION_IV_B1%', 'BIRADS_2025_SECTION_V_CAT2%'],
        '양성': ['BIRADS_2025_SECTION_IV_B1%', 'BIRADS_2025_SECTION_V_CAT2%'],
        'typically benign': ['BIRADS_2025_SECTION_IV_B1%'],
        'skin': ['BIRADS_2025_SECTION_IV_B1_CHUNK_SKIN%', 'BIRADS_2025_SECTION_IV_F%'],
        '피부': ['BIRADS_2025_SECTION_IV_B1_CHUNK_SKIN%', 'BIRADS_2025_SECTION_IV_F%'],
        'vascular': ['BIRADS_2025_SECTION_IV_B1_CHUNK_VASCULAR%'],
        '혈관': ['BIRADS_2025_SECTION_IV_B1_CHUNK_VASCULAR%'],
        'coarse': ['BIRADS_2025_SECTION_IV_B1_CHUNK_COARSE%', 'BIRADS_2025_SECTION_IV_B2_CHUNK_COARSE%'],
        '거친': ['BIRADS_2025_SECTION_IV_B1_CHUNK_COARSE%'],
        'popcorn': ['BIRADS_2025_SECTION_IV_B1_CHUNK_COARSE%'],
        '팝콘': ['BIRADS_2025_SECTION_IV_B1_CHUNK_COARSE%'],
        'large rod': ['BIRADS_2025_SECTION_IV_B1_CHUNK_LARGE_ROD%'],
        'rod-like': ['BIRADS_2025_SECTION_IV_B1_CHUNK_LARGE_ROD%'],
        '막대': ['BIRADS_2025_SECTION_IV_B1_CHUNK_LARGE_ROD%'],
        'round calcification': ['BIRADS_2025_SECTION_IV_B1_CHUNK_ROUND%'],
        'punctate': ['BIRADS_2025_SECTION_IV_B1_CHUNK_ROUND%'],
        '점상': ['BIRADS_2025_SECTION_IV_B1_CHUNK_ROUND%'],
        'rim': ['BIRADS_2025_SECTION_IV_B1_CHUNK_RIM%'],
        'eggshell': ['BIRADS_2025_SECTION_IV_B1_CHUNK_RIM%'],
        '테두리': ['BIRADS_2025_SECTION_IV_B1_CHUNK_RIM%'],
        'lucent-centered': ['BIRADS_2025_SECTION_IV_B1_CHUNK_RIM%'],
        'milk of calcium': ['BIRADS_2025_SECTION_IV_B1_CHUNK_LAYERING%'],
        'layering': ['BIRADS_2025_SECTION_IV_B1_CHUNK_LAYERING%'],
        '침전': ['BIRADS_2025_SECTION_IV_B1_CHUNK_LAYERING%'],
        'suture': ['BIRADS_2025_SECTION_IV_B1_CHUNK_SUTURE%'],
        '봉합사': ['BIRADS_2025_SECTION_IV_B1_CHUNK_SUTURE%'],
        'dystrophic': ['BIRADS_2025_SECTION_IV_B1%'],
        '이영양성': ['BIRADS_2025_SECTION_IV_B1%'],

        # Suspicious Calcifications
        'suspicious': ['BIRADS_2025_SECTION_IV_B2%', 'BIRADS_2025_SECTION_V_CAT4%'],
        '의심': ['BIRADS_2025_SECTION_IV_B2%', 'BIRADS_2025_SECTION_V_CAT4%'],
        '의심스러운': ['BIRADS_2025_SECTION_IV_B2%'],
        'amorphous': ['BIRADS_2025_SECTION_IV_B2_CHUNK_AMORPHOUS%'],
        '무정형': ['BIRADS_2025_SECTION_IV_B2_CHUNK_AMORPHOUS%'],
        'indistinct': ['BIRADS_2025_SECTION_IV_B2_CHUNK_AMORPHOUS%'],
        'coarse heterogeneous': ['BIRADS_2025_SECTION_IV_B2_CHUNK_COARSE_HETEROGENEOUS%'],
        'heterogeneous': ['BIRADS_2025_SECTION_IV_B2_CHUNK_COARSE_HETEROGENEOUS%'],
        '불균질': ['BIRADS_2025_SECTION_IV_B2_CHUNK_COARSE_HETEROGENEOUS%'],
        'fine pleomorphic': ['BIRADS_2025_SECTION_IV_B2_CHUNK_FINE_PLEOMORPHIC%'],
        'pleomorphic': ['BIRADS_2025_SECTION_IV_B2_CHUNK_FINE_PLEOMORPHIC%'],
        '다형성': ['BIRADS_2025_SECTION_IV_B2_CHUNK_FINE_PLEOMORPHIC%'],
        'fine linear': ['BIRADS_2025_SECTION_IV_B2_CHUNK_FINE_LINEAR%'],
        'fine linear-branching': ['BIRADS_2025_SECTION_IV_B2_CHUNK_FINE_LINEAR%'],
        'linear': ['BIRADS_2025_SECTION_IV_B2_CHUNK_FINE_LINEAR%', 'BIRADS_2025_SECTION_IV_B3_CHUNK_LINEAR%'],
        '선형': ['BIRADS_2025_SECTION_IV_B2_CHUNK_FINE_LINEAR%'],
        'branching': ['BIRADS_2025_SECTION_IV_B2_CHUNK_FINE_LINEAR%'],
        '분지': ['BIRADS_2025_SECTION_IV_B2_CHUNK_FINE_LINEAR%'],
        '분지형': ['BIRADS_2025_SECTION_IV_B2_CHUNK_FINE_LINEAR%'],

        # Calcification Distribution
        'distribution': ['BIRADS_2025_SECTION_IV_B3%'],
        '분포': ['BIRADS_2025_SECTION_IV_B3%'],
        'diffuse': ['BIRADS_2025_SECTION_IV_B3_CHUNK_DIFFUSE%'],
        '미만성': ['BIRADS_2025_SECTION_IV_B3_CHUNK_DIFFUSE%'],
        'scattered': ['BIRADS_2025_SECTION_IV_B3_CHUNK_DIFFUSE%', 'BIRADS_2025_SECTION_III%'],
        '산재': ['BIRADS_2025_SECTION_IV_B3_CHUNK_DIFFUSE%'],
        'regional': ['BIRADS_2025_SECTION_IV_B3_CHUNK_REGIONAL%'],
        '국소': ['BIRADS_2025_SECTION_IV_B3_CHUNK_REGIONAL%'],
        'grouped': ['BIRADS_2025_SECTION_IV_B3_CHUNK_GROUPED%'],
        'cluster': ['BIRADS_2025_SECTION_IV_B3_CHUNK_GROUPED%'],
        '군집': ['BIRADS_2025_SECTION_IV_B3_CHUNK_GROUPED%'],
        'linear distribution': ['BIRADS_2025_SECTION_IV_B3_CHUNK_LINEAR%'],
        'segmental': ['BIRADS_2025_SECTION_IV_B3_CHUNK_SEGMENTAL%'],
        '분절': ['BIRADS_2025_SECTION_IV_B3_CHUNK_SEGMENTAL%'],
        '분절성': ['BIRADS_2025_SECTION_IV_B3_CHUNK_SEGMENTAL%'],

        # ============================================================
        # === Architectural Distortion - SECTION_IV_C ===
        # ============================================================
        'architectural distortion': ['BIRADS_2025_SECTION_IV_C%'],
        'distortion': ['BIRADS_2025_SECTION_IV_C%'],
        'architectural': ['BIRADS_2025_SECTION_IV_C%'],
        '구조왜곡': ['BIRADS_2025_SECTION_IV_C%'],
        '구조 왜곡': ['BIRADS_2025_SECTION_IV_C%'],
        '왜곡': ['BIRADS_2025_SECTION_IV_C%'],
        'radial scar': ['BIRADS_2025_SECTION_IV_C%'],
        '방사상 흉터': ['BIRADS_2025_SECTION_IV_C%'],

        # ============================================================
        # === Asymmetries - SECTION_IV_D ===
        # ============================================================
        'asymmetry': ['BIRADS_2025_SECTION_IV_D%'],
        'asymmetries': ['BIRADS_2025_SECTION_IV_D%'],
        '비대칭': ['BIRADS_2025_SECTION_IV_D%'],
        'focal asymmetry': ['BIRADS_2025_SECTION_IV_D_CHUNK_FOCAL%'],
        'focal': ['BIRADS_2025_SECTION_IV_D_CHUNK_FOCAL%'],
        '국소 비대칭': ['BIRADS_2025_SECTION_IV_D_CHUNK_FOCAL%'],
        'global asymmetry': ['BIRADS_2025_SECTION_IV_D_CHUNK_GLOBAL%'],
        'global': ['BIRADS_2025_SECTION_IV_D_CHUNK_GLOBAL%'],
        '전체 비대칭': ['BIRADS_2025_SECTION_IV_D_CHUNK_GLOBAL%'],
        'developing asymmetry': ['BIRADS_2025_SECTION_IV_D%'],
        '진행성 비대칭': ['BIRADS_2025_SECTION_IV_D%'],
        'one-view': ['BIRADS_2025_SECTION_IV_D%'],

        # ============================================================
        # === Lymph Nodes - SECTION_IV_E ===
        # ============================================================
        'lymph': ['BIRADS_2025_SECTION_IV_E%'],
        'lymph node': ['BIRADS_2025_SECTION_IV_E%'],
        'lymph nodes': ['BIRADS_2025_SECTION_IV_E%'],
        '림프': ['BIRADS_2025_SECTION_IV_E%'],
        '림프절': ['BIRADS_2025_SECTION_IV_E%'],
        'axillary': ['BIRADS_2025_SECTION_IV_E%'],
        '액와': ['BIRADS_2025_SECTION_IV_E%'],
        'intramammary': ['BIRADS_2025_SECTION_IV_E%'],
        '유방내': ['BIRADS_2025_SECTION_IV_E%'],

        # ============================================================
        # === Skin Lesions - SECTION_IV_F ===
        # ============================================================
        'skin lesion': ['BIRADS_2025_SECTION_IV_F%'],
        'skin lesions': ['BIRADS_2025_SECTION_IV_F%'],
        '피부 병변': ['BIRADS_2025_SECTION_IV_F%'],
        'mole': ['BIRADS_2025_SECTION_IV_F%'],
        '점': ['BIRADS_2025_SECTION_IV_F%'],
        'sebaceous cyst': ['BIRADS_2025_SECTION_IV_F%'],
        '피지낭종': ['BIRADS_2025_SECTION_IV_F%'],

        # ============================================================
        # === Dilated Ducts - SECTION_IV_G ===
        # ============================================================
        'duct': ['BIRADS_2025_SECTION_IV_G%'],
        'ducts': ['BIRADS_2025_SECTION_IV_G%'],
        'dilated': ['BIRADS_2025_SECTION_IV_G%'],
        'dilated duct': ['BIRADS_2025_SECTION_IV_G%'],
        '유관': ['BIRADS_2025_SECTION_IV_G%'],
        '확장': ['BIRADS_2025_SECTION_IV_G%'],
        '유관 확장': ['BIRADS_2025_SECTION_IV_G%'],
        'duct ectasia': ['BIRADS_2025_SECTION_IV_G%'],

        # ============================================================
        # === Associated Features - SECTION_IV_H ===
        # ============================================================
        'associated': ['BIRADS_2025_SECTION_IV_H%'],
        'associated feature': ['BIRADS_2025_SECTION_IV_H%'],
        '동반 소견': ['BIRADS_2025_SECTION_IV_H%'],
        'skin retraction': ['BIRADS_2025_SECTION_IV_H%'],
        '피부 함몰': ['BIRADS_2025_SECTION_IV_H%'],
        'nipple retraction': ['BIRADS_2025_SECTION_IV_H%'],
        'nipple': ['BIRADS_2025_SECTION_IV_H%'],
        '유두': ['BIRADS_2025_SECTION_IV_H%'],
        '유두 함몰': ['BIRADS_2025_SECTION_IV_H%'],
        'skin thickening': ['BIRADS_2025_SECTION_IV_H%'],
        '피부 비후': ['BIRADS_2025_SECTION_IV_H%'],
        'trabecular thickening': ['BIRADS_2025_SECTION_IV_H%'],
        '소엽간격 비후': ['BIRADS_2025_SECTION_IV_H%'],
        'axillary adenopathy': ['BIRADS_2025_SECTION_IV_H%'],
        '액와 림프절 종대': ['BIRADS_2025_SECTION_IV_H%'],

        # ============================================================
        # === Special Cases - SECTION_IV_I ===
        # ============================================================
        'special case': ['BIRADS_2025_SECTION_IV_I%'],
        'special cases': ['BIRADS_2025_SECTION_IV_I%'],
        '특수 사례': ['BIRADS_2025_SECTION_IV_I%'],
        'implant': ['BIRADS_2025_SECTION_IV_I%'],
        'implants': ['BIRADS_2025_SECTION_IV_I%'],
        '보형물': ['BIRADS_2025_SECTION_IV_I%'],
        '실리콘': ['BIRADS_2025_SECTION_IV_I%'],
        'post-surgical': ['BIRADS_2025_SECTION_IV_I%'],
        '수술 후': ['BIRADS_2025_SECTION_IV_I%'],

        # ============================================================
        # === Location - SECTION_IV_J ===
        # ============================================================
        'location': ['BIRADS_2025_SECTION_IV_J%'],
        '위치': ['BIRADS_2025_SECTION_IV_J%'],
        'quadrant': ['BIRADS_2025_SECTION_IV_J%'],
        '사분면': ['BIRADS_2025_SECTION_IV_J%'],
        'clock': ['BIRADS_2025_SECTION_IV_J%'],
        '시계방향': ['BIRADS_2025_SECTION_IV_J%'],
        'depth': ['BIRADS_2025_SECTION_IV_J%'],
        '깊이': ['BIRADS_2025_SECTION_IV_J%'],
        'subareolar': ['BIRADS_2025_SECTION_IV_J%'],
        '유륜하': ['BIRADS_2025_SECTION_IV_J%'],
        'central': ['BIRADS_2025_SECTION_IV_J%'],
        '중심부': ['BIRADS_2025_SECTION_IV_J%'],

        # ============================================================
        # === Breast Density - SECTION_III ===
        # ============================================================
        'density': ['BIRADS_2025_SECTION_III%', 'BIRADS_2025_SECTION_IV_A_CHUNK_DENSITY%'],
        'breast density': ['BIRADS_2025_SECTION_III%'],
        '밀도': ['BIRADS_2025_SECTION_III%'],
        '유방밀도': ['BIRADS_2025_SECTION_III%'],
        '유방 밀도': ['BIRADS_2025_SECTION_III%'],
        'dense': ['BIRADS_2025_SECTION_III%'],
        '치밀': ['BIRADS_2025_SECTION_III%'],
        '치밀유방': ['BIRADS_2025_SECTION_III%'],
        'fatty': ['BIRADS_2025_SECTION_III%'],
        '지방형': ['BIRADS_2025_SECTION_III%'],
        'fibroglandular': ['BIRADS_2025_SECTION_III%'],
        '섬유선': ['BIRADS_2025_SECTION_III%'],
        'heterogeneously dense': ['BIRADS_2025_SECTION_III%'],
        '불균질 치밀': ['BIRADS_2025_SECTION_III%'],
        'extremely dense': ['BIRADS_2025_SECTION_III%'],
        '극도 치밀': ['BIRADS_2025_SECTION_III%'],
        'almost entirely fatty': ['BIRADS_2025_SECTION_III%'],
        'scattered fibroglandular': ['BIRADS_2025_SECTION_III%'],

        # ============================================================
        # === Views/Projection - APPENDIX_A ===
        # ============================================================
        'view': ['BIRADS_2025_APPENDIX_A%'],
        'views': ['BIRADS_2025_APPENDIX_A%'],
        'projection': ['BIRADS_2025_APPENDIX_A%'],
        '촬영': ['BIRADS_2025_APPENDIX_A%'],
        '자세': ['BIRADS_2025_APPENDIX_A%'],
        '영상': ['BIRADS_2025_APPENDIX_A%'],
        'cc': ['BIRADS_2025_APPENDIX_A%'],
        'mlo': ['BIRADS_2025_APPENDIX_A%'],
        'craniocaudal': ['BIRADS_2025_APPENDIX_A%'],
        'mediolateral': ['BIRADS_2025_APPENDIX_A%'],
        'oblique': ['BIRADS_2025_APPENDIX_A%'],
        'lateral': ['BIRADS_2025_APPENDIX_A%'],
        'spot compression': ['BIRADS_2025_APPENDIX_A%'],
        '압박': ['BIRADS_2025_APPENDIX_A%'],
        'magnification': ['BIRADS_2025_APPENDIX_A%'],
        '확대': ['BIRADS_2025_APPENDIX_A%'],
        'tangential': ['BIRADS_2025_APPENDIX_A%'],
        'rolled': ['BIRADS_2025_APPENDIX_A%'],
        'exaggerated': ['BIRADS_2025_APPENDIX_A%'],
        'cleavage': ['BIRADS_2025_APPENDIX_A%'],
        'axillary tail': ['BIRADS_2025_APPENDIX_A%'],

        # ============================================================
        # === Categories - SECTION_V ===
        # ============================================================
        'category': ['BIRADS_2025_SECTION_V%'],
        '카테고리': ['BIRADS_2025_SECTION_V%'],
        'assessment': ['BIRADS_2025_SECTION_V%'],
        '평가': ['BIRADS_2025_SECTION_V%'],
        '분류': ['BIRADS_2025_SECTION_V%'],
        'reporting': ['BIRADS_2025_SECTION_V%'],
        'report': ['BIRADS_2025_SECTION_V%'],
        '보고': ['BIRADS_2025_SECTION_V%'],

        # Category specific
        'incomplete': ['BIRADS_2025_SECTION_V_CAT0%'],
        '불완전': ['BIRADS_2025_SECTION_V_CAT0%'],
        'recall': ['BIRADS_2025_SECTION_V_CAT0%'],
        '재검': ['BIRADS_2025_SECTION_V_CAT0%'],
        'negative': ['BIRADS_2025_SECTION_V_CAT1%'],
        '음성': ['BIRADS_2025_SECTION_V_CAT1%'],
        '정상': ['BIRADS_2025_SECTION_V_CAT1%'],
        'probably benign': ['BIRADS_2025_SECTION_V_CAT3%'],
        '아마 양성': ['BIRADS_2025_SECTION_V_CAT3%'],
        'follow-up': ['BIRADS_2025_SECTION_V_CAT3%'],
        'follow up': ['BIRADS_2025_SECTION_V_CAT3%'],
        '추적': ['BIRADS_2025_SECTION_V_CAT3%'],
        '추적관찰': ['BIRADS_2025_SECTION_V_CAT3%'],
        'biopsy': ['BIRADS_2025_SECTION_V_CAT4%', 'BIRADS_2025_SECTION_V_CAT5%'],
        '생검': ['BIRADS_2025_SECTION_V_CAT4%', 'BIRADS_2025_SECTION_V_CAT5%'],
        '조직검사': ['BIRADS_2025_SECTION_V_CAT4%', 'BIRADS_2025_SECTION_V_CAT5%'],
        'malignant': ['BIRADS_2025_SECTION_IV_B2%', 'BIRADS_2025_SECTION_V_CAT5%'],
        'malignancy': ['BIRADS_2025_SECTION_IV_B2%', 'BIRADS_2025_SECTION_V_CAT5%'],
        '악성': ['BIRADS_2025_SECTION_IV_B2%', 'BIRADS_2025_SECTION_V_CAT5%'],
        'highly suggestive': ['BIRADS_2025_SECTION_V_CAT5%'],
        '강력 의심': ['BIRADS_2025_SECTION_V_CAT5%'],
        'known malignancy': ['BIRADS_2025_SECTION_V_CAT6%'],
        'proven malignancy': ['BIRADS_2025_SECTION_V_CAT6%'],
        '확진': ['BIRADS_2025_SECTION_V_CAT6%'],

        # ============================================================
        # === Screening/Guidance - SECTION_VI ===
        # ============================================================
        'screening': ['BIRADS_2025_SECTION_VI%'],
        '스크리닝': ['BIRADS_2025_SECTION_VI%'],
        '검진': ['BIRADS_2025_SECTION_VI%'],
        'diagnostic': ['BIRADS_2025_SECTION_VI%'],
        '진단': ['BIRADS_2025_SECTION_VI%'],
        'guidance': ['BIRADS_2025_SECTION_VI%'],
        '지침': ['BIRADS_2025_SECTION_VI%'],
        '가이드': ['BIRADS_2025_SECTION_VI%'],
        'management': ['BIRADS_2025_SECTION_VI%'],
        '관리': ['BIRADS_2025_SECTION_VI%'],
        'recommendation': ['BIRADS_2025_SECTION_VI%'],
        '권고': ['BIRADS_2025_SECTION_VI%'],
        'workup': ['BIRADS_2025_SECTION_VI%'],

        # ============================================================
        # === FAQ - SECTION_VII ===
        # ============================================================
        'faq': ['BIRADS_2025_SECTION_VII%'],
        'frequently asked': ['BIRADS_2025_SECTION_VII%'],
        '자주 묻는': ['BIRADS_2025_SECTION_VII%'],

        # ============================================================
        # === Lexicon - SECTION_II ===
        # ============================================================
        'lexicon': ['BIRADS_2025_SECTION_II%'],
        '용어': ['BIRADS_2025_SECTION_II%'],
        'terminology': ['BIRADS_2025_SECTION_II%'],
        'descriptor': ['BIRADS_2025_SECTION_II%'],
        '기술어': ['BIRADS_2025_SECTION_II%'],

        # ============================================================
        # === Cancer/DCIS ===
        # ============================================================
        'cancer': ['BIRADS_2025_SECTION_IV_B2%', 'BIRADS_2025_SECTION_V_CAT5%'],
        '암': ['BIRADS_2025_SECTION_IV_B2%', 'BIRADS_2025_SECTION_V_CAT5%'],
        '유방암': ['BIRADS_2025_SECTION_IV_B2%', 'BIRADS_2025_SECTION_V_CAT5%'],
        'breast cancer': ['BIRADS_2025_SECTION_IV_B2%', 'BIRADS_2025_SECTION_V_CAT5%'],
        'dcis': ['BIRADS_2025_SECTION_IV_B2%'],
        'ductal carcinoma': ['BIRADS_2025_SECTION_IV_B2%'],
        '유관암': ['BIRADS_2025_SECTION_IV_B2%'],
        '관상피내암': ['BIRADS_2025_SECTION_IV_B2%'],
        'invasive': ['BIRADS_2025_SECTION_IV_B2%', 'BIRADS_2025_SECTION_V_CAT5%'],
        '침윤성': ['BIRADS_2025_SECTION_IV_B2%', 'BIRADS_2025_SECTION_V_CAT5%'],
        'carcinoma': ['BIRADS_2025_SECTION_IV_B2%', 'BIRADS_2025_SECTION_V_CAT5%'],
        '암종': ['BIRADS_2025_SECTION_IV_B2%', 'BIRADS_2025_SECTION_V_CAT5%'],
        'idc': ['BIRADS_2025_SECTION_IV_B2%', 'BIRADS_2025_SECTION_V_CAT5%'],
        'ilc': ['BIRADS_2025_SECTION_IV_B2%', 'BIRADS_2025_SECTION_V_CAT5%'],

        # ============================================================
        # === Benign Lesions ===
        # ============================================================
        'fibroadenoma': ['BIRADS_2025_SECTION_V_CAT2%', 'BIRADS_2025_SECTION_V_CAT3%'],
        '섬유선종': ['BIRADS_2025_SECTION_V_CAT2%', 'BIRADS_2025_SECTION_V_CAT3%'],
        'cyst': ['BIRADS_2025_SECTION_V_CAT2%'],
        '낭종': ['BIRADS_2025_SECTION_V_CAT2%'],
        '물혹': ['BIRADS_2025_SECTION_V_CAT2%'],
        'lipoma': ['BIRADS_2025_SECTION_V_CAT2%'],
        '지방종': ['BIRADS_2025_SECTION_V_CAT2%'],
        'fat necrosis': ['BIRADS_2025_SECTION_V_CAT2%'],
        '지방괴사': ['BIRADS_2025_SECTION_V_CAT2%'],
        'hamartoma': ['BIRADS_2025_SECTION_V_CAT2%'],
        '과오종': ['BIRADS_2025_SECTION_V_CAT2%'],

        # ============================================================
        # === CEM (Contrast Enhanced Mammography) ===
        # ============================================================
        'cem': ['BIRADS_CEM%', 'ACR_CEM_%'],
        'contrast enhanced': ['BIRADS_CEM%', 'ACR_CEM_%'],
        'contrast-enhanced': ['BIRADS_CEM%', 'ACR_CEM_%'],
        '조영증강': ['BIRADS_CEM%', 'ACR_CEM_%'],
        '조영제': ['BIRADS_CEM%', 'ACR_CEM_%'],
        'enhancement': ['BIRADS_CEM%', 'ACR_CEM_%'],
        '증강': ['BIRADS_CEM%', 'ACR_CEM_%'],
        'nme': ['BIRADS_CEM%', 'ACR_CEM_%'],
        'non-mass enhancement': ['BIRADS_CEM%', 'ACR_CEM_%'],
        '비종괴성 조영증강': ['BIRADS_CEM%', 'ACR_CEM_%'],
        'bpe': ['BIRADS_CEM%', 'ACR_CEM_%'],
        'background parenchymal enhancement': ['BIRADS_CEM%', 'ACR_CEM_%'],
        '배경 실질 조영증강': ['BIRADS_CEM%', 'ACR_CEM_%'],
        '적응증': ['BIRADS_CEM%', 'ACR_CEM_%', 'ACR_MAMMO_%'],
        'indication': ['BIRADS_CEM%', 'ACR_CEM_%', 'ACR_MAMMO_%'],
        'indications': ['BIRADS_CEM%', 'ACR_CEM_%', 'ACR_MAMMO_%'],

        # ============================================================
        # === Audit ===
        # ============================================================
        'audit': ['BIRADS_AUDIT%'],
        '감사': ['BIRADS_AUDIT%'],
        'outcome': ['BIRADS_AUDIT%'],
        '결과': ['BIRADS_AUDIT%'],
        'sensitivity': ['BIRADS_AUDIT%'],
        '민감도': ['BIRADS_AUDIT%'],
        'specificity': ['BIRADS_AUDIT%'],
        '특이도': ['BIRADS_AUDIT%'],
        'ppv': ['BIRADS_AUDIT%'],
        'positive predictive value': ['BIRADS_AUDIT%'],
        '양성예측도': ['BIRADS_AUDIT%'],
        'cancer detection rate': ['BIRADS_AUDIT%'],
        '암발견율': ['BIRADS_AUDIT%'],
        'recall rate': ['BIRADS_AUDIT%'],
        '재검률': ['BIRADS_AUDIT%'],

        # ============================================================
        # === ACR Practice Parameters ===
        # ============================================================
        'acr': ['ACR_MAMMO_%', 'ACR_CEM_%', 'ACR_IQ_%'],
        'practice parameter': ['ACR_MAMMO_%', 'ACR_CEM_%', 'ACR_IQ_%'],
        '실무 지침': ['ACR_MAMMO_%', 'ACR_CEM_%', 'ACR_IQ_%'],
        'image quality': ['ACR_IQ_%'],
        '영상 품질': ['ACR_IQ_%'],
        'quality control': ['ACR_IQ_%', 'ACR_MAMMO_%'],
        '품질 관리': ['ACR_IQ_%', 'ACR_MAMMO_%'],
        'personnel': ['ACR_MAMMO_%', 'ACR_IQ_%'],
        '인력': ['ACR_MAMMO_%', 'ACR_IQ_%'],
        'equipment': ['ACR_MAMMO_%', 'ACR_IQ_%'],
        '장비': ['ACR_MAMMO_%', 'ACR_IQ_%'],
        'artifact': ['ACR_IQ_%'],
        '아티팩트': ['ACR_IQ_%'],
        'positioning': ['ACR_MAMMO_%', 'ACR_IQ_%'],
        '위치설정': ['ACR_MAMMO_%', 'ACR_IQ_%'],
        'iodine': ['ACR_CEM_%'],
        '요오드': ['ACR_CEM_%'],
        'contrast agent': ['ACR_CEM_%'],
        'allergic': ['ACR_CEM_%'],
        '알레르기': ['ACR_CEM_%'],
        'renal': ['ACR_CEM_%'],
        '신장': ['ACR_CEM_%'],
        'contraindication': ['ACR_CEM_%'],
        '금기': ['ACR_CEM_%'],
    }

    def search_birads_general(self, query: str, top_k: int = 3) -> List[Paper]:
        """
        BI-RADS 일반 가이드라인 검색
        (키워드 기반 섹션 매칭 + 전문 검색)
        """
        query_lower = query.lower()

        # 1. 키워드 매칭으로 타겟 섹션 찾기 (점수 기반)
        section_scores = {}  # section_pattern → score
        for keyword, sections in self.KEYWORD_SECTION_MAP.items():
            if keyword in query_lower:
                for section in sections:
                    # 더 구체적인 섹션(CHUNK)에 높은 점수
                    score = 2 if 'CHUNK' in section else 1
                    section_scores[section] = section_scores.get(section, 0) + score

        cursor = self.conn.cursor()
        paper_scores = {}  # pmid → (paper, score)

        # 2. 타겟 섹션이 있으면 해당 섹션에서 검색 (점수순)
        if section_scores:
            sorted_sections = sorted(section_scores.items(), key=lambda x: -x[1])

            for section_pattern, score in sorted_sections:
                cursor.execute("""
                    SELECT pmid, doi, title, authors, journal, year, month,
                           abstract, full_content, citation_count, journal_if
                    FROM papers
                    WHERE pmid LIKE ?
                    ORDER BY pmid
                    LIMIT ?
                """, (section_pattern, top_k * 2))

                for row in cursor.fetchall():
                    pmid = row['pmid']
                    # 중복 제거 (이미 있으면 점수만 누적)
                    if pmid in paper_scores:
                        paper_scores[pmid] = (paper_scores[pmid][0], paper_scores[pmid][1] + score)
                        continue

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
                    paper_scores[pmid] = (paper, score)

        # 점수순 정렬
        papers = [p for p, s in sorted(paper_scores.values(), key=lambda x: -x[1])]

        # 3. 섹션 매칭 결과가 부족하면 전문 검색으로 보충
        if len(papers) < top_k:
            # 검색용 키워드 추출 (공백으로 분리된 단어들)
            search_terms = [w for w in query_lower.split() if len(w) > 2]

            for term in search_terms[:5]:  # 상위 5개 단어만
                if len(papers) >= top_k:
                    break

                search_pattern = f'%{term}%'
                cursor.execute("""
                    SELECT pmid, doi, title, authors, journal, year, month,
                           abstract, full_content, citation_count, journal_if
                    FROM papers
                    WHERE (pmid LIKE 'BIRADS_%' OR pmid LIKE 'ACR_%' OR pmid LIKE 'PHYSICS_%' OR pmid LIKE 'CLINICAL_%')
                      AND (LOWER(title) LIKE ? OR LOWER(full_content) LIKE ?)
                    LIMIT ?
                """, (search_pattern, search_pattern, top_k))

                for row in cursor.fetchall():
                    if any(p.pmid == row['pmid'] for p in papers):
                        continue

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

        return papers[:top_k]


def test():
    """테스트"""
    matcher = BiradsCategoryMatcher()

    # 1. 카테고리 파싱 테스트
    test_queries = [
        "BI-RADS 5는 무엇인가요?",
        "Category 5",
        "카테고리 5",
        "Category 4A",
    ]

    print("=== Category Parsing Test ===")
    for query in test_queries:
        category = matcher.parse_category(query)
        print(f"Query: {query:40} → Category: {category}")

    # 2. 핵심 검색 테스트 - 다양한 주제
    print("\n" + "=" * 60)
    print("=== BI-RADS General Search Test ===")
    print("=" * 60)

    general_queries = [
        # 석회화 관련
        "석회화의 형태(Morphology) 중 악성 의심이 가장 높은 양상은?",
        "Fine linear calcification",
        "미세석회화 악성",
        "calcification morphology suspicious",

        # 종괴 관련
        "Mass margin spiculated",
        "종괴 경계 침상",
        "irregular mass shape",

        # 밀도 관련
        "유방 밀도 분류",
        "breast density categories",
        "extremely dense",

        # 비대칭 관련
        "focal asymmetry",
        "비대칭 소견",

        # 구조 왜곡
        "architectural distortion",
        "구조왜곡",

        # 카테고리 5 관련
        "악성 의심 소견",
        "highly suggestive of malignancy",

        # 촬영 자세
        "CC MLO view",
        "촬영 자세",
    ]

    for query in general_queries:
        papers = matcher.search_birads_general(query, top_k=3)
        print(f"\n📌 Query: {query}")
        if papers:
            for i, p in enumerate(papers, 1):
                print(f"   {i}. {p.pmid}")
                print(f"      Title: {p.title[:60]}...")
        else:
            print("   ❌ No results")

    print("\n" + "=" * 60)
    print("=== Test Complete ===")
    print("=" * 60)


if __name__ == "__main__":
    test()
