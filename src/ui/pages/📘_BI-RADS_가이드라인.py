"""
BI-RADS 가이드라인 전체 보기
============================
BI-RADS 2025 가이드라인 전문 탐색
"""

import streamlit as st
import sqlite3
from pathlib import Path
import re

st.set_page_config(
    page_title="BI-RADS 가이드라인",
    page_icon="📘",
    layout="wide"
)

# CSS 스타일
st.markdown("""
<style>
    .category-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
    .document-card {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 8px;
        border-left: 4px solid #764ba2;
        margin: 15px 0;
    }
    .document-title {
        font-size: 1.3em;
        font-weight: bold;
        color: #333;
        margin-bottom: 10px;
    }
    .document-meta {
        color: #666;
        font-size: 0.9em;
        margin-bottom: 15px;
    }
    .document-content {
        line-height: 1.8;
        white-space: pre-wrap;
    }
    .media-section {
        background: #ffffff;
        padding: 15px;
        border-radius: 8px;
        margin: 15px 0;
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# 헤더
st.markdown('<div class="category-header"><h1>📘 BI-RADS 2025 가이드라인</h1><p>Breast Imaging Reporting and Data System</p></div>', unsafe_allow_html=True)

# 챕터 선택
available_chapters = [
    'Mammography',
    'Contrast Enhanced Mammography (CEM)',
    'Auditing and Outcomes Monitoring',
    'General FAQ'
]

selected_chapter = st.selectbox(
    '챕터 선택',
    options=available_chapters,
    index=0,
    help='BI-RADS 매뉴얼의 다른 챕터를 선택하세요'
)

st.markdown("---")

# 이미지가 포함된 마크다운 렌더링 헬퍼 함수
def render_markdown_with_images(content):
    """마크다운 콘텐츠를 파싱하고 이미지를 st.image()로 렌더링"""
    if not content:
        st.markdown("_내용 없음_")
        return

    # 이미지 패턴: ![alt text](path)
    image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'

    # 콘텐츠를 이미지 기준으로 분리
    parts = re.split(image_pattern, content)

    i = 0
    while i < len(parts):
        if i % 3 == 0:
            # 텍스트 부분
            if parts[i].strip():
                st.markdown(parts[i])
        elif i % 3 == 1:
            # alt text (다음이 경로)
            alt_text = parts[i]
            image_path = parts[i + 1] if i + 1 < len(parts) else None

            if image_path:
                # 이미지 파일이 존재하는지 확인
                img_file = Path(image_path)
                if img_file.exists():
                    st.image(str(img_file), caption=alt_text, use_container_width=True)
                else:
                    st.warning(f"⚠️ 이미지를 찾을 수 없습니다: {image_path}")

            i += 1  # 경로 부분 스킵

        i += 1

# 데이터베이스에서 BI-RADS 문서 가져오기
@st.cache_data(ttl=3600)
def load_birads_documents(chapter='Mammography'):
    """BI-RADS 문서를 데이터베이스에서 로드"""
    db_path = Path('data/index/metadata.db')

    if not db_path.exists():
        return {}

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 챕터별 PMID 패턴
    chapter_patterns = {
        'Mammography': 'BIRADS_2025_%',
        'Contrast Enhanced Mammography (CEM)': 'BIRADS_CEM_%',
        'Ultrasound': 'BIRADS_US_%',
        'MRI': 'BIRADS_MRI_%',
        'Auditing and Outcomes Monitoring': 'BIRADS_AUDIT_%',
        'General FAQ': 'BIRADS_2025_SECTION_VII',
        'Online Resources': 'BIRADS_RESOURCES_%'
    }

    pattern = chapter_patterns.get(chapter, 'BIRADS_2025_%')

    # General FAQ는 정확한 pmid로 검색, 나머지는 LIKE 패턴 사용
    if '%' in pattern:
        cursor.execute('''
            SELECT pmid, title, full_content, authors, journal, year
            FROM papers
            WHERE pmid LIKE ?
            ORDER BY title
        ''', (pattern,))
    else:
        cursor.execute('''
            SELECT pmid, title, full_content, authors, journal, year
            FROM papers
            WHERE pmid = ?
            ORDER BY title
        ''', (pattern,))

    documents = cursor.fetchall()
    conn.close()

    # 카테고리별로 분류 (계층적 구조, 순서 유지)
    from collections import OrderedDict
    categories = OrderedDict()

    # 순서 정의 (Mammography + CEM + Auditing 모든 카테고리)
    category_order = [
        'Table of Contents',
        'Introduction',
        'Section I: General Considerations',
        'Section I: Glossary',
        'Section II: Breast Imaging Lexicon',
        'Section II: Basic Audit',
        'Section III: Breast Density',
        'Section III: Image Examples',
        'Section III: Complete Audit',
        'Section IV: Findings',
        'Section IV: Reporting System',
        'Section IV: Classification Examples',
        'Section V: Reporting System',
        'Section V: Guidance',
        'Section V: Data Collection',
        'Section VI: Guidance',
        'Section VI: FAQ',
        'Section VII: FAQ',
        'Section VII: Method of Detection',
        'Appendix'
    ]

    for cat in category_order:
        categories[cat] = []

    for doc in documents:
        pmid, title, content, authors, journal, year = doc

        # PMID 기반 카테고리 분류
        # Mammography 챕터
        if pmid == 'BIRADS_2025_TOC':
            category = 'Table of Contents'
        elif pmid == 'BIRADS_2025_INTRODUCTION':
            category = 'Introduction'
        elif pmid == 'BIRADS_2025_SECTION_II':
            category = 'Section II: Breast Imaging Lexicon'
        elif pmid == 'BIRADS_2025_SECTION_III':
            category = 'Section III: Breast Density'
        elif 'BIRADS_2025_SECTION_IV' in pmid:
            category = 'Section IV: Findings'
        elif pmid == 'BIRADS_2025_SECTION_V':
            category = 'Section V: Reporting System'
        elif pmid == 'BIRADS_2025_SECTION_VI':
            category = 'Section VI: Guidance'
        elif pmid == 'BIRADS_2025_SECTION_VII':
            category = 'Section VII: FAQ'
        elif 'BIRADS_2025_APPENDIX' in pmid:
            category = 'Appendix'
        # CEM 챕터
        elif pmid == 'BIRADS_CEM_PREFACE':
            category = 'Introduction'
        elif pmid == 'BIRADS_CEM_SECTION_I':
            category = 'Section I: General Considerations'
        elif pmid == 'BIRADS_CEM_SECTION_II':
            category = 'Section II: Breast Imaging Lexicon'
        elif pmid == 'BIRADS_CEM_SECTION_III':
            category = 'Section III: Image Examples'
        elif 'BIRADS_CEM_SECTION_III_' in pmid:
            category = 'Section III: Image Examples'
        elif pmid == 'BIRADS_CEM_SECTION_IV':
            category = 'Section IV: Reporting System'
        elif pmid == 'BIRADS_CEM_SECTION_V':
            category = 'Section V: Guidance'
        elif pmid == 'BIRADS_CEM_SECTION_VI':
            category = 'Section VI: FAQ'
        elif 'BIRADS_CEM_APPENDIX' in pmid:
            category = 'Appendix'
        # Auditing and Outcomes Monitoring 챕터
        elif pmid == 'BIRADS_AUDIT_INTRO':
            category = 'Introduction'
        elif pmid == 'BIRADS_AUDIT_SECTION_I':
            category = 'Section I: Glossary'
        elif pmid == 'BIRADS_AUDIT_SECTION_II':
            category = 'Section II: Basic Audit'
        elif pmid == 'BIRADS_AUDIT_SECTION_III':
            category = 'Section III: Complete Audit'
        elif pmid == 'BIRADS_AUDIT_SECTION_IV':
            category = 'Section IV: Classification Examples'
        elif pmid == 'BIRADS_AUDIT_SECTION_V':
            category = 'Section V: Data Collection'
        elif pmid == 'BIRADS_AUDIT_SECTION_VI':
            category = 'Section VI: FAQ'
        elif pmid == 'BIRADS_AUDIT_SECTION_VII':
            category = 'Section VII: Method of Detection'
        else:
            # 카테고리가 정의되지 않은 문서는 건너뜀
            continue

        categories[category].append(doc)

    # 각 카테고리 내 문서 정렬 (PMID 순서)
    pmid_order = {
        # Mammography
        'BIRADS_2025_TOC': 0,
        'BIRADS_2025_INTRODUCTION': 1,
        'BIRADS_2025_SECTION_II': 2,
        'BIRADS_2025_SECTION_III': 3,
        'BIRADS_2025_SECTION_IV_A': 4,
        'BIRADS_2025_SECTION_IV_B': 5,
        'BIRADS_2025_SECTION_IV_B1': 6,
        'BIRADS_2025_SECTION_IV_B2': 7,
        'BIRADS_2025_SECTION_IV_B3': 8,
        'BIRADS_2025_SECTION_IV_C': 9,
        'BIRADS_2025_SECTION_IV_D': 10,
        'BIRADS_2025_SECTION_IV_E': 11,
        'BIRADS_2025_SECTION_IV_F': 12,
        'BIRADS_2025_SECTION_IV_G': 13,
        'BIRADS_2025_SECTION_IV_H': 14,
        'BIRADS_2025_SECTION_IV_I': 15,
        'BIRADS_2025_SECTION_IV_J': 16,
        'BIRADS_2025_SECTION_V': 17,
        'BIRADS_2025_SECTION_VI': 18,
        'BIRADS_2025_SECTION_VII': 19,
        'BIRADS_2025_APPENDIX_A': 20,
        'BIRADS_2025_APPENDIX_B': 21,
        # CEM
        'BIRADS_CEM_PREFACE': 0,
        'BIRADS_CEM_SECTION_I': 1,
        'BIRADS_CEM_SECTION_II': 2,
        'BIRADS_CEM_SECTION_III': 3,
        'BIRADS_CEM_SECTION_III_A': 4,
        'BIRADS_CEM_SECTION_III_B': 5,
        'BIRADS_CEM_SECTION_III_C': 6,
        'BIRADS_CEM_SECTION_III_D': 7,
        'BIRADS_CEM_SECTION_III_E': 8,
        'BIRADS_CEM_SECTION_III_F': 9,
        'BIRADS_CEM_SECTION_IV': 10,
        'BIRADS_CEM_SECTION_V': 11,
        'BIRADS_CEM_SECTION_VI': 12,
        'BIRADS_CEM_APPENDIX_A': 13,
        # Auditing and Outcomes Monitoring
        'BIRADS_AUDIT_INTRO': 0,
        'BIRADS_AUDIT_SECTION_I': 1,
        'BIRADS_AUDIT_SECTION_II': 2,
        'BIRADS_AUDIT_SECTION_III': 3,
        'BIRADS_AUDIT_SECTION_IV': 4,
        'BIRADS_AUDIT_SECTION_V': 5,
        'BIRADS_AUDIT_SECTION_VI': 6,
        'BIRADS_AUDIT_SECTION_VII': 7
    }

    for cat in categories:
        categories[cat].sort(key=lambda x: pmid_order.get(x[0], 999))

    return categories


# 문서 로드
with st.spinner(f'📚 {selected_chapter} 문서 로딩 중...'):
    categories = load_birads_documents(selected_chapter)

# 챕터 상태 표시
if not any(categories.values()):
    st.warning(f"⚠️ '{selected_chapter}' 챕터는 아직 준비 중입니다. Mammography 챕터를 선택해주세요.")
    st.stop()

# 사이드바
st.sidebar.title("📑 목차")
st.sidebar.caption(f"현재 챕터: {selected_chapter}")

# 카테고리 선택
selected_category = st.sidebar.radio(
    "카테고리",
    options=[cat for cat, docs in categories.items() if docs]
)

st.sidebar.markdown("---")

# 검색 기능
search_query = st.sidebar.text_input("🔍 검색", placeholder="제목이나 내용 검색...")

# 통계 표시
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 통계")
total_docs = sum(len(docs) for docs in categories.values())
st.sidebar.metric("전체 문서", f"{total_docs}개")

# 메인 컨텐츠
if search_query:
    # 검색 결과 표시
    st.header(f"🔍 검색 결과: '{search_query}'")

    results = []
    for cat, docs in categories.items():
        for doc in docs:
            pmid, title, content, authors, journal, year = doc
            if search_query.lower() in title.lower() or (content and search_query.lower() in content.lower()):
                results.append((cat, doc))

    if results:
        st.success(f"{len(results)}개 문서 발견")

        for cat, (pmid, title, content, authors, journal, year) in results:
            with st.expander(f"**{title}** ({cat})"):
                st.markdown(f"**저자:** {authors or 'ACR Committee on BI-RADS'}")
                st.markdown(f"**출처:** {journal or 'ACR BI-RADS Atlas'} ({year or '2025'})")
                st.markdown(f"**문서 ID:** `{pmid}`")
                st.markdown("---")
                render_markdown_with_images(content)
    else:
        st.warning("검색 결과가 없습니다.")

else:
    # 선택된 카테고리 표시
    docs = categories.get(selected_category, [])

    if docs:
        st.header(f"📘 {selected_category}")
        st.caption(f"{len(docs)}개 문서")

        for pmid, title, content, authors, journal, year in docs:
            st.markdown(f'<div class="document-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="document-title">{title}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="document-meta"><strong>저자:</strong> {authors or "ACR Committee on BI-RADS"} | <strong>출처:</strong> {journal or "ACR BI-RADS Atlas"} ({year or "2025"})</div>', unsafe_allow_html=True)

            # 텍스트 내용
            with st.expander("📖 전체 내용 보기", expanded=False):
                content_len = len(content) if content else 0
                st.caption(f"💡 총 {content_len:,}글자")
                render_markdown_with_images(content)


            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info(f"{selected_category} 카테고리에 문서가 없습니다.")

# Footer
st.markdown("---")
st.caption("Generated by Sophia AI | 교육 및 연구 목적으로 제공되며, 의학적 조언을 대체할 수 없습니다.")
