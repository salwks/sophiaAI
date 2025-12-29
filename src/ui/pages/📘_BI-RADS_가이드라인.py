"""
BI-RADS 가이드라인 전체 보기
============================
BI-RADS 2025 가이드라인 전문을 카테고리별로 탐색
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
</style>
""", unsafe_allow_html=True)

# 헤더
st.markdown('<div class="category-header"><h1>📘 BI-RADS 2025 가이드라인</h1><p>Breast Imaging Reporting and Data System - 전체 문서</p></div>', unsafe_allow_html=True)

# 데이터베이스에서 BI-RADS 문서 가져오기
@st.cache_data(ttl=3600)
def load_birads_documents():
    """BI-RADS 문서를 데이터베이스에서 로드"""
    db_path = Path('data/index/metadata.db')

    if not db_path.exists():
        return {}

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
        SELECT pmid, title, full_content, authors, journal, year
        FROM papers
        WHERE pmid LIKE 'BIRADS_%'
        ORDER BY title
    ''')

    documents = cursor.fetchall()
    conn.close()

    # 카테고리별로 분류
    categories = {
        'Category 0': [],
        'Category 1': [],
        'Category 2': [],
        'Category 3': [],
        'Category 4': [],
        'Category 4A': [],
        'Category 4B': [],
        'Category 4C': [],
        'Category 5': [],
        'Category 6': [],
        '기타 (Other)': []
    }

    for doc in documents:
        pmid, title, content, authors, journal, year = doc

        # 카테고리 분류
        if 'Category 0' in title or 'category 0' in title:
            categories['Category 0'].append(doc)
        elif 'Category 1' in title or 'category 1' in title:
            categories['Category 1'].append(doc)
        elif 'Category 2' in title or 'category 2' in title:
            categories['Category 2'].append(doc)
        elif 'Category 3' in title or 'category 3' in title:
            categories['Category 3'].append(doc)
        elif 'Category 4C' in title or '4C' in title:
            categories['Category 4C'].append(doc)
        elif 'Category 4B' in title or '4B' in title:
            categories['Category 4B'].append(doc)
        elif 'Category 4A' in title or '4A' in title:
            categories['Category 4A'].append(doc)
        elif 'Category 4' in title or 'category 4' in title:
            categories['Category 4'].append(doc)
        elif 'Category 5' in title or 'category 5' in title:
            categories['Category 5'].append(doc)
        elif 'Category 6' in title or 'category 6' in title:
            categories['Category 6'].append(doc)
        else:
            categories['기타 (Other)'].append(doc)

    return categories

# 문서 로드
with st.spinner('📚 BI-RADS 문서 로딩 중...'):
    categories = load_birads_documents()

# 사이드바 네비게이션
st.sidebar.title("📑 목차")
st.sidebar.markdown("카테고리를 선택하세요:")

selected_category = st.sidebar.radio(
    "카테고리 선택",
    options=[cat for cat, docs in categories.items() if docs],
    format_func=lambda x: f"{x} ({len(categories[x])}개)"
)

# 검색 기능
st.sidebar.markdown("---")
search_query = st.sidebar.text_input("🔍 검색", placeholder="제목이나 내용 검색...")

# 통계 표시
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 통계")
total_docs = sum(len(docs) for docs in categories.values())
st.sidebar.metric("전체 문서", f"{total_docs}개")

for cat, docs in categories.items():
    if docs:
        st.sidebar.metric(cat, f"{len(docs)}개")

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
                st.markdown(content or "_내용 없음_")
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

            with st.expander("📖 전체 내용 보기", expanded=False):
                content_len = len(content) if content else 0
                st.caption(f"💡 총 {content_len:,}글자")
                st.markdown(f'<div class="document-content">{content or "_내용 없음_"}</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info(f"{selected_category} 카테고리에 문서가 없습니다.")

# Footer
st.markdown("---")
st.caption("Generated by Sophia AI - Sophia AI | 교육 및 연구 목적으로 제공되며, 의학적 조언을 대체할 수 없습니다.")
