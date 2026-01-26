"""
ACR Practice Parameters - Hierarchical Navigation
"""
import streamlit as st
import sqlite3
from pathlib import Path

st.set_page_config(
    page_title="ACR Practice Parameters",
    page_icon="📗",
    layout="wide",
)

# Custom CSS for left-aligned buttons
st.markdown("""
<style>
div.stButton > button {
    text-align: left !important;
    padding-left: 1.5rem !important;
    justify-content: flex-start !important;
}

div.stButton > button > div {
    text-align: left !important;
    justify-content: flex-start !important;
}

div.stButton > button p {
    text-align: left !important;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def get_document(pmid: str):
    """특정 문서 로드"""
    db_path = Path("data/index/metadata.db")
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.execute("""
        SELECT pmid, title, abstract, full_content, journal
        FROM papers
        WHERE pmid = ?
    """, (pmid,))

    row = cursor.fetchone()
    conn.close()

    if row:
        return {
            'pmid': row[0],
            'title': row[1],
            'abstract': row[2],
            'content': row[3] or row[2],  # full_content or abstract
            'journal': row[4]
        }
    return None


@st.cache_data
def get_acr_documents(prefix: str):
    """특정 카테고리의 ACR 문서들 조회"""
    db_path = Path("data/index/metadata.db")
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.execute("""
        SELECT pmid, title
        FROM papers
        WHERE pmid LIKE ?
        ORDER BY pmid
    """, (f"{prefix}%",))

    docs = []
    for row in cursor.fetchall():
        docs.append({'pmid': row[0], 'title': row[1]})

    conn.close()
    return docs


# URL 파라미터로 네비게이션 추적
query_params = st.query_params
category = query_params.get("category", None)
current_doc = query_params.get("doc", None)

# 헤더
st.title("📗 ACR Practice Parameters")
st.markdown("American College of Radiology Practice Guidelines for Mammography")
st.markdown("---")

# 레벨 0: Category 선택
if not category:
    st.markdown("## Select Category")
    st.markdown("")

    categories = [
        {"id": "mammo", "title": "Screening & Diagnostic Mammography", "prefix": "ACR_MAMMO_", "icon": "🏥"},
        {"id": "cem", "title": "Contrast-Enhanced Mammography (CEM)", "prefix": "ACR_CEM_", "icon": "💉"},
        {"id": "iq", "title": "Image Quality (ACR-AAPM-SIIM)", "prefix": "ACR_IQ_", "icon": "🖼️"},
    ]

    for cat in categories:
        if st.button(f"{cat['icon']} {cat['title']} ▶", key=cat['id'], use_container_width=True):
            st.query_params.category = cat['id']
            st.rerun()

# 레벨 1: 카테고리 내 문서 목록
elif category and not current_doc:
    if st.button("← Back to Categories"):
        st.query_params.clear()
        st.rerun()

    category_info = {
        "mammo": {"title": "Screening & Diagnostic Mammography", "prefix": "ACR_MAMMO_"},
        "cem": {"title": "Contrast-Enhanced Mammography (CEM)", "prefix": "ACR_CEM_"},
        "iq": {"title": "Image Quality (ACR-AAPM-SIIM)", "prefix": "ACR_IQ_"},
    }

    info = category_info.get(category, {})
    st.markdown(f"## {info.get('title', category)}")
    st.markdown("")

    docs = get_acr_documents(info.get('prefix', 'ACR_'))

    for doc in docs:
        # 제목 정리
        title = doc['title']
        title = title.replace("ACR Practice Parameter: ", "")
        title = title.replace("ACR CEM Practice Parameter - ", "")
        title = title.replace("ACR Image Quality: ", "")
        title = title.replace("ACR-AAPM-SIIM Practice Parameter: ", "")

        if st.button(f"📄 {title}", key=doc['pmid'], use_container_width=True):
            st.query_params.doc = doc['pmid']
            st.rerun()

# 레벨 2: 문서 내용 표시
elif current_doc:
    # 뒤로가기 버튼
    if st.button("← Back to Document List"):
        st.query_params.pop("doc", None)
        st.rerun()

    doc = get_document(current_doc)

    if doc:
        st.markdown(f"## {doc['title']}")

        # 출처 표시
        if doc.get('journal'):
            st.caption(f"📚 Source: {doc['journal']}")

        st.markdown("---")

        # 내용 표시
        content = doc.get('content') or doc.get('abstract') or "내용이 없습니다."
        st.markdown(content)

        # 하단에 출처 다시 표시
        st.markdown("---")
        st.info(f"📗 **ACR Practice Parameters**\n\nDocument ID: `{doc['pmid']}`\n\nSource: {doc.get('journal', 'ACR Practice Parameters')}")
    else:
        st.error(f"문서를 찾을 수 없습니다: {current_doc}")
