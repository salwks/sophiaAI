"""
BI-RADS Guidelines - Hierarchical Navigation
"""
import streamlit as st
import sqlite3
from pathlib import Path
import re
from PIL import Image

st.set_page_config(
    page_title="BI-RADS Guidelines",
    page_icon="📘",
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

button[kind="primary"],
button[kind="secondary"] {
    text-align: left !important;
    justify-content: flex-start !important;
}
</style>
""", unsafe_allow_html=True)

def render_content_with_images(content: str):
    """마크다운 컨텐츠를 이미지와 함께 렌더링"""
    if not content:
        return

    image_pattern = r'!\[([^\]]+)\]\(([^\)]+)\)'
    parts = re.split(image_pattern, content)

    i = 0
    while i < len(parts):
        if i % 3 == 0:
            if parts[i].strip():
                st.markdown(parts[i])
        elif i % 3 == 2:
            image_path = parts[i]
            image_alt = parts[i - 1] if i > 0 else "Image"
            try:
                img = Image.open(image_path)
                st.image(img, caption=image_alt, use_container_width=True)
            except Exception as e:
                st.warning(f"⚠️ 이미지를 로드할 수 없습니다: {image_path}")
        i += 1

@st.cache_data
def get_document(pmid: str):
    """특정 문서 로드"""
    db_path = Path("data/index/metadata.db")
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.execute("""
        SELECT pmid, title, full_content
        FROM papers
        WHERE pmid = ?
    """, (pmid,))

    row = cursor.fetchone()
    conn.close()

    if row:
        return {
            'pmid': row[0],
            'title': row[1],
            'content': row[2]
        }
    return None

@st.cache_data
def get_chunks_for_section(parent_pmid: str):
    """특정 섹션의 청크들 조회"""
    db_path = Path("data/index/metadata.db")
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.execute("""
        SELECT pmid, title
        FROM papers
        WHERE pmid LIKE ? AND pmid LIKE '%CHUNK%'
        ORDER BY pmid
    """, (f"{parent_pmid}%",))

    chunks = []
    for row in cursor.fetchall():
        chunk_name = row[0].split('_CHUNK_')[1] if '_CHUNK_' in row[0] else row[0]
        title = row[1].replace("BI-RADS ", "").replace("Mass ", "").replace("Calcification", "Calc").strip()
        chunks.append({'pmid': row[0], 'title': title, 'chunk_name': chunk_name})

    conn.close()
    return chunks

# URL 파라미터로 네비게이션 추적
query_params = st.query_params
modality = query_params.get("modality", None)
current_section = query_params.get("section", None)
current_subsection = query_params.get("sub", None)
target_chunk = query_params.get("chunk", None)  # 자동 펼침 대상 chunk

# 헤더
st.title("BI-RADS Atlas v2025")
st.markdown("---")

# 레벨 0: Modality 선택
if not modality:
    st.markdown("## Select Category")
    st.markdown("")

    if st.button("Mammography", key="modality_mammo", use_container_width=True):
        st.query_params.modality = "mammography"
        st.rerun()

    if st.button("Contrast-Enhanced Mammography (CEM)", key="modality_cem", use_container_width=True):
        st.query_params.modality = "cem"
        st.rerun()

    if st.button("Auditing & Outcomes Monitoring", key="modality_audit", use_container_width=True):
        st.query_params.modality = "audit"
        st.rerun()

    if st.button("General Frequently Asked Questions", key="modality_faq", use_container_width=True):
        st.query_params.modality = "general_faq"
        st.rerun()

# ============================================================================
# MAMMOGRAPHY
# ============================================================================
elif modality == "mammography":

    # 레벨 1: Mammography 메인 섹션
    if not current_section:
        if st.button("← Back to Modality Selection"):
            st.query_params.clear()
            st.rerun()

        st.markdown("## Mammography - Table of Contents")
        st.markdown("")

        sections = [
            {'pmid': 'BIRADS_2025_INTRODUCTION', 'title': 'Introduction'},
            {'pmid': 'BIRADS_2025_SECTION_II', 'title': 'II. Breast Imaging Lexicon — Mammography'},
            {'pmid': 'BIRADS_2025_SECTION_III', 'title': 'III. Breast Density'},
            {'pmid': 'BIRADS_2025_SECTION_IV', 'title': 'IV. Findings - Mammography', 'has_subsections': True},
            {'pmid': 'BIRADS_2025_SECTION_V', 'title': 'V. Reporting System', 'has_subsections': True},
            {'pmid': 'BIRADS_2025_SECTION_VI', 'title': 'VI. Guidance'},
            {'pmid': 'BIRADS_2025_APPENDIX_A', 'title': 'Appendix A: Mammographic Views'},
            {'pmid': 'BIRADS_2025_APPENDIX_B', 'title': 'Appendix B: Mammography Lexicon Summary Form'},
        ]

        for section in sections:
            title = section['title']
            pmid = section['pmid']
            has_sub = section.get('has_subsections', False)
            arrow = " ▶" if has_sub else ""

            if st.button(f"{title}{arrow}", key=pmid, use_container_width=True):
                if has_sub:
                    st.query_params.section = pmid
                else:
                    st.query_params.section = pmid
                    st.query_params.view = "content"
                st.rerun()

    # 레벨 2: Section IV 서브섹션
    elif current_section == "BIRADS_2025_SECTION_IV" and not current_subsection:
        if st.button("← Back to Mammography"):
            st.query_params.modality = modality
            st.query_params.pop("section", None)
            st.rerun()

        st.markdown("## Section IV. Findings - Mammography")
        st.markdown("")

        subsections = [
            {'pmid': 'BIRADS_2025_SECTION_IV_A', 'title': 'A. Masses', 'has_chunks': True},
            {'pmid': 'BIRADS_2025_SECTION_IV_B', 'title': 'B. Calcifications', 'has_subsections': True},
            {'pmid': 'BIRADS_2025_SECTION_IV_C', 'title': 'C. Architectural Distortion'},
            {'pmid': 'BIRADS_2025_SECTION_IV_D', 'title': 'D. Asymmetries', 'has_chunks': True},
            {'pmid': 'BIRADS_2025_SECTION_IV_E', 'title': 'E. Lymph Nodes'},
            {'pmid': 'BIRADS_2025_SECTION_IV_F', 'title': 'F. Skin Lesions'},
            {'pmid': 'BIRADS_2025_SECTION_IV_G', 'title': 'G. Dilated Ducts'},
            {'pmid': 'BIRADS_2025_SECTION_IV_H', 'title': 'H. Associated Features'},
            {'pmid': 'BIRADS_2025_SECTION_IV_I', 'title': 'I. Special Cases'},
            {'pmid': 'BIRADS_2025_SECTION_IV_J', 'title': 'J. Location of Finding'},
        ]

        for subsection in subsections:
            title = subsection['title']
            pmid = subsection['pmid']
            has_chunks = subsection.get('has_chunks', False)
            has_sub = subsection.get('has_subsections', False)
            arrow = " ▶" if (has_chunks or has_sub) else ""

            if st.button(f"{title}{arrow}", key=pmid, use_container_width=True):
                st.query_params.sub = pmid
                st.rerun()

    # 레벨 2: Section V 서브섹션
    elif current_section == "BIRADS_2025_SECTION_V" and not current_subsection:
        if st.button("← Back to Mammography"):
            st.query_params.modality = modality
            st.query_params.pop("section", None)
            st.rerun()

        doc = get_document("BIRADS_2025_SECTION_V")
        if doc:
            st.markdown("## Section V. Reporting System")
            if doc['content']:
                render_content_with_images(doc['content'])
            st.markdown("---")

        st.markdown("### BI-RADS Categories")
        st.markdown("")

        categories = [
            {'pmid': 'BIRADS_2025_SECTION_V_CAT0', 'title': 'Category 0: Incomplete Assessment'},
            {'pmid': 'BIRADS_2025_SECTION_V_CAT1', 'title': 'Category 1: Negative'},
            {'pmid': 'BIRADS_2025_SECTION_V_CAT2', 'title': 'Category 2: Benign'},
            {'pmid': 'BIRADS_2025_SECTION_V_CAT3', 'title': 'Category 3: Probably Benign'},
            {'pmid': 'BIRADS_2025_SECTION_V_CAT4', 'title': 'Category 4: Suspicious'},
            {'pmid': 'BIRADS_2025_SECTION_V_CAT5', 'title': 'Category 5: Highly Suggestive of Malignancy'},
            {'pmid': 'BIRADS_2025_SECTION_V_CAT6', 'title': 'Category 6: Known Biopsy-Proven Malignancy'},
        ]

        for cat in categories:
            if st.button(cat['title'], key=cat['pmid'], use_container_width=True):
                st.query_params.sub = cat['pmid']
                st.rerun()

    # 레벨 3: Section IV.B (Calcifications)
    elif current_subsection == "BIRADS_2025_SECTION_IV_B":
        if st.button("← Back to Section IV"):
            st.query_params.pop("sub", None)
            st.rerun()

        st.markdown("## B. Calcifications")

        doc = get_document(current_subsection)
        if doc and doc['content']:
            render_content_with_images(doc['content'])
            st.markdown("---")

        st.markdown("### Subcategories:")
        st.markdown("")

        subsections = [
            {'pmid': 'BIRADS_2025_SECTION_IV_B1', 'title': '1. Typically Benign', 'has_chunks': True},
            {'pmid': 'BIRADS_2025_SECTION_IV_B2', 'title': '2. Suspicious Morphology', 'has_chunks': True},
            {'pmid': 'BIRADS_2025_SECTION_IV_B3', 'title': '3. Distribution', 'has_chunks': True},
        ]

        for sub in subsections:
            arrow = " ▶" if sub.get('has_chunks', False) else ""
            if st.button(f"{sub['title']}{arrow}", key=sub['pmid'], use_container_width=True):
                st.query_params.sub = sub['pmid']
                st.rerun()

    # 레벨 3/4: 청크가 있는 서브섹션
    elif current_subsection and current_subsection.startswith("BIRADS_2025_SECTION_IV"):
        if current_subsection.startswith("BIRADS_2025_SECTION_IV_B"):
            back_label = "← Back to B. Calcifications"
            back_target = "BIRADS_2025_SECTION_IV_B"
        else:
            back_label = "← Back to Section IV"
            back_target = None

        if st.button(back_label):
            if back_target:
                st.query_params.sub = back_target
            else:
                st.query_params.pop("sub", None)
            st.rerun()

        doc = get_document(current_subsection)
        if doc:
            short_title = doc['title'].replace("BI-RADS v2025 Manual - ", "").replace("IV. Findings - ", "")
            st.markdown(f"## {short_title}")

            if doc['content']:
                render_content_with_images(doc['content'])
                st.markdown("---")

        chunks = get_chunks_for_section(current_subsection)

        if chunks:
            st.markdown("### Details:")

            for chunk in chunks:
                # target_chunk과 일치하면 자동으로 펼치기
                is_target = target_chunk and chunk['chunk_name'] == target_chunk
                with st.expander(f"{chunk['title']}", expanded=is_target):
                    chunk_doc = get_document(chunk['pmid'])
                    if chunk_doc and chunk_doc['content']:
                        render_content_with_images(chunk_doc['content'])
                    else:
                        st.warning("내용 없음")

    # 레벨 3: Section V 카테고리
    elif current_subsection and current_subsection.startswith("BIRADS_2025_SECTION_V_CAT"):
        if st.button("← Back to Section V"):
            st.query_params.pop("sub", None)
            st.rerun()

        doc = get_document(current_subsection)
        if doc:
            short_title = doc['title'].replace("BI-RADS ", "")
            st.markdown(f"## {short_title}")

            if doc['content']:
                render_content_with_images(doc['content'])
            else:
                st.info("내용 없음")

    # 단일 문서 보기
    elif query_params.get("view") == "content":
        if st.button("← Back to Mammography"):
            st.query_params.modality = modality
            st.query_params.pop("section", None)
            st.query_params.pop("view", None)
            st.rerun()

        doc = get_document(current_section)
        if doc:
            short_title = doc['title'].replace("BI-RADS v2025 Manual - ", "")
            st.markdown(f"## {short_title}")

            if doc['content']:
                render_content_with_images(doc['content'])
            else:
                st.info("내용 없음")

# ============================================================================
# CONTRAST-ENHANCED MAMMOGRAPHY (CEM)
# ============================================================================
elif modality == "cem":

    if not current_section:
        if st.button("← Back to Modality Selection"):
            st.query_params.clear()
            st.rerun()

        st.markdown("## Contrast-Enhanced Mammography (CEM) - Table of Contents")
        st.markdown("")

        sections = [
            {'pmid': 'BIRADS_CEM_PREFACE', 'title': 'Preface and Introduction'},
            {'pmid': 'BIRADS_CEM_SECTION_I', 'title': 'I. General Considerations'},
            {'pmid': 'BIRADS_CEM_SECTION_II', 'title': 'II. Breast Imaging Lexicon'},
            {'pmid': 'BIRADS_CEM_SECTION_III_A', 'title': 'III. Image Examples - A. Background Parenchymal Enhancement'},
            {'pmid': 'BIRADS_CEM_SECTION_III_B', 'title': 'III. Image Examples - B. Lesion Conspicuity'},
            {'pmid': 'BIRADS_CEM_SECTION_III_C', 'title': 'III. Image Examples - C. Masses on Recombined Images'},
            {'pmid': 'BIRADS_CEM_SECTION_III_D', 'title': 'III. Image Examples - D. Non-mass Enhancement'},
            {'pmid': 'BIRADS_CEM_SECTION_III_E', 'title': 'III. Image Examples - E. Enhancing Asymmetry'},
            {'pmid': 'BIRADS_CEM_SECTION_III_F', 'title': 'III. Image Examples - F. Abnormality on Both Images'},
            {'pmid': 'BIRADS_CEM_SECTION_IV', 'title': 'IV. Reporting System'},
            {'pmid': 'BIRADS_CEM_SECTION_V', 'title': 'V. Guidance'},
            {'pmid': 'BIRADS_CEM_SECTION_VI', 'title': 'VI. Frequently Asked Questions'},
            {'pmid': 'BIRADS_CEM_APPENDIX_A', 'title': 'Appendix A: Lexicon Summary Form'},
        ]

        for section in sections:
            if st.button(section['title'], key=section['pmid'], use_container_width=True):
                st.query_params.section = section['pmid']
                st.query_params.view = "content"
                st.rerun()

    elif query_params.get("view") == "content":
        if st.button("← Back to CEM Table of Contents"):
            st.query_params.modality = modality
            st.query_params.pop("section", None)
            st.query_params.pop("view", None)
            st.rerun()

        doc = get_document(current_section)
        if doc:
            short_title = doc['title'].replace("BI-RADS v2025 Manual - CEM: ", "").replace("BI-RADS v2025 Manual - ", "")
            st.markdown(f"## {short_title}")

            if doc['content']:
                render_content_with_images(doc['content'])
            else:
                st.info("내용 없음")

# ============================================================================
# AUDITING & OUTCOMES MONITORING
# ============================================================================
elif modality == "audit":

    if not current_section:
        if st.button("← Back to Modality Selection"):
            st.query_params.clear()
            st.rerun()

        st.markdown("## Auditing & Outcomes Monitoring - Table of Contents")
        st.markdown("")

        sections = [
            {'pmid': 'BIRADS_AUDIT_INTRO', 'title': 'Preface and Introduction'},
            {'pmid': 'BIRADS_AUDIT_SECTION_I', 'title': 'I. Glossary'},
            {'pmid': 'BIRADS_AUDIT_SECTION_II', 'title': 'II. The Basic Clinically Relevant Audit'},
            {'pmid': 'BIRADS_AUDIT_SECTION_III', 'title': 'III. The More Complete Audit'},
            {'pmid': 'BIRADS_AUDIT_SECTION_IV', 'title': 'IV. Examples of How to Classify Examinations'},
            {'pmid': 'BIRADS_AUDIT_SECTION_V', 'title': 'V. Areas of Confusion in Data Collection'},
            {'pmid': 'BIRADS_AUDIT_SECTION_VI', 'title': 'VI. Frequently Asked Questions'},
            {'pmid': 'BIRADS_AUDIT_SECTION_VII', 'title': 'VII. Initial Method of Detection'},
        ]

        for section in sections:
            if st.button(section['title'], key=section['pmid'], use_container_width=True):
                st.query_params.section = section['pmid']
                st.query_params.view = "content"
                st.rerun()

    elif query_params.get("view") == "content":
        if st.button("← Back to Audit Table of Contents"):
            st.query_params.modality = modality
            st.query_params.pop("section", None)
            st.query_params.pop("view", None)
            st.rerun()

        doc = get_document(current_section)
        if doc:
            short_title = doc['title'].replace("BI-RADS v2025 Manual - Auditing: ", "").replace("BI-RADS v2025 Manual - ", "").replace("BI-RADS v2025 Auditing - ", "")
            st.markdown(f"## {short_title}")

            if doc['content']:
                render_content_with_images(doc['content'])
            else:
                st.info("내용 없음")

# ============================================================================
# GENERAL FAQ
# ============================================================================
elif modality == "general_faq":

    if st.button("← Back to Category Selection"):
        st.query_params.clear()
        st.rerun()

    doc = get_document("BIRADS_2025_SECTION_VII")
    if doc:
        st.markdown("## General Frequently Asked Questions")

        if doc['content']:
            render_content_with_images(doc['content'])
        else:
            st.info("내용 없음")

# 푸터
st.markdown("""
---
**BI-RADS Atlas v2025**
American College of Radiology (ACR)
© 2025 All rights reserved.
""")
