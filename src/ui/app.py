"""
Sophia AI Alpha: RAG-based Medical AI Assistant
===============================================
논문 검색 기반 의료 AI 어시스턴트 (할루시네이션 최소화)
"""

import os
import sys
from pathlib import Path
import streamlit as st
import requests
import json

# 프로젝트 루트를 path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.search.engine import SearchEngine

# =============================================================================
# 페이지 설정
# =============================================================================

st.set_page_config(
    page_title="Sophia AI Alpha",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# 검색 엔진 & LLM 초기화
# =============================================================================

@st.cache_resource(ttl=3600)  # 1시간마다 캐시 갱신
def get_search_engine():
    """검색 엔진 싱글톤 (BI-RADS 포함)"""
    return SearchEngine(
        db_path=Path("data/index"),
        parser_mode="smart",
        ollama_url="http://localhost:11434",
        llm_model="qwen2.5:14b",
        use_reranker=True,
    )

def call_llm_with_context(question: str, context: str, model="qwen2.5:14b", temperature=0.7):
    """
    RAG: 검색된 논문 컨텍스트를 기반으로 LLM 답변 생성

    Args:
        question: 사용자 질문
        context: 검색된 논문 내용
        model: LLM 모델명
        temperature: 온도 설정

    Returns:
        LLM 답변 (generator)
    """
    url = "http://localhost:11434/api/chat"

    system_message = """당신은 유방영상의학 전문 AI 어시스턴트입니다.

**중요한 규칙:**
1. 제공된 논문과 BI-RADS 가이드라인 내용만을 기반으로 답변하세요
2. 추측하지 말고, 제공된 정보에 없으면 "제공된 자료에서 해당 정보를 찾을 수 없습니다"라고 답변하세요
3. 답변 시 반드시 출처를 명시하세요 (예: [1], [2] 형식)
4. 의학적 조언이 아닌 연구 정보 제공임을 명확히 하세요
5. 한국어로 답변하세요"""

    user_message = f"""다음 논문들과 가이드라인을 참고하여 질문에 답변해주세요.

**참고 자료:**
{context}

**질문:** {question}

**답변 형식:**
- 제공된 자료를 기반으로 답변
- 출처 번호 [1], [2] 등으로 명시
- 의학적 조언이 아닌 연구 정보임을 명확히 표시"""

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": temperature,
        }
    }

    try:
        response = requests.post(url, json=payload, stream=True, timeout=180)
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                if "message" in chunk and "content" in chunk["message"]:
                    yield chunk["message"]["content"]
    except requests.exceptions.RequestException as e:
        yield f"⚠️ LLM 연결 오류: {str(e)}"

# =============================================================================
# 사이드바
# =============================================================================

def render_sidebar():
    """사이드바 렌더링"""
    with st.sidebar:
        st.markdown("### ⚙️ Settings")

        st.markdown("#### Model")
        model = st.selectbox(
            "LLM Model",
            options=["qwen2.5:14b"],
            index=0,
        )

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1,
            help="낮을수록 일관적, 높을수록 창의적"
        )

        st.markdown("#### Search")
        top_k = st.slider(
            "참고 논문 수",
            min_value=3,
            max_value=10,
            value=5,
            step=1,
            help="답변 생성 시 참고할 논문 수"
        )

        st.markdown("---")

        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.markdown("---")
        st.markdown("#### About")
        st.markdown("""
        <div style="font-size: 0.85rem; line-height: 1.6;">
        <b>Sophia AI Alpha</b><br>
        유방영상의학 논문 기반 RAG 시스템<br><br>

        ✅ 할루시네이션 최소화<br>
        ✅ 논문 출처 명시<br>
        ✅ BI-RADS 가이드라인 참조
        </div>
        """, unsafe_allow_html=True)

        return {
            "model": model,
            "temperature": temperature,
            "top_k": top_k,
        }

# =============================================================================
# 메인 UI
# =============================================================================

def main():
    """메인 RAG 챗봇"""

    # 사이드바
    options = render_sidebar()

    # 헤더
    st.markdown("""
    <h1 style='font-size: 2.5rem; margin-bottom: 0;'>
        💬 Sophia AI
        <span style='font-size: 0.9rem; color: #888888; font-weight: normal; vertical-align: super;'>Alpha</span>
    </h1>
    """, unsafe_allow_html=True)
    st.caption("🚀 유방영상의학 논문 기반 AI 어시스턴트 (RAG)")

    # 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{
            "role": "assistant",
            "content": "안녕하세요! 유방영상의학 관련 질문을 해주시면, 관련 논문과 BI-RADS 가이드라인을 참고하여 답변해드리겠습니다.\n\n예시:\n- Mammography에 대한 기본 개념 설명\n- DBT와 FFDM의 차이점\n- BI-RADS 카테고리 설명"
        }]

    # 대화 히스토리 표시
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # 출처 표시
            if "sources" in msg:
                with st.expander("📚 참고 자료", expanded=False):
                    for i, source in enumerate(msg["sources"], 1):
                        icon = "📘" if source.get("is_birads", False) else "📄"

                        if source.get("is_birads", False):
                            # BI-RADS 문서는 내용을 직접 표시
                            st.markdown(f"**{icon} [{i}] {source['title']}**")
                            st.markdown(f"{source['authors']} - {source['journal']} ({source['year']})")

                            # 전체 내용을 expander로 표시
                            if source.get("full_content"):
                                with st.expander("📖 전체 내용 보기", expanded=False):
                                    st.markdown(source["full_content"])
                            else:
                                st.caption("_내용을 불러올 수 없습니다_")
                        else:
                            # 일반 논문은 PubMed 링크
                            st.markdown(f"""
                            **{icon} [{i}] {source['title']}**
                            {source['authors']} - {source['journal']} ({source['year']})
                            [PubMed 보기]({source['url']})
                            """)

    # 채팅 입력
    if prompt := st.chat_input("질문을 입력하세요..."):
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # 검색 엔진으로 관련 논문 검색
        with st.spinner("📚 관련 논문 검색 중..."):
            try:
                engine = get_search_engine()

                # 검색 쿼리 최적화
                search_query = prompt
                prompt_lower = prompt.lower()

                # 1. BI-RADS 카테고리 질문 처리
                is_birads_concept = (
                    'bi-rads' in prompt_lower or 'birads' in prompt_lower or '카테고리' in prompt_lower
                ) and any(keyword in prompt_lower for keyword in ['기본', '개념', '정의', '설명', 'basic', 'concept', 'definition', '무엇'])

                if is_birads_concept and 'bi-rads' not in prompt_lower:
                    search_query = f"BI-RADS {prompt}"

                # 2. 기술 용어 질문 처리 (한영 혼용 → 영문 키워드 강화)
                technical_terms = {
                    'exposure': ['kVp', 'mAs', 'radiation dose', 'technique'],
                    'positioning': ['positioning', 'technique', 'procedure'],
                    'protocol': ['protocol', 'procedure', 'technique'],
                    '노출': ['exposure', 'kVp', 'mAs', 'radiation dose'],
                    '촬영': ['acquisition', 'technique', 'imaging'],
                    '포지셔닝': ['positioning', 'technique'],
                    '프로토콜': ['protocol', 'procedure'],
                }

                for term, keywords in technical_terms.items():
                    if term in prompt_lower:
                        # 기술 용어가 있으면 영문 키워드 추가
                        search_query = f"{prompt} {' '.join(keywords)}"
                        break

                search_response = engine.search(search_query, top_k=options["top_k"], use_rerank=True)

                if not search_response.results:
                    with st.chat_message("assistant"):
                        error_msg = "죄송합니다. 관련 논문을 찾을 수 없습니다. 다른 질문을 시도해보세요."
                        st.markdown(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    st.stop()

                # 컨텍스트 구성 (검색된 논문)
                context_parts = []
                sources = []

                for i, result in enumerate(search_response.results, 1):
                    paper = result.paper

                    # BI-RADS 문서 여부
                    is_birads = paper.pmid.startswith("BIRADS_")
                    doc_type = "📘 BI-RADS 가이드라인" if is_birads else "📄 논문"

                    # 컨텍스트에 추가
                    # BI-RADS는 full_content 사용, 일반 논문은 abstract 사용
                    if is_birads:
                        content_text = getattr(paper, 'full_content', paper.abstract or '내용 없음')
                    else:
                        content_text = (paper.abstract[:500] + '...' if paper.abstract and len(paper.abstract) > 500 else paper.abstract or '초록 없음')

                    context_parts.append(f"""
[{i}] {doc_type}
제목: {paper.title}
{'저자: ' + paper.author_string if paper.author_string else ''}
{'저널: ' + paper.journal + ' (' + str(paper.year) + ')' if paper.journal else ''}
내용: {content_text}
""")

                    # 출처 정보 저장
                    sources.append({
                        "title": paper.title,
                        "authors": paper.author_string or "저자 정보 없음",
                        "journal": paper.journal or "저널 정보 없음",
                        "year": paper.year or "연도 정보 없음",
                        "url": paper.pubmed_url if not is_birads else None,
                        "is_birads": is_birads,
                        "full_content": getattr(paper, 'full_content', None) if is_birads else None
                    })

                context = "\n".join(context_parts)

            except Exception as e:
                with st.chat_message("assistant"):
                    error_msg = f"⚠️ 검색 중 오류 발생: {str(e)}"
                    st.markdown(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                st.stop()

        # LLM으로 답변 생성 (RAG)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            # 스트리밍 답변
            for chunk in call_llm_with_context(
                question=prompt,
                context=context,
                model=options["model"],
                temperature=options["temperature"]
            ):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)

            # 출처 표시
            with st.expander("📚 참고 자료", expanded=False):
                for i, source in enumerate(sources, 1):
                    icon = "📘" if source["is_birads"] else "📄"

                    if source["is_birads"]:
                        # BI-RADS 문서는 내용을 직접 표시
                        st.markdown(f"**{icon} [{i}] {source['title']}**")
                        st.markdown(f"{source['authors']} - {source['journal']} ({source['year']})")

                        # 전체 내용을 expander로 표시
                        if source.get("full_content"):
                            with st.expander("📖 전체 내용 보기", expanded=False):
                                st.markdown(source["full_content"])
                        else:
                            st.caption("_내용을 불러올 수 없습니다_")
                    else:
                        # 일반 논문은 PubMed 링크
                        st.markdown(f"""
                        **{icon} [{i}] {source['title']}**
                        {source['authors']} - {source['journal']} ({source['year']})
                        [PubMed 보기]({source['url']})
                        """)

        # 어시스턴트 메시지 저장 (출처 포함)
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "sources": sources
        })


if __name__ == "__main__":
    main()
