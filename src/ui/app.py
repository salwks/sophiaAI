"""
Sophia AI Alpha: RAG-based Medical AI Assistant
===============================================
논문 검색 기반 의료 AI 어시스턴트 (할루시네이션 최소화)
"""

import os
import sys
import copy
from pathlib import Path
import streamlit as st
import requests
import json

# 프로젝트 루트를 path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.search.engine import SearchEngine
from src.search.query_translator import get_translator

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

@st.cache_resource(ttl=3600)
def get_query_translator():
    """쿼리 번역기 싱글톤"""
    return get_translator(
        ollama_url="http://localhost:11434",
        model="qwen2.5:14b"
    )

def get_birads_nav_params(pmid: str) -> dict:
    """
    PMID로부터 BI-RADS 가이드라인 페이지 네비게이션 파라미터 생성

    예: BIRADS_2025_SECTION_IV_A_CHUNK_MARGIN
    → {"modality": "mammography", "section": "BIRADS_2025_SECTION_IV", "sub": "BIRADS_2025_SECTION_IV_A", "chunk": "MARGIN"}
    """
    if not pmid or not pmid.startswith("BIRADS"):
        return {"modality": "mammography"}

    params = {"modality": "mammography"}

    if "_CHUNK_" in pmid:
        parts = pmid.split("_CHUNK_")
        parent = parts[0]
        chunk_name = parts[1]

        if "SECTION_IV" in parent:
            params["section"] = "BIRADS_2025_SECTION_IV"
            params["sub"] = parent
            params["chunk"] = chunk_name
        elif "SECTION_V" in parent:
            params["section"] = "BIRADS_2025_SECTION_V"
            params["sub"] = parent
            params["chunk"] = chunk_name
    else:
        if "SECTION_IV" in pmid and len(pmid) > len("BIRADS_2025_SECTION_IV"):
            params["section"] = "BIRADS_2025_SECTION_IV"
            params["sub"] = pmid
        elif "SECTION_V" in pmid and len(pmid) > len("BIRADS_2025_SECTION_V"):
            params["section"] = "BIRADS_2025_SECTION_V"
            params["sub"] = pmid
        else:
            params["section"] = pmid
            params["view"] = "content"

    return params


def is_korean(text: str) -> bool:
    """텍스트에 한글이 포함되어 있는지 확인"""
    import re
    return bool(re.search(r'[가-힣]', text))


def get_messages(is_ko: bool) -> dict:
    """언어별 메시지 반환"""
    if is_ko:
        return {
            "found_high": "📘 **BI-RADS 가이드라인에서 관련 내용을 찾았습니다.**\n\n아래 원문을 확인해주세요.",
            "found_medium": "📋 **BI-RADS 가이드라인에서 관련될 수 있는 내용을 찾았습니다.**\n\n⚠️ _{reason}_\n\n아래 원문을 확인해주세요.",
            "not_found": "📭 **BI-RADS 가이드라인에서 관련 내용을 찾지 못했습니다.**\n\n_{reason}_",
            "no_results": "검색 결과가 없습니다. 다른 키워드로 검색해 주세요.",
            "view_source": "📘 원문 확인하기",
            "papers_high": "📄 관련 연구 논문",
            "papers_medium": "📄 관련될 수 있는 연구 논문 ⚠️",
            "verifying": "🔍 문서 관련성 검증 중...",
            "searching": "📚 관련 논문 검색 중..."
        }
    else:
        return {
            "found_high": "📘 **Found relevant content in BI-RADS Guidelines.**\n\nPlease check the original text below.",
            "found_medium": "📋 **Found possibly relevant content in BI-RADS Guidelines.**\n\n⚠️ _{reason}_\n\nPlease check the original text below.",
            "not_found": "📭 **No relevant content found in BI-RADS Guidelines.**\n\n_{reason}_",
            "no_results": "No search results. Please try different keywords.",
            "view_source": "📘 View Original",
            "papers_high": "📄 Related Research Papers",
            "papers_medium": "📄 Possibly Related Research Papers ⚠️",
            "verifying": "🔍 Verifying document relevance...",
            "searching": "📚 Searching for related papers..."
        }


def enhance_query_with_context(current_question: str, chat_history: list, model="qwen2.5:14b") -> str:
    """
    이전 대화 맥락을 참고해서 검색 쿼리를 보강

    Args:
        current_question: 현재 질문
        chat_history: 이전 대화 기록
        model: LLM 모델명

    Returns:
        보강된 검색 쿼리
    """
    # 이전 대화가 없으면 원본 질문 반환
    if not chat_history:
        return current_question

    # 질문이 충분히 길고 참조 단어가 없으면 새 질문으로 간주 (대화 보강 건너뛰기)
    reference_words = ["더", "그거", "그것", "위에서", "아까", "방금", "이전", "그러면", "그럼"]
    has_reference = any(word in current_question for word in reference_words)
    if len(current_question) > 20 and not has_reference:
        return current_question

    # 최근 4개 메시지만 사용 (토큰 절약)
    recent_history = chat_history[-4:]

    # 대화 기록 포맷팅
    history_text = ""
    for msg in recent_history:
        role = "사용자" if msg["role"] == "user" else "어시스턴트"
        content = msg["content"][:200]  # 너무 길면 자르기
        history_text += f"{role}: {content}\n"

    url = "http://localhost:11434/api/chat"

    system_message = """당신은 검색 쿼리 보강 전문가입니다.
사용자의 현재 질문이 이전 대화를 참조하는지 판단하고, 검색에 적합한 쿼리를 생성하세요.

규칙:
1. 현재 질문이 이전 맥락을 참조하면 → 맥락을 포함한 완전한 쿼리 생성
2. 현재 질문이 새로운 주제면 → 원본 질문 그대로 반환
3. 검색 쿼리만 출력 (설명 없이)
4. **중요**: 입력 언어(한국어)를 그대로 유지. 절대 다른 언어로 번역하지 마세요!

예시:
- 이전: "mass의 margin 분류는?" / 현재: "더 자세히" → "mass margin 분류 상세 설명"
- 이전: "calcification 종류" / 현재: "MRI 원리는?" → "MRI 원리" (새 주제)
- 이전: "BI-RADS 카테고리" / 현재: "3은 뭐야?" → "BI-RADS 카테고리 3 의미"
"""

    user_message = f"""이전 대화:
{history_text}

현재 질문: {current_question}

검색 쿼리:"""

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.1}
    }

    try:
        response = requests.post(url, json=payload, timeout=15)
        response.raise_for_status()
        result = response.json()
        enhanced = result.get("message", {}).get("content", "").strip()

        # 빈 결과면 원본 반환
        if not enhanced:
            return current_question

        return enhanced
    except Exception:
        # 오류시 원본 질문 반환
        return current_question


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
1. 절대로 내용을 요약하거나 해석하지 마세요 - 할루시네이션 방지
2. 질문에 해당하는 출처 번호만 안내하세요 (예: "[1]번 BI-RADS 가이드라인을 참조하세요")
3. 아래에 원문이 표시되니 사용자가 직접 확인하도록 안내하세요
4. 한국어로 간단히 답변하세요 (1-2문장)"""

    user_message = f"""다음 자료 중 질문에 답할 수 있는 출처 번호를 안내해주세요.
내용을 요약하지 말고, 출처 번호만 알려주세요.

**참고 자료:**
{context}

**질문:** {question}

**답변 예시:**
"[1]번 BI-RADS 가이드라인에서 해당 내용을 확인하실 수 있습니다. 아래 원문을 참조해주세요."
"""

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


def verify_relevance(question: str, documents: list, model="qwen2.5:14b") -> dict:
    """
    검색된 문서가 질문에 관련이 있는지 LLM으로 3단계 검증

    Returns:
        {"level": "high"/"medium"/"low", "reason": str, "relevant_indices": list}
    """
    url = "http://localhost:11434/api/chat"

    # 문서 내용 요약 (첫 500자씩)
    doc_summaries = []
    for i, doc in enumerate(documents, 1):
        content = doc.get('content', '')[:500]
        title = doc.get('title', '')
        doc_summaries.append(f"[{i}] {title}\n{content}...")

    docs_text = "\n\n".join(doc_summaries)

    system_message = """당신은 문서 관련성 검증 전문가입니다.
사용자의 질문에 대해 제공된 문서들이 **직접적인 답**을 포함하고 있는지 3단계로 판단하세요.

**판단 기준:**
- high: 문서가 질문에 대한 직접적인 답을 포함함
  예: "margin 분류는?" → 문서에 margin 분류 목록과 설명이 있음
- medium: 문서가 관련 주제를 다루지만 직접적 답은 없음
  예: "Mammography 기본 개념?" → 문서에 Mammography 언급만 있고 기본 개념 설명(원리, 방법 등)은 없음
- low: 문서가 질문과 거의 관련 없음
  예: "MRI 촬영 방법?" → 문서에 Mammography만 있고 MRI 정보 없음

반드시 다음 JSON 형식으로만 응답하세요:
{"level": "high/medium/low", "reason": "판단 이유", "relevant_indices": [관련 문서 번호들]}"""

    user_message = f"""질문: {question}

검색된 문서들:
{docs_text}

이 문서들이 질문에 대한 답을 포함하고 있나요? JSON으로 응답하세요."""

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.1}
    }

    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        content = result.get("message", {}).get("content", "")

        # JSON 파싱 시도
        import re
        json_match = re.search(r'\{[^}]+\}', content)
        if json_match:
            parsed = json.loads(json_match.group())
            return {
                "level": parsed.get("level", "medium"),
                "reason": parsed.get("reason", ""),
                "relevant_indices": parsed.get("relevant_indices", [])
            }

        # 파싱 실패시 기본값 (medium)
        return {"level": "medium", "reason": "검증 불가", "relevant_indices": list(range(1, len(documents)+1))}

    except Exception as e:
        # 오류시 medium으로 처리
        return {"level": "medium", "reason": f"검증 오류: {e}", "relevant_indices": list(range(1, len(documents)+1))}


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
                            # BI-RADS 문서는 마크다운 링크
                            pmid = source.get('pmid', '')
                            nav_params = get_birads_nav_params(pmid)
                            param_str = "&".join([f"{k}={v}" for k, v in nav_params.items()])
                            page_url = f"/BI-RADS_Guidelines?{param_str}"
                            st.markdown(f"""
                            **{icon} [{i}] {source['title']}**
                            {source['authors']} - {source['journal']} ({source['year']})
                            [📘 원문 확인하기]({page_url})
                            """)
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

        # 언어 감지 및 메시지 설정 (검색 전에 미리 설정)
        is_ko = is_korean(prompt)
        msg = get_messages(is_ko)

        with st.chat_message("user"):
            st.markdown(prompt)

        # 검색 엔진으로 관련 논문 검색
        with st.spinner(msg["searching"]):
            try:
                engine = get_search_engine()
                translator = get_query_translator()

                # 대화형 쿼리 보강 (이전 맥락 참조)
                enhanced_prompt = enhance_query_with_context(
                    current_question=prompt,
                    chat_history=st.session_state.messages[:-1],  # 현재 질문 제외
                    model=options["model"]
                )
                if enhanced_prompt != prompt:
                    st.caption(f"💬 대화 맥락 반영: `{enhanced_prompt}`")

                # 검색 쿼리 최적화
                search_query = enhanced_prompt
                prompt_lower = enhanced_prompt.lower()

                # 0. LLM 기반 쿼리 번역 (한글 → 영문 의학 키워드)
                if translator.needs_translation(enhanced_prompt):
                    with st.spinner("🔄 쿼리 최적화 중..."):
                        translated_query = translator.translate(enhanced_prompt)
                        if translated_query != enhanced_prompt:
                            search_query = translated_query
                            st.caption(f"🔍 검색 키워드: `{translated_query}`")

                # 1. 쿼리 확장: 모든 한글 번역 쿼리에 BI-RADS 추가
                search_query_lower = search_query.lower()
                if 'bi-rads' not in search_query_lower and 'birads' not in search_query_lower:
                    if translator.needs_translation(enhanced_prompt):
                        search_query = f"BI-RADS {search_query}"
                        st.caption(f"✨ 쿼리 확장: `{search_query}`")

                # 이중 검색: BI-RADS + 연구논문
                birads_response, papers_response = engine.search_dual(
                    search_query,
                    birads_k=3,
                    papers_k=5
                )

                if not birads_response.results and not papers_response.results:
                    with st.chat_message("assistant"):
                        error_msg = msg["no_results"]
                        st.markdown(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    st.stop()

                # BI-RADS 컨텍스트 구성
                birads_context_parts = []
                birads_sources = []

                for i, result in enumerate(birads_response.results, 1):
                    paper = result.paper
                    content_text = getattr(paper, 'full_content', paper.abstract or '내용 없음')

                    birads_context_parts.append(f"""
[{i}] 📘 BI-RADS 가이드라인
제목: {paper.title}
내용: {content_text}
""")

                    birads_sources.append({
                        "title": paper.title,
                        "authors": paper.author_string or "American College of Radiology",
                        "journal": paper.journal or "ACR BI-RADS Atlas v2025",
                        "year": paper.year or "2025",
                        "pmid": paper.pmid,
                        "is_birads": True,
                        "full_content": getattr(paper, 'full_content', None)
                    })

                birads_context = "\n".join(birads_context_parts) if birads_context_parts else ""

                # 연구논문 컨텍스트 구성
                papers_context_parts = []
                papers_sources = []

                for i, result in enumerate(papers_response.results, 1):
                    paper = result.paper
                    content_text = (paper.abstract[:500] + '...' if paper.abstract and len(paper.abstract) > 500 else paper.abstract or '초록 없음')

                    papers_context_parts.append(f"""
[{i}] 📄 연구논문
제목: {paper.title}
저자: {paper.author_string}
저널: {paper.journal} ({paper.year})
내용: {content_text}
""")

                    papers_sources.append({
                        "title": paper.title,
                        "authors": paper.author_string or "저자 정보 없음",
                        "journal": paper.journal or "저널 정보 없음",
                        "year": paper.year or "연도 정보 없음",
                        "url": paper.pubmed_url,
                        "is_birads": False
                    })

                papers_context = "\n".join(papers_context_parts) if papers_context_parts else ""

                # 전체 컨텍스트 결합 (BI-RADS 우선, 연구논문 후순위)
                context_parts = []
                if birads_context:
                    context_parts.append("### 📘 BI-RADS 가이드라인\n" + birads_context)
                if papers_context:
                    context_parts.append("\n### 📄 관련 연구 논문\n" + papers_context)

                context = "\n\n".join(context_parts)
                # sources는 검증 후에 추가됨 (빈 리스트로 초기화)
                sources = []

            except Exception as e:
                with st.chat_message("assistant"):
                    error_msg = f"⚠️ 검색 중 오류 발생: {str(e)}"
                    st.markdown(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                st.stop()

        # 답변 생성 (BI-RADS가 있으면 관련성 검증 후 표시, 없으면 LLM)
        with st.chat_message("assistant"):
            if birads_sources:
                # BI-RADS 문서 관련성 검증
                with st.spinner(msg["verifying"]):
                    relevance = verify_relevance(
                        question=prompt,
                        documents=birads_sources,
                        model=options["model"]
                    )

                level = relevance.get("level", "medium")
                reason = relevance.get("reason", "")

                if level == "low":
                    # 관련 없음 - BI-RADS를 건너뛰고 일반 논문으로 진행
                    full_response = msg["not_found"].format(reason=reason)
                    st.markdown(full_response)
                    birads_sources = []  # 소스에서 제거 (논문은 아래에서 검증 후 추가됨)
                else:
                    # high 또는 medium - 문서 표시
                    relevant_indices = relevance.get("relevant_indices", [])
                    if relevant_indices:
                        filtered_sources = [birads_sources[i-1] for i in relevant_indices if 0 < i <= len(birads_sources)]
                    else:
                        filtered_sources = birads_sources

                    # sources를 filtered_sources로 업데이트 (저장용)
                    sources = filtered_sources

                    if level == "high":
                        full_response = msg["found_high"]
                    else:  # medium
                        full_response = msg["found_medium"].format(reason=reason)
                    st.markdown(full_response)

                    st.markdown("---")
                    for i, source in enumerate(filtered_sources, 1):
                        pmid = source.get('pmid', '')
                        nav_params = get_birads_nav_params(pmid)
                        param_str = "&".join([f"{k}={v}" for k, v in nav_params.items()])
                        page_url = f"/BI-RADS_Guidelines?{param_str}"

                        st.markdown(f"### [{i}] {source['title']}")
                        st.markdown(f"_{source['authors']} - {source['journal']} ({source['year']})_")
                        st.markdown(f"[{msg['view_source']}]({page_url})")
                        st.markdown("---")
            else:
                # BI-RADS 없으면 LLM으로 답변 생성
                message_placeholder = st.empty()
                full_response = ""

                for chunk in call_llm_with_context(
                    question=prompt,
                    context=context,
                    model=options["model"],
                    temperature=options["temperature"]
                ):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")

                message_placeholder.markdown(full_response)

            if papers_sources:
                # 논문 관련성 검증
                with st.spinner(msg["verifying"]):
                    paper_relevance = verify_relevance(
                        question=prompt,
                        documents=papers_sources,
                        model=options["model"]
                    )

                paper_level = paper_relevance.get("level", "medium")
                paper_reason = paper_relevance.get("reason", "")
                paper_indices = paper_relevance.get("relevant_indices", [])

                if paper_level != "low":
                    # 관련 있는 논문만 필터링
                    if paper_indices:
                        filtered_papers = [papers_sources[i-1] for i in paper_indices if 0 < i <= len(papers_sources)]
                    else:
                        filtered_papers = papers_sources

                    if filtered_papers:
                        # sources에 관련 논문 추가
                        sources = sources + filtered_papers

                        if paper_level == "high":
                            expander_title = msg["papers_high"]
                        else:  # medium
                            expander_title = msg["papers_medium"]

                        pubmed_text = "PubMed" if not is_ko else "PubMed 보기"
                        with st.expander(expander_title, expanded=False):
                            if paper_level == "medium":
                                st.caption(f"_{paper_reason}_")
                            for i, source in enumerate(filtered_papers, 1):
                                st.markdown(f"""
                                **📄 [{i}] {source['title']}**
                                {source['authors']} - {source['journal']} ({source['year']})
                                [{pubmed_text}]({source['url']})
                                """)

        # 어시스턴트 메시지 저장 (출처 포함, 딥카피로 저장)
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "sources": copy.deepcopy(sources)  # 딥카피로 참조 완전 분리
        })


if __name__ == "__main__":
    main()
