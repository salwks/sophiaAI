"""
Sophia AI Alpha: RAG-based Medical AI Assistant
===============================================
논문 검색 기반 의료 AI 어시스턴트 (할루시네이션 최소화)
"""

import os
import sys
import re
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
from src.search.relay_router import RelayRouter, get_relay_router, QueryIntent
from src.evaluation.agent_judge import (
    TextExcellencePipeline, get_text_excellence_pipeline, JudgeVerdict,
    AgentJudge, get_agent_judge  # Phase 7.6: Agent-as-a-Judge 통합
)
from src.retrieval.dynamic_evidence import DynamicEvidencePipeline, get_dynamic_evidence_pipeline


def convert_latex_for_streamlit(text: str) -> str:
    r"""
    LLM 응답의 수식을 Streamlit이 렌더링할 수 있는 형태로 변환

    패턴:
    - (수식) → $수식$ (LaTeX 문법이 포함된 경우만)
    - \(...\) → $...$
    - \[...\] → $$...$$
    """
    if not text:
        return text

    # 1. \(...\) 인라인 LaTeX → $...$
    text = re.sub(r'\\\((.+?)\\\)', r'$\1$', text)

    # 2. \[...\] 디스플레이 LaTeX → $$...$$
    text = re.sub(r'\\\[(.+?)\\\]', r'$$\1$$', text, flags=re.DOTALL)

    # 3. (수식) 패턴 변환 - LaTeX 문법이 포함된 경우만
    # LaTeX 문법 패턴: \frac, \times, \sin, \cos, \pi, \sqrt, _{, ^{, ×, ∝, ≈
    latex_indicators = r'\\(?:frac|times|sin|cos|pi|sqrt|alpha|beta|sigma|delta|Delta|mu)|_\{|\^\{|[×∝≈∼→←↑↓]'

    def replace_latex_parens(match):
        content = match.group(1)
        # LaTeX 문법이 포함되어 있으면 $...$로 변환
        if re.search(latex_indicators, content):
            return f'${content}$'
        # 아니면 그대로 유지
        return match.group(0)

    # 괄호 안에 = 또는 LaTeX 문법이 있는 경우만 변환
    # 예: (w = f \times (M-1)) → $w = f \times (M-1)$
    text = re.sub(r'\(([^()]*(?:=|' + latex_indicators + r')[^()]*)\)', replace_latex_parens, text)

    # 4. <br> 태그 전후의 수식 정리
    text = re.sub(r'\$\s*<br>\s*', '<br>\n$', text)
    text = re.sub(r'\s*<br>\s*\$', '$<br>\n', text)

    return text

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
        llm_model="gpt-oss:20b",
        use_reranker=True,
    )

@st.cache_resource(ttl=3600)
def get_query_translator():
    """쿼리 번역기 싱글톤 (레거시, RelayRouter로 대체 예정)"""
    return get_translator(
        ollama_url="http://localhost:11434",
        model="gpt-oss:20b"
    )

@st.cache_resource(ttl=3600)
def get_cached_relay_router():
    """RelayRouter 싱글톤 (SLM-LLM 협업)"""
    return get_relay_router()

@st.cache_resource(ttl=3600)
def get_cached_text_pipeline():
    """TextExcellencePipeline 싱글톤 (Answering Twice + Agent Judge)"""
    return get_text_excellence_pipeline()

@st.cache_resource(ttl=3600)
def get_cached_dynamic_pipeline(_version: str = "7.17"):
    """DynamicEvidencePipeline 싱글톤 (Phase 7.17: 단순화된 파이프라인)"""
    from src.retrieval.dynamic_evidence import DynamicEvidencePipeline
    # Phase 7.17: decomposition 비활성화 (복잡한 레이어 제거)
    return DynamicEvidencePipeline(use_summarizer=False, enable_decomposition=False)

def get_guideline_type(pmid: str) -> str:
    """PMID로부터 가이드라인 타입 반환"""
    if not pmid:
        return "unknown"
    if pmid.startswith("BIRADS_"):
        return "birads"
    elif pmid.startswith("ACR_"):
        return "acr"
    elif pmid.startswith("PHYSICS_"):
        return "physics"
    elif pmid.startswith("CLINICAL_"):
        return "clinical"
    elif pmid.startswith("DANCE_"):
        return "dance"  # Dance et al. 물리학 참조 논문
    return "paper"


def get_guideline_label(pmid: str) -> str:
    """PMID로부터 가이드라인 라벨 반환"""
    guideline_type = get_guideline_type(pmid)
    labels = {
        "birads": "📘 BI-RADS 가이드라인",
        "acr": "📗 ACR Practice Parameter",
        "physics": "📙 물리학 가이드라인",
        "clinical": "📕 임상 가이드라인",
        "dance": "📙 Dance 2011 물리 참조",  # Dance et al. MGD 논문
        "paper": "📄 연구논문",
        "unknown": "📄 문서"
    }
    return labels.get(guideline_type, "📄 문서")


def get_birads_nav_params(pmid: str) -> dict:
    """
    PMID로부터 가이드라인 페이지 네비게이션 파라미터 생성

    예: BIRADS_2025_SECTION_IV_A_CHUNK_MARGIN
    → {"modality": "mammography", "section": "BIRADS_2025_SECTION_IV", "sub": "BIRADS_2025_SECTION_IV_A", "chunk": "MARGIN"}
    """
    GUIDELINE_PREFIXES = ("BIRADS_", "ACR_", "PHYSICS_", "CLINICAL_")
    if not pmid or not pmid.startswith(GUIDELINE_PREFIXES):
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


def get_acr_nav_params(pmid: str) -> dict:
    """
    ACR PMID로부터 ACR 페이지 네비게이션 파라미터 생성

    예: ACR_CEM_INDICATIONS → {"category": "cem", "doc": "ACR_CEM_INDICATIONS"}
    예: ACR_MAMMO_SECTION_I → {"category": "mammo", "doc": "ACR_MAMMO_SECTION_I"}
    """
    if not pmid or not pmid.startswith("ACR_"):
        return {}

    params = {}

    # 카테고리 결정
    if pmid.startswith("ACR_CEM_"):
        params["category"] = "cem"
    elif pmid.startswith("ACR_MAMMO_"):
        params["category"] = "mammo"
    elif pmid.startswith("ACR_IQ_"):
        params["category"] = "iq"
    else:
        params["category"] = "mammo"  # 기본값

    params["doc"] = pmid
    return params


def get_guideline_page_url(pmid: str) -> str:
    """
    가이드라인 PMID에 따라 적절한 페이지 URL 생성
    """
    guideline_type = get_guideline_type(pmid)

    if guideline_type == "acr":
        nav_params = get_acr_nav_params(pmid)
        param_str = "&".join([f"{k}={v}" for k, v in nav_params.items()])
        return f"/ACR_Practice_Parameters?{param_str}"
    elif guideline_type in ("birads", "physics", "clinical"):
        nav_params = get_birads_nav_params(pmid)
        param_str = "&".join([f"{k}={v}" for k, v in nav_params.items()])
        return f"/BI-RADS_Guidelines?{param_str}"
    else:
        return "#"


def fetch_pmc_fulltext(pmc_url: str, max_chars: int = 8000) -> str:
    """
    PMC 논문 전문을 가져와서 텍스트로 반환

    Args:
        pmc_url: PMC 논문 URL (예: https://pmc.ncbi.nlm.nih.gov/articles/PMC7533093/)
        max_chars: 최대 문자 수 (기본 8000자)

    Returns:
        논문 전문 텍스트 (HTML 태그 제거됨)
    """
    try:
        # URL 정규화
        if "www.ncbi.nlm.nih.gov" in pmc_url:
            pmc_url = pmc_url.replace("www.ncbi.nlm.nih.gov/pmc", "pmc.ncbi.nlm.nih.gov")

        response = requests.get(pmc_url, timeout=15, headers={
            "User-Agent": "Mozilla/5.0 (compatible; SophiaAI/1.0; Medical Research Assistant)"
        })
        response.raise_for_status()

        html = response.text

        # HTML에서 본문 텍스트 추출 (간단한 방식)
        import re

        # script, style 태그 제거
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)

        # 주요 섹션 추출 시도 (article-body, main-content 등)
        article_match = re.search(r'<article[^>]*>(.*?)</article>', html, flags=re.DOTALL | re.IGNORECASE)
        if article_match:
            html = article_match.group(1)

        # HTML 태그 제거
        text = re.sub(r'<[^>]+>', ' ', html)

        # 여러 공백을 하나로
        text = re.sub(r'\s+', ' ', text)

        # 특수문자 정리
        text = text.replace('&nbsp;', ' ')
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')

        text = text.strip()

        # 최대 길이 제한
        if len(text) > max_chars:
            text = text[:max_chars] + "... [전문 일부 생략]"

        return text

    except Exception as e:
        return f"[PMC 전문 가져오기 실패: {str(e)}]"


def is_korean(text: str) -> bool:
    """텍스트에 한글이 포함되어 있는지 확인"""
    import re
    return bool(re.search(r'[가-힣]', text))


def get_messages(is_ko: bool) -> dict:
    """언어별 메시지 반환"""
    if is_ko:
        return {
            "found_high": "📚 **가이드라인에서 관련 내용을 찾았습니다.**\n\n아래 원문을 확인해주세요.",
            "found_medium": "📋 **가이드라인에서 관련될 수 있는 내용을 찾았습니다.**\n\n⚠️ _{reason}_\n\n아래 원문을 확인해주세요.",
            "not_found": "📭 **가이드라인에서 관련 내용을 찾지 못했습니다.**\n\n_{reason}_",
            "no_results": "검색 결과가 없습니다. 다른 키워드로 검색해 주세요.",
            "view_source": "📚 원문 확인하기",
            "papers_high": "📄 관련 연구 논문",
            "papers_medium": "📄 관련될 수 있는 연구 논문 ⚠️",
            "verifying": "🔍 문서 관련성 검증 중...",
            "searching": "📚 관련 문서 검색 중..."
        }
    else:
        return {
            "found_high": "📚 **Found relevant content in Guidelines.**\n\nPlease check the original text below.",
            "found_medium": "📋 **Found possibly relevant content in Guidelines.**\n\n⚠️ _{reason}_\n\nPlease check the original text below.",
            "not_found": "📭 **No relevant content found in Guidelines.**\n\n_{reason}_",
            "no_results": "No search results. Please try different keywords.",
            "view_source": "📚 View Original",
            "papers_high": "📄 Related Research Papers",
            "papers_medium": "📄 Possibly Related Research Papers ⚠️",
            "verifying": "🔍 Verifying document relevance...",
            "searching": "📚 Searching for related papers..."
        }


def enhance_query_with_context(current_question: str, chat_history: list, model="gpt-oss:20b") -> str:
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
        response = requests.post(url, json=payload, timeout=45)
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


def call_llm_with_context(
    question: str,
    context: str,
    model="gpt-oss:20b",
    temperature=0.7,
    has_guidelines: bool = True,
    physics_knowledge: str = ""  # Phase 7.18: 동적 물리 지식 별도 전달
):
    """
    RAG: 검색된 논문 컨텍스트를 기반으로 LLM 답변 생성

    Phase 7.19: UnifiedPromptBuilder 사용으로 지식 전달 경로 통합

    Args:
        question: 사용자 질문
        context: 검색된 논문 내용
        model: LLM 모델명
        temperature: 온도 설정
        has_guidelines: 가이드라인 문서 포함 여부
        physics_knowledge: KnowledgeManager에서 로드한 검증된 물리 지식 (Phase 7.18)

    Returns:
        LLM 답변 (generator)
    """
    url = "http://localhost:11434/api/chat"

    # Phase 7.19: UnifiedPromptBuilder 사용 (통합 지식 전달)
    # Phase 7.20: Query Decomposition + Grounded Values
    grounded_values_context = ""
    try:
        from src.prompts.unified_builder import UnifiedPromptBuilder, PromptLimits
        from src.knowledge.manager import get_knowledge_manager

        km = get_knowledge_manager()
        builder = UnifiedPromptBuilder(km)

        # 동적 지식 컨텍스트 생성 (physics_knowledge가 없으면 쿼리 기반 자동 검색)
        if not physics_knowledge:
            physics_knowledge = builder.build_knowledge_context(question)

        # 통합 시스템 프롬프트에서 core_physics 추출
        core_physics = builder.get_axioms()

        # Phase 7.20: Grounded Values (검증된 값 테이블)
        grounded_values_context = builder._get_grounded_values(question)
    except ImportError:
        # Fallback: 기존 방식 사용
        try:
            from src.knowledge.core_physics import get_core_physics_prompt
            core_physics = get_core_physics_prompt()
        except ImportError:
            core_physics = ""

    # Phase 7.18/7.19: 동적 물리 지식을 core_physics 앞에 배치 (우선순위)
    if physics_knowledge:
        core_physics = f"{physics_knowledge}\n\n{core_physics}"

    if has_guidelines:
        # 데이터 무결성 강화 프롬프트 (Integrity-First Prompt)
        system_message = f"""# Role
너는 의학 물리 데이터의 '무결성(Integrity)'을 검증하는 전문 감사관(Auditor)이다.
단순한 정보 요약자가 아니라, 제공된 자료의 수치와 물리 법칙이 답변에 '왜곡 없이' 반영되었는지 감시하라.

# ============================================================
# 📚 검증된 표준 참조 자료 (Standard Reference - 인용 가능)
# ============================================================
# 아래 내용은 원본 논문에서 검증된 표준 지식입니다.
# RAG 검색 결과와 동등하게 인용할 수 있습니다.
# 특히 Dance et al. 2011 논문의 t-factor/T-factor 테이블은
# 직접 인용하여 답변에 활용하세요.
# ============================================================

{core_physics}

# ============================================================
# 표준 참조 자료 끝
# ============================================================

# Strict Instruction (절대 준수 사항)
0. **검증된 수치 우선 사용 (Data Priority)**:
   - 위 [표준 물리학 참조]에 명시된 검증 수치(W값, QE, Ghosting, Lag 등)가 있으면 **반드시** 해당 값을 사용하라.
   - **⚠️ W = 50-64 eV** (a-Se 검출기). 3.6 eV는 실리콘 값이므로 **절대 사용 금지**.
   - **⚠️ 양자검출효율 (QE = QDE)**: QE(LE 28kVp) = **97%**, QE(HE 49kVp) = **56%** — CEM 질문에 필수.
   - **⚠️ 이론 공식 QE=1-e^(-μt) 사용 금지!** 위 **실측값(97%, 56%)**을 직접 인용하라.
   - **⚠️ Ghosting = 15%, Lag = 0.15%** — 100배 차이! 둘을 혼동하지 말 것.
   - 네 내부 지식이 위 수치와 다르면, **위 검증 수치를 우선**하라.

1. **결론 도출 금지 (Evidence First)**:
   - 네가 이미 알고 있는 일반 지식으로 결론을 먼저 내리지 마라.
   - 반드시 제공된 [표준 참조 자료], [가이드라인], [연구 논문]의 텍스트에서 직접적인 근거를 먼저 나열한 후, 그 데이터가 허용하는 범위 내에서만 결론을 도출하라.

2. **물리 개념의 엄격한 정의 (Grounding Physics)**:
   - 위 "필수 물리 지식"과 Mammography 물리 법칙을 혼동하지 마라.
   - [Hard Beam]: 에너지가 높음(keV↑), K-edge가 높음, 투과력 높음, 두꺼운 유방에 필수.
   - [Soft Beam]: 에너지가 낮음(keV↓), K-edge가 낮음, 투과력 낮음, 얇은 유방에 적합.
   - MGD 관련 질문에서는 Dance et al. 2011의 t-factor/T-factor 개념을 정확히 적용하라.
   - 이 정의와 반대되는 주장을 논문에 근거 없이 작성할 경우, 이는 치명적인 오류로 간주한다.

3. **인용 무결성 (Quoting & Verification)**:
   - 논문의 내용을 언급할 때는 반드시 해당 논문의 번호와 핵심 키워드를 병기하라.
   - 초록(Abstract)에 명시되지 않은 데이터(예: 구체적 수치 등)를 "합성"하거나 "유추"하여 기재하지 마라.
   - 데이터가 부족하면 "제시된 자료에는 구체적 수치가 포함되어 있지 않음"을 명시하라.
   - **제공된 자료에 없는 논문(저자명, 연도)은 절대 인용하지 마라. 이는 심각한 Hallucination 오류다.**

4. **모순 발생 시 보고**:
   - 질문의 내용과 논문의 데이터가 상충하거나, 논문 내부의 논리가 네 상식과 다를 경우 지어내지 말고 "자료 간의 논리적 모순"을 보고하라.

# Response Format (한국어로 답변)
⚠️ **반드시 한국어로만 답변하세요. 일본어/중국어 출력 금지.**
- [데이터 기반 근거]: 각 논문에서 추출한 팩트 나열 (논문 번호 명시)
- [물리적 인과관계 검증]: 추출된 팩트와 물리 법칙의 일치성 확인
- [최종 결론]: 데이터가 보장하는 범위 내에서의 설계 지침

# 답변 상세화 원칙 (Comprehensive Response)
1. **수식 포함**: 물리 개념 설명 시 관련 수식을 반드시 포함하라.
   - 예: "t(θ) = D(θ) / D(0)" 형태로 명시
   - 수식의 각 변수 의미를 설명하라.

2. **유도 과정 설명**: 개념 간 연결 관계를 유도 공식으로 보여라.
   - 예: t-factor가 T-factor로 어떻게 통합되는지
   - "T = Σ αᵢ × t(θᵢ)"와 같이 단계별로 설명

3. **테이블/수치 인용**: 논문의 테이블 데이터가 있다면 구체적 수치를 제시하라.
   - 예: "Table 6에 따르면, 6.5cm 두께에서 t(20°) = 0.929"

4. **실무 적용**: 장비 설계나 AEC 알고리즘 관점에서 실무적 의미를 설명하라.
   - 예: "이 공식은 각 각도별 mAs 배분 전략 수립에 사용됨"

5. **페이지/섹션 참조**: 가능하면 논문의 구체적 페이지나 섹션을 명시하라.
   - 예: "Dance et al. 2011, 8-9페이지 참조\""""

        # Phase 7.20: Grounded Values를 user_message 최상단에 배치
        grounded_section = ""
        if grounded_values_context:
            grounded_section = f"""
{grounded_values_context}

"""

        user_message = f"""다음 참고 자료를 분석하여 질문에 답변해주세요.
{grounded_section}
**⚠️ 데이터 우선순위 규칙 (CRITICAL):**
0. **[검증된 데이터 테이블]** (위에 표시) → **절대 최우선 - 이 값만 사용**
1. **[표준 물리학 참조]** (시스템 프롬프트 내) → **우선 사용**
   - 검증된 물리 상수와 수치 (W값, QDE, Ghosting 등)
   - **이 섹션에 명시된 수치가 있으면 반드시 해당 값을 사용할 것**
2. **[검색된 논문]**: 아래 RAG 검색 결과

**⚠️ Hallucination 금지**: 위 두 출처에 없는 논문(저자명, 연도)은 절대 인용하지 마세요.

**검색된 논문 (RAG 결과):**
{context}

**질문:** {question}

**요구사항:**
- **[검증된 데이터 테이블]의 수치를 반드시 사용** (Ghosting=15%, Lag=0.15% 등)
- **[표준 물리학 참조]에 명시된 검증 수치를 우선 사용** (W값, QDE 등)
- **수식을 포함**하여 물리적 관계를 명확히 설명
- **유도 과정**을 단계별로 보여줄 것
- **구체적 수치**를 [표준 물리학 참조]에서 정확히 인용
- **실무 적용** 관점에서 장비 설계/캘리브레이션에 어떻게 활용되는지 설명
- 위 두 출처에 없는 논문은 절대 인용하지 말 것
- 자료에 답이 없으면 "제공된 자료에는 해당 정보가 없습니다"라고 명시
"""
    else:
        # 가이드라인 없음: 보수적 안내형 (논문 초록 기반 추정)
        system_message = f"""당신은 유방영상학 전문 의학 물리 보조원입니다.

# ============================================================
# 📚 검증된 표준 참조 자료 (Standard Reference - 인용 가능)
# ============================================================
# 아래 내용은 원본 논문에서 검증된 표준 지식입니다.
# RAG 검색 결과와 동등하게 인용할 수 있습니다.
# ============================================================

{core_physics}

# ============================================================
# 표준 참조 자료 끝
# ============================================================

**핵심 원칙:**
1. **정직한 부재 고지**: 가이드라인(ACR/BI-RADS)에 직접적 답변이 없음을 먼저 밝히세요.
2. **표준 참조 + 초록 기반 추론**: 위 [표준 참조 자료]와 RAG 검색된 논문 초록을 함께 활용하세요.
3. **환각 방지**: 위 두 출처에 없는 수치를 만들지 마세요.
4. **물리 원리**: [표준 참조 자료]의 Dance et al. 2011 t-factor/T-factor 테이블은 직접 인용 가능합니다.

**출력 구조:**
1. [지침 확인]: "가이드라인에는 이 주제에 대한 직접적인 명시가 없습니다."
2. [표준 참조 + 초록 분석]: Dance et al. 테이블 및 논문 번호를 인용하며 물리적 원리 설명
3. [제한 사항]: 필요시 "ℹ️ 초록(Abstract) 기반 분석입니다" 고지
"""

        # Phase 7.20: Grounded Values (가이드라인 없는 경우도 적용)
        grounded_section_no_guide = ""
        if grounded_values_context:
            grounded_section_no_guide = f"""
{grounded_values_context}

"""

        user_message = f"""**사용자 질문:** {question}
{grounded_section_no_guide}
**⚠️ 인용 가능한 자료:**
0. **[검증된 데이터 테이블]** (위에 표시) → **절대 최우선 - 이 값만 사용**
1. **[표준 참조 자료]**: 시스템 프롬프트의 Dance et al. 2011 t-factor/T-factor 테이블 및 MGD 공식
2. **[검색된 논문]**: 아래 RAG 검색 결과

**검색된 연구 논문 (초록):**
{context}

위 세 출처를 활용하여 질문에 답변해주세요.
- **[검증된 데이터 테이블]의 수치를 반드시 사용**
- 가이드라인 부재를 먼저 언급
- **Dance et al. 2011 Table 6 수치** 직접 인용 가능 (예: "5cm 두께에서 t(20°)=0.919")
- 논문 번호를 인용하여 물리적 원리 설명
- 위 세 출처에 없는 논문은 인용 금지
"""

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

    # 🛡️ 데이터 무결성 모드: Temperature 0.0으로 결정론적 응답 강제
    # has_guidelines=True일 때는 창의성을 완전히 차단하여 hallucination 방지
    actual_temperature = 0.0 if has_guidelines else temperature

    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": actual_temperature,
        }
    }

    try:
        print(f"[DEBUG] LLM 호출 시작: model={model}, context_len={len(context)}, has_guidelines={has_guidelines}")
        response = requests.post(url, json=payload, stream=True, timeout=180)
        response.raise_for_status()

        chunk_count = 0
        for line in response.iter_lines():
            if line:
                try:
                    chunk = json.loads(line)
                    if "message" in chunk and "content" in chunk["message"]:
                        chunk_count += 1
                        yield chunk["message"]["content"]
                except json.JSONDecodeError as je:
                    print(f"[DEBUG] JSON 파싱 오류: {je}, line={line[:100]}")
                    continue
        print(f"[DEBUG] LLM 호출 완료: {chunk_count} chunks")
    except requests.exceptions.RequestException as e:
        print(f"[DEBUG] LLM 연결 오류: {e}")
        yield f"⚠️ LLM 연결 오류: {str(e)}"
    except Exception as e:
        print(f"[DEBUG] LLM 예외 발생: {type(e).__name__}: {e}")
        yield f"⚠️ 예상치 못한 오류: {str(e)}"


def verify_hallucination(response: str, context: str) -> dict:
    """
    LLM 응답에서 Hallucination(허위 인용) 및 자료 무시를 탐지

    Args:
        response: LLM 응답 텍스트
        context: 제공된 참고 자료 (원본)

    Returns:
        {
            "has_hallucination": bool,
            "suspicious_citations": list,  # 의심되는 인용 목록
            "ignored_context": bool,  # 자료 무시 여부
            "physics_errors": list,  # 물리 개념 오류
            "warning_message": str  # 경고 메시지
        }
    """
    import re

    suspicious = []
    physics_errors = []
    ignored_context = False
    context_lower = context.lower()
    response_lower = response.lower()

    # ========================================
    # 1. 허위 인용 탐지 (기존)
    # ========================================

    # 1-1. "저자 et al." 패턴 탐지
    et_al_pattern = r'([A-Z][a-z]+)\s+et\s+al\.?'
    et_al_matches = re.findall(et_al_pattern, response)

    for author in et_al_matches:
        if author.lower() not in context_lower:
            suspicious.append(f"{author} et al.")

    # 1-2. "저자 (연도)" 패턴 탐지
    author_year_pattern = r'([A-Z][a-z]+)\s*\(\s*(19|20)\d{2}\s*\)'
    author_year_matches = re.findall(author_year_pattern, response)

    for author, _ in author_year_matches:
        if author.lower() not in context_lower:
            citation = f"{author} (연도)"
            if f"{author} et al." not in suspicious:
                suspicious.append(citation)

    # 1-3. "~에 따르면" 패턴 탐지
    according_pattern = r'([A-Z][a-z]+)에\s*따르면'
    according_matches = re.findall(according_pattern, response)

    for author in according_matches:
        if author.lower() not in context_lower and author not in ["Reference", "참고", "자료", "논문"]:
            if not any(author in s for s in suspicious):
                suspicious.append(f"{author}에 따르면")

    # ========================================
    # 2. 자료 무시 탐지 (신규)
    # ========================================

    # "자료 없음" 패턴 탐지
    no_data_patterns = [
        r"내용을\s*찾을\s*수\s*없",
        r"해당\s*정보가?\s*없",
        r"자료에서?\s*확인되지\s*않",
        r"직접적인\s*답을?\s*포함하지\s*않",
        r"관련된\s*내용을?\s*찾을\s*수\s*없",
        r"제공된\s*자료에는?\s*없",
    ]

    claims_no_data = any(re.search(p, response_lower) for p in no_data_patterns)

    # Context에 관련 키워드가 실제로 있는지 확인
    relevant_keywords_in_context = []
    important_keywords = [
        "denoising", "deep learning", "microcalcification", "noise reduction",
        "quantum", "mottle", "mtf", "spatial resolution", "cnn", "wavelet",
        "디노이징", "딥러닝", "미세석회화", "노이즈"
    ]

    for kw in important_keywords:
        if kw.lower() in context_lower:
            relevant_keywords_in_context.append(kw)

    # "자료 없다"고 했는데 실제로 관련 키워드가 있으면 = 자료 무시
    if claims_no_data and len(relevant_keywords_in_context) >= 2:
        ignored_context = True

    # ========================================
    # 3. 가짜 참고자료 탐지 (신규)
    # ========================================

    fake_references = set()  # 중복 방지를 위해 set 사용

    # "참고 자료:" 섹션에서 가짜 문서 제목 탐지
    # 실제 논문이 아닌 일반적인 설명 형태의 참고자료

    # 3-1. 개별 라인에서 가짜 참고자료 탐지 (더 정밀한 패턴)
    # 실제 예시: "[1] 제조업체의 Post-processing 알고리즘 설명서"
    fake_ref_line_patterns = [
        # 제조업체/업체 관련
        r'\[?\d+\]?\s*.{0,30}(제조업체|제조사|업체).{0,50}(설명서|문서|매뉴얼|가이드)',
        # 일반적인 보고서/문서
        r'\[?\d+\]?\s*.{0,50}(기술\s*보고서|분석\s*보고서|연구\s*보고서|분석\s*문서)',
        r'\[?\d+\]?\s*.{0,50}(알고리즘\s*설명서|기법\s*문서|처리\s*문서)',
        # 노이즈/이미지 관련 일반 문서 (저자명 없이)
        r'\[?\d+\]?\s*(노이즈|이미지|영상).{0,30}(기법|방법|처리).{0,20}(문서|설명|보고서)',
        # 대비/향상 관련 일반 문서
        r'\[?\d+\]?\s*.{0,30}(대비|contrast).{0,20}(향상|enhancement).{0,20}(문서|보고서|기술)',
        # Post-processing 관련
        r'\[?\d+\]?\s*.{0,30}(post-?processing|후처리).{0,30}(설명서|문서|알고리즘)',
    ]

    for pattern in fake_ref_line_patterns:
        matches = re.finditer(pattern, response, re.IGNORECASE)
        for match in matches:
            matched_text = match.group(0).strip()
            # Context에 이 내용이 없으면 가짜
            if matched_text.lower() not in context_lower:
                fake_references.add(f"'{matched_text}'")

    # "참고 자료:" 섹션 전체 검사 - 실제 논문 제목이 없으면 가짜
    ref_section_match = re.search(r'참고\s*자료[:\s]*\n?(.*?)(?:\n\n|$)', response, re.DOTALL | re.IGNORECASE)
    if ref_section_match:
        ref_section = ref_section_match.group(1)
        # Context에서 실제 논문 제목 추출
        real_titles = re.findall(r'제목:\s*([^\n]+)', context)

        # 참고자료 섹션에 실제 논문 제목이 하나도 없으면 경고
        has_real_title = False
        for title in real_titles:
            title_words = title.lower().split()[:3]  # 첫 3단어만 비교
            if any(word in ref_section.lower() for word in title_words if len(word) > 3):
                has_real_title = True
                break

        if not has_real_title and len(real_titles) > 0:
            fake_references.add("참고자료 섹션이 실제 제공된 논문과 일치하지 않습니다")

    # set을 list로 변환
    fake_references = list(fake_references)

    # ========================================
    # 4. 물리 개념 오류 탐지 (규칙 엔진 사용)
    # ========================================

    try:
        from src.validation.physics_rules import check_physics_errors
        detected_errors = check_physics_errors(response)
        for error in detected_errors:
            physics_errors.append(f"{error['description']} → {error['correct_statement']}")
    except ImportError:
        # 규칙 엔진 없으면 기본 검사 (fallback)
        pass

    # ========================================
    # 5. 경고 메시지 생성
    # ========================================

    has_issues = (
        len(suspicious) > 0 or
        ignored_context or
        len(physics_errors) > 0 or
        len(fake_references) > 0
    )

    warning_parts = []

    if suspicious:
        warning_parts.append("⚠️ **허위 인용 의심**: 다음 인용이 자료에서 확인되지 않습니다:")
        for i, citation in enumerate(suspicious, 1):
            warning_parts.append(f"  {i}. `{citation}`")
        warning_parts.append("")

    if fake_references:
        warning_parts.append("📝 **가짜 참고자료 탐지**: AI가 실제 논문이 아닌 가짜 참고자료를 생성했습니다:")
        for i, fake_ref in enumerate(fake_references, 1):
            warning_parts.append(f"  {i}. {fake_ref}")
        warning_parts.append("  _실제 제공된 논문: Context의 '제목:' 항목을 확인하세요._")
        warning_parts.append("")

    if ignored_context:
        warning_parts.append("🚨 **자료 무시 경고**: AI가 '자료 없음'이라 답했지만, 실제로 관련 내용이 있습니다:")
        warning_parts.append(f"  발견된 키워드: `{', '.join(relevant_keywords_in_context[:5])}`")
        warning_parts.append("  _제공된 논문을 다시 확인하세요._")
        warning_parts.append("")

    if physics_errors:
        warning_parts.append("🔴 **물리 개념 오류**: 다음 내용이 물리 법칙과 충돌합니다:")
        for i, error in enumerate(physics_errors, 1):
            warning_parts.append(f"  {i}. {error}")
        warning_parts.append("")

    warning_message = "\n".join(warning_parts) if warning_parts else ""

    return {
        "has_hallucination": len(suspicious) > 0 or len(fake_references) > 0,
        "suspicious_citations": suspicious,
        "fake_references": fake_references,
        "ignored_context": ignored_context,
        "physics_errors": physics_errors,
        "warning_message": warning_message
    }


def verify_relevance(question: str, documents: list, model="gpt-oss:20b", has_physics_knowledge: bool = False) -> dict:
    """
    검색된 문서가 질문에 관련이 있는지 LLM으로 3단계 검증

    Args:
        question: 사용자 질문
        documents: 검색된 문서 리스트
        model: LLM 모델명
        has_physics_knowledge: KnowledgeManager에서 관련 물리 지식을 찾았는지 여부

    Returns:
        {"level": "high"/"medium"/"low", "reason": str, "relevant_indices": list}
    """
    # KnowledgeManager에서 관련 지식을 찾았으면 바로 high 반환
    if has_physics_knowledge:
        return {
            "level": "high",
            "reason": "표준 참조 자료(물리 지식 모듈)에서 관련 정보를 확인했습니다.",
            "relevant_indices": list(range(1, len(documents)+1))
        }

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
        response = requests.post(url, json=payload, timeout=90)  # deepseek-r1 응답 대기
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
            options=["gpt-oss:20b"],
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

        st.markdown("#### 🛡️ 검증 모드")
        critic_mode = st.checkbox(
            "전문가 검증 (Critic Agent)",
            value=True,  # 기본 활성화
            help="AI 응답을 물리 법칙/인용 정확성 기준으로 자동 검증 및 교정합니다"
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
        ✅ BI-RADS 가이드라인 참조<br>
        ⚖️ <b>Critic Agent 자동 검증</b>
        </div>
        """, unsafe_allow_html=True)

        return {
            "model": model,
            "temperature": temperature,
            "top_k": top_k,
            "critic_mode": critic_mode,
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
                        pmid = source.get('pmid', '')
                        guideline_type = get_guideline_type(pmid)
                        is_guideline = source.get("is_birads", False) or guideline_type in ("birads", "acr", "physics", "clinical")

                        if guideline_type == "acr":
                            icon = "📗"
                        elif guideline_type == "birads":
                            icon = "📘"
                        elif is_guideline:
                            icon = "📙"
                        else:
                            icon = "📄"

                        if is_guideline:
                            # 가이드라인 문서 - BIRADS만 원문 링크 표시 (실제 full_content 있음)
                            if guideline_type == "birads":
                                page_url = get_guideline_page_url(pmid)
                                st.markdown(f"""
                                **{icon} [{i}] {source['title']}**
                                {source['authors']} - {source['journal']} ({source['year']})
                                [📘 원문 확인하기]({page_url})
                                """)
                            else:
                                # ACR, PHYSICS, CLINICAL - full_content 없음, PMC 출처 표시
                                journal_info = source.get('journal', '')
                                # PMC 번호가 있으면 링크로 변환
                                pmc_ids = re.findall(r'PMC\d+', journal_info)
                                if pmc_ids:
                                    pmc_links = " | ".join([f"[{pmc}](https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc}/)" for pmc in pmc_ids])
                                    st.markdown(f"""
                                    **{icon} [{i}] {source['title']}**
                                    {source['authors']} ({source['year']})
                                    📚 출처: {pmc_links}
                                    """)
                                else:
                                    st.markdown(f"""
                                    **{icon} [{i}] {source['title']}**
                                    {source['authors']} - {journal_info} ({source['year']})
                                    """)
                        else:
                            # 일반 논문은 PubMed + Google Scholar + PMC 링크
                            links = [f"[PubMed]({source['url']})"]
                            if source.get('google_scholar_url'):
                                links.append(f"[🔍 Scholar]({source['google_scholar_url']})")
                            if source.get('pmc_url'):
                                links.append(f"[✅ PMC]({source['pmc_url']})")

                            st.markdown(f"""
                            **{icon} [{i}] {source['title']}**
                            {source['authors']} - {source['journal']} ({source['year']})
                            {' | '.join(links)}
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
                relay_router = get_cached_relay_router()

                # 대화형 쿼리 보강 (이전 맥락 참조)
                enhanced_prompt = enhance_query_with_context(
                    current_question=prompt,
                    chat_history=st.session_state.messages[:-1],  # 현재 질문 제외
                    model=options["model"]
                )
                if enhanced_prompt != prompt:
                    st.caption(f"💬 대화 맥락 반영: `{enhanced_prompt}`")

                # =====================================================
                # RelayRouter: SLM 기반 빠른 전처리
                # =====================================================
                with st.spinner("🚀 SLM 분석 중..."):
                    dispatch_result = relay_router.dispatch(enhanced_prompt)
                    dispatch_result = relay_router.enrich_with_knowledge(dispatch_result)

                # 검색 쿼리 최적화 (SLM 번역 결과 사용)
                search_query = dispatch_result.translated_query
                prompt_lower = enhanced_prompt.lower()

                # SLM 분석 결과 표시
                intent_labels = {
                    QueryIntent.SIMPLE_LOOKUP: "📖 단순 조회",
                    QueryIntent.PHYSICS_CALCULATION: "🔬 물리 계산",
                    QueryIntent.CLINICAL_GUIDELINE: "🏥 임상 가이드라인",
                    QueryIntent.COMPLEX_REASONING: "🧠 복합 추론",
                    QueryIntent.UNKNOWN: "❓ 분류 불가",
                }
                st.caption(f"🔍 검색 키워드: `{search_query}`")
                st.caption(f"📊 질문 유형: {intent_labels.get(dispatch_result.intent, '알 수 없음')} | 사용 모델: {relay_router.get_model_used(dispatch_result)}")

                # 1. 쿼리 확장: 모든 한글 번역 쿼리에 BI-RADS 추가
                search_query_lower = search_query.lower()
                if 'bi-rads' not in search_query_lower and 'birads' not in search_query_lower:
                    if translator.needs_translation(enhanced_prompt):
                        search_query = f"BI-RADS {search_query}"
                        st.caption(f"✨ 쿼리 확장: `{search_query}`")

                # 이중 검색: BI-RADS + 연구논문
                # birads_k=5: Dance 논문 등 물리 참조 문서가 포함되도록 확대
                birads_response, papers_response = engine.search_dual(
                    search_query,
                    birads_k=5,
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
                import re as re_module  # 로컬 스코프에서 사용

                for i, result in enumerate(birads_response.results, 1):
                    paper = result.paper
                    guideline_label = get_guideline_label(paper.pmid)

                    # PMC ID 추출 (pmc_id 필드 또는 journal 필드에서)
                    pmc_id = getattr(paper, 'pmc_id', None)
                    if not pmc_id and paper.journal:
                        pmc_match = re_module.search(r'(PMC\d+)', paper.journal)
                        if pmc_match:
                            pmc_id = pmc_match.group(1)

                    # PMC 전문이 있으면 fetch, 없으면 기존 내용 사용
                    if pmc_id:
                        pmc_url = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmc_id}/"
                        content_text = fetch_pmc_fulltext(pmc_url, max_chars=8000)
                        content_text = f"[PMC 전문 - {pmc_id}]\n{content_text}"
                    else:
                        content_text = getattr(paper, 'full_content', paper.abstract or '내용 없음')

                    birads_context_parts.append(f"""
[{i}] {guideline_label}
제목: {paper.title}
내용: {content_text}
""")

                    birads_sources.append({
                        "title": paper.title,
                        "authors": paper.author_string or "American College of Radiology",
                        "journal": paper.journal or "ACR BI-RADS Atlas v2025",
                        "year": paper.year or "2025",
                        "pmid": paper.pmid,
                        "pmc_id": pmc_id,  # ⚠️ Phase 7.1 Fix: 추출된 pmc_id 사용 (journal에서 추출한 값 포함)
                        "is_birads": True,
                        "full_content": getattr(paper, 'full_content', None)
                    })

                birads_context = "\n".join(birads_context_parts) if birads_context_parts else ""

                # 연구논문 컨텍스트 구성
                papers_context_parts = []
                papers_sources = []

                for i, result in enumerate(papers_response.results, 1):
                    paper = result.paper

                    # full_content가 있으면 우선 사용 (Dance 논문 등 직접 청킹된 문서)
                    full_content = getattr(paper, 'full_content', None)
                    pmc_id = getattr(paper, 'pmc_id', None)

                    if full_content:
                        # 직접 청킹된 문서 (Dance 논문, 물리학 참조 등)
                        content_text = full_content[:8000] if len(full_content) > 8000 else full_content
                    elif pmc_id:
                        # PMC 전문 fetch
                        pmc_url = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmc_id}/"
                        content_text = fetch_pmc_fulltext(pmc_url, max_chars=6000)
                        content_text = f"[PMC 전문 - {pmc_id}]\n{content_text}"
                    else:
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
                        "pmid": getattr(paper, 'pmid', ''),
                        "pmc_id": pmc_id,  # ⚠️ Phase 7.1 Fix: DynamicEvidencePipeline에 전달할 pmc_id
                        "url": paper.pubmed_url,
                        "google_scholar_url": paper.google_scholar_url,
                        "pmc_url": paper.pmc_url,  # None if not available
                        "doi_url": paper.doi_url,  # None if not available
                        "is_birads": False
                    })

                papers_context = "\n".join(papers_context_parts) if papers_context_parts else ""

                # 전체 컨텍스트 결합 (가이드라인 우선, 연구논문 후순위)
                context_parts = []
                if birads_context:
                    context_parts.append("### 📚 가이드라인 문서\n" + birads_context)
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
            # RelayRouter의 dispatch_result에서 KnowledgeManager 정보 사용
            from src.knowledge.manager import get_knowledge_manager
            km = get_knowledge_manager()

            # dispatch_result에서 지식 정보 추출 (이미 enrich_with_knowledge()에서 처리됨)
            matched_modules = km.get_relevant_knowledge(prompt)  # 컨텍스트용
            # Phase 7.18: prompt(query) 전달하여 동적 섹션 배치
            relevant_knowledge = km.format_for_context(matched_modules, query=prompt) if matched_modules else ""
            has_physics_knowledge = dispatch_result.has_knowledge

            # Knowledge Status Bar 표시
            status_parts = []
            if dispatch_result.knowledge_modules:
                status_parts.append(f"📚 표준 지식: {', '.join(dispatch_result.knowledge_modules)}")
            if birads_sources:
                status_parts.append(f"🔍 가이드라인: {len(birads_sources)}개")
            if papers_sources:
                status_parts.append(f"📄 논문: {len(papers_sources)}개")

            if status_parts:
                status_text = " | ".join(status_parts)
                st.info(f"**근거 자료 현황**: {status_text}")

            if birads_sources:
                # BI-RADS 문서 관련성 검증 (물리 지식 유무도 고려)
                with st.spinner(msg["verifying"]):
                    relevance = verify_relevance(
                        question=prompt,
                        documents=birads_sources,
                        model=options["model"],
                        has_physics_knowledge=has_physics_knowledge
                    )

                level = relevance.get("level", "medium")
                reason = relevance.get("reason", "")

                # 관련성 낮아도 검색된 내용이 있으면 분석 수행 (버리지 않음)
                if level == "low":
                    st.markdown(f"⚠️ 관련성이 낮을 수 있지만, 검색된 내용으로 분석을 시도합니다.\n\n_{reason}_")

                if True:  # 무조건 분석 수행
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
                        pmc_id = source.get('pmc_id', '')
                        journal = source.get('journal', '')
                        guideline_type = get_guideline_type(pmid)

                        # journal 필드에서 PMC ID 추출 (PMC로 시작하는 것)
                        if not pmc_id and journal:
                            pmc_match = re_module.search(r'(PMC\d+)', journal)
                            if pmc_match:
                                pmc_id = pmc_match.group(1)

                        st.markdown(f"### [{i}] {source['title']}")
                        st.markdown(f"_{source['authors']} - {source['journal']} ({source['year']})_")

                        # PMC ID가 있으면 PMC 링크 표시 (전문은 이미 LLM context에 포함됨)
                        if pmc_id:
                            pmc_url = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmc_id}/"
                            st.markdown(f"✅ **PMC 전문이 AI 분석에 사용되었습니다** | [🔗 PMC 원문 보기]({pmc_url})")
                        elif guideline_type != "paper":
                            # 가이드라인은 기존 방식
                            page_url = get_guideline_page_url(pmid)
                            link_icon = "📗" if guideline_type == "acr" else "📘"
                            st.markdown(f"[{link_icon} {msg['view_source']}]({page_url})")
                        else:
                            # 일반 논문 (PMC 없음)
                            pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                            st.markdown(f"[📄 PubMed 보기]({pubmed_url})")

                        st.markdown("---")

                    # =====================================================
                    # RelayLLM: 지능형 응답 라우팅
                    # =====================================================
                    st.markdown("### 🤖 AI 분석")

                    # KnowledgeManager가 직접 답변 가능한지 확인
                    if dispatch_result.knowledge_answer and dispatch_result.intent == QueryIntent.SIMPLE_LOOKUP:
                        # 📚 KnowledgeManager 직접 응답 (LLM 호출 스킵)
                        st.caption("⚡ **고속 응답**: 검증된 표준 지식에서 직접 답변")
                        full_response = dispatch_result.knowledge_answer
                        st.markdown(convert_latex_for_streamlit(full_response))

                    elif (dispatch_result.intent in [QueryIntent.PHYSICS_CALCULATION, QueryIntent.COMPLEX_REASONING]
                          or (dispatch_result.intent == QueryIntent.UNKNOWN
                              and any(kw in prompt.lower() for kw in ['증명', '수식', '계산', '도출', 'snr', 'cnr', 'mgd', 'pcd', 'eid', 'dqe', '논하시오', '기술하시오']))):
                        # 🔬 DynamicEvidencePipeline: PMC 전문 + Answering Twice + Evidence Mapping
                        st.caption("🔬 **정밀 분석**: Phase 7 Dynamic Evidence (PMC 전문 + 2단계 검증)")

                        # papers 데이터 구성 (DynamicEvidencePipeline 입력 형식)
                        papers_for_pipeline = []
                        for source in filtered_sources:
                            papers_for_pipeline.append({
                                "pmid": source.get("pmid", ""),
                                "pmc_id": source.get("pmc_id"),
                                "title": source.get("title", ""),
                                "abstract": source.get("abstract", source.get("content", ""))[:2000]
                            })

                        # PMC ID 보유 논문 수 표시 (디버깅 포함)
                        pmc_count = sum(1 for p in papers_for_pipeline if p.get("pmc_id"))
                        pmc_ids_found = [p.get("pmc_id") for p in papers_for_pipeline if p.get("pmc_id")]

                        if pmc_count > 0:
                            st.info(f"📑 **PMC 전문 인출 예정**: {pmc_count}개 ({', '.join(pmc_ids_found[:3])}...)")
                        else:
                            st.warning("⚠️ PMC ID가 없어 초록 기반으로 분석합니다.")

                        import asyncio
                        with st.spinner("🧠 단순화 모드: 지식 기반 직접 추론 중..."):
                            dynamic_pipeline = get_cached_dynamic_pipeline(_version="7.17")
                            # Phase 7.17: 단순화된 파이프라인 (복잡한 레이어 모두 제거)
                            result = asyncio.run(dynamic_pipeline.process_simple_async(
                                question=prompt,
                                papers=papers_for_pipeline,
                                physics_knowledge=relevant_knowledge,
                                max_pmc_fetch=2
                            ))

                        # Phase 7.16: 구조화된 답변 출력 (빈 응답 검증 포함)
                        full_response = result.answer
                        if not full_response or not full_response.strip() or "비어있습니다" in full_response:
                            st.error("⚠️ 응답 생성에 실패했습니다. 다시 시도해 주세요.")
                            st.info("💡 팁: 질문을 더 간단하게 다시 작성하거나, 잠시 후 다시 시도하세요.")
                        else:
                            # Placeholder로 답변 표시 (재생성 시 교체 가능)
                            response_placeholder = st.empty()
                            response_placeholder.markdown(convert_latex_for_streamlit(full_response))

                        # =====================================================
                        # Phase 7.6: Agent-as-a-Judge 평가 + 자동 재생성
                        # =====================================================
                        judge_result = None
                        regenerated = False
                        original_response = full_response  # 원본 보관 (로그용)

                        if options.get("enable_judge", True):
                            with st.spinner("🔍 Agent-as-a-Judge 품질 검증 중..."):
                                judge = get_agent_judge()
                                judge_result = judge.evaluate(
                                    question=prompt,
                                    answer=full_response,
                                    reference_knowledge=relevant_knowledge,
                                    context=""
                                )

                            # 자동 재생성: corrections가 있고 심각한 오류가 있으면
                            critical_errors = [c for c in (judge_result.corrections or [])
                                              if "Ghosting" in c or "혼동" in c or "오류" in c
                                              or "누락" in c or "QDE" in c or "트랩" in c]

                            if critical_errors:
                                # 로그에 수정 내역 기록
                                import logging
                                regen_logger = logging.getLogger("regeneration")
                                regen_logger.info(f"[Auto-Regeneration] Question: {prompt[:100]}...")
                                regen_logger.info(f"[Auto-Regeneration] Critical errors: {critical_errors}")
                                regen_logger.info(f"[Auto-Regeneration] Original response (first 500 chars): {original_response[:500]}...")

                                # 수정 가이드를 프롬프트 맨 앞에 강제 주입
                                correction_prefix = "\n".join([
                                    "🚨🚨🚨 최우선 준수 사항 (이 지침을 무시하면 답변이 거부됩니다) 🚨🚨🚨",
                                    "",
                                    "## 반드시 지켜야 할 물리 법칙:",
                                    "1. **Ghosting ≠ Lag** (서로 다른 현상입니다!)",
                                    "   - Ghosting = 15% (민감도 저하, 트랩 전자와 홀 재결합)",
                                    "   - Lag = 0.15% (신호 잔류, 프레임 간 전하 이월)",
                                    "   - 두 값은 100배 차이! 절대 혼동하지 마세요.",
                                    "",
                                    "2. **QDE 에너지 의존성:**",
                                    "   - QDE(LE, 28kVp) = 97%",
                                    "   - QDE(HE, 49kVp) = 56%",
                                    "   - 41%p 차이가 Gain Map 불일치의 핵심 원인",
                                    "",
                                    "## 이전 답변의 오류:",
                                    *[f"• {c}" for c in critical_errors],
                                    "",
                                    "위 오류를 반드시 수정하여 답변하세요.",
                                    "═══════════════════════════════════════════════════",
                                    ""
                                ])

                                # 수정 가이드를 질문 앞에 배치
                                corrected_prompt = correction_prefix + "\n\n질문: " + prompt

                                with st.spinner("🔄 답변 품질 개선 중..."):
                                    result_v2 = asyncio.run(dynamic_pipeline.process_simple_async(
                                        question=corrected_prompt,
                                        papers=papers_for_pipeline,
                                        physics_knowledge=relevant_knowledge,
                                        max_pmc_fetch=2
                                    ))

                                # 재생성 성공 시: 이전 답변 교체 (삭제 후 새로 표시)
                                if result_v2.answer and result_v2.answer.strip():
                                    full_response = result_v2.answer
                                    regenerated = True

                                    # 이전 답변 삭제하고 새 답변으로 교체
                                    response_placeholder.empty()
                                    response_placeholder.markdown(convert_latex_for_streamlit(full_response))

                                    # 로그에 수정 완료 기록
                                    regen_logger.info(f"[Auto-Regeneration] Corrected response (first 500 chars): {full_response[:500]}...")
                                    regen_logger.info(f"[Auto-Regeneration] Regeneration successful")

                                    # 재생성된 답변 재평가
                                    with st.spinner("🔍 재평가 중..."):
                                        judge_result = judge.evaluate(
                                            question=prompt,
                                            answer=full_response,
                                            reference_knowledge=relevant_knowledge,
                                            context=""
                                        )

                        # Phase 7.1 상태 정보 표시
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            if result.used_fulltext:
                                st.success(f"✅ PMC: {result.enriched_context.fetched_count}개 ({result.enriched_context.total_chars:,}자)")
                            else:
                                st.info(f"📄 초록 기반 ({result.enriched_context.total_chars:,}자)")
                        with col2:
                            if regenerated:
                                st.success("🔄 자동 수정됨")
                            elif result.used_summarizer:
                                st.success("🔬 SLM 요약 완료")
                            else:
                                st.info("📝 원본 사용")
                        with col3:
                            # Agent-as-a-Judge 결과 사용 (있으면)
                            if judge_result:
                                verdict_badges = {
                                    JudgeVerdict.APPROVED: "🏆",
                                    JudgeVerdict.REVISION_REQUIRED: "⚠️",
                                    JudgeVerdict.REJECTED: "❌",
                                }
                                st.metric("품질", f"{judge_result.total_score:.0f}",
                                         delta=f"{verdict_badges.get(judge_result.verdict, '')}")
                            else:
                                st.metric("품질", f"{result.judge_result.total_score:.0f}",
                                         delta="🏆" if result.judge_result.total_score >= 70 else "⚠️")
                        with col4:
                            if judge_result:
                                issues_count = len(judge_result.issues_found) if judge_result.issues_found else 0
                                st.metric("발견 이슈", f"{issues_count}개")
                            else:
                                st.metric("검증",
                                         f"{result.evidence_report.verified_claims}/{result.evidence_report.total_claims}")

                        # 수정 가이드 표시 (재생성 후에도 이슈가 남아있으면)
                        # Phase 7.22: issues_found 또는 corrections 중 하나라도 있으면 표시
                        has_issues = judge_result and (judge_result.corrections or judge_result.issues_found)
                        if has_issues and not regenerated:
                            with st.expander("🔧 수정 가이드 (클릭하여 확인)", expanded=True):
                                # corrections가 있으면 corrections 표시
                                if judge_result.corrections:
                                    for corr in judge_result.corrections[:5]:
                                        st.warning(f"⚠️ {corr}")
                                # corrections가 없고 issues_found만 있으면 issues 표시
                                elif judge_result.issues_found:
                                    for issue in judge_result.issues_found[:5]:
                                        st.warning(f"⚠️ {issue}")

                        # 심각한 문제 경고
                        if judge_result and judge_result.verdict == JudgeVerdict.REJECTED:
                            st.error("⚠️ 답변에 심각한 문제가 발견되었습니다. 수정 가이드를 참고하세요.")

                    else:
                        # 🧠 일반 LLM 심층 추론
                        st.caption(f"🧠 **심층 분석**: {relay_router.get_model_used(dispatch_result)}")

                        # Phase 7.18: 검증된 물리 지식과 검색된 논문 분리
                        # 검색된 논문만 context_parts에 추가 (물리 지식은 별도 전달)
                        context_parts = []
                        for i, source in enumerate(filtered_sources, 1):
                            context_parts.append(f"[문서 {i}] {source['title']}\n저자: {source['authors']}\n내용: {source.get('abstract', source.get('content', ''))[:2000]}")

                        context = "\n\n".join(context_parts)

                        # LLM 답변 스트리밍
                        response_placeholder = st.empty()
                        full_response = ""

                        for chunk in call_llm_with_context(
                            question=prompt,
                            context=context,
                            model=options["model"],
                            temperature=options["temperature"],
                            has_guidelines=True,
                            physics_knowledge=relevant_knowledge  # Phase 7.18: 별도 전달
                        ):
                            full_response += chunk
                            response_placeholder.markdown(full_response + "▌")

                        response_placeholder.markdown(convert_latex_for_streamlit(full_response))

                        # =====================================================
                        # Phase 7.6: Agent-as-a-Judge 평가
                        # =====================================================
                        if options.get("enable_judge", True):  # 기본 활성화
                            with st.spinner("🔍 Agent-as-a-Judge 품질 검증 중..."):
                                judge = get_agent_judge()
                                judge_result = judge.evaluate(
                                    question=prompt,
                                    answer=full_response,
                                    reference_knowledge=relevant_knowledge,
                                    context=context
                                )

                            # 품질 지표 표시
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                verdict_badges = {
                                    JudgeVerdict.APPROVED: ("✅", "success"),
                                    JudgeVerdict.REVISION_REQUIRED: ("⚠️", "warning"),
                                    JudgeVerdict.REJECTED: ("❌", "error"),
                                }
                                badge, status = verdict_badges.get(judge_result.verdict, ("❓", "info"))
                                st.metric("품질 점수", f"{judge_result.total_score:.0f}/100", delta=badge)
                            with col2:
                                st.metric("도구 검증", f"{judge_result.tool_score:.0f}" if hasattr(judge_result, 'tool_score') else "N/A")
                            with col3:
                                issues_count = len(judge_result.issues_found) if judge_result.issues_found else 0
                                st.metric("발견 이슈", f"{issues_count}개")

                            # 수정 가이드 표시 (있는 경우)
                            if hasattr(judge_result, 'corrections') and judge_result.corrections:
                                with st.expander("🔧 수정 가이드 (클릭하여 확인)", expanded=False):
                                    for corr in judge_result.corrections[:5]:
                                        st.markdown(f"- ✓ {corr}")

                            # 심각한 문제 경고
                            if judge_result.verdict == JudgeVerdict.REJECTED:
                                st.error("⚠️ 답변에 심각한 문제가 발견되었습니다. 수정 가이드를 참고하여 재질문을 권장합니다.")

            else:
                # BI-RADS 없음 - 근거 자료 안내만 표시
                full_response = "가이드라인에 직접적인 답변이 없습니다. 아래 연구 논문을 참고하세요."
                st.markdown(full_response)

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

                        pubmed_text = "PubMed" if not is_ko else "PubMed"
                        scholar_text = "Scholar" if not is_ko else "Scholar"
                        pmc_text = "PMC (무료 전문)" if is_ko else "PMC (Free)"

                        with st.expander(expander_title, expanded=False):
                            if paper_level == "medium":
                                st.caption(f"_{paper_reason}_")
                            for i, source in enumerate(filtered_papers, 1):
                                # 링크 구성: PubMed + Google Scholar + PMC(있으면)
                                links = [f"[{pubmed_text}]({source['url']})"]
                                links.append(f"[🔍 {scholar_text}]({source.get('google_scholar_url', '')})")
                                if source.get('pmc_url'):
                                    links.append(f"[✅ {pmc_text}]({source['pmc_url']})")

                                st.markdown(f"""
                                **📄 [{i}] {source['title']}**
                                {source['authors']} - {source['journal']} ({source['year']})
                                {' | '.join(links)}
                                """)

        # 어시스턴트 메시지 저장 (출처 포함, 딥카피로 저장)
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "sources": copy.deepcopy(sources)  # 딥카피로 참조 완전 분리
        })


if __name__ == "__main__":
    main()
