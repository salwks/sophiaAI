"""
Sophia AI: Unified Prompt Builder
=================================
Phase 7.19: Single prompt builder for all LLM calls.

This module consolidates the 4 separate knowledge delivery paths into one:
- app.py: core_physics.py + KnowledgeManager
- dynamic_evidence.py: 3 prompt builders (CQO, QOCO, CoT)
- relay_router.py: _build_llm_user_prompt()
- orchestrator.py: explain_prompt + physics_knowledge

All LLM calls now use UnifiedPromptBuilder for consistent:
- Knowledge source (KnowledgeManager only)
- Truncation limits (standardized)
- Priority ordering (verified knowledge > searched papers)
- Format consistency
"""

import logging
from dataclasses import dataclass
from typing import Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..knowledge.manager import KnowledgeManager

logger = logging.getLogger(__name__)


@dataclass
class PromptLimits:
    """Unified truncation limits for all prompt components."""
    KNOWLEDGE_CONTEXT: int = 6000    # Verified physics knowledge
    SEARCH_CONTEXT: int = 4000       # Searched papers/documents
    TOTAL_PROMPT: int = 12000        # Total user prompt limit
    AXIOMS: int = 8000               # Constitutional axioms

    # Strategy-specific limits
    COT_KNOWLEDGE: int = 5000        # Chain-of-Thought allows more context
    SIMPLE_KNOWLEDGE: int = 4000     # Simple queries need less


class UnifiedPromptBuilder:
    """
    Unified prompt builder for all LLM calls.

    Consolidates knowledge injection from multiple sources into a single,
    consistent interface. All LLM calling code should use this class.

    Usage:
        builder = UnifiedPromptBuilder(knowledge_manager)

        # For system prompt
        system_prompt = builder.build_system_prompt()

        # For user prompt with knowledge context
        knowledge_ctx = builder.build_knowledge_context(query)
        user_prompt = builder.build_user_prompt(
            query=query,
            knowledge_context=knowledge_ctx,
            search_context=search_results
        )
    """

    def __init__(self, knowledge_manager: "KnowledgeManager"):
        """
        Initialize with KnowledgeManager instance.

        Args:
            knowledge_manager: The single source of truth for all knowledge
        """
        self.km = knowledge_manager
        self._cached_axioms: Optional[str] = None
        self._cached_strict_rules: Optional[str] = None

    def build_system_prompt(self, include_axioms: bool = True) -> str:
        """
        Build system prompt with physics laws enforcement.

        This system prompt ensures LLM responses adhere to physical laws
        and use verified numerics.

        Args:
            include_axioms: Whether to include constitutional axioms (default True)

        Returns:
            Complete system prompt string
        """
        parts = []

        # Role definition
        parts.append("# Role: Mammography Physics Integrity Expert")
        parts.append("당신은 유방영상의학 물리학 무결성 검증 전문가입니다.")
        parts.append("모든 답변은 검증된 물리 법칙과 수치에 기반해야 합니다.")
        parts.append("")

        # Constitutional axioms
        if include_axioms:
            axioms = self.get_axioms()
            if axioms:
                parts.append("# Constitutional Physics Axioms")
                parts.append("다음 물리 법칙은 절대 위반할 수 없습니다:")
                parts.append(axioms)
                parts.append("")

        # Strict rules
        rules = self.get_strict_rules()
        if rules:
            parts.append("# Strict Rules")
            parts.append(rules)
            parts.append("")

        # Response requirements
        parts.append("# Response Requirements")
        parts.append("1. 검증된 수치 우선 사용 (표준 참조 자료의 값)")
        parts.append("2. 물리 법칙 준수 (CONSTITUTIONAL AXIOMS)")
        parts.append("3. 출처 명시 (논문 또는 표준)")
        parts.append("4. 불확실한 경우 명시적으로 표현")
        parts.append("5. 물리 법칙 위반 시 자동 거부")

        return "\n".join(parts)

    def build_knowledge_context(
        self,
        query: str,
        max_chars: int = PromptLimits.KNOWLEDGE_CONTEXT,
        max_modules: int = 3,
        strategy: str = "default"
    ) -> str:
        """
        Build query-based dynamic knowledge context.

        Retrieves relevant knowledge modules based on query keywords
        and formats them for LLM context.

        Args:
            query: User question
            max_chars: Maximum characters for knowledge context
            max_modules: Maximum number of knowledge modules to include
            strategy: Prompt strategy ("default", "cot", "simple")

        Returns:
            Formatted knowledge context string
        """
        # Adjust limits based on strategy
        if strategy == "cot":
            max_chars = min(max_chars, PromptLimits.COT_KNOWLEDGE)
        elif strategy == "simple":
            max_chars = min(max_chars, PromptLimits.SIMPLE_KNOWLEDGE)

        # Get relevant knowledge modules
        modules = self.km.get_relevant_knowledge(query, max_modules=max_modules)

        if not modules:
            logger.debug(f"No knowledge modules matched for query: {query[:50]}...")
            return ""

        # Format modules for context
        formatted = self.km.format_for_context(modules, query=query)

        # Smart truncation preserving important sections
        if len(formatted) > max_chars:
            formatted = self._smart_truncate(formatted, max_chars)

        logger.info(f"Knowledge context: {len(modules)} modules, {len(formatted)} chars")
        return formatted

    def build_user_prompt(
        self,
        query: str,
        knowledge_context: str = "",
        search_context: str = "",
        include_priority_note: bool = True
    ) -> str:
        """
        Build unified user prompt with proper section ordering.

        Priority ordering:
        1. Verified physics knowledge (highest priority)
        2. Searched papers (supplementary)
        3. User question

        Args:
            query: User question
            knowledge_context: Formatted knowledge from KnowledgeManager
            search_context: Searched papers/documents
            include_priority_note: Whether to include data priority reminder

        Returns:
            Complete user prompt string
        """
        parts = []

        # Priority note
        if include_priority_note and knowledge_context:
            parts.append("## Data Priority (반드시 준수)")
            parts.append("1순위: 아래 '검증된 물리학 참조'의 수치")
            parts.append("2순위: 검색된 논문의 수치")
            parts.append("3순위: 일반 지식 (출처 명시 필수)")
            parts.append("")

        # Verified knowledge (highest priority)
        if knowledge_context:
            parts.append("## 검증된 물리학 참조 (반드시 사용)")
            parts.append(knowledge_context)
            parts.append("")

        # Searched papers (supplementary)
        if search_context:
            # Truncate search context if needed
            truncated_search = search_context[:PromptLimits.SEARCH_CONTEXT]
            parts.append("## 검색된 논문")
            parts.append(truncated_search)
            parts.append("")

        # User question
        parts.append("## 질문")
        parts.append(query)

        return "\n".join(parts)

    def get_axioms(self) -> str:
        """
        Get constitutional physics axioms from core_physics.json.

        Returns:
            Formatted axioms string
        """
        if self._cached_axioms is not None:
            return self._cached_axioms

        # Load core_physics module
        core_physics = self.km.get_knowledge_by_id("core_physics")
        if not core_physics:
            logger.warning("core_physics module not found in KnowledgeManager")
            return ""

        axioms_data = core_physics.get("constitutional_axioms", {})
        if not axioms_data:
            return ""

        parts = []

        # Phase 1: Fundamental laws
        phase1 = axioms_data.get("phase1_fundamental", {})
        if phase1:
            parts.append(f"### {phase1.get('title', 'Phase 1')}")
            for law in phase1.get("laws", []):
                parts.append(f"- **{law['name']}**: {law['statement']}")
                if law.get("formula"):
                    parts.append(f"  {law['formula']}")
                if law.get("warning"):
                    parts.append(f"  (Warning: {law['warning']})")
            derivations = phase1.get("key_derivations", [])
            if derivations:
                parts.append("Key derivations:")
                for d in derivations:
                    parts.append(f"  - {d}")
            parts.append("")

        # Phase 2: PCD Contrast
        phase2 = axioms_data.get("phase2_contrast", {})
        if phase2:
            parts.append(f"### {phase2.get('title', 'Phase 2')}")
            for law in phase2.get("laws", []):
                parts.append(f"- **{law['name']}**: {law['statement']}")
            if phase2.get("warning"):
                parts.append(f"Warning: {phase2['warning']}")
            parts.append("")

        # Phase 3: DQE/NPS
        phase3 = axioms_data.get("phase3_dqe", {})
        if phase3:
            parts.append(f"### {phase3.get('title', 'Phase 3')}")
            for law in phase3.get("laws", []):
                parts.append(f"- **{law['name']}**: {law.get('statement', law.get('formula', ''))}")
            if phase3.get("warning"):
                parts.append(f"Warning: {phase3['warning']}")
            parts.append("")

        # Phase 4: MTF
        phase4 = axioms_data.get("phase4_mtf", {})
        if phase4:
            parts.append(f"### {phase4.get('title', 'Phase 4')}")
            for law in phase4.get("laws", []):
                parts.append(f"- **{law['name']}**: {law.get('statement', law.get('formula', ''))}")
            if phase4.get("warning"):
                parts.append(f"Warning: {phase4['warning']}")
            parts.append("")

        # Phase 5: Tomo
        phase5 = axioms_data.get("phase5_tomo", {})
        if phase5:
            parts.append(f"### {phase5.get('title', 'Phase 5')}")
            for law in phase5.get("laws", []):
                parts.append(f"- **{law['name']}**")
            # PCD advantage table
            adv = phase5.get("pcd_snr_advantage_by_projections", {})
            if adv:
                parts.append("PCD SNR advantage: " + ", ".join([f"{k}:{v}" for k, v in adv.items()]))
            parts.append("")

        result = "\n".join(parts)

        # Truncate if too long
        if len(result) > PromptLimits.AXIOMS:
            result = result[:PromptLimits.AXIOMS]
            # Find last complete line
            last_newline = result.rfind("\n")
            if last_newline > 0:
                result = result[:last_newline]

        self._cached_axioms = result
        return result

    def get_strict_rules(self) -> str:
        """
        Get strict rules for LLM responses.

        Returns:
            Formatted rules string
        """
        if self._cached_strict_rules is not None:
            return self._cached_strict_rules

        core_physics = self.km.get_knowledge_by_id("core_physics")
        if not core_physics:
            # Fallback to default rules
            return self._default_strict_rules()

        rules_data = core_physics.get("strict_rules", {})
        rules = rules_data.get("rules", [])

        if not rules:
            return self._default_strict_rules()

        parts = []
        for rule in rules:
            parts.append(f"{rule['id']}. {rule['name']}: {rule['statement']}")

        self._cached_strict_rules = "\n".join(parts)
        return self._cached_strict_rules

    def _default_strict_rules(self) -> str:
        """Default strict rules if core_physics.json is unavailable."""
        return """0. Data Priority: 검증된 수치 우선 사용
1. Evidence First: 결론 도출 전 근거 제시
2. Grounding Physics: 물리 정의 준수
3. Quoting Integrity: 인용 무결성 유지
4. Contradiction Reporting: 물리 법칙 충돌 시 보고"""

    def _smart_truncate(self, text: str, max_chars: int) -> str:
        """
        Smart truncation preserving important sections.

        Priority:
        1. Warning/misconception sections
        2. Formulas sections
        3. Beginning of text

        Args:
            text: Text to truncate
            max_chars: Maximum characters

        Returns:
            Truncated text
        """
        if len(text) <= max_chars:
            return text

        # Find priority sections
        priority_markers = ["Warning", "warning", "오류", "주의", "WRONG", "공식", "formula"]

        # Collect priority sections
        priority_sections = []
        for marker in priority_markers:
            idx = text.find(marker)
            if idx != -1:
                # Find section boundaries (next double newline or end)
                section_end = text.find("\n\n", idx)
                if section_end == -1:
                    section_end = len(text)
                section = text[idx:min(section_end + 2, len(text))]
                if len(section) < 1000:  # Don't include huge sections
                    priority_sections.append(section)

        # Combine priority sections
        priority_text = "\n".join(priority_sections)
        remaining_budget = max_chars - len(priority_text) - 100  # Buffer

        if remaining_budget > 0:
            # Add beginning of text
            truncated = text[:remaining_budget]
            last_newline = truncated.rfind("\n")
            if last_newline > remaining_budget * 0.5:
                truncated = truncated[:last_newline]
            return truncated + "\n\n[...truncated...]\n\n" + priority_text
        else:
            # Just use priority sections
            return priority_text[:max_chars]

    def build_cot_prompt(
        self,
        query: str,
        knowledge_context: str,
        search_context: str,
        causal_chain_hint: str = ""
    ) -> str:
        """
        Build Chain-of-Thought prompt for composite questions.

        Uses 4-stage causal reasoning:
        1. Phenomenon Analysis (현상 분석)
        2. Causal Mechanism (물리적 메커니즘)
        3. Parameter Constraints (파라미터 제약)
        4. Evidence Synthesis (근거 통합)

        Args:
            query: User question
            knowledge_context: Physics knowledge
            search_context: Searched papers
            causal_chain_hint: Detected causal chain hint

        Returns:
            Formatted CoT prompt
        """
        parts = []

        # Role
        parts.append("# Physics Chain-of-Thought Reasoning")
        parts.append("복합 물리 질문에 대해 4단계 인과 추론을 수행합니다.")
        parts.append("")

        # Causal chain hint
        if causal_chain_hint:
            parts.append("## 핵심 인과 관계")
            parts.append(causal_chain_hint)
            parts.append("")

        # Add knowledge with higher limit for CoT
        if knowledge_context:
            parts.append("## 검증된 물리학 참조")
            parts.append(knowledge_context[:PromptLimits.COT_KNOWLEDGE])
            parts.append("")

        # Add search context
        if search_context:
            parts.append("## 검색된 논문")
            parts.append(search_context[:PromptLimits.SEARCH_CONTEXT])
            parts.append("")

        # Question
        parts.append("## 질문")
        parts.append(query)
        parts.append("")

        # Response format
        parts.append("## 응답 형식 (4단계 인과 추론)")
        parts.append("### 1. 현상 분석 (Phenomenon Analysis)")
        parts.append("- 관찰된/질문된 현상은 무엇인가?")
        parts.append("")
        parts.append("### 2. 물리적 메커니즘 (Causal Mechanism)")
        parts.append("- 어떤 물리 법칙이 적용되는가?")
        parts.append("- 변수들 간의 인과 관계는?")
        parts.append("")
        parts.append("### 3. 파라미터 제약 (Parameter Constraints)")
        parts.append("- 어떤 제한 조건이 존재하는가?")
        parts.append("- 수치적 범위는?")
        parts.append("")
        parts.append("### 4. 근거 통합 (Evidence Synthesis)")
        parts.append("- 최종 결론과 수치")
        parts.append("- 출처 명시")

        return "\n".join(parts)

    def build_qoco_prompt(
        self,
        query: str,
        knowledge_context: str,
        search_context: str
    ) -> str:
        """
        Build QOCO (Question-Options-Context-Output) prompt for calculation questions.

        Args:
            query: User question
            knowledge_context: Physics knowledge
            search_context: Searched papers

        Returns:
            Formatted QOCO prompt
        """
        parts = []

        # [Q] Question framing
        parts.append("## [Q] Question")
        parts.append(f"질문: {query}")
        parts.append("의도: 물리량 계산 또는 비교")
        parts.append("")

        # [O] Options/Constraints
        parts.append("## [O] Constraints")
        parts.append("- 검증된 공식만 사용")
        parts.append("- 단위 명시 필수")
        parts.append("- 중간 계산 과정 표시")
        parts.append("")

        # [C] Context
        parts.append("## [C] Reference Context")
        if knowledge_context:
            parts.append("### 검증된 물리 참조")
            parts.append(knowledge_context[:PromptLimits.SIMPLE_KNOWLEDGE])
        if search_context:
            parts.append("### 검색된 문서")
            parts.append(search_context[:PromptLimits.SEARCH_CONTEXT])
        parts.append("")

        # [O] Output format
        parts.append("## [O] Output Format")
        parts.append("1. 사용 공식")
        parts.append("2. 변수 대입")
        parts.append("3. 계산 결과 (단위 포함)")
        parts.append("4. 출처")

        return "\n".join(parts)

    def clear_cache(self):
        """Clear cached axioms and rules."""
        self._cached_axioms = None
        self._cached_strict_rules = None
        logger.debug("UnifiedPromptBuilder cache cleared")


# Convenience function for creating builder with default KnowledgeManager
def get_unified_builder(knowledge_manager: Optional["KnowledgeManager"] = None) -> UnifiedPromptBuilder:
    """
    Get UnifiedPromptBuilder instance.

    Args:
        knowledge_manager: Optional KnowledgeManager instance.
                          If None, creates a new one with default path.

    Returns:
        UnifiedPromptBuilder instance
    """
    if knowledge_manager is None:
        from ..knowledge.manager import KnowledgeManager
        knowledge_manager = KnowledgeManager()

    return UnifiedPromptBuilder(knowledge_manager)
