"""
Sophia AI: Knowledge Module
===========================
물리 지식 및 도메인 지식 관리
"""

from src.knowledge.core_physics import (
    get_core_physics_prompt,
    GOLDEN_FORMULAS,
    get_formula,
    get_all_formula_ids,
)

from src.knowledge.manager import (
    KnowledgeManager,
    get_knowledge_manager,
    get_relevant_physics_knowledge,
)

__all__ = [
    # Core Physics
    "get_core_physics_prompt",
    "GOLDEN_FORMULAS",
    "get_formula",
    "get_all_formula_ids",
    # Knowledge Manager
    "KnowledgeManager",
    "get_knowledge_manager",
    "get_relevant_physics_knowledge",
]
