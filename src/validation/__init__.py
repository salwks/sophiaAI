"""
Sophia AI: Validation Module
==============================
Phase 2: 검증 모듈 - 수식 무결성, 할루시네이션 검사, Critic Agent
"""

from src.validation.formula_checker import (
    FormulaChecker,
    FormulaCheckResult,
    FormulaMatch,
    check_formulas,
    extract_and_validate_formulas,
)

from src.validation.hallucination import (
    HallucinationChecker,
    HallucinationCheckResult,
    CitationMatch,
    CitationExtractor,
    check_hallucination,
    extract_citations,
)

from src.validation.critic_agent import (
    CriticResult,
    IntegratedAuditResult,
    IntegratedAuditor,
    run_critic,
    dual_agent_workflow,
    integrated_dual_agent_workflow,
)

__all__ = [
    # Formula Checker
    "FormulaChecker",
    "FormulaCheckResult",
    "FormulaMatch",
    "check_formulas",
    "extract_and_validate_formulas",

    # Hallucination Checker
    "HallucinationChecker",
    "HallucinationCheckResult",
    "CitationMatch",
    "CitationExtractor",
    "check_hallucination",
    "extract_citations",

    # Critic Agent
    "CriticResult",
    "IntegratedAuditResult",
    "IntegratedAuditor",
    "run_critic",
    "dual_agent_workflow",
    "integrated_dual_agent_workflow",
]
