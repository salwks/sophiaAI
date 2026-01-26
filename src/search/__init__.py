"""
Sophia AI: Search Module
=========================
검색 관련 모듈 - 쿼리 파싱, 검색, 리랭킹
"""

from src.search.query_parser import (
    QueryParser,
    LLMQueryParser,
    SmartQueryParser,
    EnhancedQueryParser,
)
from src.search.query_expander import (
    MedicalCoTQueryExpander,
    QueryExpansion,
)
from src.search.retriever import HybridRetriever
from src.search.reranker import Reranker
from src.search.weighting_logic import (
    WMRConfig,
    WeightCalculator,
    WeightedMedicalReranker,
    apply_wmr,
    get_paper_weight_breakdown,
)

__all__ = [
    # Query Parsing
    "QueryParser",
    "LLMQueryParser",
    "SmartQueryParser",
    "EnhancedQueryParser",

    # Query Expansion (Phase 1)
    "MedicalCoTQueryExpander",
    "QueryExpansion",

    # Retrieval
    "HybridRetriever",

    # Reranking (Phase 3: WMR)
    "Reranker",
    "WMRConfig",
    "WeightCalculator",
    "WeightedMedicalReranker",
    "apply_wmr",
    "get_paper_weight_breakdown",
]
