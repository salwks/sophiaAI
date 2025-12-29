"""
Sophia AI: API Schemas
========================
API 요청/응답 스키마
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """검색 요청"""
    query: str = Field(..., min_length=1, description="검색 쿼리")
    top_k: int = Field(10, ge=1, le=100, description="반환할 결과 수")
    use_rerank: bool = Field(True, description="리랭킹 사용 여부")

    # 필터
    year_min: Optional[int] = Field(None, description="최소 연도")
    year_max: Optional[int] = Field(None, description="최대 연도")
    modality: Optional[List[str]] = Field(None, description="Modality 필터")
    pathology: Optional[List[str]] = Field(None, description="Pathology 필터")
    study_type: Optional[str] = Field(None, description="Study Type 필터")

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "DBT vs FFDM for microcalcification detection",
                "top_k": 10,
                "use_rerank": True,
            }
        }
    }


class PaperResponse(BaseModel):
    """논문 응답"""
    pmid: str
    doi: Optional[str]
    title: str
    authors: List[str]
    author_string: str
    journal: str
    journal_abbrev: str
    year: int
    abstract: str
    modality: List[str]
    pathology: List[str]
    study_type: Optional[str]
    population: Optional[str]
    citation_count: int
    journal_if: Optional[float]
    pubmed_url: str
    doi_url: Optional[str]
    pmc_url: Optional[str]


class SearchResultItem(BaseModel):
    """검색 결과 아이템"""
    paper: PaperResponse
    score: float
    score_percent: int
    rank: int
    matched_terms: List[str]


class ParsedQueryResponse(BaseModel):
    """파싱된 쿼리"""
    original_query: str
    keywords: List[str]
    mesh_terms: List[str]
    intent: str
    has_filters: bool


class SearchResponse(BaseModel):
    """검색 응답"""
    query: ParsedQueryResponse
    results: List[SearchResultItem]
    total_count: int
    time_ms: int


class StatsResponse(BaseModel):
    """통계 응답"""
    total_papers: int
    total_vectors: int
    by_year: dict


class HealthResponse(BaseModel):
    """헬스체크 응답"""
    status: str
    version: str
    papers_count: int
    llm_status: str  # "available" or "unavailable"
    parser_mode: str  # "rule", "llm", "smart"


class ParseRequest(BaseModel):
    """쿼리 파싱 요청"""
    query: str = Field(..., min_length=1, description="파싱할 쿼리")

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "DBT vs FFDM sensitivity comparison"
            }
        }
    }


class ParseResponse(BaseModel):
    """쿼리 파싱 응답"""
    original_query: str
    keywords: List[str]
    mesh_terms: List[str]
    filters: dict
    intent: str
    parser_used: str  # "llm" or "rule"
    time_ms: int
