"""
Sophia AI: Data Models
========================
Pydantic 모델 정의 - 논문 메타데이터, 검색 쿼리, 검색 결과
"""

from datetime import datetime
from typing import List, Optional
from urllib.parse import quote

from pydantic import BaseModel, Field, computed_field, field_validator


class QueryFilters(BaseModel):
    """검색 필터"""
    year_min: Optional[int] = None
    year_max: Optional[int] = None
    modality: Optional[List[str]] = None  # DBT, FFDM, CEM, SM
    pathology: Optional[List[str]] = None  # mass, calcification, density, distortion
    study_type: Optional[str] = None  # prospective, retrospective, meta-analysis, review, rct
    population: Optional[str] = None  # Asian, Western, Mixed
    min_citations: Optional[int] = None

    @field_validator("modality", "pathology", mode="before")
    @classmethod
    def ensure_list(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            return [v]
        return v


class Paper(BaseModel):
    """논문 메타데이터"""
    # 기본 식별자
    pmid: str = Field(..., description="PubMed ID (Primary Key)")
    doi: Optional[str] = Field(None, description="Digital Object Identifier")

    # 기본 정보
    title: str = Field(..., description="논문 제목")
    authors: List[str] = Field(default_factory=list, description="저자 목록")
    journal: str = Field(..., description="저널 전체 이름")
    journal_abbrev: str = Field("", description="저널 약어")
    year: int = Field(..., description="출판 연도")
    month: Optional[int] = Field(None, description="출판 월")
    abstract: str = Field("", description="초록")
    full_content: Optional[str] = Field(None, description="전체 내용 (BI-RADS 가이드라인용)")

    # MeSH 및 키워드
    mesh_terms: List[str] = Field(default_factory=list, description="MeSH 용어")
    keywords: List[str] = Field(default_factory=list, description="저자 키워드")
    publication_types: List[str] = Field(default_factory=list, description="출판 유형")

    # 자동 분류 필드
    modality: List[str] = Field(default_factory=list, description="영상 방식: DBT, FFDM, CEM, SM")
    pathology: List[str] = Field(default_factory=list, description="병변 유형: mass, calcification 등")
    study_type: Optional[str] = Field(None, description="연구 유형: prospective, retrospective 등")
    population: Optional[str] = Field(None, description="인구 집단: Asian, Western, Mixed")

    # 품질 지표
    citation_count: int = Field(0, description="피인용 횟수")
    journal_if: Optional[float] = Field(None, description="저널 Impact Factor")

    # 추가 식별자
    pmc_id: Optional[str] = Field(None, description="PubMed Central ID")

    # 데이터 소스
    source: str = Field("pubmed", description="데이터 소스: pubmed, koreamed, semantic_scholar, rsna")

    # 임베딩 (별도 저장, 선택적)
    embedding: Optional[List[float]] = Field(None, exclude=True)

    # 타임스탬프
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None

    @computed_field
    @property
    def pubmed_url(self) -> str:
        """PubMed 링크"""
        return f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/"

    @computed_field
    @property
    def doi_url(self) -> Optional[str]:
        """DOI 링크"""
        if self.doi:
            return f"https://doi.org/{self.doi}"
        return None

    @computed_field
    @property
    def pmc_url(self) -> Optional[str]:
        """PMC 링크 (전문 무료 접근 가능)"""
        if self.pmc_id:
            return f"https://www.ncbi.nlm.nih.gov/pmc/articles/{self.pmc_id}/"
        return None

    @computed_field
    @property
    def google_scholar_url(self) -> str:
        """Google Scholar 검색 링크 (제목 기반)"""
        return f"https://scholar.google.com/scholar?q={quote(self.title)}"

    @computed_field
    @property
    def author_string(self) -> str:
        """저자 문자열 (첫 3명 + et al.)"""
        if len(self.authors) <= 3:
            return ", ".join(self.authors)
        return f"{', '.join(self.authors[:3])}, et al."

    @computed_field
    @property
    def citation_text(self) -> str:
        """인용 형식 문자열"""
        return f"{self.author_string}. {self.title}. {self.journal}. {self.year}."

    @field_validator("abstract", mode="before")
    @classmethod
    def clean_abstract(cls, v):
        if v is None:
            return ""
        return str(v).strip()

    @field_validator("authors", "mesh_terms", "keywords", "publication_types",
                     "modality", "pathology", mode="before")
    @classmethod
    def ensure_list_fields(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return list(v)

    class Config:
        json_schema_extra = {
            "example": {
                "pmid": "12345678",
                "doi": "10.1000/example",
                "title": "Digital Breast Tomosynthesis vs Full-Field Digital Mammography",
                "authors": ["Kim JH", "Lee SY", "Park HJ"],
                "journal": "Radiology",
                "journal_abbrev": "Radiology",
                "year": 2024,
                "abstract": "Purpose: To compare the diagnostic performance...",
                "mesh_terms": ["Mammography", "Breast Neoplasms"],
                "modality": ["DBT", "FFDM"],
                "pathology": ["mass", "calcification"],
                "study_type": "retrospective",
                "citation_count": 15,
                "journal_if": 19.7
            }
        }


class SearchQuery(BaseModel):
    """파싱된 검색 쿼리"""
    original_query: str = Field(..., description="원본 검색어")
    keywords: List[str] = Field(default_factory=list, description="추출된 키워드")
    mesh_terms: List[str] = Field(default_factory=list, description="관련 MeSH 용어")
    filters: QueryFilters = Field(default_factory=QueryFilters)
    intent: str = Field("", description="검색 의도 설명")
    recommended_exclusions: List[str] = Field(default_factory=list, description="제외 권장 출판 유형: Case Reports, Letter")

    # Phase 1: Medical-Logic-CoT 확장 필드
    expanded_keywords: List[str] = Field(default_factory=list, description="CoT 추론으로 확장된 의학 키워드")
    reasoning_trace: str = Field("", description="DeepSeek-R1 <think> 추론 과정")
    semantic_variations: List[str] = Field(default_factory=list, description="동의어/유사어 확장")

    @computed_field
    @property
    def has_filters(self) -> bool:
        """필터가 적용되어 있는지"""
        f = self.filters
        return any([
            f.year_min, f.year_max, f.modality, f.pathology,
            f.study_type, f.population, f.min_citations
        ])


class SearchResult(BaseModel):
    """검색 결과 아이템"""
    paper: Paper
    score: float = Field(..., description="검색 점수 (0-1)")
    rank: int = Field(..., description="검색 순위")
    matched_terms: List[str] = Field(default_factory=list, description="매칭된 검색어")

    @computed_field
    @property
    def score_percent(self) -> int:
        """점수를 퍼센트로"""
        return int(self.score * 100)


class SearchResponse(BaseModel):
    """검색 API 응답"""
    query: SearchQuery
    results: List[SearchResult]
    total_count: int
    time_ms: int = Field(..., description="검색 소요 시간 (밀리초)")

    @computed_field
    @property
    def has_results(self) -> bool:
        return len(self.results) > 0


class CollectionStats(BaseModel):
    """수집 통계"""
    total_papers: int
    by_year: dict[int, int]
    by_modality: dict[str, int]
    by_pathology: dict[str, int]
    by_study_type: dict[str, int]
    top_journals: List[tuple[str, int]]
    collected_at: datetime


class EvalQuery(BaseModel):
    """평가용 쿼리"""
    id: str
    query: str
    relevant_pmids: List[str]
    category: str = "general"


class EvalMetrics(BaseModel):
    """평가 지표"""
    query_id: str
    precision_at_5: float
    precision_at_10: float
    recall_at_10: float
    mrr: float  # Mean Reciprocal Rank
    ndcg_at_10: float


class EvalReport(BaseModel):
    """평가 리포트"""
    evaluated_at: datetime
    total_queries: int
    avg_precision_at_5: float
    avg_precision_at_10: float
    avg_recall_at_10: float
    avg_mrr: float
    avg_ndcg_at_10: float
    per_query_metrics: List[EvalMetrics]
