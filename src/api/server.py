"""
MARIA-Mammo: FastAPI Server
===========================
검색 API 서버 (LLM 쿼리 파서 지원)
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    HealthResponse,
    PaperResponse,
    ParsedQueryResponse,
    ParseRequest,
    ParseResponse,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    StatsResponse,
)
from src.search.engine import SearchEngine

logger = logging.getLogger(__name__)

# 환경 변수에서 설정 읽기
PARSER_MODE = os.getenv("PARSER_MODE", "smart")  # rule, llm, smart
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")

# 글로벌 검색 엔진
search_engine: Optional[SearchEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 라이프사이클"""
    global search_engine

    logger.info("Starting MARIA-Mammo API server...")
    logger.info(f"Parser mode: {PARSER_MODE}, Ollama URL: {OLLAMA_URL}, LLM model: {LLM_MODEL}")

    # 검색 엔진 초기화
    try:
        search_engine = SearchEngine(
            db_path=Path("data/index"),
            parser_mode=PARSER_MODE,
            ollama_url=OLLAMA_URL,
            llm_model=LLM_MODEL,
        )
        logger.info("Search engine initialized")
    except Exception as e:
        logger.error(f"Failed to initialize search engine: {e}")
        search_engine = None

    yield

    # 리소스 정리
    if search_engine and hasattr(search_engine.parser, 'close'):
        search_engine.parser.close()

    logger.info("Shutting down...")


app = FastAPI(
    title="MARIA-Mammo API",
    description="Mammography Literature Search Engine API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_engine() -> SearchEngine:
    """검색 엔진 의존성"""
    if search_engine is None:
        raise HTTPException(
            status_code=503,
            detail="Search engine not initialized. Please check the index.",
        )
    return search_engine


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """헬스체크"""
    engine = get_engine()
    stats = engine.get_stats()

    # LLM 상태 확인
    llm_available = False
    if hasattr(engine, 'parser'):
        if hasattr(engine.parser, 'using_llm'):
            llm_available = engine.parser.using_llm
        elif hasattr(engine.parser, 'is_available'):
            llm_available = engine.parser.is_available()

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        papers_count=stats.get("total_papers", 0),
        llm_status="available" if llm_available else "unavailable",
        parser_mode=PARSER_MODE,
    )


@app.post("/parse", response_model=ParseResponse)
async def parse_query(request: ParseRequest):
    """쿼리 파싱만 테스트 (검색 없이)"""
    import time

    engine = get_engine()

    start = time.time()
    try:
        parsed = engine.parser.parse(request.query)
        elapsed_ms = int((time.time() - start) * 1000)

        # parser_used 결정
        parser_used = "rule"
        if hasattr(engine.parser, 'using_llm') and engine.parser.using_llm:
            parser_used = "llm"
        elif hasattr(engine.parser, '_llm_parser') and engine.parser._llm_parser:
            parser_used = "llm"

        return ParseResponse(
            original_query=parsed.original_query,
            keywords=parsed.keywords,
            mesh_terms=parsed.mesh_terms,
            filters=parsed.filters.model_dump() if parsed.filters else {},
            intent=parsed.intent,
            parser_used=parser_used,
            time_ms=elapsed_ms,
        )

    except Exception as e:
        logger.error(f"Parse error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """논문 검색"""
    engine = get_engine()

    try:
        response = engine.search(
            query=request.query,
            top_k=request.top_k,
            use_rerank=request.use_rerank,
        )

        # 응답 변환
        results = []
        for r in response.results:
            paper = r.paper
            results.append(
                SearchResultItem(
                    paper=PaperResponse(
                        pmid=paper.pmid,
                        doi=paper.doi,
                        title=paper.title,
                        authors=paper.authors,
                        author_string=paper.author_string,
                        journal=paper.journal,
                        journal_abbrev=paper.journal_abbrev,
                        year=paper.year,
                        abstract=paper.abstract,
                        modality=paper.modality,
                        pathology=paper.pathology,
                        study_type=paper.study_type,
                        population=paper.population,
                        citation_count=paper.citation_count,
                        journal_if=paper.journal_if,
                        pubmed_url=paper.pubmed_url,
                        doi_url=paper.doi_url,
                        pmc_url=paper.pmc_url,
                    ),
                    score=r.score,
                    score_percent=r.score_percent,
                    rank=r.rank,
                    matched_terms=r.matched_terms,
                )
            )

        return SearchResponse(
            query=ParsedQueryResponse(
                original_query=response.query.original_query,
                keywords=response.query.keywords,
                mesh_terms=response.query.mesh_terms,
                intent=response.query.intent,
                has_filters=response.query.has_filters,
            ),
            results=results,
            total_count=response.total_count,
            time_ms=response.time_ms,
        )

    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/paper/{pmid}", response_model=PaperResponse)
async def get_paper(pmid: str):
    """논문 상세 조회"""
    engine = get_engine()

    paper = engine.db.get_paper(pmid)
    if paper is None:
        raise HTTPException(status_code=404, detail="Paper not found")

    return PaperResponse(
        pmid=paper.pmid,
        doi=paper.doi,
        title=paper.title,
        authors=paper.authors,
        author_string=paper.author_string,
        journal=paper.journal,
        journal_abbrev=paper.journal_abbrev,
        year=paper.year,
        abstract=paper.abstract,
        modality=paper.modality,
        pathology=paper.pathology,
        study_type=paper.study_type,
        population=paper.population,
        citation_count=paper.citation_count,
        journal_if=paper.journal_if,
        pubmed_url=paper.pubmed_url,
        doi_url=paper.doi_url,
        pmc_url=paper.pmc_url,
    )


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """인덱스 통계"""
    engine = get_engine()
    stats = engine.get_stats()

    return StatsResponse(
        total_papers=stats.get("total_papers", 0),
        total_vectors=stats.get("total_vectors", 0),
        by_year=stats.get("by_year", {}),
    )


def main():
    """서버 실행"""
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    uvicorn.run(
        "src.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    main()
