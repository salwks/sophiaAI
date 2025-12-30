"""
Sophia AI: Search Engine
==========================
통합 검색 엔진 (Rule-based + LLM 쿼리 파서 지원)
"""

import logging
import time
from pathlib import Path
from typing import List, Optional, Literal

from src.indexing.database import DatabaseManager
from src.indexing.embedder import PaperEmbedder
from src.models import Paper, SearchQuery, SearchResponse, SearchResult
from src.search.birads_matcher import BiradsCategoryMatcher
from src.search.query_parser import QueryParser, LLMQueryParser, SmartQueryParser
from src.search.reranker import Reranker
from src.search.retriever import HybridRetriever

logger = logging.getLogger(__name__)


class SearchEngine:
    """통합 검색 엔진"""

    def __init__(
        self,
        db: Optional[DatabaseManager] = None,
        embedder: Optional[PaperEmbedder] = None,
        db_path: Path = Path("data/index"),
        use_reranker: bool = True,
        parser_mode: Literal["rule", "llm", "smart"] = "smart",
        ollama_url: str = "http://localhost:11434",
        llm_model: str = "llama3.2",
    ):
        """
        Args:
            db: 데이터베이스 매니저
            embedder: 임베딩 생성기
            db_path: 데이터베이스 경로
            use_reranker: 리랭커 사용 여부
            parser_mode: 파서 모드 (rule: 규칙 기반, llm: LLM 기반, smart: 자동 선택)
            ollama_url: Ollama 서버 URL
            llm_model: LLM 모델 이름
        """
        logger.info("Initializing search engine...")

        # 컴포넌트 초기화
        self.db = db or DatabaseManager(db_path)
        self.embedder = embedder or PaperEmbedder()

        # 파서 선택
        self.parser_mode = parser_mode
        if parser_mode == "llm":
            self.parser = LLMQueryParser(
                ollama_url=ollama_url,
                model=llm_model,
                fallback_to_rule=True,
            )
            logger.info("Using LLM query parser")
        elif parser_mode == "smart":
            self.parser = SmartQueryParser(
                prefer_llm=True,
                ollama_url=ollama_url,
                model=llm_model,
            )
            if hasattr(self.parser, 'using_llm') and self.parser.using_llm:
                logger.info("Using Smart parser (LLM mode)")
            else:
                logger.info("Using Smart parser (Rule-based mode)")
        else:
            self.parser = QueryParser()
            logger.info("Using rule-based query parser")

        self.retriever = HybridRetriever(self.db, self.embedder)
        self.reranker = Reranker() if use_reranker else None
        self.birads_matcher = BiradsCategoryMatcher(db_path)

        self.use_reranker = use_reranker

        logger.info("Search engine initialized with BI-RADS category matcher")

    def _protect_guidelines(
        self,
        reranked: List[tuple],
        parsed_query,
    ) -> List[tuple]:
        """
        가이드라인 쿼리일 때 BI-RADS 문서를 상위에 유지

        Args:
            reranked: (Paper, score) 튜플 리스트
            parsed_query: 파싱된 쿼리

        Returns:
            BI-RADS 문서가 보호된 결과
        """
        # 가이드라인 의도 확인
        intent_lower = parsed_query.intent.lower() if parsed_query.intent else ""
        keywords_lower = [k.lower() for k in parsed_query.keywords]
        original_lower = parsed_query.original_query.lower()

        is_guideline_query = (
            "guideline" in intent_lower or
            "definition" in intent_lower or
            "criteria" in intent_lower or
            "guideline" in keywords_lower or
            "criteria" in keywords_lower or
            "bi-rads" in keywords_lower or
            "기준" in original_lower or
            "정의" in original_lower or
            "가이드라인" in original_lower
        )

        if not is_guideline_query:
            return reranked

        # BI-RADS 문서와 일반 문서 분리
        birads_docs = []
        other_docs = []

        for paper, score in reranked:
            if paper.pmid.startswith("BIRADS_"):
                birads_docs.append((paper, score))
            else:
                other_docs.append((paper, score))

        if not birads_docs:
            return reranked

        logger.info(f"Protecting {len(birads_docs)} BI-RADS documents after reranking")

        # BI-RADS 문서 점수를 최고 점수 * 1.2로 조정
        if other_docs:
            max_score = other_docs[0][1]
            adjusted_birads = [
                (paper, max_score * (1.2 - 0.05 * i))
                for i, (paper, _) in enumerate(birads_docs[:3])
            ]
        else:
            adjusted_birads = birads_docs[:3]

        # BI-RADS 먼저, 그 다음 일반 문서
        combined = adjusted_birads + other_docs

        # 점수로 재정렬
        combined.sort(key=lambda x: x[1], reverse=True)

        return combined

    def search(
        self,
        query: str,
        top_k: int = 10,
        use_rerank: bool = True,
    ) -> SearchResponse:
        """
        통합 검색

        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수
            use_rerank: 리랭킹 사용 여부

        Returns:
            검색 응답
        """
        start_time = time.time()

        # 0. BI-RADS 카테고리 직접 매칭 시도 (구조화 검색)
        if self.birads_matcher.is_birads_query(query):
            birads_doc = self.birads_matcher.search(query)
            if birads_doc:
                logger.info(f"BI-RADS category query detected and matched: {birads_doc.title}")

                # BI-RADS 문서를 최상위 결과로 반환
                elapsed_ms = int((time.time() - start_time) * 1000)

                # 간단한 파싱된 쿼리 생성
                parsed_query = SearchQuery(
                    original_query=query,
                    intent="guideline_lookup",
                    keywords=[],
                )

                return SearchResponse(
                    query=parsed_query,
                    results=[
                        SearchResult(
                            paper=birads_doc,
                            score=1.0,  # 완벽한 매칭
                            rank=1,
                            matched_terms=["BI-RADS", "Category"],
                        )
                    ],
                    total_count=1,
                    time_ms=elapsed_ms,
                )

        # 0-1. BI-RADS 일반 가이드라인 검색 (positioning, views 등)
        if self.birads_matcher.is_birads_general_query(query):
            birads_docs = self.birads_matcher.search_birads_general(query, top_k=min(top_k, 3))
            if birads_docs:
                logger.info(f"BI-RADS general query detected, found {len(birads_docs)} documents")

                elapsed_ms = int((time.time() - start_time) * 1000)

                parsed_query = SearchQuery(
                    original_query=query,
                    intent="guideline_lookup",
                    keywords=[],
                )

                results = [
                    SearchResult(
                        paper=doc,
                        score=1.0 - (i * 0.1),  # 순서대로 점수 부여
                        rank=i + 1,
                        matched_terms=["BI-RADS", "Guideline"],
                    )
                    for i, doc in enumerate(birads_docs)
                ]

                return SearchResponse(
                    query=parsed_query,
                    results=results,
                    total_count=len(results),
                    time_ms=elapsed_ms,
                )

        # 1. 쿼리 파싱
        parsed_query = self.parser.parse(query)
        logger.info(f"Query parsed: {parsed_query.intent}")

        # 2. 하이브리드 검색
        retrieval_k = top_k * 5 if use_rerank and self.reranker else top_k
        retrieved = self.retriever.retrieve(parsed_query, top_k=retrieval_k)

        if not retrieved:
            return SearchResponse(
                query=parsed_query,
                results=[],
                total_count=0,
                time_ms=int((time.time() - start_time) * 1000),
            )

        # PMID로 논문 조회
        pmids = [pmid for pmid, _ in retrieved]
        papers = self.db.get_papers(pmids)

        # PMID -> Paper 매핑
        paper_dict = {p.pmid: p for p in papers}

        # 3. 리랭킹
        if use_rerank and self.reranker and papers:
            # BI-RADS 보호를 위해 전체 결과 리랭킹
            reranked = self.reranker.rerank(query, papers, top_k=len(papers))
            reranked = self.reranker.normalize_scores(reranked)

            # 가이드라인 쿼리일 때 BI-RADS 문서 보호
            reranked = self._protect_guidelines(reranked, parsed_query)

            # 최종 top_k로 자르기
            reranked = reranked[:top_k]

            results = [
                SearchResult(
                    paper=paper,
                    score=score,
                    rank=i + 1,
                    matched_terms=parsed_query.keywords[:5],
                )
                for i, (paper, score) in enumerate(reranked)
            ]
        else:
            # 리랭킹 없이 결과 생성
            results = []
            max_score = max(score for _, score in retrieved) if retrieved else 1

            for i, (pmid, score) in enumerate(retrieved[:top_k]):
                if pmid in paper_dict:
                    results.append(
                        SearchResult(
                            paper=paper_dict[pmid],
                            score=score / max_score,  # 정규화
                            rank=i + 1,
                            matched_terms=parsed_query.keywords[:5],
                        )
                    )

        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.info(f"Search completed: {len(results)} results in {elapsed_ms}ms")

        return SearchResponse(
            query=parsed_query,
            results=results,
            total_count=len(results),
            time_ms=elapsed_ms,
        )

    def search_simple(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Paper]:
        """
        간단한 검색 (Paper 리스트 반환)

        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수

        Returns:
            Paper 리스트
        """
        response = self.search(query, top_k)
        return [r.paper for r in response.results]

    def format_results(self, results: List[SearchResult]) -> str:
        """
        CLI용 결과 포맷팅

        Args:
            results: 검색 결과 리스트

        Returns:
            포맷된 문자열
        """
        if not results:
            return "No results found."

        lines = []

        for result in results:
            paper = result.paper
            score_pct = int(result.score * 100)

            lines.append(f"\n{'='*70}")
            lines.append(f"[{result.rank}] Score: {score_pct}%")
            lines.append(f"{'='*70}")
            lines.append(f"Title: {paper.title}")
            lines.append(f"Authors: {paper.author_string}")
            lines.append(f"Journal: {paper.journal} ({paper.year})")

            if paper.journal_if:
                lines.append(f"Impact Factor: {paper.journal_if}")
            if paper.citation_count:
                lines.append(f"Citations: {paper.citation_count}")

            # 분류 태그
            tags = []
            if paper.modality:
                tags.extend(paper.modality)
            if paper.pathology:
                tags.extend(paper.pathology)
            if paper.study_type:
                tags.append(paper.study_type)
            if tags:
                lines.append(f"Tags: [{'] ['.join(tags)}]")

            # 초록 (처음 300자)
            if paper.abstract:
                abstract_preview = paper.abstract[:300]
                if len(paper.abstract) > 300:
                    abstract_preview += "..."
                lines.append(f"\nAbstract: {abstract_preview}")

            # 링크
            lines.append(f"\nPubMed: {paper.pubmed_url}")
            if paper.doi_url:
                lines.append(f"DOI: {paper.doi_url}")

        return "\n".join(lines)

    def get_stats(self) -> dict:
        """데이터베이스 통계"""
        return self.db.get_stats()


def main():
    """CLI 검색"""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Search mammography papers")
    parser.add_argument("query", nargs="?", help="Search query")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results")
    parser.add_argument("--no-rerank", action="store_true", help="Disable reranking")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--index-dir", type=str, default="data/index", help="Index directory")
    parser.add_argument(
        "--parser",
        type=str,
        choices=["rule", "llm", "smart"],
        default="smart",
        help="Query parser mode (rule: rule-based, llm: LLM-based, smart: auto-select)",
    )
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434", help="Ollama server URL")
    parser.add_argument("--llm-model", type=str, default="llama3.2", help="LLM model name")

    args = parser.parse_args()

    # 검색 엔진 초기화
    engine = SearchEngine(
        db_path=Path(args.index_dir),
        parser_mode=args.parser,
        ollama_url=args.ollama_url,
        llm_model=args.llm_model,
    )

    if args.interactive:
        print("\n" + "="*70)
        print("Sophia AI Interactive Search")
        print("Type 'quit' or 'exit' to quit")
        print("="*70)

        while True:
            try:
                query = input("\nSearch: ").strip()
                if query.lower() in ("quit", "exit", "q"):
                    break
                if not query:
                    continue

                response = engine.search(query, top_k=args.top_k, use_rerank=not args.no_rerank)
                print(engine.format_results(response.results))
                print(f"\n[{response.total_count} results in {response.time_ms}ms]")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")

        print("\nGoodbye!")

    elif args.query:
        response = engine.search(args.query, top_k=args.top_k, use_rerank=not args.no_rerank)
        print(engine.format_results(response.results))
        print(f"\n[{response.total_count} results in {response.time_ms}ms]")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
