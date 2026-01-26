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
from src.search.query_parser import QueryParser, LLMQueryParser, SmartQueryParser, EnhancedQueryParser
from src.search.reranker import Reranker
from src.search.retriever import HybridRetriever
from src.search.weighting_logic import WeightedMedicalReranker, WMRConfig

logger = logging.getLogger(__name__)


class SearchEngine:
    """통합 검색 엔진"""

    def __init__(
        self,
        db: Optional[DatabaseManager] = None,
        embedder: Optional[PaperEmbedder] = None,
        db_path: Path = Path("data/index"),
        use_reranker: bool = True,  # Skip reranking for BI-RADS guideline queries
        parser_mode: Literal["rule", "llm", "smart", "enhanced"] = "enhanced",
        ollama_url: str = "http://localhost:11434",
        llm_model: str = "llama3.2",
        cot_model: str = "glm4:9b",  # Phase 7.7: CoT용 경량 모델
    ):
        """
        Args:
            db: 데이터베이스 매니저
            embedder: 임베딩 생성기
            db_path: 데이터베이스 경로
            use_reranker: 리랭커 사용 여부
            parser_mode: 파서 모드 (rule: 규칙 기반, llm: LLM 기반, smart: 자동 선택, enhanced: CoT 확장)
            ollama_url: Ollama 서버 URL
            llm_model: LLM 모델 이름
            cot_model: CoT 확장용 경량 모델 이름
        """
        logger.info("Initializing search engine...")

        # 컴포넌트 초기화
        self.db = db or DatabaseManager(db_path)
        self.embedder = embedder or PaperEmbedder()

        # 파서 선택
        self.parser_mode = parser_mode
        if parser_mode == "enhanced":
            # Phase 1: Medical-Logic-CoT 확장 파서 (권장)
            self.parser = EnhancedQueryParser(
                ollama_url=ollama_url,
                llm_model=llm_model,
                cot_model=cot_model,
                enable_cot=True,
            )
            if hasattr(self.parser, 'using_cot') and self.parser.using_cot:
                logger.info("Using Enhanced parser with Medical-Logic-CoT (DeepSeek-R1)")
            else:
                logger.info("Using Enhanced parser (CoT unavailable, LLM fallback)")
        elif parser_mode == "llm":
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

        # 가중치 조정: BM25 우선 (문서 길이 정규화 강화)
        self.retriever = HybridRetriever(
            self.db,
            self.embedder,
            bm25_weight=0.6,  # 0.4 → 0.6
            vector_weight=0.4,  # 0.6 → 0.4
        )

        # Phase 3: Weighted Medical Reranker (WMR) 사용
        self.use_reranker = use_reranker
        self.use_wmr = True  # WMR 활성화

        if use_reranker:
            if self.use_wmr:
                # WMR: Cross-Encoder + 의학적 가중치
                self.reranker = WeightedMedicalReranker(
                    config=WMRConfig(),
                    use_cross_encoder=True,
                )
                logger.info("Using Weighted Medical Reranker (WMR)")
            else:
                # 기존 Cross-Encoder only
                self.reranker = Reranker()
                logger.info("Using standard Cross-Encoder reranker")
        else:
            self.reranker = None

        self.birads_matcher = BiradsCategoryMatcher(db_path)

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

        # 가이드라인 문서와 일반 문서 분리 (모든 가이드라인 접두사 포함)
        GUIDELINE_PREFIXES = ("ACR_", "BIRADS_", "PHYSICS_", "CLINICAL_", "DANCE_")
        guideline_docs = []
        other_docs = []

        for paper, score in reranked:
            if paper.pmid.startswith(GUIDELINE_PREFIXES):
                guideline_docs.append((paper, score))
            else:
                other_docs.append((paper, score))

        if not guideline_docs:
            return reranked

        logger.info(f"Protecting {len(guideline_docs)} guideline documents after reranking")

        # 가이드라인 문서 점수를 최고 점수 * 1.2로 조정
        if other_docs:
            max_score = other_docs[0][1]
            adjusted_guidelines = [
                (paper, max_score * (1.2 - 0.05 * i))
                for i, (paper, _) in enumerate(guideline_docs[:3])
            ]
        else:
            adjusted_guidelines = guideline_docs[:3]

        # 가이드라인 먼저, 그 다음 일반 문서
        combined = adjusted_guidelines + other_docs

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

        # 가이드라인 쿼리 감지
        intent_lower = parsed_query.intent.lower() if parsed_query.intent else ""
        keywords_lower = [k.lower() for k in parsed_query.keywords]
        original_lower = parsed_query.original_query.lower()

        is_guideline_query = (
            "guideline" in intent_lower or
            "bi-rads" in keywords_lower or
            "bi-rads" in original_lower or
            "기준" in original_lower or
            "가이드라인" in original_lower or
            "분류" in original_lower or  # classification
            "정의" in original_lower
        )

        # 3. 리랭킹 (Phase 3: WMR 적용)
        if use_rerank and self.reranker and papers:
            if self.use_wmr and isinstance(self.reranker, WeightedMedicalReranker):
                # WMR: Cross-Encoder + 의학적 가중치
                # intent를 전달하여 쿼리 의도 기반 가중치 적용
                reranked_with_breakdown = self.reranker.rerank(
                    query,
                    papers,
                    intent=parsed_query.intent,
                    top_k=top_k,
                )
                reranked_with_breakdown = self.reranker.normalize_scores(reranked_with_breakdown)

                results = [
                    SearchResult(
                        paper=paper,
                        score=score,
                        rank=i + 1,
                        matched_terms=parsed_query.keywords[:5],
                    )
                    for i, (paper, score, _breakdown) in enumerate(reranked_with_breakdown)
                ]

                logger.info(f"WMR reranking applied with intent: {parsed_query.intent[:50]}")

            elif not is_guideline_query:
                # 기존 Cross-Encoder (가이드라인 쿼리는 건너뛰기)
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
                # 가이드라인 쿼리 + 기존 리랭커 = 리랭킹 스킵
                results = []
                max_score = max(score for _, score in retrieved) if retrieved else 1

                for i, (pmid, score) in enumerate(retrieved[:top_k]):
                    if pmid in paper_dict:
                        results.append(
                            SearchResult(
                                paper=paper_dict[pmid],
                                score=score / max_score,
                                rank=i + 1,
                                matched_terms=parsed_query.keywords[:5],
                            )
                        )
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

    def search_dual(
        self,
        query: str,
        birads_k: int = 3,
        papers_k: int = 5,
    ) -> tuple[SearchResponse, SearchResponse]:
        """
        이중 검색: 가이드라인(BIRADS, ACR, PHYSICS, CLINICAL) + 연구 논문 분리

        벡터 검색을 메인으로 사용하고, 소스 다양성을 보장함.

        Args:
            query: 검색 쿼리
            birads_k: 가이드라인 결과 수
            papers_k: 논문 결과 수

        Returns:
            (가이드라인 응답, 논문 응답) 튜플
        """
        start_time = time.time()

        # 쿼리 파싱
        parsed_query = self.parser.parse(query)

        GUIDELINE_PREFIXES = ("ACR_", "BIRADS_", "PHYSICS_", "CLINICAL_", "DANCE_")

        # === 1. 가이드라인 검색 (키워드 매칭 + 벡터 검색 하이브리드) ===

        # 1-1. BI-RADS 카테고리 직접 질문인지 확인 (예: "BI-RADS 5", "Category 4A")
        category_doc = None
        if self.birads_matcher.is_birads_query(query):
            category_doc = self.birads_matcher.search(query)
            if category_doc:
                logger.info(f"Direct category match: {category_doc.pmid}")

        # 1-1b. 키워드 기반 BI-RADS 섹션 매칭 (Fine Linear, Amorphous 등 정확 매칭)
        # 벡터 검색으로 놓칠 수 있는 특정 BI-RADS 용어를 키워드로 정확히 검색
        # birads_k * 2개를 가져와서 관련 섹션이 누락되지 않도록 함
        keyword_matched_docs = []
        if self.birads_matcher.is_birads_general_query(query):
            keyword_matched_docs = self.birads_matcher.search_birads_general(query, top_k=birads_k * 2)
            if keyword_matched_docs:
                logger.info(f"Keyword-matched BI-RADS docs: {[d.pmid for d in keyword_matched_docs]}")

        # 1-2. 가이드라인 전용 벡터 검색 (LanceDB 직접 쿼리)
        query_embedding = self.embedder.embed_query(query)

        import lancedb
        lance_db = lancedb.connect(str(self.db.db_path / "lancedb"))
        table = lance_db.open_table("papers")

        # 가이드라인 및 참조 문서만 대상으로 벡터 검색 (WHERE 절 사용)
        # DANCE_: Dance et al. 2011 등 물리학 참조 논문 포함
        guideline_results = (
            table.search(query_embedding)
            .where(f"pmid LIKE 'ACR_%' OR pmid LIKE 'BIRADS_%' OR pmid LIKE 'PHYSICS_%' OR pmid LIKE 'CLINICAL_%' OR pmid LIKE 'DANCE_%'", prefilter=True)
            .limit(birads_k * 5)
            .to_pandas()
        )

        guideline_candidates = [
            (row['pmid'], 1.0 - row['_distance'])  # distance → similarity
            for _, row in guideline_results.iterrows()
        ]

        # 중복 제거
        seen_pmids = set()
        unique_candidates = []
        for pmid, score in guideline_candidates:
            if pmid not in seen_pmids:
                seen_pmids.add(pmid)
                unique_candidates.append((pmid, score))
        guideline_candidates = unique_candidates

        # 1-3. 소스별로 그룹화 (다양성 보장)
        # DANCE_: Dance et al. 물리학 참조 논문 포함
        source_groups = {"BIRADS_": [], "ACR_": [], "PHYSICS_": [], "CLINICAL_": [], "DANCE_": []}
        for pmid, score in guideline_candidates:
            for prefix in source_groups.keys():
                if pmid.startswith(prefix):
                    source_groups[prefix].append((pmid, score))
                    break

        # 1-4. 각 소스에서 균형있게 선택 (키워드 매칭 우선 + 라운드 로빈)
        selected_pmids = []

        # 1) 카테고리 직접 매칭이면 최우선
        if category_doc:
            selected_pmids.append(category_doc.pmid)

        # 2) 키워드 매칭 결과 우선 포함 (Fine Linear, Amorphous 등 정확 매칭)
        for doc in keyword_matched_docs:
            if doc.pmid not in selected_pmids:
                selected_pmids.append(doc.pmid)
                logger.info(f"Priority keyword match added: {doc.pmid}")

        # 3) 벡터 검색 결과에서 라운드 로빈으로 보충
        max_rounds = birads_k * 2
        for round_idx in range(max_rounds):
            if len(selected_pmids) >= birads_k * 2:
                break
            for prefix, candidates in source_groups.items():
                if round_idx < len(candidates):
                    pmid, score = candidates[round_idx]
                    if pmid not in selected_pmids:
                        selected_pmids.append(pmid)

        # 1-5. 선택된 문서들 로드 및 Reranker 적용
        # 키워드 매칭된 핵심 문서는 Reranker와 무관하게 상위 보장
        priority_pmids = [doc.pmid for doc in keyword_matched_docs[:3]]  # 상위 3개 보장

        if selected_pmids:
            guideline_papers = self.db.get_papers(selected_pmids)
            paper_dict = {p.pmid: p for p in guideline_papers}

            if self.reranker and guideline_papers:
                # Reranker로 순위 결정 (전체 대상)
                if self.use_wmr and isinstance(self.reranker, WeightedMedicalReranker):
                    reranked = self.reranker.rerank(query, guideline_papers, top_k=len(guideline_papers))
                    reranked = self.reranker.normalize_scores(reranked)
                    reranked_papers = [(paper, score) for paper, score, _ in reranked]
                else:
                    reranked = self.reranker.rerank(query, guideline_papers, top_k=len(guideline_papers))
                    reranked = self.reranker.normalize_scores(reranked)
                    reranked_papers = reranked

                # 키워드 매칭 문서 우선 배치 + Reranker 결과로 보충
                birads_results = []
                used_pmids = set()

                # 1) 키워드 매칭 핵심 문서 상위 배치 (Fine Linear, Amorphous 등)
                for pmid in priority_pmids:
                    if pmid in paper_dict and len(birads_results) < birads_k:
                        paper = paper_dict[pmid]
                        # Reranker 점수 찾기, 없으면 높은 점수 부여
                        score = next((s for p, s in reranked_papers if p.pmid == pmid), 0.95)
                        birads_results.append(SearchResult(
                            paper=paper,
                            score=max(score, 0.9),  # 최소 0.9 보장
                            rank=len(birads_results) + 1,
                            matched_terms=parsed_query.keywords[:5],
                        ))
                        used_pmids.add(pmid)
                        logger.info(f"Priority BI-RADS section guaranteed: {pmid}")

                # 2) Reranker 결과로 나머지 슬롯 채우기
                for paper, score in reranked_papers:
                    if paper.pmid not in used_pmids and len(birads_results) < birads_k:
                        birads_results.append(SearchResult(
                            paper=paper,
                            score=score,
                            rank=len(birads_results) + 1,
                            matched_terms=parsed_query.keywords[:5],
                        ))
                        used_pmids.add(paper.pmid)
            else:
                # Reranker 없으면 벡터 검색 점수 사용
                paper_dict = {p.pmid: p for p in guideline_papers}
                pmid_to_score = {pmid: score for pmid, score in guideline_candidates}
                max_score = max(pmid_to_score.values()) if pmid_to_score else 1

                birads_results = []
                for i, pmid in enumerate(selected_pmids[:birads_k]):
                    if pmid in paper_dict:
                        score = pmid_to_score.get(pmid, 0.5) / max_score
                        birads_results.append(
                            SearchResult(
                                paper=paper_dict[pmid],
                                score=score,
                                rank=i + 1,
                                matched_terms=parsed_query.keywords[:5],
                            )
                        )
        else:
            birads_results = []

        # 2. 연구 논문 검색 (가이드라인 문서 제외, reranker 사용)
        GUIDELINE_PREFIXES = ("ACR_", "BIRADS_", "PHYSICS_", "CLINICAL_", "DANCE_")
        papers_retrieved = self.retriever.retrieve(parsed_query, top_k=papers_k * 5)
        papers_pmids = [pmid for pmid, _ in papers_retrieved if not pmid.startswith(GUIDELINE_PREFIXES)][:papers_k * 2]

        if papers_pmids:
            papers_list = self.db.get_papers(papers_pmids)

            # Reranker 사용
            if self.reranker and papers_list:
                # WMR은 3개 값 반환, 기존 Reranker는 2개 값 반환
                if self.use_wmr and isinstance(self.reranker, WeightedMedicalReranker):
                    reranked = self.reranker.rerank(query, papers_list, top_k=papers_k)
                    reranked = self.reranker.normalize_scores(reranked)
                    papers_results = [
                        SearchResult(
                            paper=paper,
                            score=score,
                            rank=i + 1,
                            matched_terms=parsed_query.keywords[:5],
                        )
                        for i, (paper, score, _breakdown) in enumerate(reranked)
                    ]
                else:
                    reranked = self.reranker.rerank(query, papers_list, top_k=papers_k)
                    reranked = self.reranker.normalize_scores(reranked)
                    papers_results = [
                        SearchResult(
                            paper=paper,
                            score=score,
                            rank=i + 1,
                            matched_terms=parsed_query.keywords[:5],
                        )
                        for i, (paper, score) in enumerate(reranked)
                    ]
            else:
                papers_results = [
                    SearchResult(
                        paper=paper,
                        score=1.0 - (i * 0.1),
                        rank=i + 1,
                        matched_terms=parsed_query.keywords[:5],
                    )
                    for i, paper in enumerate(papers_list[:papers_k])
                ]
        else:
            papers_results = []

        elapsed_ms = int((time.time() - start_time) * 1000)

        birads_response = SearchResponse(
            query=parsed_query,
            results=birads_results,
            total_count=len(birads_results),
            time_ms=elapsed_ms,
        )

        papers_response = SearchResponse(
            query=parsed_query,
            results=papers_results,
            total_count=len(papers_results),
            time_ms=elapsed_ms,
        )

        return birads_response, papers_response

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
        choices=["rule", "llm", "smart", "enhanced"],
        default="enhanced",
        help="Query parser mode (rule: rule-based, llm: LLM-based, smart: auto-select, enhanced: CoT expansion)",
    )
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434", help="Ollama server URL")
    parser.add_argument("--llm-model", type=str, default="llama3.2", help="LLM model name")
    parser.add_argument("--cot-model", type=str, default="glm4:9b", help="CoT expansion model")

    args = parser.parse_args()

    # 검색 엔진 초기화
    engine = SearchEngine(
        db_path=Path(args.index_dir),
        parser_mode=args.parser,
        ollama_url=args.ollama_url,
        llm_model=args.llm_model,
        cot_model=args.cot_model,
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
