"""
MARIA-Mammo: Hybrid Retriever
=============================
BM25 + Vector 하이브리드 검색
"""

import logging
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from src.indexing.database import DatabaseManager
from src.indexing.embedder import PaperEmbedder
from src.models import Paper, SearchQuery

logger = logging.getLogger(__name__)


class HybridRetriever:
    """BM25 + Vector 하이브리드 검색기"""

    def __init__(
        self,
        db: DatabaseManager,
        embedder: PaperEmbedder,
        bm25_weight: float = 0.4,
        vector_weight: float = 0.6,
    ):
        """
        Args:
            db: 데이터베이스 매니저
            embedder: 임베딩 생성기
            bm25_weight: BM25 가중치
            vector_weight: Vector 검색 가중치
        """
        self.db = db
        self.embedder = embedder
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight

        # BM25 인덱스 (lazy loading)
        self._bm25_index: Optional[BM25Okapi] = None
        self._bm25_pmids: List[str] = []
        self._bm25_corpus: List[List[str]] = []

    def _tokenize(self, text: str) -> List[str]:
        """BM25용 토큰화"""
        text = text.lower()
        tokens = re.findall(r'\b[a-z0-9]+\b', text)
        return tokens

    def _build_bm25_index(self, papers: List[Paper]):
        """BM25 인덱스 구축"""
        logger.info(f"Building BM25 index for {len(papers)} papers...")

        self._bm25_pmids = []
        self._bm25_corpus = []

        for paper in papers:
            text = f"{paper.title} {paper.abstract}"
            tokens = self._tokenize(text)

            self._bm25_pmids.append(paper.pmid)
            self._bm25_corpus.append(tokens)

        self._bm25_index = BM25Okapi(self._bm25_corpus)
        logger.info("BM25 index built")

    def _ensure_bm25_index(self, candidate_pmids: Optional[List[str]] = None):
        """BM25 인덱스 확인 및 구축"""
        if candidate_pmids:
            # 특정 후보만으로 임시 인덱스 구축
            papers = self.db.get_papers(candidate_pmids)
            self._build_bm25_index(papers)
        elif self._bm25_index is None:
            # 전체 인덱스 구축
            all_pmids = self.db.get_all_pmids()
            papers = self.db.get_papers(all_pmids)
            self._build_bm25_index(papers)

    def search_bm25(
        self,
        keywords: List[str],
        k: int = 100,
        candidate_pmids: Optional[List[str]] = None,
    ) -> List[Tuple[str, float]]:
        """
        BM25 검색

        Args:
            keywords: 검색 키워드
            k: 반환할 결과 수
            candidate_pmids: 검색 대상 PMID (None이면 전체)

        Returns:
            (pmid, score) 튜플 리스트
        """
        if not keywords:
            return []

        self._ensure_bm25_index(candidate_pmids)

        if self._bm25_index is None:
            return []

        # 쿼리 토큰화
        query_tokens = []
        for kw in keywords:
            query_tokens.extend(self._tokenize(kw))

        if not query_tokens:
            return []

        # BM25 스코어 계산
        scores = self._bm25_index.get_scores(query_tokens)

        # 상위 k개 선택
        top_indices = np.argsort(scores)[-k:][::-1]
        results = [
            (self._bm25_pmids[i], float(scores[i]))
            for i in top_indices
            if scores[i] > 0
        ]

        return results[:k]

    def search_vector(
        self,
        query: str,
        k: int = 100,
        candidate_pmids: Optional[List[str]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Vector 검색

        Args:
            query: 검색 쿼리
            k: 반환할 결과 수
            candidate_pmids: 검색 대상 PMID (None이면 전체)

        Returns:
            (pmid, score) 튜플 리스트
        """
        # 쿼리 임베딩
        query_embedding = self.embedder.embed_query(query)

        # LanceDB 검색
        results = self.db.search_vector(
            query_embedding,
            k=k,
            pmid_filter=candidate_pmids,
        )

        return results

    def weighted_rrf(
        self,
        bm25_results: List[Tuple[str, float]],
        vector_results: List[Tuple[str, float]],
        k: int = 60,
        bm25_weight: Optional[float] = None,
        vector_weight: Optional[float] = None,
    ) -> List[Tuple[str, float]]:
        """
        가중치 적용 Reciprocal Rank Fusion

        Args:
            bm25_results: BM25 검색 결과
            vector_results: Vector 검색 결과
            k: RRF 상수 (기본 60)
            bm25_weight: BM25 가중치 (기본값: self.bm25_weight)
            vector_weight: Vector 가중치 (기본값: self.vector_weight)

        Returns:
            융합된 (pmid, score) 튜플 리스트
        """
        bm25_w = bm25_weight if bm25_weight is not None else self.bm25_weight
        vector_w = vector_weight if vector_weight is not None else self.vector_weight

        fusion_scores: Dict[str, float] = defaultdict(float)

        # BM25 결과에 가중치 적용
        for rank, (pmid, _) in enumerate(bm25_results, 1):
            fusion_scores[pmid] += bm25_w * (1.0 / (k + rank))

        # Vector 결과에 가중치 적용
        for rank, (pmid, _) in enumerate(vector_results, 1):
            fusion_scores[pmid] += vector_w * (1.0 / (k + rank))

        # 정렬
        sorted_results = sorted(
            fusion_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return sorted_results

    def reciprocal_rank_fusion(
        self,
        results_list: List[List[Tuple[str, float]]],
        k: int = 60,
    ) -> List[Tuple[str, float]]:
        """
        Reciprocal Rank Fusion (레거시 호환용)

        Args:
            results_list: 여러 검색 결과 리스트
            k: RRF 상수 (기본 60)

        Returns:
            융합된 (pmid, score) 튜플 리스트
        """
        fusion_scores: Dict[str, float] = defaultdict(float)

        for results in results_list:
            for rank, (pmid, _) in enumerate(results, 1):
                fusion_scores[pmid] += 1 / (k + rank)

        # 정렬
        sorted_results = sorted(
            fusion_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return sorted_results

    def _inject_guidelines(
        self,
        results: List[Tuple[str, float]],
        query: SearchQuery,
        top_k: int,
    ) -> List[Tuple[str, float]]:
        """
        가이드라인 의도 감지 시 BI-RADS 문서를 상위에 강제 주입

        Args:
            results: 기존 검색 결과
            query: 파싱된 쿼리
            top_k: 반환할 결과 수

        Returns:
            BI-RADS 문서가 주입된 결과
        """
        # 가이드라인 의도 확인
        intent_lower = query.intent.lower() if query.intent else ""
        keywords_lower = [k.lower() for k in query.keywords]
        original_lower = query.original_query.lower()

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
            return results

        logger.info("Guideline query detected, injecting BI-RADS documents")

        # BI-RADS 문서 별도 검색 (Vector만 사용, 필터 없이)
        # 영어 키워드로 검색 (더 정확한 매칭)
        search_query = " ".join(query.keywords) if query.keywords else query.original_query
        birads_results = self.search_vector(
            search_query,
            k=100,  # 충분히 많이
            candidate_pmids=None,  # 필터 없이 전체 검색
        )

        # BI-RADS 문서만 필터링 (중복 제거)
        seen_pmids = set()
        birads_docs = []
        for pmid, score in birads_results:
            if pmid.startswith("BIRADS_") and pmid not in seen_pmids:
                seen_pmids.add(pmid)
                birads_docs.append((pmid, score))
                if len(birads_docs) >= 3:  # 상위 3개만
                    break

        if not birads_docs:
            logger.debug("No BI-RADS documents found in vector search")
            return results

        logger.info(f"Injecting {len(birads_docs)} BI-RADS documents: {[p for p, _ in birads_docs]}")

        # 기존 결과에서 BI-RADS 문서 제거 (중복 방지) + 중복 제거
        birads_pmids = {pmid for pmid, _ in birads_docs}
        seen_in_results = set()
        filtered_results = []
        for pmid, score in results:
            if pmid not in birads_pmids and pmid not in seen_in_results:
                seen_in_results.add(pmid)
                filtered_results.append((pmid, score))

        # 최고 점수 기준으로 BI-RADS 문서 점수 조정
        if filtered_results:
            max_score = filtered_results[0][1]
            # BI-RADS 문서를 최고 점수 * 1.2로 설정 (확실히 상위에 오도록)
            adjusted_birads = [
                (pmid, max_score * (1.2 - 0.05 * i))  # 1위: 1.2x, 2위: 1.15x, 3위: 1.1x
                for i, (pmid, _) in enumerate(birads_docs)
            ]
        else:
            adjusted_birads = [
                (pmid, 1.0 - 0.1 * i)
                for i, (pmid, _) in enumerate(birads_docs)
            ]

        # BI-RADS 먼저, 그 다음 기존 결과
        combined = adjusted_birads + filtered_results

        # 점수로 정렬
        combined.sort(key=lambda x: x[1], reverse=True)

        return combined

    def _demote_case_reports(
        self,
        results: List[Tuple[str, float]],
        query: SearchQuery,
        demote_factor: float = 0.5,
        max_check: int = 100,
    ) -> List[Tuple[str, float]]:
        """
        케이스 리포트를 하위로 다운랭킹

        LLM이 recommended_exclusions에 "Case Reports"를 포함시킨 경우 또는
        연구 관련 의도가 감지된 경우 동작

        Args:
            results: 검색 결과
            query: 파싱된 쿼리 (recommended_exclusions 포함)
            demote_factor: 점수 감소 비율 (0.5 = 50% 감소)
            max_check: 성능을 위해 상위 N개만 체크

        Returns:
            다운랭킹된 결과
        """
        # recommended_exclusions 확인
        exclusions = [e.lower() for e in query.recommended_exclusions] if query.recommended_exclusions else []

        should_demote_cases = any(
            "case report" in exc or "case reports" in exc
            for exc in exclusions
        )

        # LLM이 recommended_exclusions를 출력하지 않은 경우 의도 기반 폴백
        if not should_demote_cases:
            intent_lower = query.intent.lower() if query.intent else ""
            keywords_lower = [k.lower() for k in query.keywords] if query.keywords else []

            # 연구 관련 의도 패턴 (케이스 리포트 제외 대상)
            research_intent_patterns = [
                "comparison", "performance", "accuracy", "sensitivity", "specificity",
                "outcome", "prediction", "correlation", "risk", "malignancy rate",
                "ppv", "positive predictive value", "diagnostic",
            ]

            # 가이드라인/정의 패턴 (케이스 리포트 제외 안함)
            guideline_patterns = ["guideline", "definition", "criteria", "classification"]

            is_guideline = any(p in intent_lower for p in guideline_patterns)
            is_research = any(p in intent_lower or p in " ".join(keywords_lower) for p in research_intent_patterns)

            # 가이드라인이 아니고 연구 의도면 케이스 리포트 다운랭킹
            should_demote_cases = is_research and not is_guideline

        if not should_demote_cases:
            return results

        logger.info("Demoting case reports based on recommended_exclusions")

        # 케이스 리포트 publication_type 패턴
        case_report_patterns = ["case report", "case study", "letter", "comment", "editorial"]

        # 성능을 위해 상위 max_check 개만 체크
        results_to_check = results[:max_check]
        remaining = results[max_check:]

        # PMID 목록으로 논문 조회 (배치 조회)
        pmids = [pmid for pmid, _ in results_to_check]
        papers = self.db.get_papers(pmids)
        paper_dict = {p.pmid: p for p in papers}

        demoted = []
        for pmid, score in results_to_check:
            paper = paper_dict.get(pmid)
            if paper and paper.publication_types:
                pub_types_lower = [pt.lower() for pt in paper.publication_types]
                is_case_report = any(
                    pattern in pt
                    for pt in pub_types_lower
                    for pattern in case_report_patterns
                )
                if is_case_report:
                    demoted.append((pmid, score * demote_factor))
                    logger.debug(f"Demoted case report: {pmid}")
                else:
                    demoted.append((pmid, score))
            else:
                demoted.append((pmid, score))

        # 재정렬 후 나머지 추가
        demoted.sort(key=lambda x: x[1], reverse=True)

        return demoted + remaining

    def retrieve(
        self,
        query: SearchQuery,
        top_k: int = 50,
    ) -> List[Tuple[str, float]]:
        """
        하이브리드 검색

        Args:
            query: 파싱된 검색 쿼리
            top_k: 반환할 결과 수

        Returns:
            (pmid, score) 튜플 리스트
        """
        # 1. 메타데이터 필터로 후보 제한
        candidate_pmids = None
        if query.has_filters:
            candidate_pmids = self.db.search_metadata(query.filters)
            logger.info(f"Metadata filter: {len(candidate_pmids)} candidates")

            if not candidate_pmids:
                # 필터 결과가 없으면 전체 검색으로 폴백
                logger.warning("No papers match the filters, falling back to full search")
                candidate_pmids = None

        # 2. BM25 검색 - 파싱된 영어 키워드 사용
        # LLM이 한국어를 영어로 변환했으므로 영어 키워드로 BM25 검색
        bm25_keywords = query.keywords if query.keywords else []

        # MeSH 용어도 BM25에 포함
        if query.mesh_terms:
            bm25_keywords = bm25_keywords + query.mesh_terms

        bm25_results = self.search_bm25(
            bm25_keywords,
            k=top_k * 2,
            candidate_pmids=candidate_pmids,
        )
        logger.debug(f"BM25 results: {len(bm25_results)} (keywords: {bm25_keywords[:5]})")

        # 3. Vector 검색 - 원본 쿼리 사용 (다국어 임베딩 지원)
        vector_results = self.search_vector(
            query.original_query,
            k=top_k * 2,
            candidate_pmids=candidate_pmids,
        )
        logger.debug(f"Vector results: {len(vector_results)}")

        # 4. 가중치 적용 RRF 융합 (Vector 우선)
        fused_results = self.weighted_rrf(
            bm25_results,
            vector_results,
        )

        # 5. 가이드라인 쿼리일 때 BI-RADS 문서 강제 주입
        fused_results = self._inject_guidelines(fused_results, query, top_k)

        # 6. 케이스 리포트 다운랭킹 (recommended_exclusions 기반)
        fused_results = self._demote_case_reports(fused_results, query)

        logger.info(f"Retrieved {len(fused_results[:top_k])} papers")
        return fused_results[:top_k]

    def retrieve_simple(
        self,
        query_text: str,
        top_k: int = 50,
    ) -> List[Tuple[str, float]]:
        """
        간단한 텍스트 검색 (파싱 없이)

        Args:
            query_text: 검색 텍스트
            top_k: 반환할 결과 수

        Returns:
            (pmid, score) 튜플 리스트
        """
        # 간단한 키워드 추출
        keywords = re.findall(r'\b[a-zA-Z0-9-]+\b', query_text.lower())

        # BM25 검색
        bm25_results = self.search_bm25(keywords, k=top_k * 2)

        # Vector 검색
        vector_results = self.search_vector(query_text, k=top_k * 2)

        # 가중치 적용 RRF 융합
        fused_results = self.weighted_rrf(
            bm25_results,
            vector_results,
        )

        return fused_results[:top_k]
