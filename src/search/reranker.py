"""
Sophia AI: Reranker
=====================
Cross-encoder 기반 리랭킹
"""

import logging
from typing import List, Tuple

from sentence_transformers import CrossEncoder

from src.models import Paper

logger = logging.getLogger(__name__)

# 기본 리랭커 모델
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    """Cross-encoder 리랭커"""

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL):
        """
        Args:
            model_name: Cross-encoder 모델 이름
        """
        logger.info(f"Loading reranker: {model_name}")
        self.model = CrossEncoder(model_name)
        logger.info("Reranker loaded")

    def _prepare_text(self, paper: Paper) -> str:
        """리랭킹용 텍스트 준비"""
        text = f"{paper.title}\n\n{paper.abstract}"
        # 최대 512 토큰 정도로 제한
        if len(text) > 1000:
            text = text[:1000]
        return text

    def rerank(
        self,
        query: str,
        papers: List[Paper],
        top_k: int = 20,
    ) -> List[Tuple[Paper, float]]:
        """
        논문 리랭킹

        Args:
            query: 검색 쿼리
            papers: 논문 리스트
            top_k: 반환할 상위 개수

        Returns:
            (Paper, score) 튜플 리스트
        """
        if not papers:
            return []

        # 쿼리-문서 쌍 생성
        pairs = [(query, self._prepare_text(paper)) for paper in papers]

        # 점수 계산
        scores = self.model.predict(pairs, show_progress_bar=False)

        # Paper와 점수 결합
        paper_scores = list(zip(papers, scores))

        # 점수 기준 정렬
        paper_scores.sort(key=lambda x: x[1], reverse=True)

        logger.debug(f"Reranked {len(papers)} papers, returning top {top_k}")

        return paper_scores[:top_k]

    def rerank_with_pmids(
        self,
        query: str,
        papers: List[Paper],
        top_k: int = 20,
    ) -> List[Tuple[str, float]]:
        """
        리랭킹 후 PMID와 점수 반환

        Args:
            query: 검색 쿼리
            papers: 논문 리스트
            top_k: 반환할 상위 개수

        Returns:
            (pmid, score) 튜플 리스트
        """
        reranked = self.rerank(query, papers, top_k)
        return [(paper.pmid, float(score)) for paper, score in reranked]

    def normalize_scores(
        self,
        results: List[Tuple[Paper, float]],
    ) -> List[Tuple[Paper, float]]:
        """
        점수를 0-1 범위로 정규화

        Args:
            results: (Paper, score) 리스트

        Returns:
            정규화된 (Paper, score) 리스트
        """
        if not results:
            return results

        scores = [score for _, score in results]
        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            # 모든 점수가 같으면 1.0
            return [(paper, 1.0) for paper, _ in results]

        normalized = [
            (paper, (score - min_score) / (max_score - min_score))
            for paper, score in results
        ]

        return normalized
