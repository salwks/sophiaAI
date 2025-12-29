"""
Sophia AI: Paper Embedder
===========================
의료 도메인 특화 임베딩 생성
"""

import gc
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.models import Paper

logger = logging.getLogger(__name__)

# 기본 모델 (의료 도메인 + 검색 특화)
DEFAULT_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"

# 대안 모델들
ALTERNATIVE_MODELS = {
    "pubmedbert": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
    "bge": "BAAI/bge-large-en-v1.5",
    "minilm": "all-MiniLM-L6-v2",
}


class PaperEmbedder:
    """논문 임베딩 생성기"""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ):
        """
        Args:
            model_name: 임베딩 모델 이름
            device: 디바이스 (cuda, mps, cpu)
            cache_dir: 임베딩 캐시 디렉토리
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else None

        # 디바이스 결정
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        logger.info(f"Loading embedding model: {model_name}")
        logger.info(f"Device: {self.device}")

        self.model = SentenceTransformer(model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        logger.info(f"Embedding dimension: {self.embedding_dim}")

    def _prepare_text(self, paper: Paper) -> str:
        """임베딩용 텍스트 준비"""
        # BI-RADS 문서는 full_content 사용, 일반 논문은 제목 + 초록
        if hasattr(paper, 'full_content') and paper.full_content:
            # BI-RADS 가이드라인: full_content 사용
            text = f"{paper.title}\n\n{paper.full_content}"
        else:
            # 일반 논문: 제목 + 초록
            text = f"{paper.title}\n\n{paper.abstract}"

        # 최대 길이 제한 (모델별로 다름, 보통 512 토큰)
        # 대략 1500자 정도로 제한
        if len(text) > 1500:
            text = text[:1500]

        return text

    def embed_paper(self, paper: Paper) -> np.ndarray:
        """단일 논문 임베딩"""
        text = self._prepare_text(paper)
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding

    def embed_query(self, query: str) -> np.ndarray:
        """검색 쿼리 임베딩"""
        embedding = self.model.encode(query, normalize_embeddings=True)
        return embedding

    def embed_batch(
        self,
        papers: List[Paper],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> List[np.ndarray]:
        """
        배치 임베딩 생성

        Args:
            papers: 논문 리스트
            batch_size: 배치 크기
            show_progress: 진행률 표시

        Returns:
            임베딩 리스트
        """
        texts = [self._prepare_text(paper) for paper in papers]

        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size

        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, total=total_batches, desc="Embedding")

        for i in iterator:
            batch_texts = texts[i : i + batch_size]

            try:
                batch_embeddings = self.model.encode(
                    batch_texts,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                all_embeddings.extend(batch_embeddings)

            except Exception as e:
                logger.error(f"Batch embedding failed at {i}: {e}")
                # 개별 처리 폴백
                for text in batch_texts:
                    try:
                        emb = self.model.encode(text, normalize_embeddings=True)
                        all_embeddings.append(emb)
                    except:
                        # 빈 임베딩
                        all_embeddings.append(np.zeros(self.embedding_dim))

            # 메모리 정리
            if (i // batch_size) % 10 == 0:
                gc.collect()
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                elif self.device == "mps":
                    torch.mps.empty_cache()

        logger.info(f"Generated {len(all_embeddings)} embeddings")
        return all_embeddings

    def compute_similarity(
        self,
        query_embedding: np.ndarray,
        doc_embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        코사인 유사도 계산

        Args:
            query_embedding: 쿼리 임베딩 (1D)
            doc_embeddings: 문서 임베딩들 (2D)

        Returns:
            유사도 배열
        """
        # 정규화 (이미 되어 있으면 스킵)
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)

        # 내적 = 코사인 유사도 (정규화된 경우)
        similarities = np.dot(doc_embeddings, query_norm)

        return similarities


def main():
    """테스트 실행"""
    logging.basicConfig(level=logging.INFO)

    embedder = PaperEmbedder()

    # 테스트 논문
    test_paper = Paper(
        pmid="12345678",
        title="Digital Breast Tomosynthesis vs Full-Field Digital Mammography",
        abstract="Purpose: To compare the diagnostic performance of DBT and FFDM...",
        journal="Radiology",
        year=2024,
        authors=["Kim JH", "Lee SY"],
    )

    # 임베딩 생성
    embedding = embedder.embed_paper(test_paper)
    print(f"\nEmbedding shape: {embedding.shape}")
    print(f"Embedding sample: {embedding[:5]}")

    # 쿼리 임베딩
    query_emb = embedder.embed_query("DBT screening performance")
    print(f"\nQuery embedding shape: {query_emb.shape}")

    # 유사도
    similarity = np.dot(embedding, query_emb)
    print(f"Similarity: {similarity:.4f}")


if __name__ == "__main__":
    main()
