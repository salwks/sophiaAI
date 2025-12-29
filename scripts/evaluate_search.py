#!/usr/bin/env python3
"""
Sophia AI: Search Quality Evaluation
=======================================
검색 품질 평가 스크립트

평가 지표:
- Precision@K: 상위 K개 결과 중 관련 문서 비율
- MRR (Mean Reciprocal Rank): 첫 번째 관련 문서의 순위 역수 평균
- Response Time: 검색 응답 시간
- Intent Match: 쿼리 의도 파싱 정확도
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.search.engine import SearchEngine
from src.models import SearchResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class SearchEvaluator:
    """검색 품질 평가기"""

    def __init__(
        self,
        engine: SearchEngine,
        benchmark_path: Path = Path("data/eval/benchmark_queries.json"),
    ):
        self.engine = engine
        self.benchmark = self._load_benchmark(benchmark_path)

    def _load_benchmark(self, path: Path) -> Dict:
        """벤치마크 쿼리 로드"""
        if not path.exists():
            raise FileNotFoundError(f"Benchmark file not found: {path}")

        with open(path) as f:
            return json.load(f)

    def evaluate_query(
        self,
        query_spec: Dict,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """
        단일 쿼리 평가

        Args:
            query_spec: 쿼리 스펙 (벤치마크에서)
            top_k: 평가할 결과 수

        Returns:
            평가 결과
        """
        query = query_spec["query"]
        query_id = query_spec["id"]

        # 검색 실행
        start_time = time.time()
        response = self.engine.search(query, top_k=top_k)
        elapsed_ms = (time.time() - start_time) * 1000

        # 결과 분석
        results = response.results
        parsed_query = response.query

        # Intent Match 평가
        intent_score = self._evaluate_intent(query_spec, parsed_query)

        # Modality Match 평가
        modality_score = self._evaluate_modality(query_spec, parsed_query)

        # Pathology Match 평가
        pathology_score = self._evaluate_pathology(query_spec, parsed_query)

        # Study Type Match 평가
        study_type_score = self._evaluate_study_type(query_spec, parsed_query)

        # 결과 관련성 평가 (키워드 기반 휴리스틱)
        relevance_scores = []
        for r in results:
            rel_score = self._estimate_relevance(query_spec, r.paper)
            relevance_scores.append(rel_score)

        # Precision@K 계산
        precision_at_5 = self._precision_at_k(relevance_scores, 5)
        precision_at_10 = self._precision_at_k(relevance_scores, 10)

        # MRR 계산
        mrr = self._calculate_mrr(relevance_scores)

        return {
            "query_id": query_id,
            "query": query,
            "category": query_spec.get("category", "general"),
            "num_results": len(results),
            "response_time_ms": elapsed_ms,
            "intent": parsed_query.intent,
            "intent_match_score": intent_score,
            "modality_match_score": modality_score,
            "pathology_match_score": pathology_score,
            "study_type_match_score": study_type_score,
            "precision_at_5": precision_at_5,
            "precision_at_10": precision_at_10,
            "mrr": mrr,
            "avg_relevance": sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0,
        }

    def _evaluate_intent(self, query_spec: Dict, parsed_query) -> float:
        """의도 파싱 정확도 평가"""
        expected_keywords = query_spec.get("expected_keywords", [])
        if not expected_keywords:
            return 1.0

        parsed_text = f"{parsed_query.intent} {' '.join(parsed_query.keywords)}"
        parsed_lower = parsed_text.lower()

        matches = sum(1 for kw in expected_keywords if kw.lower() in parsed_lower)
        return matches / len(expected_keywords)

    def _evaluate_modality(self, query_spec: Dict, parsed_query) -> float:
        """모달리티 추출 정확도 평가"""
        expected = set(query_spec.get("expected_modalities", []))
        if not expected:
            return 1.0

        filters = parsed_query.filters
        parsed = set(filters.modality or [])

        if not expected:
            return 1.0

        intersection = expected & parsed
        return len(intersection) / len(expected)

    def _evaluate_pathology(self, query_spec: Dict, parsed_query) -> float:
        """병변 유형 추출 정확도 평가"""
        expected = set(query_spec.get("expected_pathologies", []))
        if not expected:
            return 1.0

        filters = parsed_query.filters
        parsed = set(filters.pathology or [])

        intersection = expected & parsed
        return len(intersection) / len(expected)

    def _evaluate_study_type(self, query_spec: Dict, parsed_query) -> float:
        """연구 유형 추출 정확도 평가"""
        expected = query_spec.get("expected_study_type")
        if not expected:
            return 1.0

        filters = parsed_query.filters
        parsed = filters.study_type

        return 1.0 if parsed == expected else 0.0

    def _estimate_relevance(self, query_spec: Dict, paper) -> float:
        """
        결과 관련성 추정 (휴리스틱)

        실제 평가에서는 수동 레이블링이 필요하지만,
        여기서는 키워드 매칭 기반으로 추정
        """
        expected_keywords = query_spec.get("expected_keywords", [])
        if not expected_keywords:
            return 1.0

        # 논문 텍스트 결합
        paper_text = f"{paper.title} {paper.abstract}".lower()

        # 키워드 매칭
        matches = sum(1 for kw in expected_keywords if kw.lower() in paper_text)
        base_score = matches / len(expected_keywords)

        # 모달리티 보너스
        expected_modalities = set(query_spec.get("expected_modalities", []))
        if expected_modalities and paper.modality:
            if set(paper.modality) & expected_modalities:
                base_score += 0.1

        # 병변 유형 보너스
        expected_pathologies = set(query_spec.get("expected_pathologies", []))
        if expected_pathologies and paper.pathology:
            if set(paper.pathology) & expected_pathologies:
                base_score += 0.1

        return min(base_score, 1.0)

    def _precision_at_k(self, relevance_scores: List[float], k: int, threshold: float = 0.5) -> float:
        """Precision@K 계산"""
        if not relevance_scores:
            return 0.0

        top_k_scores = relevance_scores[:k]
        relevant_count = sum(1 for s in top_k_scores if s >= threshold)
        return relevant_count / min(k, len(top_k_scores))

    def _calculate_mrr(self, relevance_scores: List[float], threshold: float = 0.5) -> float:
        """Mean Reciprocal Rank 계산"""
        for i, score in enumerate(relevance_scores):
            if score >= threshold:
                return 1.0 / (i + 1)
        return 0.0

    def evaluate_all(self, top_k: int = 10) -> Dict[str, Any]:
        """전체 벤치마크 평가"""
        queries = self.benchmark.get("queries", [])
        results = []

        logger.info(f"Evaluating {len(queries)} benchmark queries...")

        for query_spec in queries:
            try:
                result = self.evaluate_query(query_spec, top_k=top_k)
                results.append(result)
                logger.info(
                    f"  [{result['query_id']}] P@5={result['precision_at_5']:.2f}, "
                    f"MRR={result['mrr']:.2f}, Time={result['response_time_ms']:.0f}ms"
                )
            except Exception as e:
                logger.error(f"  [{query_spec['id']}] Error: {e}")
                results.append({
                    "query_id": query_spec["id"],
                    "error": str(e),
                })

        # 집계
        valid_results = [r for r in results if "error" not in r]

        if valid_results:
            avg_p5 = sum(r["precision_at_5"] for r in valid_results) / len(valid_results)
            avg_p10 = sum(r["precision_at_10"] for r in valid_results) / len(valid_results)
            avg_mrr = sum(r["mrr"] for r in valid_results) / len(valid_results)
            avg_time = sum(r["response_time_ms"] for r in valid_results) / len(valid_results)
            avg_intent = sum(r["intent_match_score"] for r in valid_results) / len(valid_results)
            avg_modality = sum(r["modality_match_score"] for r in valid_results) / len(valid_results)
        else:
            avg_p5 = avg_p10 = avg_mrr = avg_time = avg_intent = avg_modality = 0

        return {
            "evaluated_at": datetime.now().isoformat(),
            "num_queries": len(queries),
            "num_successful": len(valid_results),
            "metrics": {
                "avg_precision_at_5": round(avg_p5, 4),
                "avg_precision_at_10": round(avg_p10, 4),
                "avg_mrr": round(avg_mrr, 4),
                "avg_response_time_ms": round(avg_time, 2),
                "avg_intent_match": round(avg_intent, 4),
                "avg_modality_match": round(avg_modality, 4),
            },
            "per_query_results": results,
        }


def main():
    """메인 실행"""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate search quality")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results to evaluate")
    parser.add_argument("--parser", type=str, choices=["rule", "llm", "smart"], default="smart")
    parser.add_argument("--output", type=str, default="data/eval/eval_report.json", help="Output file")
    parser.add_argument("--index-dir", type=str, default="data/index", help="Index directory")

    args = parser.parse_args()

    # 검색 엔진 초기화
    logger.info("Initializing search engine...")
    engine = SearchEngine(
        db_path=Path(args.index_dir),
        parser_mode=args.parser,
    )

    # 평가 실행
    evaluator = SearchEvaluator(engine)
    report = evaluator.evaluate_all(top_k=args.top_k)

    # 결과 출력
    print("\n" + "=" * 60)
    print("Sophia AI Search Quality Report")
    print("=" * 60)
    print(f"\nEvaluated: {report['num_queries']} queries")
    print(f"Parser mode: {args.parser}")
    print("\nMetrics:")
    print(f"  Precision@5:    {report['metrics']['avg_precision_at_5']:.2%}")
    print(f"  Precision@10:   {report['metrics']['avg_precision_at_10']:.2%}")
    print(f"  MRR:            {report['metrics']['avg_mrr']:.4f}")
    print(f"  Avg Time:       {report['metrics']['avg_response_time_ms']:.0f}ms")
    print(f"  Intent Match:   {report['metrics']['avg_intent_match']:.2%}")
    print(f"  Modality Match: {report['metrics']['avg_modality_match']:.2%}")

    # 결과 저장
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
