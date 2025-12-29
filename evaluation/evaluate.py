"""
MARIA-Mammo: Evaluation System
==============================
검색 품질 평가
"""

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set

from src.models import EvalMetrics, EvalQuery, EvalReport
from src.search.engine import SearchEngine

logger = logging.getLogger(__name__)


class Evaluator:
    """검색 품질 평가기"""

    def __init__(self, engine: SearchEngine):
        self.engine = engine

    def precision_at_k(
        self,
        retrieved: List[str],
        relevant: Set[str],
        k: int,
    ) -> float:
        """Precision@K 계산"""
        if k == 0 or not retrieved:
            return 0.0

        retrieved_k = retrieved[:k]
        relevant_count = sum(1 for pmid in retrieved_k if pmid in relevant)

        return relevant_count / min(k, len(retrieved_k))

    def recall_at_k(
        self,
        retrieved: List[str],
        relevant: Set[str],
        k: int,
    ) -> float:
        """Recall@K 계산"""
        if not relevant:
            return 0.0

        retrieved_k = retrieved[:k]
        relevant_count = sum(1 for pmid in retrieved_k if pmid in relevant)

        return relevant_count / len(relevant)

    def mrr(
        self,
        retrieved: List[str],
        relevant: Set[str],
    ) -> float:
        """Mean Reciprocal Rank 계산"""
        for i, pmid in enumerate(retrieved):
            if pmid in relevant:
                return 1.0 / (i + 1)
        return 0.0

    def dcg_at_k(
        self,
        retrieved: List[str],
        relevant: Set[str],
        k: int,
    ) -> float:
        """DCG@K 계산"""
        dcg = 0.0
        for i, pmid in enumerate(retrieved[:k]):
            if pmid in relevant:
                # Binary relevance (1 if relevant, 0 otherwise)
                dcg += 1.0 / math.log2(i + 2)  # +2 because index starts at 0
        return dcg

    def ndcg_at_k(
        self,
        retrieved: List[str],
        relevant: Set[str],
        k: int,
    ) -> float:
        """NDCG@K 계산"""
        dcg = self.dcg_at_k(retrieved, relevant, k)

        # Ideal DCG
        ideal_retrieved = list(relevant)[:k]
        idcg = self.dcg_at_k(ideal_retrieved, relevant, k)

        if idcg == 0:
            return 0.0

        return dcg / idcg

    def evaluate_query(
        self,
        query: EvalQuery,
        top_k: int = 20,
    ) -> EvalMetrics:
        """단일 쿼리 평가"""
        # 검색 실행
        response = self.engine.search(query.query, top_k=top_k)

        # 검색된 PMID
        retrieved = [r.paper.pmid for r in response.results]

        # 관련 문서 집합
        relevant = set(query.relevant_pmids)

        # 지표 계산
        return EvalMetrics(
            query_id=query.id,
            precision_at_5=self.precision_at_k(retrieved, relevant, 5),
            precision_at_10=self.precision_at_k(retrieved, relevant, 10),
            recall_at_10=self.recall_at_k(retrieved, relevant, 10),
            mrr=self.mrr(retrieved, relevant),
            ndcg_at_10=self.ndcg_at_k(retrieved, relevant, 10),
        )

    def evaluate(
        self,
        queries: List[EvalQuery],
        top_k: int = 20,
    ) -> EvalReport:
        """전체 쿼리 평가"""
        logger.info(f"Evaluating {len(queries)} queries...")

        metrics_list = []
        for query in queries:
            try:
                metrics = self.evaluate_query(query, top_k)
                metrics_list.append(metrics)
                logger.debug(f"Query {query.id}: P@10={metrics.precision_at_10:.3f}")
            except Exception as e:
                logger.error(f"Failed to evaluate query {query.id}: {e}")

        if not metrics_list:
            return EvalReport(
                evaluated_at=datetime.now(),
                total_queries=0,
                avg_precision_at_5=0.0,
                avg_precision_at_10=0.0,
                avg_recall_at_10=0.0,
                avg_mrr=0.0,
                avg_ndcg_at_10=0.0,
                per_query_metrics=[],
            )

        # 평균 계산
        n = len(metrics_list)
        return EvalReport(
            evaluated_at=datetime.now(),
            total_queries=n,
            avg_precision_at_5=sum(m.precision_at_5 for m in metrics_list) / n,
            avg_precision_at_10=sum(m.precision_at_10 for m in metrics_list) / n,
            avg_recall_at_10=sum(m.recall_at_10 for m in metrics_list) / n,
            avg_mrr=sum(m.mrr for m in metrics_list) / n,
            avg_ndcg_at_10=sum(m.ndcg_at_10 for m in metrics_list) / n,
            per_query_metrics=metrics_list,
        )


def load_eval_queries(path: Path) -> List[EvalQuery]:
    """평가 쿼리 로드"""
    if not path.exists():
        return []

    data = json.loads(path.read_text())
    return [EvalQuery(**q) for q in data]


def save_report(report: EvalReport, path: Path):
    """평가 리포트 저장"""
    data = {
        "evaluated_at": report.evaluated_at.isoformat(),
        "total_queries": report.total_queries,
        "avg_precision_at_5": report.avg_precision_at_5,
        "avg_precision_at_10": report.avg_precision_at_10,
        "avg_recall_at_10": report.avg_recall_at_10,
        "avg_mrr": report.avg_mrr,
        "avg_ndcg_at_10": report.avg_ndcg_at_10,
        "per_query_metrics": [
            {
                "query_id": m.query_id,
                "precision_at_5": m.precision_at_5,
                "precision_at_10": m.precision_at_10,
                "recall_at_10": m.recall_at_10,
                "mrr": m.mrr,
                "ndcg_at_10": m.ndcg_at_10,
            }
            for m in report.per_query_metrics
        ],
    }

    path.write_text(json.dumps(data, indent=2))


def generate_report_markdown(report: EvalReport) -> str:
    """마크다운 리포트 생성"""
    lines = [
        "# MARIA-Mammo Evaluation Report",
        "",
        f"Evaluated at: {report.evaluated_at.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total queries: {report.total_queries}",
        "",
        "## Summary Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Precision@5 | {report.avg_precision_at_5:.3f} |",
        f"| Precision@10 | {report.avg_precision_at_10:.3f} |",
        f"| Recall@10 | {report.avg_recall_at_10:.3f} |",
        f"| MRR | {report.avg_mrr:.3f} |",
        f"| NDCG@10 | {report.avg_ndcg_at_10:.3f} |",
        "",
        "## Per-Query Metrics",
        "",
        "| Query ID | P@5 | P@10 | R@10 | MRR | NDCG@10 |",
        "|----------|-----|------|------|-----|---------|",
    ]

    for m in report.per_query_metrics:
        lines.append(
            f"| {m.query_id} | {m.precision_at_5:.3f} | {m.precision_at_10:.3f} | "
            f"{m.recall_at_10:.3f} | {m.mrr:.3f} | {m.ndcg_at_10:.3f} |"
        )

    return "\n".join(lines)


def main():
    """CLI 실행"""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Evaluate search quality")
    parser.add_argument(
        "--queries",
        type=str,
        default="evaluation/queries.json",
        help="Evaluation queries file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation/report.json",
        help="Output report file",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default="data/index",
        help="Index directory",
    )

    args = parser.parse_args()

    # 검색 엔진 초기화
    engine = SearchEngine(db_path=Path(args.index_dir))

    # 쿼리 로드
    queries = load_eval_queries(Path(args.queries))

    if not queries:
        print("No evaluation queries found. Create evaluation/queries.json first.")
        return

    # 평가 실행
    evaluator = Evaluator(engine)
    report = evaluator.evaluate(queries)

    # 리포트 저장
    save_report(report, Path(args.output))

    # 마크다운 리포트
    md_report = generate_report_markdown(report)
    Path(args.output.replace(".json", ".md")).write_text(md_report)

    # 결과 출력
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Queries evaluated: {report.total_queries}")
    print(f"Precision@5:  {report.avg_precision_at_5:.3f}")
    print(f"Precision@10: {report.avg_precision_at_10:.3f}")
    print(f"Recall@10:    {report.avg_recall_at_10:.3f}")
    print(f"MRR:          {report.avg_mrr:.3f}")
    print(f"NDCG@10:      {report.avg_ndcg_at_10:.3f}")
    print("=" * 50)
    print(f"Report saved to: {args.output}")


if __name__ == "__main__":
    main()
