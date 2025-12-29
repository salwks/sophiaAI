#!/usr/bin/env python3
"""
MARIA-Mammo: Automated Question Testing
========================================
테스트 질문으로 시스템 검증

Usage:
    uv run python scripts/test_questions.py --category birads
    uv run python scripts/test_questions.py --category all
    uv run python scripts/test_questions.py --sample 10
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import time
from typing import List, Dict
from src.search.engine import SearchEngine

# 테스트 질문들
TEST_QUESTIONS = {
    "birads": [
        "BI-RADS 0은 무엇인가요?",
        "BI-RADS Category 1이란?",
        "카테고리 2는 무엇을 의미하나요?",
        "BI-RADS 3는 무엇인가요?",
        "Category 4A는 무엇인가요?",
        "BI-RADS 4B 카테고리 설명해주세요",
        "카테고리 4C는 무엇인가요?",
        "BI-RADS 5는 무엇인가요?",
        "Category 6는 어떤 경우인가요?",
        "What is BI-RADS Category 5?",
    ],

    "basics": [
        "맘모그래피 검사는 어떻게 하나요?",
        "mammography positioning techniques는 무엇인가요?",
        "맘모그래피 표준 뷰는 무엇인가요?",
        "breast density는 무엇인가요?",
        "유방 밀도 분류 방법은?",
    ],

    "technology": [
        "Digital Breast Tomosynthesis는 무엇인가요?",
        "DBT의 장점은 무엇인가요?",
        "AI in mammography는 어떻게 사용되나요?",
        "CAD 시스템이란 무엇인가요?",
        "딥러닝으로 맘모그래피를 판독할 수 있나요?",
    ],

    "clinical": [
        "맘모그래피 스크리닝은 몇 살부터 하나요?",
        "breast cancer screening guidelines",
        "맘모그래피 위양성률은 얼마나 되나요?",
        "dense breast에서 맘모그래피 제한점은?",
        "interval cancer란 무엇인가요?",
    ],

    "comparison": [
        "맘모그래피 vs 초음파 차이는?",
        "mammography vs MRI for breast cancer",
        "맘모그래피와 초음파를 함께 하는 이유는?",
    ],
}


def test_questions(
    questions: List[str],
    engine: SearchEngine,
    verbose: bool = True,
) -> Dict:
    """질문 테스트 실행"""
    results = {
        "total": len(questions),
        "passed": 0,
        "failed": 0,
        "avg_time_ms": 0,
        "details": []
    }

    total_time = 0

    for i, question in enumerate(questions, 1):
        if verbose:
            print(f"\n{'='*70}")
            print(f"[{i}/{len(questions)}] {question}")
            print('='*70)

        start_time = time.time()

        try:
            response = engine.search(question, top_k=3, use_rerank=False)
            elapsed_ms = int((time.time() - start_time) * 1000)
            total_time += elapsed_ms

            # 결과 평가
            has_results = len(response.results) > 0
            top_score = response.results[0].score if has_results else 0

            # BI-RADS 질문은 즉각 응답 확인
            is_birads = "BI-RADS" in question or "Category" in question or "카테고리" in question
            fast_response = elapsed_ms < 100

            passed = has_results and (not is_birads or fast_response)

            if passed:
                results["passed"] += 1
                status = "✅ PASS"
            else:
                results["failed"] += 1
                status = "❌ FAIL"

            detail = {
                "question": question,
                "passed": passed,
                "time_ms": elapsed_ms,
                "num_results": len(response.results),
                "top_score": top_score,
            }
            results["details"].append(detail)

            if verbose:
                print(f"{status} | Time: {elapsed_ms}ms | Results: {len(response.results)}")

                if has_results:
                    for j, result in enumerate(response.results[:2], 1):
                        paper = result.paper
                        print(f"\n  [{j}] Score: {result.score:.3f}")
                        print(f"      {paper.title[:60]}")
                        print(f"      {paper.journal[:40] if paper.journal else 'N/A'} ({paper.year})")

        except Exception as e:
            results["failed"] += 1
            if verbose:
                print(f"❌ ERROR: {e}")

            detail = {
                "question": question,
                "passed": False,
                "error": str(e)
            }
            results["details"].append(detail)

    results["avg_time_ms"] = total_time // len(questions) if questions else 0

    return results


def main():
    parser = argparse.ArgumentParser(description="Test MARIA-Mammo with predefined questions")
    parser.add_argument(
        "--category",
        type=str,
        choices=["birads", "basics", "technology", "clinical", "comparison", "all"],
        default="birads",
        help="Question category to test",
    )
    parser.add_argument(
        "--sample",
        type=int,
        help="Number of random questions to sample",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Show detailed output",
    )

    args = parser.parse_args()

    # 검색 엔진 초기화
    print("🔍 Initializing search engine...")
    engine = SearchEngine(
        db_path=Path("data/index"),
        parser_mode="smart",
        use_reranker=False,
    )

    # 질문 선택
    if args.category == "all":
        questions = []
        for cat_questions in TEST_QUESTIONS.values():
            questions.extend(cat_questions)
    else:
        questions = TEST_QUESTIONS.get(args.category, [])

    if args.sample and args.sample < len(questions):
        import random
        questions = random.sample(questions, args.sample)

    print(f"\n📝 Testing {len(questions)} questions from category: {args.category}")
    print("="*70)

    # 테스트 실행
    results = test_questions(questions, engine, verbose=args.verbose)

    # 결과 요약
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    print(f"Total:     {results['total']}")
    print(f"✅ Passed:  {results['passed']} ({results['passed']/results['total']*100:.1f}%)")
    print(f"❌ Failed:  {results['failed']} ({results['failed']/results['total']*100:.1f}%)")
    print(f"⏱️  Avg Time: {results['avg_time_ms']}ms")

    # BI-RADS 질문 특별 분석
    birads_questions = [d for d in results['details']
                        if 'BI-RADS' in d['question'] or 'Category' in d['question']]
    if birads_questions:
        avg_birads_time = sum(d['time_ms'] for d in birads_questions) // len(birads_questions)
        fast_birads = sum(1 for d in birads_questions if d['time_ms'] < 100)
        print(f"\n🎯 BI-RADS Performance:")
        print(f"   Average time: {avg_birads_time}ms")
        print(f"   Fast responses (<100ms): {fast_birads}/{len(birads_questions)}")

    return 0 if results['failed'] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
