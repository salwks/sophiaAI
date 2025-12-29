#!/usr/bin/env python3
"""
MARIA-Mammo: LLM Query Parser Test Script
==========================================
LLM 파서 종합 테스트
"""

import json
import sys
import time
from pathlib import Path

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.search.query_parser import OllamaClient, LLMQueryParser, QueryParser


# 테스트 쿼리 (한국어/영어 혼합)
TEST_QUERIES = [
    # === 기본 영어 ===
    {
        "query": "DBT vs FFDM sensitivity comparison",
        "expected": {
            "modality": ["DBT", "FFDM"],
            "keywords_contain": ["dbt", "ffdm", "sensitivity"],
        }
    },
    {
        "query": "microcalcification detection in dense breast",
        "expected": {
            "pathology": ["calcification", "density"],
        }
    },

    # === 기본 한국어 ===
    {
        "query": "유방 밀도 분류 기준",
        "expected": {
            "keywords_contain": ["breast", "density"],
        }
    },
    {
        "query": "치밀유방에서 미세석회화 검출",
        "expected": {
            "pathology": ["calcification"],
        }
    },

    # === 복잡한 조건 ===
    {
        "query": "Korean women DBT prospective study since 2020",
        "expected": {
            "modality": ["DBT"],
            "population": "Asian",
            "study_type": "prospective",
            "year_min": 2020,
        }
    },
    {
        "query": "BI-RADS 4 lesion positive predictive value",
        "expected": {
            "keywords_contain": ["bi-rads", "predictive"],
        }
    },

    # === 비교 연구 ===
    {
        "query": "contrast enhanced mammography vs MRI comparison",
        "expected": {
            "modality": ["CEM", "MRI"],
        }
    },
    {
        "query": "3D tomosynthesis vs 2D mammography sensitivity specificity",
        "expected": {
            "modality": ["DBT", "FFDM"],
            "keywords_contain": ["sensitivity", "specificity"],
        }
    },

    # === AI/CAD ===
    {
        "query": "AI CAD breast cancer detection performance",
        "expected": {
            "keywords_contain": ["ai", "cad", "detection"],
        }
    },
    {
        "query": "deep learning breast density classification",
        "expected": {
            "keywords_contain": ["deep", "learning", "density"],
        }
    },

    # === 엣지 케이스 ===
    {
        "query": "mammography screening",
        "expected": {
            "keywords_contain": ["mammography", "screening"],
        }
    },
    {
        "query": "breast cancer",
        "expected": {
            "keywords_contain": ["breast", "cancer"],
        }
    },
]


def check_expected(result, expected: dict) -> tuple[bool, list[str]]:
    """결과가 기대값과 일치하는지 확인"""
    passed = True
    issues = []

    # modality 확인
    if "modality" in expected:
        result_modality = set(result.filters.modality or [])
        expected_modality = set(expected["modality"])
        if not expected_modality.issubset(result_modality):
            issues.append(f"Modality: expected {expected_modality}, got {result_modality}")
            passed = False

    # pathology 확인
    if "pathology" in expected:
        result_pathology = set(result.filters.pathology or [])
        expected_pathology = set(expected["pathology"])
        if not expected_pathology.issubset(result_pathology):
            issues.append(f"Pathology: expected {expected_pathology}, got {result_pathology}")
            passed = False

    # population 확인
    if "population" in expected:
        if result.filters.population != expected["population"]:
            issues.append(f"Population: expected {expected['population']}, got {result.filters.population}")
            passed = False

    # study_type 확인
    if "study_type" in expected:
        if result.filters.study_type != expected["study_type"]:
            issues.append(f"Study type: expected {expected['study_type']}, got {result.filters.study_type}")
            passed = False

    # year_min 확인
    if "year_min" in expected:
        if result.filters.year_min != expected["year_min"]:
            issues.append(f"Year min: expected {expected['year_min']}, got {result.filters.year_min}")
            passed = False

    # keywords 포함 여부 확인
    if "keywords_contain" in expected:
        result_keywords = set(k.lower() for k in result.keywords)
        for kw in expected["keywords_contain"]:
            if not any(kw.lower() in rk for rk in result_keywords):
                issues.append(f"Keyword '{kw}' not found in {result.keywords}")
                passed = False

    return passed, issues


def test_parser(use_llm: bool = True):
    """파서 테스트 실행"""
    print("=" * 70)
    print(f"MARIA-Mammo Query Parser Test ({'LLM' if use_llm else 'Rule-based'})")
    print("=" * 70)

    # Ollama 상태 확인
    if use_llm:
        client = OllamaClient()
        if not client.is_available():
            print("\n❌ Ollama not available!")
            print("   Run: ollama serve")
            print("   Run: ollama pull llama3.2")
            print("\nFalling back to rule-based parser...")
            use_llm = False
        else:
            print(f"\n✅ Ollama available (model: {client.model})")
        client.close()

    # 파서 초기화
    if use_llm:
        parser = LLMQueryParser(fallback_to_rule=True)
    else:
        parser = QueryParser()

    # 결과 수집
    results = []
    success_count = 0
    total_time = 0

    print("\n" + "-" * 70)

    for i, test_case in enumerate(TEST_QUERIES, 1):
        query = test_case["query"]
        expected = test_case["expected"]

        print(f"\n[{i}/{len(TEST_QUERIES)}] Query: {query}")

        start = time.time()
        try:
            parsed = parser.parse(query)
            elapsed = time.time() - start
            total_time += elapsed

            # 결과 출력
            print(f"   Keywords: {parsed.keywords[:5]}{'...' if len(parsed.keywords) > 5 else ''}")
            print(f"   MeSH: {parsed.mesh_terms[:3]}{'...' if len(parsed.mesh_terms) > 3 else ''}")
            print(f"   Filters: mod={parsed.filters.modality}, path={parsed.filters.pathology}, "
                  f"pop={parsed.filters.population}, study={parsed.filters.study_type}")
            print(f"   Intent: {parsed.intent[:60]}{'...' if len(parsed.intent) > 60 else ''}")
            print(f"   Time: {elapsed:.2f}s")

            # 검증
            passed, issues = check_expected(parsed, expected)

            if passed:
                success_count += 1
                print("   ✅ PASS")
            else:
                print("   ⚠️  PARTIAL")
                for issue in issues:
                    print(f"      - {issue}")

            results.append({
                "query": query,
                "parsed": {
                    "keywords": parsed.keywords,
                    "mesh_terms": parsed.mesh_terms,
                    "filters": parsed.filters.model_dump() if hasattr(parsed.filters, 'model_dump') else str(parsed.filters),
                    "intent": parsed.intent,
                },
                "expected": expected,
                "time": elapsed,
                "passed": passed,
                "issues": issues,
            })

        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            results.append({
                "query": query,
                "error": str(e),
                "passed": False,
            })

    # 요약
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Parser: {'LLM (llama3.2)' if use_llm else 'Rule-based'}")
    print(f"Total queries: {len(TEST_QUERIES)}")
    print(f"Passed: {success_count}/{len(TEST_QUERIES)} ({100*success_count/len(TEST_QUERIES):.1f}%)")
    print(f"Average time: {total_time/len(TEST_QUERIES):.2f}s")
    print(f"Total time: {total_time:.2f}s")

    # 결과 저장
    output_dir = Path("data/eval")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "parser_test_results.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    # 정리
    if use_llm and hasattr(parser, 'close'):
        parser.close()

    return success_count, len(TEST_QUERIES)


def test_single(query: str, use_llm: bool = True):
    """단일 쿼리 테스트"""
    print(f"Query: {query}\n")

    if use_llm:
        client = OllamaClient()
        if not client.is_available():
            print("❌ Ollama not available, using rule-based parser")
            use_llm = False
        client.close()

    if use_llm:
        parser = LLMQueryParser(fallback_to_rule=True)
    else:
        parser = QueryParser()

    start = time.time()
    parsed = parser.parse(query)
    elapsed = time.time() - start

    print(f"Keywords: {parsed.keywords}")
    print(f"MeSH terms: {parsed.mesh_terms}")
    print(f"Filters:")
    print(f"  - modality: {parsed.filters.modality}")
    print(f"  - pathology: {parsed.filters.pathology}")
    print(f"  - study_type: {parsed.filters.study_type}")
    print(f"  - population: {parsed.filters.population}")
    print(f"  - year_min: {parsed.filters.year_min}")
    print(f"  - year_max: {parsed.filters.year_max}")
    print(f"Intent: {parsed.intent}")
    print(f"Parser: {'LLM' if use_llm else 'Rule-based'}")
    print(f"Time: {elapsed:.2f}s")

    if use_llm and hasattr(parser, 'close'):
        parser.close()


def compare_parsers():
    """LLM vs Rule-based 파서 비교"""
    print("=" * 70)
    print("Parser Comparison: LLM vs Rule-based")
    print("=" * 70)

    # Ollama 확인
    client = OllamaClient()
    llm_available = client.is_available()
    client.close()

    if not llm_available:
        print("\n❌ Ollama not available. Cannot compare.")
        return

    llm_parser = LLMQueryParser(fallback_to_rule=False)
    rule_parser = QueryParser()

    test_queries = [
        "DBT vs FFDM for microcalcification",
        "유방 밀도 분류 BI-RADS",
        "Korean women breast screening since 2020",
        "meta-analysis contrast enhanced mammography",
    ]

    print("\n" + "-" * 70)

    for query in test_queries:
        print(f"\n🔍 Query: {query}")

        # Rule-based
        start = time.time()
        rule_result = rule_parser.parse(query)
        rule_time = time.time() - start

        # LLM
        start = time.time()
        try:
            llm_result = llm_parser.parse(query)
            llm_time = time.time() - start
            llm_error = None
        except Exception as e:
            llm_time = 0
            llm_result = None
            llm_error = str(e)

        print(f"\n   [Rule-based] ({rule_time:.2f}s)")
        print(f"   Keywords: {rule_result.keywords[:5]}")
        print(f"   Filters: {rule_result.filters.modality}, {rule_result.filters.pathology}")

        if llm_result:
            print(f"\n   [LLM] ({llm_time:.2f}s)")
            print(f"   Keywords: {llm_result.keywords[:5]}")
            print(f"   Filters: {llm_result.filters.modality}, {llm_result.filters.pathology}")
            print(f"   Intent: {llm_result.intent[:50]}...")
        else:
            print(f"\n   [LLM] Error: {llm_error}")

    llm_parser.close()


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser(description="MARIA-Mammo Query Parser Test")
    argparser.add_argument("query", nargs="?", help="Single query to test")
    argparser.add_argument("--rule", action="store_true", help="Use rule-based parser only")
    argparser.add_argument("--compare", action="store_true", help="Compare LLM vs Rule-based")

    args = argparser.parse_args()

    if args.compare:
        compare_parsers()
    elif args.query:
        test_single(args.query, use_llm=not args.rule)
    else:
        test_parser(use_llm=not args.rule)
