#!/usr/bin/env python3
"""
Query Translator 테스트
======================
한글 의학 질문 → 영문 기술 키워드 변환 테스트
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.search.query_translator import QueryTranslator

def test_translator():
    """쿼리 번역 테스트"""

    translator = QueryTranslator(
        ollama_url="http://localhost:11434",
        model="qwen2.5:14b"
    )

    # 테스트 케이스
    test_cases = [
        "유방촬영 노출 기법을 설명해줘",
        "BI-RADS 카테고리 5는 무엇인가요?",
        "맘모그래피 포지셔닝 방법",
        "DBT와 일반 맘모그래피 차이",
        "mammography의 기본 exposure procedure를 설명해줘",
        "유방 밀도는 무엇인가요?",
        "맘모그래피 스크리닝은 몇 살부터 하나요?",
        "mammography exposure technique",  # 이미 영어
    ]

    print("=" * 80)
    print("Query Translator 테스트")
    print("=" * 80)

    for i, query in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}]")
        print(f"원본: {query}")

        needs_translation = translator.needs_translation(query)
        print(f"번역 필요: {'예' if needs_translation else '아니오'}")

        if needs_translation:
            translated = translator.translate(query)
            print(f"번역: {translated}")
            print(f"길이: {len(query)} → {len(translated)} 글자")
        else:
            print(f"번역: (원본 유지) {query}")

        print("-" * 80)

    print("\n✅ 테스트 완료!")

if __name__ == "__main__":
    test_translator()
