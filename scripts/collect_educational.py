#!/usr/bin/env python3
"""
Sophia AI: Educational Content Collection Script
===================================================
맘모그래피 기초 및 교육 자료 수집

Usage:
    # 모든 카테고리 수집
    uv run python scripts/collect_educational.py

    # 특정 카테고리만 수집
    uv run python scripts/collect_educational.py --category basics
    uv run python scripts/collect_educational.py --category guidelines

    # 수집 개수 제한
    uv run python scripts/collect_educational.py --max-results 200
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.collection.educational_collector import main

if __name__ == "__main__":
    main()
