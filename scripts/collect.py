#!/usr/bin/env python3
"""
MARIA-Mammo: Data Collection Script
===================================
PubMed에서 맘모그래피 관련 논문 수집

Usage:
    uv run python scripts/collect.py --start-year 2020 --end-year 2025
    uv run python scripts/collect.py --stats
"""

import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.collection.collector import main

if __name__ == "__main__":
    main()
