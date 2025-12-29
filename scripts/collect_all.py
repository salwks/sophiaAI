#!/usr/bin/env python3
"""
MARIA-Mammo: Multi-Source Collection Script
============================================
PubMed, KoreaMed, Semantic Scholar, RSNA에서 논문 수집

Usage:
    uv run python scripts/collect_all.py
    uv run python scripts/collect_all.py --start-year 2010 --end-year 2025
    uv run python scripts/collect_all.py --sources pubmed,semantic_scholar
"""

import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.collection.multi_source_collector import main

if __name__ == "__main__":
    main()
