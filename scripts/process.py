#!/usr/bin/env python3
"""
MARIA-Mammo: Data Processing Script
====================================
수집된 논문 데이터 정제 및 분류

Usage:
    uv run python scripts/process.py
    uv run python scripts/process.py --stats-only
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.processing.pipeline import main

if __name__ == "__main__":
    main()
