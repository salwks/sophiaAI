#!/usr/bin/env python3
"""
MARIA-Mammo: Streamlit UI 실행
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def main():
    """Streamlit 앱 실행"""
    app_path = PROJECT_ROOT / "src" / "ui" / "app.py"

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        "8501",
        "--server.address",
        "0.0.0.0",
    ]

    subprocess.run(cmd, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    main()
