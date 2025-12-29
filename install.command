#!/bin/bash
#
# Sophia AI Installer
# ===================
# 초기 설정 및 설치 스크립트
#

cd "$(dirname "$0")"

echo "🧠 Sophia AI 설치 시작"
echo "================================"
echo ""

# 1. Homebrew 확인 (macOS 패키지 매니저)
echo "📦 Homebrew 확인 중..."
if ! command -v brew &> /dev/null; then
    echo "⚠️  Homebrew가 설치되지 않았습니다."
    echo "📥 Homebrew 설치 중..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
    echo "✅ Homebrew 설치됨"
fi

# 2. Python 확인
echo ""
echo "🐍 Python 확인 중..."
if ! command -v python3 &> /dev/null; then
    echo "⚠️  Python3가 설치되지 않았습니다."
    echo "📥 Python3 설치 중..."
    brew install python@3.12
else
    PYTHON_VERSION=$(python3 --version)
    echo "✅ $PYTHON_VERSION 설치됨"
fi

# 3. uv 확인 (빠른 Python 패키지 매니저)
echo ""
echo "⚡ uv 확인 중..."
if ! command -v uv &> /dev/null; then
    echo "⚠️  uv가 설치되지 않았습니다."
    echo "📥 uv 설치 중..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # PATH에 추가
    export PATH="$HOME/.cargo/bin:$PATH"
else
    echo "✅ uv 설치됨"
fi

# 4. Ollama 확인
echo ""
echo "🤖 Ollama 확인 중..."
if ! command -v ollama &> /dev/null; then
    echo "⚠️  Ollama가 설치되지 않았습니다."
    echo ""
    echo "Ollama 수동 설치가 필요합니다:"
    echo "1. https://ollama.com 방문"
    echo "2. macOS용 다운로드 (약 600MB)"
    echo "3. .dmg 파일 실행 및 설치"
    echo ""
    read -p "Ollama를 설치한 후 Enter를 눌러 계속하세요..."

    # 재확인
    if ! command -v ollama &> /dev/null; then
        echo "❌ Ollama가 여전히 설치되지 않았습니다."
        echo "설치 후 이 스크립트를 다시 실행하세요."
        exit 1
    fi
else
    echo "✅ Ollama 설치됨"
fi

# 5. Ollama 서버 시작 및 모델 다운로드
echo ""
echo "🚀 Ollama 서버 시작 중..."
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    ollama serve > /dev/null 2>&1 &
    sleep 3
fi

echo "📥 qwen2.5:14b 모델 다운로드 중..."
echo "(약 8GB, 인터넷 속도에 따라 5-30분 소요)"
ollama pull qwen2.5:14b

if [ $? -eq 0 ]; then
    echo "✅ 모델 다운로드 완료!"
else
    echo "❌ 모델 다운로드 실패!"
    echo "터미널에서 수동으로 실행하세요: ollama pull qwen2.5:14b"
    exit 1
fi

# 6. Python 패키지 설치
echo ""
echo "📦 Python 패키지 설치 중..."
if [ -f "pyproject.toml" ]; then
    uv sync
    echo "✅ 패키지 설치 완료!"
else
    echo "❌ pyproject.toml 파일을 찾을 수 없습니다."
    exit 1
fi

# 7. 데이터베이스 확인
echo ""
echo "💾 데이터베이스 확인 중..."
if [ ! -f "data/index/metadata.db" ]; then
    echo "⚠️  데이터베이스가 없습니다."
    echo "📥 샘플 데이터를 다운로드하거나 직접 인덱싱해야 합니다."
    echo ""
    echo "옵션 1: 기존 데이터베이스 복사"
    echo "  - data/index/ 폴더에 metadata.db 파일 복사"
    echo ""
    echo "옵션 2: 직접 인덱싱 (시간 소요)"
    echo "  - 터미널: uv run python scripts/collect_all.py"
    echo "  - 터미널: uv run python scripts/index.py"
    echo ""
    read -p "나중에 설정하시겠습니까? (Y/n): " choice
    case "$choice" in
        n|N )
            echo "설치를 중단합니다."
            exit 1
            ;;
        * )
            echo "⏭️  데이터베이스는 나중에 설정하세요."
            ;;
    esac
else
    echo "✅ 데이터베이스 준비됨"
fi

# 8. 설치 완료
echo ""
echo "================================"
echo "✅ 설치 완료!"
echo "================================"
echo ""
echo "🚀 실행 방법:"
echo "  1. 'start-sophiaai.command' 파일을 더블클릭"
echo "  또는"
echo "  2. 터미널: ./start-sophiaai.command"
echo ""
echo "📚 데이터베이스 설정이 필요하면:"
echo "  uv run python scripts/index.py"
echo ""
echo "💡 도움말:"
echo "  README.md 파일 참조"
echo ""
read -p "Enter 키를 눌러 종료..."
