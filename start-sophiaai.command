#!/bin/bash
#
# Sophia AI Launcher
# ==================
# 더블클릭으로 Sophia AI 실행
#

# 현재 스크립트의 디렉토리로 이동
cd "$(dirname "$0")"

echo "🧠 Sophia AI 시작 중..."
echo "================================"

# 1. Ollama 서버 확인 및 시작
echo "📡 Ollama 서버 확인 중..."
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "⚠️  Ollama 서버가 실행되지 않았습니다. 시작 중..."

    # Ollama가 설치되어 있는지 확인
    if ! command -v ollama &> /dev/null; then
        echo "❌ Ollama가 설치되지 않았습니다!"
        echo ""
        echo "Ollama 설치 방법:"
        echo "1. https://ollama.com 방문"
        echo "2. macOS용 다운로드 및 설치"
        echo "3. 터미널에서 'ollama pull qwen2.5:14b' 실행"
        echo ""
        read -p "Enter 키를 눌러 종료..."
        exit 1
    fi

    # Ollama 서버 백그라운드 실행
    ollama serve > /dev/null 2>&1 &

    # 서버 시작 대기
    echo "⏳ Ollama 서버 시작 대기 중..."
    sleep 5

    # 서버 확인
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "✅ Ollama 서버 시작 완료!"
    else
        echo "❌ Ollama 서버 시작 실패!"
        read -p "Enter 키를 눌러 종료..."
        exit 1
    fi
else
    echo "✅ Ollama 서버가 이미 실행 중입니다."
fi

# 2. qwen2.5:14b 모델 확인
echo ""
echo "🤖 LLM 모델 확인 중..."
if ! ollama list | grep -q "qwen2.5:14b"; then
    echo "⚠️  qwen2.5:14b 모델이 설치되지 않았습니다."
    echo "📥 모델 다운로드 중... (약 8GB, 시간이 걸릴 수 있습니다)"
    ollama pull qwen2.5:14b

    if [ $? -eq 0 ]; then
        echo "✅ 모델 다운로드 완료!"
    else
        echo "❌ 모델 다운로드 실패!"
        read -p "Enter 키를 눌러 종료..."
        exit 1
    fi
else
    echo "✅ qwen2.5:14b 모델이 설치되어 있습니다."
fi

# 3. Python 환경 확인
echo ""
echo "🐍 Python 환경 확인 중..."

# uv 사용 여부 확인
if [ -f "pyproject.toml" ] && command -v uv &> /dev/null; then
    echo "✅ uv 환경 사용"
    PYTHON_CMD="uv run python"
    STREAMLIT_CMD="uv run streamlit"
else
    # 일반 Python 사용
    if [ -d ".venv" ]; then
        echo "✅ 가상환경 활성화"
        source .venv/bin/activate
    fi
    PYTHON_CMD="python3"
    STREAMLIT_CMD="streamlit"
fi

# 4. 데이터베이스 확인
echo ""
echo "💾 데이터베이스 확인 중..."
if [ ! -f "data/index/metadata.db" ]; then
    echo "⚠️  데이터베이스가 없습니다!"
    echo "📥 초기 설정이 필요합니다."
    echo ""
    echo "다음 명령어를 터미널에서 실행하세요:"
    echo "  cd $(pwd)"
    echo "  uv run python scripts/index.py"
    echo ""
    read -p "Enter 키를 눌러 종료..."
    exit 1
else
    echo "✅ 데이터베이스 준비됨"
fi

# 5. Streamlit 앱 실행
echo ""
echo "🚀 Sophia AI 실행 중..."
echo "================================"
echo ""
echo "✅ 브라우저가 자동으로 열립니다!"
echo "✅ URL: http://localhost:8501"
echo ""
echo "⚠️  종료하려면 이 창에서 Ctrl+C 를 누르세요"
echo ""

# 기존 8501 포트 프로세스 종료
lsof -ti:8501 | xargs kill -9 2>/dev/null

# Streamlit 실행
$STREAMLIT_CMD run src/ui/app.py \
    --server.port 8501 \
    --server.headless true \
    --browser.gatherUsageStats false \
    --browser.serverAddress localhost

# 스크립트 종료 시 정리
echo ""
echo "👋 Sophia AI 종료됨"
