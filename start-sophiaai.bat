@echo off
REM Sophia AI Launcher for Windows
REM ================================
REM 더블클릭으로 Sophia AI 실행

setlocal enabledelayedexpansion

echo.
echo ============================================
echo    🧠 Sophia AI 시작 중...
echo ============================================
echo.

REM 현재 디렉토리로 이동
cd /d "%~dp0"

REM 1. Ollama 서버 확인
echo 📡 Ollama 서버 확인 중...
curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Ollama 서버가 실행되지 않았습니다.
    echo.

    REM Ollama 설치 확인
    where ollama >nul 2>&1
    if errorlevel 1 (
        echo ❌ Ollama가 설치되지 않았습니다!
        echo.
        echo Ollama 설치 방법:
        echo 1. https://ollama.com 방문
        echo 2. Windows용 다운로드 및 설치
        echo 3. 명령 프롬프트에서 'ollama pull qwen2.5:14b' 실행
        echo.
        pause
        exit /b 1
    )

    echo 🚀 Ollama 서버 시작 중...
    start /B ollama serve >nul 2>&1

    REM 서버 시작 대기
    echo ⏳ Ollama 서버 시작 대기 중...
    timeout /t 5 /nobreak >nul

    REM 서버 확인
    curl -s http://localhost:11434/api/tags >nul 2>&1
    if errorlevel 1 (
        echo ❌ Ollama 서버 시작 실패!
        pause
        exit /b 1
    ) else (
        echo ✅ Ollama 서버 시작 완료!
    )
) else (
    echo ✅ Ollama 서버가 이미 실행 중입니다.
)

REM 2. qwen2.5:14b 모델 확인
echo.
echo 🤖 LLM 모델 확인 중...
ollama list | findstr "qwen2.5:14b" >nul 2>&1
if errorlevel 1 (
    echo ⚠️  qwen2.5:14b 모델이 설치되지 않았습니다.
    echo 📥 모델 다운로드 중... (약 8GB, 시간이 걸릴 수 있습니다)
    ollama pull qwen2.5:14b

    if errorlevel 1 (
        echo ❌ 모델 다운로드 실패!
        pause
        exit /b 1
    ) else (
        echo ✅ 모델 다운로드 완료!
    )
) else (
    echo ✅ qwen2.5:14b 모델이 설치되어 있습니다.
)

REM 3. Python 환경 확인
echo.
echo 🐍 Python 환경 확인 중...

REM uv 확인
where uv >nul 2>&1
if not errorlevel 1 (
    echo ✅ uv 환경 사용
    set PYTHON_CMD=uv run python
    set STREAMLIT_CMD=uv run streamlit
) else (
    REM 일반 Python 사용
    where python >nul 2>&1
    if errorlevel 1 (
        echo ❌ Python이 설치되지 않았습니다!
        echo.
        echo Python 설치 방법:
        echo 1. https://python.org 방문
        echo 2. Python 3.11 이상 다운로드
        echo 3. "Add to PATH" 옵션 체크하여 설치
        echo.
        pause
        exit /b 1
    )

    echo ✅ Python 사용
    set PYTHON_CMD=python
    set STREAMLIT_CMD=streamlit

    REM 가상환경 활성화
    if exist ".venv\Scripts\activate.bat" (
        echo ✅ 가상환경 활성화
        call .venv\Scripts\activate.bat
    )
)

REM 4. 데이터베이스 확인
echo.
echo 💾 데이터베이스 확인 중...
if not exist "data\index\metadata.db" (
    echo ⚠️  데이터베이스가 없습니다!
    echo 📥 초기 설정이 필요합니다.
    echo.
    echo 다음 명령어를 명령 프롬프트에서 실행하세요:
    echo   cd %CD%
    echo   uv run python scripts/index.py
    echo.
    pause
    exit /b 1
) else (
    echo ✅ 데이터베이스 준비됨
)

REM 5. 기존 프로세스 종료
echo.
echo 🔄 기존 프로세스 확인 중...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8501"') do (
    taskkill /F /PID %%a >nul 2>&1
)

REM 6. Streamlit 앱 실행
echo.
echo ============================================
echo    🚀 Sophia AI 실행 중...
echo ============================================
echo.
echo ✅ 브라우저가 자동으로 열립니다!
echo ✅ URL: http://localhost:8501
echo.
echo ⚠️  종료하려면 이 창에서 Ctrl+C 를 누르세요
echo.

%STREAMLIT_CMD% run src/ui/app.py --server.port 8501 --server.headless true --browser.gatherUsageStats false --browser.serverAddress localhost

REM 종료 메시지
echo.
echo ============================================
echo    👋 Sophia AI 종료됨
echo ============================================
pause
