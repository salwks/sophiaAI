@echo off
REM Sophia AI Installer for Windows
REM ================================
REM 초기 설정 및 설치 스크립트

setlocal enabledelayedexpansion

echo.
echo ============================================
echo    🧠 Sophia AI 설치 시작
echo ============================================
echo.

cd /d "%~dp0"

REM 1. 관리자 권한 확인
net session >nul 2>&1
if errorlevel 1 (
    echo ⚠️  관리자 권한이 필요합니다!
    echo 이 파일을 마우스 우클릭 후 "관리자 권한으로 실행"을 선택하세요.
    echo.
    pause
    exit /b 1
)

REM 2. Chocolatey 확인 및 설치
echo 📦 Chocolatey 확인 중...
where choco >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Chocolatey가 설치되지 않았습니다.
    echo 📥 Chocolatey 설치 중...

    powershell -NoProfile -ExecutionPolicy Bypass -Command "Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))"

    if errorlevel 1 (
        echo ❌ Chocolatey 설치 실패!
        pause
        exit /b 1
    )

    REM PATH 새로고침
    call refreshenv
    echo ✅ Chocolatey 설치 완료!
) else (
    echo ✅ Chocolatey 설치됨
)

REM 3. Python 확인 및 설치
echo.
echo 🐍 Python 확인 중...
where python >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Python이 설치되지 않았습니다.
    echo 📥 Python 3.12 설치 중...
    choco install python312 -y

    if errorlevel 1 (
        echo ❌ Python 설치 실패!
        pause
        exit /b 1
    )

    REM PATH 새로고침
    call refreshenv
    echo ✅ Python 설치 완료!
) else (
    for /f "tokens=*" %%i in ('python --version') do set PYTHON_VERSION=%%i
    echo ✅ !PYTHON_VERSION! 설치됨
)

REM 4. Git 확인 및 설치 (선택사항)
echo.
echo 📚 Git 확인 중...
where git >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Git이 설치되지 않았습니다.
    choice /C YN /M "Git을 설치하시겠습니까? (Y/N)"
    if errorlevel 2 (
        echo ⏭️  Git 설치 건너뜀
    ) else (
        echo 📥 Git 설치 중...
        choco install git -y
        call refreshenv
    )
) else (
    echo ✅ Git 설치됨
)

REM 5. uv 설치
echo.
echo ⚡ uv 확인 중...
where uv >nul 2>&1
if errorlevel 1 (
    echo ⚠️  uv가 설치되지 않았습니다.
    echo 📥 uv 설치 중...

    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"

    if errorlevel 1 (
        echo ❌ uv 설치 실패!
        pause
        exit /b 1
    )

    REM PATH에 추가
    set PATH=%USERPROFILE%\.cargo\bin;%PATH%
    echo ✅ uv 설치 완료!
) else (
    echo ✅ uv 설치됨
)

REM 6. Ollama 확인
echo.
echo 🤖 Ollama 확인 중...
where ollama >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Ollama가 설치되지 않았습니다.
    echo.
    echo Ollama 수동 설치가 필요합니다:
    echo 1. https://ollama.com 방문
    echo 2. Windows용 다운로드 (약 600MB)
    echo 3. 실행 파일 설치
    echo.
    echo 설치 후 Enter를 눌러 계속하세요...
    pause

    REM 재확인
    where ollama >nul 2>&1
    if errorlevel 1 (
        echo ❌ Ollama가 여전히 설치되지 않았습니다.
        echo 설치 후 이 스크립트를 다시 실행하세요.
        pause
        exit /b 1
    )
) else (
    echo ✅ Ollama 설치됨
)

REM 7. Ollama 서버 시작 및 모델 다운로드
echo.
echo 🚀 Ollama 서버 시작 중...
curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    start /B ollama serve >nul 2>&1
    timeout /t 3 /nobreak >nul
)

echo 📥 qwen2.5:14b 모델 다운로드 중...
echo (약 8GB, 인터넷 속도에 따라 5-30분 소요)
ollama pull qwen2.5:14b

if errorlevel 1 (
    echo ❌ 모델 다운로드 실패!
    echo 명령 프롬프트에서 수동으로 실행하세요: ollama pull qwen2.5:14b
    pause
    exit /b 1
) else (
    echo ✅ 모델 다운로드 완료!
)

REM 8. Python 패키지 설치
echo.
echo 📦 Python 패키지 설치 중...
if exist "pyproject.toml" (
    uv sync
    if errorlevel 1 (
        echo ❌ 패키지 설치 실패!
        pause
        exit /b 1
    )
    echo ✅ 패키지 설치 완료!
) else (
    echo ❌ pyproject.toml 파일을 찾을 수 없습니다.
    pause
    exit /b 1
)

REM 9. 데이터베이스 확인
echo.
echo 💾 데이터베이스 확인 중...
if not exist "data\index\metadata.db" (
    echo ⚠️  데이터베이스가 없습니다.
    echo 📥 샘플 데이터를 다운로드하거나 직접 인덱싱해야 합니다.
    echo.
    echo 옵션 1: 기존 데이터베이스 복사
    echo   - data\index\ 폴더에 metadata.db 파일 복사
    echo.
    echo 옵션 2: 직접 인덱싱 (시간 소요)
    echo   - 명령 프롬프트: uv run python scripts\collect_all.py
    echo   - 명령 프롬프트: uv run python scripts\index.py
    echo.
    choice /C YN /M "나중에 설정하시겠습니까? (Y/N)"
    if errorlevel 2 (
        echo 설치를 중단합니다.
        pause
        exit /b 1
    ) else (
        echo ⏭️  데이터베이스는 나중에 설정하세요.
    )
) else (
    echo ✅ 데이터베이스 준비됨
)

REM 10. 설치 완료
echo.
echo ============================================
echo    ✅ 설치 완료!
echo ============================================
echo.
echo 🚀 실행 방법:
echo   1. 'start-sophiaai.bat' 파일을 더블클릭
echo   또는
echo   2. 명령 프롬프트: start-sophiaai.bat
echo.
echo 📚 데이터베이스 설정이 필요하면:
echo   uv run python scripts\index.py
echo.
echo 💡 도움말:
echo   README.md 파일 참조
echo.
pause
