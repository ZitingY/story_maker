@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

echo ==========================================
echo StoryWeaver Bootstrap ^& Launch (Windows)
echo ==========================================
echo.

REM Create virtual environment if it doesn't exist
if not exist ".venv\Scripts\python.exe" (
    echo [1/5] Creating virtual environment...
    py -3 -m venv .venv 2>nul || python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        exit /b 1
    )
)

set "PYTHON_CMD=.venv\Scripts\python.exe"

echo [2/5] Upgrading pip...
"%PYTHON_CMD%" -m pip install --upgrade pip

echo [3/5] Installing dependencies...
"%PYTHON_CMD%" -m pip install -r requirements.txt "streamlit>=1.30.0"

echo [4/5] Downloading spaCy model (optional)...
"%PYTHON_CMD%" -m spacy download en_core_web_sm 2>nul || echo [WARN] spaCy download skipped

echo [5/5] Launching app on http://127.0.0.1:7860...
"%PYTHON_CMD%" -m streamlit run app.py --server.port=7860

endlocal
