@echo off
setlocal
cd /d "%~dp0"
echo ============================================================
echo Smart AI DataWarehouse - Backend API
echo ============================================================
echo.

if not exist "backend\app\main.py" (
  echo ERROR: backend\app\main.py not found.
  echo Run this BAT from the project root folder.
  pause
  exit /b 1
)

set "PYTHONPATH=%CD%"

if not exist "backend\.venv\Scripts\python.exe" (
  echo Creating backend virtual environment...
  python -m venv backend\.venv
  if errorlevel 1 (
    echo Python failed. Trying py launcher...
    py -m venv backend\.venv
    if errorlevel 1 (
      echo ERROR: Could not create virtual environment. Install Python 3.10+ and tick Add to PATH.
      pause
      exit /b 1
    )
  )
  echo Installing requirements...
  call backend\.venv\Scripts\activate
  python -m pip install --upgrade pip
  python -m pip install -r backend\requirements.txt
) else (
  call backend\.venv\Scripts\activate
)

echo.
echo Backend API will run at: http://127.0.0.1:8001
echo API docs:                 http://127.0.0.1:8001/docs
echo Health check:             http://127.0.0.1:8001/health
echo.
echo Leave this window open.
echo.

python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8001

echo.
echo Backend stopped.
pause
