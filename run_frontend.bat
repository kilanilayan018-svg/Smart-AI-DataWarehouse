@echo off
setlocal
cd /d "%~dp0"
echo ============================================================
echo Smart AI DataWarehouse - Frontend Website
echo ============================================================
echo.

if not exist "frontend\package.json" (
  echo ERROR: frontend\package.json not found.
  echo Run this BAT from the project root folder.
  pause
  exit /b 1
)

where node >nul 2>nul
if errorlevel 1 (
  echo ERROR: Node.js not found. Install it with:
  echo   winget install OpenJS.NodeJS.LTS
  echo Then close and reopen this terminal.
  pause
  exit /b 1
)

cd frontend

echo Installing frontend packages if needed...
call npm install
if errorlevel 1 (
  echo ERROR: npm install failed.
  pause
  exit /b 1
)

echo.
echo Frontend will run at: http://localhost:3000
echo Make sure backend is running on port 8001 first!
echo Leave this window open.
echo.

call npm run dev

echo.
echo Frontend stopped.
pause
