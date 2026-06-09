@echo off
setlocal
cd /d "%~dp0"
echo ============================================================
echo Smart AI DataWarehouse - Start Website + Backend
echo ============================================================
echo.
echo This opens TWO windows:
echo   1. Backend API  - http://127.0.0.1:8000/docs
echo   2. Frontend UI  - http://localhost:3000
echo.
echo Model API is external/ngrok and already configured in backend\.env.
echo Supabase is optional and currently uses local SQLite demo fallback.
echo.

start "Smart AI DW Backend" cmd /k "cd /d %~dp0 && call run_backend.bat"
timeout /t 5 /nobreak >nul
start "Smart AI DW Frontend" cmd /k "cd /d %~dp0 && call run_frontend.bat"

echo.
echo Open after both windows finish installing:
echo   http://localhost:3000

echo.
pause
