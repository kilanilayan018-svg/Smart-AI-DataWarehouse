@echo off
echo Starting Smart AI DataWarehouse...

start "Backend" cmd /k "cd backend && .venv\Scripts\activate && python -m uvicorn app.main:app --reload"

start "Frontend" cmd /k "cd frontend && npm run dev"

echo Backend: http://localhost:8000/docs
echo Frontend: http://localhost:3000
pause