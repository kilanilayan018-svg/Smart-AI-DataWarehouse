@echo off
setlocal
cd /d "%~dp0"
echo Checking backend health...
powershell -NoProfile -Command "try { Invoke-RestMethod http://127.0.0.1:8000/health | ConvertTo-Json -Depth 5 } catch { Write-Host $_.Exception.Message; exit 1 }"
pause
