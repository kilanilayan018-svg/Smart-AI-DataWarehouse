@echo off
setlocal
cd /d "%~dp0"
echo Testing configured MODEL_API_URL from backend\.env...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$envFile='backend\.env'; $url=(Get-Content $envFile | Where-Object {$_ -match '^MODEL_API_URL='}) -replace '^MODEL_API_URL=',''; if(-not $url){Write-Host 'MODEL_API_URL missing'; exit 1}; $body=@{schema_description='Dataset: test.csv. Shape: 10 rows x 3 columns. Prediction target: label. Column details: - age: int64, 10 unique, 0%% missing - city: object, 3 unique, 0%% missing'; max_new_tokens=128} | ConvertTo-Json; try { Invoke-RestMethod -Uri $url -Method Post -ContentType 'application/json' -Body $body | ConvertTo-Json -Depth 8 } catch { Write-Host $_.Exception.Message; exit 1 }"
pause
