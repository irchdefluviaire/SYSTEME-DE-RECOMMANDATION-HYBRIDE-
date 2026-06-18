$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
Set-Location -LiteralPath "D:\DATA SCIENCES\SYSTEME-DE-RECOMMANDATION-HYBRIDE-"
"$(Get-Date -Format o) starting FastAPI UTF-8" | Out-File -FilePath "fastapi-utf8.launcher.log" -Append -Encoding utf8
& ".\.venv\Scripts\python.exe" -m uvicorn src.06_api.main:app --host 0.0.0.0 --port 8000
"$(Get-Date -Format o) FastAPI process exited with code $LASTEXITCODE" | Out-File -FilePath "fastapi-utf8.launcher.log" -Append -Encoding utf8
