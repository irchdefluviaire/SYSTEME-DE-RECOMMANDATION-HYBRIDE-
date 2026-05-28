$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$Poetry = "C:\Users\Ultra Tech\AppData\Roaming\pypoetry\venv\Scripts\poetry.exe"
$LogPath = Join-Path $ProjectRoot "streamlit-chatbot.poetry.log"

Set-Location -LiteralPath $ProjectRoot
& $Poetry run streamlit run chatbot_app.py `
    --server.port=8501 `
    --server.address=127.0.0.1 `
    --server.headless=true `
    --browser.gatherUsageStats=false *>> $LogPath
