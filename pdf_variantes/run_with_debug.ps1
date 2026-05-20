# Run API with debugging enabled (PowerShell)

Write-Host "🚀 Starting API with debugging..." -ForegroundColor Green
Write-Host ""

# Set debugging environment variables
$env:DEBUG = "true"
$env:LOG_LEVEL = "debug"
$env:DETAILED_ERRORS = "true"
$env:LOG_REQUESTS = "true"
$env:LOG_RESPONSES = "true"
$env:ENABLE_METRICS = "true"
$env:ENABLE_PROFILING = "true"

# Run the API
python run_api_debug.py



