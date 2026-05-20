# Quick start script with debugging (PowerShell)

Write-Host "🚀 Quick Start - API with Debugging" -ForegroundColor Green
Write-Host ""

# Check if API is already running
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -Method Get -TimeoutSec 2 -ErrorAction SilentlyContinue
    if ($response.StatusCode -eq 200) {
        Write-Host "✅ API is already running on http://localhost:8000" -ForegroundColor Green
        Write-Host ""
        Write-Host "You can now:"
        Write-Host "  - Use debug tool: python debug_api.py"
        Write-Host "  - View docs: http://localhost:8000/docs"
        Write-Host "  - View ReDoc: http://localhost:8000/redoc"
        Write-Host ""
        Read-Host "Press Enter to open debug tool or Ctrl+C to exit"
        python debug_api.py
    }
} catch {
    Write-Host "Starting API with debugging..." -ForegroundColor Yellow
    Write-Host ""
    python run_api_debug.py
}



