# PowerShell stop script
# Usage: .\stop.ps1

Write-Host "🛑 Stopping Manuales Hogar AI..." -ForegroundColor Yellow

docker-compose down

Write-Host "✅ Services stopped" -ForegroundColor Green




