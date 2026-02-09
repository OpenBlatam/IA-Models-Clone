# PowerShell status script
# Usage: .\status.ps1

Write-Host "📊 Manuales Hogar AI - Service Status" -ForegroundColor Cyan
Write-Host ""

# Check Docker
try {
    docker info | Out-Null
} catch {
    Write-Host "❌ Docker is not running" -ForegroundColor Red
    exit 1
}

# Check which compose file to use
$prodRunning = docker-compose -f docker-compose.prod.yml ps 2>$null | Select-String "Up"
$devRunning = docker-compose ps 2>$null | Select-String "Up"

if ($prodRunning) {
    $ComposeFile = "-f docker-compose.prod.yml"
    $Env = "production"
    $DockerCompose = "docker-compose"
} elseif ($devRunning) {
    $ComposeFile = ""
    $Env = "development"
    $DockerCompose = "docker-compose"
} else {
    Write-Host "⚠️  No services are running" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Start services with: .\start.ps1"
    exit 0
}

Write-Host "Environment: $Env" -ForegroundColor Cyan
Write-Host ""

# Show container status
Write-Host "Container Status:" -ForegroundColor Cyan
docker-compose $ComposeFile ps
Write-Host ""

# Check health endpoint
Write-Host "Health Check:" -ForegroundColor Cyan
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/api/v1/health" -UseBasicParsing -TimeoutSec 2
    Write-Host "✅ Service is healthy" -ForegroundColor Green
    $response.Content | ConvertFrom-Json | ConvertTo-Json -Depth 10
} catch {
    Write-Host "❌ Service is not responding" -ForegroundColor Red
}
Write-Host ""

# Show resource usage
Write-Host "Resource Usage:" -ForegroundColor Cyan
$containerIds = docker-compose $ComposeFile ps -q
if ($containerIds) {
    docker stats --no-stream $containerIds
}
Write-Host ""

# Show recent logs
Write-Host "Recent Logs (last 5 lines):" -ForegroundColor Cyan
docker-compose $ComposeFile logs --tail=5 app 2>$null
Write-Host ""




