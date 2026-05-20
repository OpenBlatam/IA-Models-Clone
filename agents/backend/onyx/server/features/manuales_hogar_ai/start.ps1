# PowerShell start script - Run everything with one command
# Usage: .\start.ps1 [dev|prod]

param(
    [string]$Environment = "dev"
)

Write-Host "🚀 Starting Manuales Hogar AI..." -ForegroundColor Green
Write-Host "Environment: $Environment" -ForegroundColor Cyan

# Check if Docker is running
try {
    docker info | Out-Null
} catch {
    Write-Host "❌ Docker is not running. Please start Docker and try again." -ForegroundColor Red
    exit 1
}

# Check if .env file exists
if (-not (Test-Path .env)) {
    Write-Host "⚠️  .env file not found. Creating from .env.example..." -ForegroundColor Yellow
    if (Test-Path .env.example) {
        Copy-Item .env.example .env
        Write-Host "✅ Created .env file. Please edit it with your configuration." -ForegroundColor Green
        Write-Host "   Required: OPENROUTER_API_KEY" -ForegroundColor Yellow
        Read-Host "Press Enter to continue or Ctrl+C to edit .env first"
    } else {
        Write-Host "❌ .env.example not found. Please create .env manually." -ForegroundColor Red
        exit 1
    }
}

# Load environment variables
if (Test-Path .env) {
    Get-Content .env | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]*)=(.*)$') {
            $name = $matches[1].Trim()
            $value = $matches[2].Trim()
            [Environment]::SetEnvironmentVariable($name, $value, "Process")
        }
    }
}

# Check required variables
if ([string]::IsNullOrEmpty($env:OPENROUTER_API_KEY)) {
    Write-Host "⚠️  OPENROUTER_API_KEY not set in .env" -ForegroundColor Yellow
    Write-Host "   The service will start but API calls will fail." -ForegroundColor Yellow
    Read-Host "Press Enter to continue anyway or Ctrl+C to set it first"
}

# Start services
if ($Environment -eq "prod") {
    Write-Host "📦 Starting production environment..." -ForegroundColor Cyan
    docker-compose -f docker-compose.prod.yml up -d --build
} else {
    Write-Host "🔧 Starting development environment..." -ForegroundColor Cyan
    docker-compose up -d --build
}

# Wait for services to be ready
Write-Host "⏳ Waiting for services to be ready..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Check health
$MaxRetries = 30
$RetryCount = 0
$IsHealthy = $false

while ($RetryCount -lt $MaxRetries) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/api/v1/health" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
        if ($response.StatusCode -eq 200) {
            Write-Host "✅ Service is healthy!" -ForegroundColor Green
            $IsHealthy = $true
            break
        }
    } catch {
        # Service not ready yet
    }
    
    $RetryCount++
    Write-Host "   Waiting... ($RetryCount/$MaxRetries)" -ForegroundColor Gray
    Start-Sleep -Seconds 2
}

if (-not $IsHealthy) {
    Write-Host "⚠️  Service may not be ready yet. Check logs with: docker-compose logs" -ForegroundColor Yellow
} else {
    Write-Host ""
    Write-Host "🎉 Manuales Hogar AI is running!" -ForegroundColor Green
    Write-Host ""
    Write-Host "📍 API URL: http://localhost:8000" -ForegroundColor Cyan
    Write-Host "📚 API Docs: http://localhost:8000/docs" -ForegroundColor Cyan
    Write-Host "❤️  Health: http://localhost:8000/api/v1/health" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "📋 Useful commands:" -ForegroundColor Yellow
    Write-Host "   View logs:    docker-compose logs -f"
    Write-Host "   Stop:         docker-compose down"
    Write-Host "   Restart:      docker-compose restart"
    Write-Host "   Shell:        docker-compose exec app bash"
    Write-Host ""
}




