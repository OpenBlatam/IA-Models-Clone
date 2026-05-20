# Deployment Script for Bulk TruthGPT (PowerShell)
# ================================================

Write-Host "🚀 Starting deployment..." -ForegroundColor Green

# Check Python version
$pythonVersion = python --version 2>&1
Write-Host "Python version: $pythonVersion" -ForegroundColor Cyan

# Install dependencies
Write-Host "📦 Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

# Setup environment
Write-Host "⚙️  Setting up environment..." -ForegroundColor Yellow
if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
    Write-Host "✅ Created .env file from .env.example" -ForegroundColor Green
}

# Create directories
Write-Host "📁 Creating directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path storage, backups, logs, cache, temp | Out-Null

# Run setup
Write-Host "🔧 Running setup..." -ForegroundColor Yellow
python setup.py

# Verify setup
Write-Host "✅ Verifying setup..." -ForegroundColor Yellow
python verify_setup.py

Write-Host "`n🎉 Deployment completed successfully!`n" -ForegroundColor Green
Write-Host "To start the service:" -ForegroundColor Cyan
Write-Host "  python start.py" -ForegroundColor White
Write-Host "  or" -ForegroundColor White
Write-Host "  uvicorn main:app --host 0.0.0.0 --port 8000" -ForegroundColor White
















