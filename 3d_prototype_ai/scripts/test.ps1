# Script de testing automatizado para 3D Prototype AI (PowerShell)

param(
    [switch]$Coverage = $false,
    [switch]$Verbose = $true
)

Write-Host "🧪 Ejecutando tests de 3D Prototype AI..." -ForegroundColor Yellow

# Activar entorno virtual si existe
if (Test-Path "venv") {
    & .\venv\Scripts\Activate.ps1
}

# Instalar dependencias de testing
Write-Host "Instalando dependencias de testing..." -ForegroundColor Yellow
pip install -q pytest pytest-asyncio pytest-cov pytest-mock

# Ejecutar tests
if ($Coverage) {
    Write-Host "Ejecutando tests con coverage..." -ForegroundColor Yellow
    pytest tests/ -v --cov=. --cov-report=html --cov-report=term
    Write-Host "✅ Coverage report generado en htmlcov/index.html" -ForegroundColor Green
} else {
    if ($Verbose) {
        pytest tests/ -v
    } else {
        pytest tests/
    }
}

# Verificar linting
Write-Host "Verificando linting..." -ForegroundColor Yellow
if (Get-Command flake8 -ErrorAction SilentlyContinue) {
    flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
}

# Verificar tipos
Write-Host "Verificando tipos..." -ForegroundColor Yellow
if (Get-Command mypy -ErrorAction SilentlyContinue) {
    mypy . --ignore-missing-imports
}

Write-Host "✅ Tests completados!" -ForegroundColor Green




