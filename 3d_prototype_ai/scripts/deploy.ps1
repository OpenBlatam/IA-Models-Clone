# Script de deployment automatizado para 3D Prototype AI (PowerShell)

param(
    [string]$Environment = "production",
    [string]$Version = "latest"
)

Write-Host "🚀 Iniciando deployment de 3D Prototype AI..." -ForegroundColor Yellow
Write-Host "Environment: $Environment" -ForegroundColor Yellow
Write-Host "Version: $Version" -ForegroundColor Yellow

# Verificar dependencias
Write-Host "Verificando dependencias..." -ForegroundColor Yellow
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "Python no encontrado" -ForegroundColor Red
    exit 1
}

# Crear entorno virtual si no existe
if (-not (Test-Path "venv")) {
    Write-Host "Creando entorno virtual..." -ForegroundColor Yellow
    python -m venv venv
}

# Activar entorno virtual
Write-Host "Activando entorno virtual..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1

# Instalar dependencias
Write-Host "Instalando dependencias..." -ForegroundColor Yellow
python -m pip install --upgrade pip
pip install -r requirements.txt

# Ejecutar tests
Write-Host "Ejecutando tests..." -ForegroundColor Yellow
pytest tests/ -v
if ($LASTEXITCODE -ne 0) {
    Write-Host "Tests fallaron, continuando..." -ForegroundColor Yellow
}

# Crear directorios necesarios
Write-Host "Creando directorios..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "storage\prototypes" | Out-Null
New-Item -ItemType Directory -Force -Path "storage\backups" | Out-Null
New-Item -ItemType Directory -Force -Path "logs" | Out-Null

# Verificar configuración
Write-Host "Verificando configuración..." -ForegroundColor Yellow
if (-not (Test-Path ".env")) {
    Write-Host "Creando archivo .env desde template..." -ForegroundColor Yellow
    if (Test-Path ".env.example") {
        Copy-Item ".env.example" ".env"
    }
}

# Configurar variables de entorno
if ($Environment -eq "production") {
    $env:DEBUG = "false"
    $env:LOG_LEVEL = "INFO"
} else {
    $env:DEBUG = "true"
    $env:LOG_LEVEL = "DEBUG"
}

Write-Host "✅ Deployment completado!" -ForegroundColor Green
Write-Host "Iniciando servidor..." -ForegroundColor Green

if ($Environment -eq "production") {
    gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8030
} else {
    uvicorn main:app --host 0.0.0.0 --port 8030 --reload
}




