# ============================================================================
# GitHub Autonomous Agent - Setup Script (PowerShell)
# ============================================================================
# Script de instalación y configuración automática para Windows
# Uso: .\scripts\setup.ps1 [-Dev] [-Prod] [-Minimal]
# ============================================================================

param(
    [switch]$Dev,
    [switch]$Prod,
    [switch]$Minimal
)

$ErrorActionPreference = "Stop"

# Colores para output
function Write-Info {
    Write-Host "ℹ️  $args" -ForegroundColor Blue
}

function Write-Success {
    Write-Host "✅ $args" -ForegroundColor Green
}

function Write-Warning {
    Write-Host "⚠️  $args" -ForegroundColor Yellow
}

function Write-Error {
    Write-Host "❌ $args" -ForegroundColor Red
    exit 1
}

# Verificar Python
Write-Info "Verificando Python..."
try {
    $pythonVersion = python --version 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Python no encontrado"
    }
    Write-Success "Python detectado: $pythonVersion"
} catch {
    Write-Error "Python 3 no está instalado. Por favor instala Python 3.10+ desde python.org"
}

# Verificar versión de Python
$version = python --version 2>&1 | ForEach-Object { $_ -replace 'Python ', '' }
$major = [int]($version -split '\.')[0]
$minor = [int]($version -split '\.')[1]

if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 10)) {
    Write-Error "Python 3.10+ requerido. Versión actual: $version"
}

# Verificar pip
Write-Info "Verificando pip..."
try {
    python -m pip --version | Out-Null
    Write-Success "pip detectado"
} catch {
    Write-Warning "pip no encontrado, intentando instalar..."
    python -m ensurepip --upgrade
}

# Crear entorno virtual si no existe
if (-not (Test-Path "venv")) {
    Write-Info "Creando entorno virtual..."
    python -m venv venv
    Write-Success "Entorno virtual creado"
} else {
    Write-Info "Entorno virtual ya existe"
}

# Activar entorno virtual
Write-Info "Activando entorno virtual..."
& .\venv\Scripts\Activate.ps1
if ($LASTEXITCODE -ne 0) {
    Write-Error "Error activando entorno virtual"
}
Write-Success "Entorno virtual activado"

# Actualizar pip
Write-Info "Actualizando pip..."
python -m pip install --upgrade pip setuptools wheel
Write-Success "pip actualizado"

# Determinar qué requirements instalar
$installType = "base"

if ($Dev) {
    $installType = "dev"
    Write-Info "Modo: Desarrollo"
} elseif ($Prod) {
    $installType = "prod"
    Write-Info "Modo: Producción"
} elseif ($Minimal) {
    $installType = "minimal"
    Write-Info "Modo: Minimal"
} else {
    Write-Info "Modo: Base (usa -Dev para desarrollo o -Prod para producción)"
}

# Instalar dependencias
Write-Info "Instalando dependencias..."

switch ($installType) {
    "dev" {
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
        Write-Success "Dependencias de desarrollo instaladas"
    }
    "prod" {
        pip install -r requirements.txt
        pip install -r requirements-prod.txt
        Write-Success "Dependencias de producción instaladas"
    }
    "minimal" {
        pip install -r requirements.txt
        Write-Success "Dependencias mínimas instaladas"
    }
    default {
        pip install -r requirements.txt
        Write-Success "Dependencias base instaladas"
    }
}

# Verificar .env
if (-not (Test-Path ".env")) {
    Write-Warning ".env no encontrado"
    if (Test-Path ".env.example") {
        Write-Info "Copiando .env.example a .env..."
        Copy-Item .env.example .env
        Write-Success ".env creado desde .env.example"
        Write-Warning "⚠️  IMPORTANTE: Edita .env con tus credenciales antes de continuar"
    } else {
        Write-Warning "No se encontró .env.example. Crea un archivo .env manualmente."
    }
} else {
    Write-Success ".env encontrado"
}

# Crear directorios necesarios
Write-Info "Creando directorios necesarios..."
@("storage/tasks", "storage/logs", "storage/cache") | ForEach-Object {
    if (-not (Test-Path $_)) {
        New-Item -ItemType Directory -Path $_ -Force | Out-Null
    }
}
Write-Success "Directorios creados"

# Verificar Redis (opcional)
try {
    $redisCheck = redis-cli ping 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Redis está corriendo"
    } else {
        Write-Warning "Redis no está corriendo. Inicia Redis para usar Celery"
    }
} catch {
    Write-Warning "Redis no está instalado o no está en PATH. Instálalo para usar Celery."
}

# Verificar Git
try {
    git --version | Out-Null
    Write-Success "Git detectado"
} catch {
    Write-Warning "Git no está instalado. Necesario para operaciones Git."
}

# Verificar dependencias críticas
Write-Info "Verificando dependencias críticas instaladas..."
try {
    python -c "import fastapi, uvicorn, pydantic, PyGithub, gitpython, celery, redis, sqlalchemy" 2>&1 | Out-Null
    Write-Success "Todas las dependencias críticas están instaladas"
} catch {
    Write-Error "Algunas dependencias críticas no están instaladas correctamente"
}

# Resumen
Write-Host ""
Write-Host "════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Success "🎉 Instalación completada!"
Write-Host "════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Info "Próximos pasos:"
Write-Host "  1. Edita .env con tus credenciales de GitHub"
Write-Host "  2. Inicia Redis si usas Celery: redis-server"
Write-Host "  3. Ejecuta la aplicación: python main.py"
Write-Host ""
Write-Info "Para activar el entorno virtual en el futuro:"
Write-Host "  .\venv\Scripts\Activate.ps1"
Write-Host ""
Write-Info "Para desarrollo, instala dependencias adicionales:"
Write-Host "  pip install -r requirements-dev.txt"
Write-Host ""




