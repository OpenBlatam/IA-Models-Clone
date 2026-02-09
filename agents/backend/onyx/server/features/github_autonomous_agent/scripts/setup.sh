#!/bin/bash

# ============================================================================
# GitHub Autonomous Agent - Setup Script
# ============================================================================
# Script de instalación y configuración automática
# Uso: ./scripts/setup.sh [--dev] [--prod] [--minimal]
# ============================================================================

set -e  # Exit on error

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Función para imprimir mensajes
info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
    exit 1
}

# Verificar Python
info "Verificando Python..."
if ! command -v python3 &> /dev/null; then
    error "Python 3 no está instalado. Por favor instala Python 3.10+"
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED_VERSION="3.10"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    error "Python 3.10+ requerido. Versión actual: $PYTHON_VERSION"
fi

success "Python $PYTHON_VERSION detectado"

# Verificar pip
info "Verificando pip..."
if ! command -v pip3 &> /dev/null; then
    warning "pip3 no encontrado, intentando instalar..."
    python3 -m ensurepip --upgrade
fi

success "pip detectado"

# Crear entorno virtual si no existe
if [ ! -d "venv" ]; then
    info "Creando entorno virtual..."
    python3 -m venv venv
    success "Entorno virtual creado"
else
    info "Entorno virtual ya existe"
fi

# Activar entorno virtual
info "Activando entorno virtual..."
source venv/bin/activate || error "Error activando entorno virtual"
success "Entorno virtual activado"

# Actualizar pip
info "Actualizando pip..."
pip install --upgrade pip setuptools wheel
success "pip actualizado"

# Determinar qué requirements instalar
INSTALL_TYPE="base"

if [[ "$*" == *"--dev"* ]]; then
    INSTALL_TYPE="dev"
    info "Modo: Desarrollo"
elif [[ "$*" == *"--prod"* ]]; then
    INSTALL_TYPE="prod"
    info "Modo: Producción"
elif [[ "$*" == *"--minimal"* ]]; then
    INSTALL_TYPE="minimal"
    info "Modo: Minimal"
else
    info "Modo: Base (usa --dev para desarrollo o --prod para producción)"
fi

# Instalar dependencias
info "Instalando dependencias..."

case $INSTALL_TYPE in
    dev)
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
        success "Dependencias de desarrollo instaladas"
        ;;
    prod)
        pip install -r requirements.txt
        pip install -r requirements-prod.txt
        success "Dependencias de producción instaladas"
        ;;
    minimal)
        pip install -r requirements.txt
        success "Dependencias mínimas instaladas"
        ;;
    *)
        pip install -r requirements.txt
        success "Dependencias base instaladas"
        ;;
esac

# Verificar .env
if [ ! -f ".env" ]; then
    warning ".env no encontrado"
    if [ -f ".env.example" ]; then
        info "Copiando .env.example a .env..."
        cp .env.example .env
        success ".env creado desde .env.example"
        warning "⚠️  IMPORTANTE: Edita .env con tus credenciales antes de continuar"
    else
        warning "No se encontró .env.example. Crea un archivo .env manualmente."
    fi
else
    success ".env encontrado"
fi

# Crear directorios necesarios
info "Creando directorios necesarios..."
mkdir -p storage/tasks
mkdir -p storage/logs
mkdir -p storage/cache
success "Directorios creados"

# Verificar Redis (opcional)
if command -v redis-cli &> /dev/null; then
    if redis-cli ping &> /dev/null; then
        success "Redis está corriendo"
    else
        warning "Redis no está corriendo. Inicia Redis para usar Celery: redis-server"
    fi
else
    warning "Redis no está instalado. Instálalo para usar Celery."
fi

# Verificar Git
if command -v git &> /dev/null; then
    success "Git detectado"
else
    warning "Git no está instalado. Necesario para operaciones Git."
fi

# Verificar dependencias críticas
info "Verificando dependencias críticas instaladas..."
python3 -c "import fastapi, uvicorn, pydantic, PyGithub, gitpython, celery, redis, sqlalchemy" 2>/dev/null
if [ $? -eq 0 ]; then
    success "Todas las dependencias críticas están instaladas"
else
    error "Algunas dependencias críticas no están instaladas correctamente"
fi

# Resumen
echo ""
echo "════════════════════════════════════════════════════════════"
success "🎉 Instalación completada!"
echo "════════════════════════════════════════════════════════════"
echo ""
info "Próximos pasos:"
echo "  1. Edita .env con tus credenciales de GitHub"
echo "  2. Inicia Redis si usas Celery: redis-server"
echo "  3. Ejecuta la aplicación: python main.py"
echo ""
info "Para activar el entorno virtual en el futuro:"
echo "  source venv/bin/activate"
echo ""
info "Para desarrollo, instala dependencias adicionales:"
echo "  pip install -r requirements-dev.txt"
echo ""




