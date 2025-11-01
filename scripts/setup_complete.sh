#!/bin/bash
# Setup Completo del Sistema Blatam Academy Features
# Este script configura todo el sistema desde cero

set -e

echo "🚀 Blatam Academy Features - Setup Completo"
echo "============================================"

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Función para imprimir mensajes
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

# Verificar prerrequisitos
print_info "Verificando prerrequisitos..."

# Verificar Docker
if ! command -v docker &> /dev/null; then
    print_error "Docker no está instalado"
    exit 1
fi
print_success "Docker encontrado: $(docker --version)"

# Verificar Docker Compose
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose no está instalado"
    exit 1
fi
print_success "Docker Compose encontrado: $(docker-compose --version)"

# Verificar Python
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 no está instalado"
    exit 1
fi
print_success "Python encontrado: $(python3 --version)"

# Crear archivo .env si no existe
if [ ! -f .env ]; then
    print_info "Creando archivo .env..."
    python3 start_system.py create-env
    print_success "Archivo .env creado"
else
    print_info "Archivo .env ya existe"
fi

# Construir imágenes Docker
print_info "Construyendo imágenes Docker..."
docker-compose build
print_success "Imágenes construidas"

# Iniciar servicios
print_info "Iniciando servicios..."
docker-compose up -d
print_success "Servicios iniciados"

# Esperar a que los servicios estén listos
print_info "Esperando a que los servicios estén listos..."
sleep 10

# Verificar estado
print_info "Verificando estado de servicios..."
python3 start_system.py status

print_success "Setup completo finalizado!"
print_info "Documentación disponible en:"
echo "  - README.md"
echo "  - QUICK_START_GUIDE.md"
echo "  - DOCUMENTATION_INDEX.md"
echo ""
print_info "Accesos:"
echo "  - API Principal: http://localhost:8000"
echo "  - API Docs: http://localhost:8000/docs"
echo "  - Grafana: http://localhost:3000"
echo "  - Prometheus: http://localhost:9090"

