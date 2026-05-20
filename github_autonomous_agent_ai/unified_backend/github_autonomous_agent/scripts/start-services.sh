#!/bin/bash

# ============================================================================
# GitHub Autonomous Agent - Start Services Script
# ============================================================================
# Inicia todos los servicios necesarios (Redis, etc.)
# ============================================================================

set -e

# Colores
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
success() { echo -e "${GREEN}✅ $1${NC}"; }
warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
error() { echo -e "${RED}❌ $1${NC}"; }

info "Iniciando servicios para GitHub Autonomous Agent..."

# Verificar Redis
if command -v redis-cli &> /dev/null; then
    if redis-cli ping &> /dev/null; then
        success "Redis ya está corriendo"
    else
        info "Iniciando Redis..."
        redis-server --daemonize yes
        sleep 2
        if redis-cli ping &> /dev/null; then
            success "Redis iniciado"
        else
            error "No se pudo iniciar Redis"
            exit 1
        fi
    fi
else
    warning "Redis no está instalado. Instálalo para usar Celery."
fi

# Verificar PostgreSQL (opcional)
if command -v psql &> /dev/null; then
    info "PostgreSQL detectado (opcional)"
else
    info "PostgreSQL no detectado (usando SQLite por defecto)"
fi

success "Servicios listos!"
info "Ahora puedes ejecutar: python main.py"




