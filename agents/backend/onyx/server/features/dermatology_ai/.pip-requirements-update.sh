#!/bin/bash
# Script para actualizar y gestionar dependencias de Dermatology AI

set -e

echo "=========================================="
echo "Dermatology AI - Dependency Manager"
echo "=========================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Función para mostrar ayuda
show_help() {
    echo "Uso: $0 [comando]"
    echo ""
    echo "Comandos disponibles:"
    echo "  install          - Instalar dependencias de producción"
    echo "  install-dev      - Instalar dependencias de desarrollo"
    echo "  install-min      - Instalar dependencias mínimas"
    echo "  install-opt      - Instalar dependencias optimizadas"
    echo "  update           - Actualizar todas las dependencias"
    echo "  outdated         - Mostrar dependencias desactualizadas"
    echo "  check            - Verificar vulnerabilidades"
    echo "  compile          - Generar requirements-lock.txt"
    echo "  clean            - Limpiar cache de pip"
    echo "  size             - Mostrar tamaño de dependencias"
    echo "  help             - Mostrar esta ayuda"
}

# Instalar dependencias
install_prod() {
    echo -e "${GREEN}Instalando dependencias de producción...${NC}"
    pip install -r requirements.txt
    echo -e "${GREEN}✓ Instalación completada${NC}"
}

install_dev() {
    echo -e "${GREEN}Instalando dependencias de desarrollo...${NC}"
    pip install -r requirements-dev.txt
    echo -e "${GREEN}✓ Instalación completada${NC}"
}

install_min() {
    echo -e "${GREEN}Instalando dependencias mínimas...${NC}"
    pip install -r requirements-minimal.txt
    echo -e "${GREEN}✓ Instalación completada${NC}"
}

install_opt() {
    echo -e "${GREEN}Instalando dependencias optimizadas...${NC}"
    pip install -r requirements-optimized.txt
    echo -e "${GREEN}✓ Instalación completada${NC}"
}

# Actualizar dependencias
update_deps() {
    echo -e "${YELLOW}Actualizando dependencias...${NC}"
    pip list --outdated
    echo ""
    read -p "¿Actualizar todas las dependencias? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pip install --upgrade -r requirements.txt
        echo -e "${GREEN}✓ Dependencias actualizadas${NC}"
    fi
}

# Mostrar dependencias desactualizadas
show_outdated() {
    echo -e "${YELLOW}Dependencias desactualizadas:${NC}"
    pip list --outdated
}

# Verificar vulnerabilidades
check_vulnerabilities() {
    echo -e "${YELLOW}Verificando vulnerabilidades...${NC}"
    
    if command -v safety &> /dev/null; then
        safety check -r requirements.txt
    else
        echo -e "${RED}safety no está instalado. Instalando...${NC}"
        pip install safety
        safety check -r requirements.txt
    fi
    
    if command -v pip-audit &> /dev/null; then
        echo ""
        echo -e "${YELLOW}Verificando con pip-audit...${NC}"
        pip-audit -r requirements.txt
    fi
}

# Compilar requirements-lock.txt
compile_lock() {
    echo -e "${YELLOW}Generando requirements-lock.txt...${NC}"
    
    if ! command -v pip-compile &> /dev/null; then
        echo -e "${YELLOW}Instalando pip-tools...${NC}"
        pip install pip-tools
    fi
    
    pip-compile requirements.txt -o requirements-lock.txt
    echo -e "${GREEN}✓ requirements-lock.txt generado${NC}"
}

# Limpiar cache
clean_cache() {
    echo -e "${YELLOW}Limpiando cache de pip...${NC}"
    pip cache purge
    echo -e "${GREEN}✓ Cache limpiado${NC}"
}

# Mostrar tamaño
show_size() {
    echo -e "${YELLOW}Tamaño estimado de dependencias:${NC}"
    echo ""
    echo "requirements.txt:        ~2-3 GB"
    echo "requirements-optimized.txt: ~500 MB"
    echo "requirements-dev.txt:    ~3-4 GB"
    echo "requirements-minimal.txt: ~50 MB"
}

# Procesar comando
case "${1:-help}" in
    install)
        install_prod
        ;;
    install-dev)
        install_dev
        ;;
    install-min)
        install_min
        ;;
    install-opt)
        install_opt
        ;;
    update)
        update_deps
        ;;
    outdated)
        show_outdated
        ;;
    check)
        check_vulnerabilities
        ;;
    compile)
        compile_lock
        ;;
    clean)
        clean_cache
        ;;
    size)
        show_size
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo -e "${RED}Comando desconocido: $1${NC}"
        echo ""
        show_help
        exit 1
        ;;
esac



