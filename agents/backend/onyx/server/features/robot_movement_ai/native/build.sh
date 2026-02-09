#!/bin/bash
# Build script para extensiones nativas
# ======================================

set -e  # Salir si hay error

echo "=========================================="
echo "Building Native Extensions for Robot Movement AI"
echo "=========================================="

# Colores para output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Función para imprimir mensajes
info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Verificar Python
if ! command -v python3 &> /dev/null; then
    error "Python 3 no encontrado. Por favor instala Python 3.8+"
    exit 1
fi

info "Python encontrado: $(python3 --version)"

# Verificar pybind11
if ! python3 -c "import pybind11" 2>/dev/null; then
    warn "pybind11 no encontrado. Instalando..."
    pip install pybind11
fi

# Compilar extensiones C++
info "Compilando extensiones C++..."
cd "$(dirname "$0")"

if [ -f "cpp_extensions.cpp" ]; then
    python3 setup.py build_ext --inplace
    if [ $? -eq 0 ]; then
        info "Extensiones C++ compiladas exitosamente"
    else
        warn "Error compilando extensiones C++. Continuando sin ellas..."
    fi
else
    warn "cpp_extensions.cpp no encontrado. Saltando compilación C++..."
fi

# Compilar extensiones Rust
if command -v cargo &> /dev/null; then
    info "Compilando extensiones Rust..."
    cd rust_extensions
    
    if [ -f "Cargo.toml" ]; then
        if command -v maturin &> /dev/null; then
            maturin develop --release
            if [ $? -eq 0 ]; then
                info "Extensiones Rust compiladas exitosamente"
            else
                warn "Error compilando extensiones Rust. Continuando sin ellas..."
            fi
        else
            warn "maturin no encontrado. Instalando..."
            pip install maturin
            maturin develop --release
        fi
    else
        warn "Cargo.toml no encontrado. Saltando compilación Rust..."
    fi
    cd ..
else
    warn "Rust no encontrado. Saltando compilación Rust..."
    warn "Para instalar Rust: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
fi

echo ""
echo "=========================================="
info "Build completado!"
echo "=========================================="
echo ""
echo "Para verificar la instalación, ejecuta:"
echo "  python3 -c \"from robot_movement_ai.native import CPP_AVAILABLE, RUST_AVAILABLE; print(f'C++: {CPP_AVAILABLE}, Rust: {RUST_AVAILABLE}')\""

