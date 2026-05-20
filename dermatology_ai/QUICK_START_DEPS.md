# Quick Start - Dependencias

## 🚀 Inicio Rápido

### Opción 1: Setup Automático (Recomendado)

```bash
# Linux/Mac
./scripts/setup-dev-environment.sh

# Windows
# Ejecutar en PowerShell o usar Makefile
make install-dev
```

### Opción 2: Manual

```bash
# 1. Crear entorno virtual
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate     # Windows

# 2. Instalar dependencias
pip install -r requirements-dev.txt

# 3. Configurar pre-commit
pre-commit install
```

## 📦 Instalación por Caso de Uso

### Producción
```bash
pip install -r requirements.txt
# o
make install
```

### Producción Optimizada (Recomendado)
```bash
pip install -r requirements-optimized.txt
# o
make install-opt
```

### Desarrollo
```bash
pip install -r requirements-dev.txt
# o
make install-dev
```

### GPU Support
```bash
pip install -r requirements-gpu.txt
# o
make install-gpu
```

### Docker
```bash
pip install -r requirements-docker.txt
# o
make install-docker
```

## ✅ Verificación

```bash
# Verificar instalación
make check

# Verificar salud de dependencias
./scripts/check-dependencies.sh

# Ejecutar tests
make test
```

## 📚 Documentación Completa

- `DEPENDENCIES_GUIDE.md` - Guía detallada
- `DEPENDENCY_MANAGEMENT.md` - Gestión completa
- `Makefile` - Todos los comandos disponibles



