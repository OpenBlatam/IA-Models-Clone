# Gestión de Dependencias - Guía Completa

## 🎯 Visión General

Este proyecto utiliza un sistema completo de gestión de dependencias con múltiples archivos, herramientas de automatización y mejores prácticas.

## 📦 Archivos de Dependencias

### Archivos Principales

1. **`requirements.txt`** - Producción completa (~2-3 GB)
2. **`requirements-optimized.txt`** - Producción optimizada (~500 MB)
3. **`requirements-dev.txt`** - Desarrollo completo (~3-4 GB)
4. **`requirements-minimal.txt`** - Mínimo para CI (~50 MB)
5. **`requirements-gpu.txt`** - Con soporte GPU (~4-5 GB)
6. **`requirements-docker.txt`** - Optimizado para Docker (~600 MB)
7. **`requirements-lock.txt`** - Versiones fijas (generado)

## 🛠️ Herramientas Disponibles

### 1. Makefile (Recomendado)

```bash
# Instalación
make install          # Producción completa
make install-opt      # Optimizado
make install-dev      # Desarrollo
make install-gpu      # Con GPU
make install-docker   # Docker

# Gestión
make update           # Actualizar dependencias
make outdated         # Ver desactualizadas
make check            # Verificar vulnerabilidades
make compile          # Generar lock file
make clean            # Limpiar cache

# Desarrollo
make test             # Ejecutar tests
make test-cov         # Tests con cobertura
make lint             # Linters
make format           # Formatear código
make type-check       # Type checking
make pre-commit       # Todo antes de commit
```

### 2. Scripts de Gestión

#### Linux/Mac:
```bash
# Script principal
./.pip-requirements-update.sh install
./.pip-requirements-update.sh check
./.pip-requirements-update.sh compile

# Scripts adicionales
./scripts/check-dependencies.sh
./scripts/update-dependencies.sh
./scripts/analyze-dependencies.py
./scripts/setup-dev-environment.sh
```

#### Windows:
```cmd
.pip-requirements-update.bat install
.pip-requirements-update.bat check
.pip-requirements-update.bat compile
```

### 3. Pre-commit Hooks

Instalación automática de hooks de Git:

```bash
pre-commit install
```

Hooks incluidos:
- ✅ Formateo de código (Black)
- ✅ Linting (Ruff)
- ✅ Type checking (MyPy)
- ✅ Security checks (Bandit, Safety)
- ✅ Validación de archivos (YAML, JSON, TOML)
- ✅ Validación de commits

### 4. Dependabot

Actualización automática de dependencias vía GitHub:

- Actualización semanal
- Agrupación de updates
- Pull requests automáticos
- Revisión de seguridad

## 🔄 Flujos de Trabajo

### Desarrollo Local

```bash
# 1. Setup inicial
./scripts/setup-dev-environment.sh

# 2. Activar entorno
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate     # Windows

# 3. Trabajar normalmente
make test
make format
make lint
```

### Actualización de Dependencias

```bash
# Opción 1: Script automatizado (recomendado)
./scripts/update-dependencies.sh

# Opción 2: Manual
make outdated
make update
make check
make test
```

### Verificación de Seguridad

```bash
# Verificación completa
make check

# O manualmente
safety check -r requirements.txt
pip-audit -r requirements.txt
```

### Análisis de Dependencias

```bash
# Análisis completo
python scripts/analyze-dependencies.py

# Verificar salud
./scripts/check-dependencies.sh
```

## 📊 Monitoreo y Mantenimiento

### Checklist Semanal

- [ ] Revisar dependencias desactualizadas: `make outdated`
- [ ] Verificar vulnerabilidades: `make check`
- [ ] Actualizar si es necesario: `make update`
- [ ] Ejecutar tests: `make test`
- [ ] Revisar PRs de Dependabot

### Checklist Mensual

- [ ] Revisar y actualizar `requirements-lock.txt`
- [ ] Analizar dependencias: `python scripts/analyze-dependencies.py`
- [ ] Revisar tamaño de dependencias
- [ ] Optimizar archivos de requirements si es necesario
- [ ] Actualizar documentación

### Checklist Antes de Release

- [ ] Todas las dependencias actualizadas
- [ ] Sin vulnerabilidades conocidas
- [ ] `requirements-lock.txt` actualizado
- [ ] Tests pasando
- [ ] Documentación actualizada
- [ ] Tamaño de imagen Docker verificado

## 🔒 Seguridad

### Verificación Regular

```bash
# Verificación básica
safety check -r requirements.txt

# Verificación avanzada
pip-audit -r requirements.txt --desc

# Verificación completa
make check
```

### Actualización de Seguridad

```bash
# Actualizar paquetes críticos
pip install --upgrade cryptography python-jose passlib

# Regenerar requirements
pip freeze > requirements-new.txt
```

## 🚀 Optimización

### Reducir Tamaño

1. **Usar requirements-optimized.txt** para producción
2. **Multi-stage Docker builds**
3. **Eliminar dependencias no usadas**
4. **Usar ONNX en lugar de PyTorch completo**

### Acelerar Instalación

1. **Usar cache de pip**
2. **Mirrors locales**
3. **Instalación paralela**
4. **Pre-compilar wheels**

## 📝 Mejores Prácticas

### 1. Versionado

- ✅ Usar `>=` para flexibilidad
- ✅ Fijar versiones críticas en producción
- ✅ Mantener `requirements-lock.txt` actualizado

### 2. Organización

- ✅ Separar dev y prod
- ✅ Usar archivos específicos (gpu, docker)
- ✅ Documentar dependencias opcionales

### 3. Seguridad

- ✅ Verificar regularmente
- ✅ Actualizar rápidamente vulnerabilidades
- ✅ Revisar PRs de Dependabot

### 4. Testing

- ✅ Ejecutar tests después de actualizar
- ✅ Verificar compatibilidad
- ✅ Probar en CI/CD

## 🐛 Troubleshooting

### Problemas Comunes

**Error: "No module named X"**
```bash
pip install -r requirements.txt --force-reinstall
```

**Conflictos de versiones**
```bash
pip-compile requirements.txt
pip install -r requirements-lock.txt
```

**Dependencias de sistema faltantes**
```bash
# Ubuntu/Debian
sudo apt-get install libgl1 libglib2.0-0 libpq-dev

# macOS
brew install postgresql
```

## 📚 Recursos

- [pip documentation](https://pip.pypa.io/)
- [pip-tools](https://github.com/jazzband/pip-tools)
- [Poetry](https://python-poetry.org/)
- [Safety](https://github.com/pyupio/safety)
- [pip-audit](https://github.com/trailofbits/pip-audit)

## 🔗 Archivos Relacionados

- `DEPENDENCIES_GUIDE.md` - Guía detallada de dependencias
- `pyproject.toml` - Configuración moderna
- `.pre-commit-config.yaml` - Hooks de pre-commit
- `.github/dependabot.yml` - Configuración Dependabot
- `Makefile` - Comandos convenientes



