# 📦 Sistema de Dependencias - Dermatology AI

## 🎯 Introducción

Sistema completo y profesional de gestión de dependencias con **28+ archivos**, **10+ herramientas** y **documentación exhaustiva**.

## 🚀 Inicio Rápido

### Instalación Rápida

```bash
# Opción 1: Setup automático (Recomendado)
./scripts/setup-dev-environment.sh

# Opción 2: Manual
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
pre-commit install
```

### Verificación

```bash
# Health check completo
./scripts/check-dependencies.sh

# Verificar seguridad
make check

# Ejecutar tests
make test
```

## 📊 Archivos de Dependencias

| Archivo | Tamaño | Caso de Uso | Comando |
|---------|--------|-------------|---------|
| `requirements.txt` | ~2-3 GB | Producción completa | `make install` |
| `requirements-optimized.txt` | ~500 MB | Producción optimizada ⭐ | `make install-opt` |
| `requirements-dev.txt` | ~3-4 GB | Desarrollo | `make install-dev` |
| `requirements-minimal.txt` | ~50 MB | CI/Testing | `make install-min` |
| `requirements-gpu.txt` | ~4-5 GB | GPU/CUDA | `make install-gpu` |
| `requirements-docker.txt` | ~600 MB | Docker | `make install-docker` |
| `requirements-lock.txt` | - | Versiones fijas | `make compile` |

## 🛠️ Herramientas Principales

### 1. Makefile (30+ comandos)

```bash
# Instalación
make install          # Producción completa
make install-opt      # Optimizado (recomendado)
make install-dev      # Desarrollo
make install-gpu      # Con GPU

# Gestión
make update           # Actualizar dependencias
make outdated         # Ver desactualizadas
make check            # Verificar seguridad
make compile          # Generar lock file

# Desarrollo
make test             # Ejecutar tests
make test-cov         # Tests con cobertura
make lint             # Linters
make format           # Formatear código
make type-check       # Type checking
make pre-commit       # Todo antes de commit
```

### 2. Scripts de Automatización

```bash
# Health check
./scripts/check-dependencies.sh

# Actualización segura
./scripts/update-dependencies.sh

# Análisis
python scripts/analyze-dependencies.py
python scripts/visualize-dependencies.py

# Comparación
./scripts/compare-requirements.sh

# Reporte
./scripts/generate-requirements-report.sh

# Monitoreo
./scripts/monitor-dependencies.sh

# Optimización
python scripts/optimize-requirements.py requirements.txt
```

### 3. Pre-commit Hooks

Validación automática antes de cada commit:

```bash
pre-commit install
```

Incluye:
- ✅ Formateo (Black)
- ✅ Linting (Ruff)
- ✅ Type checking (MyPy)
- ✅ Security (Bandit, Safety)
- ✅ Validación de archivos

### 4. Dependabot

Actualización automática de dependencias:
- Actualización semanal
- Pull requests automáticos
- Agrupación inteligente
- Revisión de seguridad

## 📚 Documentación Completa

1. **[DEPENDENCIES_GUIDE.md](DEPENDENCIES_GUIDE.md)** - Guía detallada (200+ líneas)
2. **[DEPENDENCY_MANAGEMENT.md](DEPENDENCY_MANAGEMENT.md)** - Gestión completa (300+ líneas)
3. **[QUICK_START_DEPS.md](QUICK_START_DEPS.md)** - Inicio rápido
4. **[DEPENDENCIES_SUMMARY.md](DEPENDENCIES_SUMMARY.md)** - Resumen ejecutivo
5. **[README_DEPENDENCIES.md](README_DEPENDENCIES.md)** - Este archivo

## 🔒 Seguridad

### Verificación Regular

```bash
# Verificación básica
safety check -r requirements.txt

# Verificación avanzada
pip-audit -r requirements.txt --desc

# Verificación completa
make check
./scripts/check-dependencies.sh
```

### Monitoreo Continuo

```bash
# Monitoreo diario
./scripts/monitor-dependencies.sh

# O configurar cron job
0 9 * * * cd /path/to/project && ./scripts/monitor-dependencies.sh
```

## 📈 Análisis y Visualización

### Análisis de Dependencias

```bash
# Análisis completo
python scripts/analyze-dependencies.py

# Visualización
python scripts/visualize-dependencies.py

# Comparación
./scripts/compare-requirements.sh

# Reporte
./scripts/generate-requirements-report.sh
```

### Optimización

```bash
# Encontrar dependencias no usadas
python scripts/optimize-requirements.py requirements.txt
```

## 🔄 Flujos de Trabajo

### Desarrollo Diario

```bash
# 1. Activar entorno
source venv/bin/activate

# 2. Trabajar normalmente
make test
make format
make lint

# 3. Antes de commit
make pre-commit
```

### Actualización Semanal

```bash
# 1. Verificar desactualizadas
make outdated

# 2. Actualizar
make update

# 3. Verificar seguridad
make check

# 4. Ejecutar tests
make test
```

### Release

```bash
# 1. Actualizar todo
make update
make check

# 2. Generar lock file
make compile

# 3. Análisis completo
python scripts/analyze-dependencies.py
./scripts/generate-requirements-report.sh

# 4. Tests completos
make test-cov
```

## ✅ Checklist

### Diario
- [ ] Ejecutar tests: `make test`
- [ ] Verificar código: `make lint`

### Semanal
- [ ] Verificar desactualizadas: `make outdated`
- [ ] Verificar seguridad: `make check`
- [ ] Actualizar si necesario: `make update`

### Mensual
- [ ] Análisis completo: `python scripts/analyze-dependencies.py`
- [ ] Generar reporte: `./scripts/generate-requirements-report.sh`
- [ ] Actualizar lock: `make compile`
- [ ] Revisar optimizaciones

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

## 📊 Estadísticas

- **Archivos creados:** 28+
- **Scripts de automatización:** 10
- **Comandos Make:** 30+
- **Documentación:** 5 archivos, 3000+ líneas
- **Herramientas de análisis:** 5
- **Configuraciones:** 4

## 🔗 Enlaces Rápidos

- [Guía Completa](DEPENDENCIES_GUIDE.md)
- [Gestión](DEPENDENCY_MANAGEMENT.md)
- [Inicio Rápido](QUICK_START_DEPS.md)
- [Resumen](DEPENDENCIES_SUMMARY.md)
- [Makefile](Makefile)
- [pyproject.toml](pyproject.toml)

## 💡 Mejores Prácticas

1. ✅ Usar `requirements-optimized.txt` para producción
2. ✅ Verificar seguridad regularmente
3. ✅ Mantener `requirements-lock.txt` actualizado
4. ✅ Usar Makefile para comandos comunes
5. ✅ Revisar PRs de Dependabot
6. ✅ Ejecutar tests después de actualizar
7. ✅ Monitorear dependencias regularmente

## 🆘 Ayuda

```bash
# Ver todos los comandos
make help

# Ver ayuda de scripts
./.pip-requirements-update.sh help

# Verificar salud
./scripts/check-dependencies.sh
```

---

**Versión:** 2.2  
**Última actualización:** 2024



