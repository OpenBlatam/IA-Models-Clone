# 📑 Índice Rápido - Sistema de Dependencias

## 🚀 Inicio Rápido (30 segundos)

```bash
# 1. Setup automático
./scripts/setup-dev-environment.sh

# 2. Verificar
make check

# 3. Listo!
```

## 📦 Archivos de Requirements

| Archivo | Comando | Tamaño | Uso |
|---------|---------|--------|-----|
| `requirements.txt` | `make install` | ~2-3 GB | Producción completa |
| `requirements-optimized.txt` | `make install-opt` | ~500 MB | ⭐ Producción (recomendado) |
| `requirements-dev.txt` | `make install-dev` | ~3-4 GB | Desarrollo |
| `requirements-minimal.txt` | `make install-min` | ~50 MB | CI/Testing |
| `requirements-gpu.txt` | `make install-gpu` | ~4-5 GB | GPU/CUDA |
| `requirements-docker.txt` | `make install-docker` | ~600 MB | Docker |
| `requirements-lock.txt` | `make compile` | - | Versiones fijas |

## 🛠️ Comandos Esenciales

### Instalación
```bash
make install          # Producción
make install-opt      # Optimizado ⭐
make install-dev      # Desarrollo
```

### Verificación
```bash
make check            # Seguridad
make outdated         # Desactualizadas
make validate-reqs    # Validar formato
```

### Desarrollo
```bash
make test             # Tests
make lint             # Linters
make format           # Formatear
make pre-commit       # Todo
```

### Análisis
```bash
make dashboard        # Dashboard
make dep-tree         # Árbol
make check-all        # Todos los checks
```

### Gestión
```bash
make update           # Actualizar
make backup           # Backup
make restore          # Restore
make cleanup          # Limpiar
```

## 📚 Documentación

1. **[README_DEPENDENCIES.md](README_DEPENDENCIES.md)** - 📖 Guía principal
2. **[DEPENDENCIES_GUIDE.md](DEPENDENCIES_GUIDE.md)** - 📘 Guía detallada
3. **[DEPENDENCY_MANAGEMENT.md](DEPENDENCY_MANAGEMENT.md)** - 📗 Gestión completa
4. **[QUICK_START_DEPS.md](QUICK_START_DEPS.md)** - ⚡ Inicio rápido
5. **[DEPENDENCIES_SUMMARY.md](DEPENDENCIES_SUMMARY.md)** - 📊 Resumen
6. **[DEPENDENCIES_CHANGELOG.md](DEPENDENCIES_CHANGELOG.md)** - 📝 Changelog
7. **[FINAL_DEPENDENCIES_SUMMARY.md](FINAL_DEPENDENCIES_SUMMARY.md)** - 🎉 Resumen final

## 🔧 Scripts Principales

### Gestión
- `check-dependencies.sh` - Health check
- `update-dependencies.sh` - Actualizar
- `backup-requirements.sh` - Backup
- `restore-requirements.sh` - Restore
- `monitor-dependencies.sh` - Monitoreo

### Análisis
- `analyze-dependencies.py` - Análisis
- `visualize-dependencies.py` - Visualización
- `compare-requirements.sh` - Comparar
- `dependency-tree.py` - Árbol
- `dependency-dashboard.py` - Dashboard

### Validación
- `validate-requirements.py` - Validar
- `all-checks.sh` - Todos los checks
- `ci-check-dependencies.sh` - CI/CD

### Utilidades
- `optimize-requirements.py` - Optimizar
- `migrate-requirements.py` - Migrar
- `benchmark-install.sh` - Benchmark
- `cleanup-requirements.sh` - Limpiar

## 🎯 Casos de Uso Comunes

### Primera vez
```bash
./scripts/setup-dev-environment.sh
```

### Desarrollo diario
```bash
make test
make format
make pre-commit
```

### Actualización semanal
```bash
make outdated
make update
make check
make backup
```

### Antes de release
```bash
make check-all
make backup
make compile
make test-cov
```

## ⚠️ Troubleshooting Rápido

**Error: "No module named X"**
```bash
pip install -r requirements.txt --force-reinstall
```

**Conflictos de versiones**
```bash
make compile
pip install -r requirements-lock.txt
```

**Verificar salud**
```bash
./scripts/check-dependencies.sh
```

## 🔗 Enlaces Rápidos

- [Makefile](Makefile) - Todos los comandos
- [pyproject.toml](pyproject.toml) - Configuración
- [.pre-commit-config.yaml](.pre-commit-config.yaml) - Hooks
- [.github/workflows/dependencies.yml](.github/workflows/dependencies.yml) - CI/CD

## 📊 Estadísticas

- **Archivos:** 44+
- **Scripts:** 20
- **Comandos Make:** 44+
- **Documentación:** 4500+ líneas
- **Versión:** 2.2

---

**Última actualización:** 2024  
**Estado:** ✅ Completo



