# 🛠️ Scripts de Utilidad - GitHub Autonomous Agent

> Colección completa de scripts para facilitar desarrollo, deployment y mantenimiento

## 📋 Tabla de Contenidos

- [Scripts Disponibles](#-scripts-disponibles)
- [Setup y Configuración](#-setup-y-configuración)
- [Validación y Verificación](#-validación-y-verificación)
- [Base de Datos](#-base-de-datos)
- [Servicios](#-servicios)
- [Seguridad](#-seguridad)
- [Mantenimiento](#-mantenimiento)
- [Flujo de Trabajo](#-flujo-de-trabajo)
- [Troubleshooting](#-troubleshooting)

---

## 🚀 Scripts Disponibles

### Categorías

| Categoría | Scripts | Descripción |
|----------|---------|-------------|
| **Setup** | `setup.sh`, `setup.ps1` | Setup automático del proyecto |
| **Validación** | `check-dependencies.py`, `validate-env.py`, `health-check.py` | Verificación de configuración y dependencias |
| **Base de Datos** | `migrate-db.py` | Migraciones de base de datos |
| **Seguridad** | `security-check.py`, `generate-secret.py` | Verificación de seguridad y generación de secretos |
| **Mantenimiento** | `update-dependencies.py`, `requirements-lock.py`, `backup.py`, `cleanup.py` | Actualización y mantenimiento |
| **Servicios** | `start-services.sh` | Iniciar servicios necesarios |
| **Utilidades** | `init-project.py` | Inicialización de proyecto |

---

## ⚙️ Setup y Configuración

### `setup.sh` / `setup.ps1`

**Propósito:** Setup automático completo del proyecto

**Uso:**
```bash
# Linux/macOS
chmod +x scripts/setup.sh
./scripts/setup.sh --dev    # Desarrollo
./scripts/setup.sh --prod   # Producción

# Windows (PowerShell)
.\scripts\setup.ps1 -Dev    # Desarrollo
.\scripts\setup.ps1 -Prod   # Producción
```

**Features:**
- ✅ Crea entorno virtual si no existe
- ✅ Instala dependencias base y de desarrollo
- ✅ Configura `.env` desde `.env.example`
- ✅ Crea directorios necesarios (`storage/`, `logs/`, etc.)
- ✅ Configura pre-commit hooks
- ✅ Verifica servicios (Redis, PostgreSQL)
- ✅ Ejecuta validaciones iniciales
- ✅ Genera `SECRET_KEY` si no existe

**Opciones:**
- `--dev` / `-Dev`: Setup para desarrollo
- `--prod` / `-Prod`: Setup para producción
- `--skip-venv`: No crear entorno virtual
- `--skip-deps`: No instalar dependencias

---

## ✅ Validación y Verificación

### `check-dependencies.py`

**Propósito:** Verificar que todas las dependencias están instaladas

**Uso:**
```bash
python scripts/check-dependencies.py
```

**Output:**
- ✅ Lista de dependencias instaladas con versiones
- ❌ Dependencias faltantes
- 📊 Score de instalación (porcentaje)
- ⚠️ Advertencias sobre versiones desactualizadas
- 📋 Resumen detallado

**Opciones:**
```bash
# Ver solo dependencias faltantes
python scripts/check-dependencies.py --missing-only

# Verificar dependencias específicas
python scripts/check-dependencies.py --package fastapi --package pydantic

# Output en JSON
python scripts/check-dependencies.py --json
```

---

### `validate-env.py`

**Propósito:** Validar archivo `.env` y variables de entorno

**Uso:**
```bash
python scripts/validate-env.py
```

**Verifica:**
- ✅ Variables requeridas presentes
- ✅ Variables con valores válidos
- ⚠️ Valores por defecto inseguros
- ⚠️ Variables faltantes pero recomendadas
- ✅ Formato de URLs y conexiones
- ✅ Permisos de archivos

**Opciones:**
```bash
# Validar archivo específico
python scripts/validate-env.py --file .env.production

# Generar reporte
python scripts/validate-env.py --report

# Modo estricto (falla si hay errores)
python scripts/validate-env.py --strict
```

**Ejemplo de Output:**
```
✅ GITHUB_TOKEN: Configurado
✅ SECRET_KEY: Configurado (32 caracteres)
✅ DATABASE_URL: Configurado
⚠️  REDIS_URL: No configurado (recomendado)
❌ OPENROUTER_API_KEY: No configurado (requerido para LLM)
```

---

### `health-check.py`

**Propósito:** Health check completo del sistema

**Uso:**
```bash
python scripts/health-check.py
```

**Verifica:**
- ✅ Aplicación corriendo
- ✅ Base de datos conectada
- ✅ Redis conectado
- ✅ Servicios externos (GitHub API)
- ✅ Espacio en disco
- ✅ Memoria disponible
- ✅ Logs recientes

**Opciones:**
```bash
# Health check básico
python scripts/health-check.py --basic

# Health check detallado
python scripts/health-check.py --detailed

# Output en JSON
python scripts/health-check.py --json

# Verificar servicio específico
python scripts/health-check.py --service database
```

---

## 🗄️ Base de Datos

### `migrate-db.py`

**Propósito:** Gestionar migraciones de base de datos

**Uso:**
```bash
# Aplicar todas las migraciones pendientes
python scripts/migrate-db.py upgrade

# Aplicar hasta una versión específica
python scripts/migrate-db.py upgrade <revision>

# Crear nueva migración
python scripts/migrate-db.py create "descripción de la migración"

# Ver historial de migraciones
python scripts/migrate-db.py history

# Ver estado actual
python scripts/migrate-db.py current

# Revertir última migración
python scripts/migrate-db.py downgrade -1

# Revertir a versión específica
python scripts/migrate-db.py downgrade <revision>

# Generar migración automática (desde modelos)
python scripts/migrate-db.py autogenerate "descripción"
```

**Opciones:**
```bash
# Modo verbose
python scripts/migrate-db.py upgrade --verbose

# SQL sin ejecutar (dry-run)
python scripts/migrate-db.py upgrade --sql

# Forzar migración
python scripts/migrate-db.py upgrade --force
```

---

## 🔒 Seguridad

### `security-check.py`

**Propósito:** Verificación completa de seguridad

**Uso:**
```bash
python scripts/security-check.py
```

**Verifica:**
- 🔍 Vulnerabilidades conocidas (usando `safety` y `pip-audit`)
- 📦 Paquetes críticos desactualizados
- 🔐 Secretos hardcodeados en código
- ⚙️ Configuración insegura
- 📋 Permisos de archivos
- 🔗 Dependencias con vulnerabilidades conocidas

**Requisitos:**
```bash
pip install safety pip-audit bandit
```

**Opciones:**
```bash
# Solo vulnerabilidades críticas
python scripts/security-check.py --critical-only

# Generar reporte
python scripts/security-check.py --report security_report.txt

# Verificar solo secretos
python scripts/security-check.py --secrets-only

# Excluir directorios
python scripts/security-check.py --exclude tests/ venv/
```

---

### `generate-secret.py`

**Propósito:** Generar secretos seguros

**Uso:**
```bash
# Generar SECRET_KEY
python scripts/generate-secret.py

# Generar token JWT
python scripts/generate-secret.py --type jwt

# Generar password
python scripts/generate-secret.py --type password --length 32

# Generar API key
python scripts/generate-secret.py --type api-key
```

**Opciones:**
- `--type`: Tipo de secreto (key, jwt, password, api-key)
- `--length`: Longitud del secreto (default: 32)
- `--output`: Archivo de salida
- `--format`: Formato (hex, base64, url-safe)

---

## 🔧 Mantenimiento

### `update-dependencies.py`

**Propósito:** Actualizar dependencias de forma segura

**Uso:**
```bash
# Modo dry-run (solo muestra qué se actualizaría)
python scripts/update-dependencies.py --dry-run

# Actualizar realmente
python scripts/update-dependencies.py

# Actualizar solo dependencias específicas
python scripts/update-dependencies.py --package fastapi --package pydantic

# Actualizar a versiones específicas
python scripts/update-dependencies.py --package fastapi==0.115.0
```

**Features:**
- 📋 Lista paquetes desactualizados
- 🔍 Verifica vulnerabilidades antes de actualizar
- 📊 Genera reporte de cambios
- ✅ Actualización interactiva con confirmación
- 🔄 Backup de requirements antes de actualizar

**Opciones:**
- `--dry-run`: Solo mostrar cambios sin aplicar
- `--package`: Actualizar paquete específico
- `--major`: Permitir actualizaciones major
- `--report`: Generar reporte de actualización

---

### `requirements-lock.py`

**Propósito:** Generar `requirements-lock.txt` con versiones exactas

**Uso:**
```bash
python scripts/requirements-lock.py
```

**Genera:**
- `requirements-lock.txt` con versiones exactas de todas las dependencias
- Útil para deployments reproducibles
- Incluye dependencias transitivas

**Opciones:**
```bash
# Incluir dependencias de desarrollo
python scripts/requirements-lock.py --include-dev

# Output a archivo específico
python scripts/requirements-lock.py --output requirements-lock.txt
```

---

### `backup.py`

**Propósito:** Crear backup de base de datos y archivos

**Uso:**
```bash
# Backup completo
python scripts/backup.py

# Backup solo base de datos
python scripts/backup.py --db-only

# Backup solo archivos
python scripts/backup.py --files-only

# Backup a ubicación específica
python scripts/backup.py --output /path/to/backup/
```

**Features:**
- 💾 Backup de base de datos (PostgreSQL/SQLite)
- 📁 Backup de archivos importantes
- 🗜️ Compresión automática
- 🗑️ Limpieza de backups antiguos
- 📊 Reporte de backup

---

### `cleanup.py`

**Propósito:** Limpiar archivos temporales y cache

**Uso:**
```bash
# Modo dry-run (solo muestra qué se limpiaría)
python scripts/cleanup.py

# Limpiar realmente
python scripts/cleanup.py --force

# Limpiar solo cache
python scripts/cleanup.py --cache-only

# Limpiar solo logs antiguos
python scripts/cleanup.py --logs-only --days 30
```

**Opciones:**
- `--force`: Ejecutar limpieza (sin esto es dry-run)
- `--cache-only`: Solo limpiar cache
- `--logs-only`: Solo limpiar logs
- `--days`: Mantener archivos de últimos N días
- `--size`: Limpiar archivos mayores a tamaño específico

---

## 🚀 Servicios

### `start-services.sh`

**Propósito:** Iniciar servicios necesarios (Redis, PostgreSQL, etc.)

**Uso:**
```bash
chmod +x scripts/start-services.sh
./scripts/start-services.sh
```

**Inicia:**
- ✅ Redis (si está instalado localmente)
- ✅ PostgreSQL (verifica si está corriendo)
- ✅ Verifica servicios con Docker (si está disponible)

**Opciones:**
```bash
# Solo Redis
./scripts/start-services.sh --redis-only

# Solo PostgreSQL
./scripts/start-services.sh --postgres-only

# Con Docker
./scripts/start-services.sh --docker
```

---

## 🎯 Flujo de Trabajo Recomendado

### Setup Inicial

```bash
# 1. Setup automático
./scripts/setup.sh --dev

# 2. Validar configuración
python scripts/validate-env.py

# 3. Verificar dependencias
python scripts/check-dependencies.py

# 4. Health check
python scripts/health-check.py
```

### Desarrollo Diario

```bash
# 1. Iniciar servicios
./scripts/start-services.sh

# 2. Ejecutar aplicación
make run-dev

# 3. Verificar antes de commit
python scripts/security-check.py
python scripts/check-dependencies.py
```

### Antes de Commit

```bash
# 1. Verificar seguridad
python scripts/security-check.py

# 2. Verificar dependencias
python scripts/check-dependencies.py

# 3. Validar configuración
python scripts/validate-env.py

# 4. Ejecutar tests
make test
```

### Actualización de Dependencias

```bash
# 1. Ver qué se actualizaría
python scripts/update-dependencies.py --dry-run

# 2. Actualizar
python scripts/update-dependencies.py

# 3. Verificar seguridad
python scripts/security-check.py

# 4. Ejecutar tests
make test

# 5. Si todo está bien, generar lock file
python scripts/requirements-lock.py
```

### Mantenimiento Regular

```bash
# 1. Backup
python scripts/backup.py

# 2. Limpiar archivos temporales
python scripts/cleanup.py --force

# 3. Verificar seguridad
python scripts/security-check.py

# 4. Health check
python scripts/health-check.py
```

---

## 📊 Reportes Generados

Los scripts generan varios reportes:

| Script | Reporte | Ubicación |
|--------|---------|-----------|
| `update-dependencies.py` | `dependency_update_report.txt` | `storage/reports/` |
| `security-check.py` | `security_report.txt` | `storage/reports/` |
| `requirements-lock.py` | `requirements-lock.txt` | Raíz del proyecto |
| `backup.py` | `backup_report.txt` | `storage/backups/` |
| `health-check.py` | `health_check_report.txt` | `storage/reports/` |

---

## 🔧 Requisitos Adicionales

Algunos scripts requieren herramientas adicionales:

```bash
# Para security-check.py
pip install safety pip-audit bandit

# Para update-dependencies.py
# No requiere dependencias adicionales (usa pip internamente)

# Para health-check.py
# No requiere dependencias adicionales
```

---

## 🐛 Troubleshooting

### Error: "Command not found"

**Problema:** Script no se encuentra o no es ejecutable

**Solución:**
```bash
# Verificar que estás en el directorio correcto
pwd

# Hacer scripts ejecutables (Linux/macOS)
chmod +x scripts/*.sh

# Verificar permisos
ls -la scripts/
```

### Error: "Module not found"

**Problema:** Módulos Python no encontrados

**Solución:**
```bash
# Asegúrate de estar en el entorno virtual
source venv/bin/activate  # Linux/macOS
.\venv\Scripts\Activate.ps1  # Windows

# Verificar Python
which python  # Debe mostrar venv/bin/python

# Reinstalar dependencias
pip install -r requirements.txt -r requirements-dev.txt
```

### Error: "Permission denied"

**Problema:** Scripts no tienen permisos de ejecución

**Solución:**
```bash
# Linux/macOS
chmod +x scripts/setup.sh
chmod +x scripts/start-services.sh

# Windows (PowerShell)
# Los scripts .ps1 deberían ejecutarse directamente
```

### Error: "Script fails silently"

**Problema:** Script no muestra errores

**Solución:**
```bash
# Ejecutar con verbose
python scripts/script.py --verbose

# Verificar logs
tail -f storage/logs/app.log

# Ejecutar con Python directamente
python -u scripts/script.py
```

---

## 📚 Documentación Relacionada

- [DEVELOPMENT.md](../DEVELOPMENT.md) - Guía de desarrollo
- [DEPLOYMENT.md](../DEPLOYMENT.md) - Guía de deployment
- [REQUIREMENTS_GUIDE.md](../REQUIREMENTS_GUIDE.md) - Guía de dependencias
- [TROUBLESHOOTING.md](../TROUBLESHOOTING.md) - Guía de troubleshooting
- [README.md](../README.md) - Documentación principal

---

## 🎯 Mejores Prácticas

### Para Desarrolladores

1. ✅ Ejecutar `validate-env.py` antes de empezar a desarrollar
2. ✅ Ejecutar `check-dependencies.py` después de `git pull`
3. ✅ Ejecutar `security-check.py` antes de cada commit
4. ✅ Usar `update-dependencies.py --dry-run` antes de actualizar

### Para DevOps

1. ✅ Ejecutar `health-check.py` regularmente
2. ✅ Configurar backups automáticos con `backup.py`
3. ✅ Ejecutar `security-check.py` en CI/CD
4. ✅ Usar `requirements-lock.py` para deployments reproducibles

### Para Mantenimiento

1. ✅ Ejecutar `cleanup.py` regularmente
2. ✅ Revisar reportes de seguridad semanalmente
3. ✅ Actualizar dependencias mensualmente
4. ✅ Verificar health checks diariamente

---

**Última actualización:** Diciembre 2024  
**Mantenido por:** GitHub Autonomous Agent Team
