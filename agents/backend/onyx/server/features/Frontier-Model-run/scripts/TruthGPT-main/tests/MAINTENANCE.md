# 🔧 Mantenimiento - TruthGPT Test Suite

## Scripts de Mantenimiento

### 1. Health Check (`health_check.py`)

Verificación completa de salud del sistema:

```bash
python health_check.py
make health
```

**Verifica:**
- ✅ Estructura de directorios
- ✅ Archivos de configuración
- ✅ Dependencias instaladas
- ✅ Scripts principales
- ✅ Archivos de test
- ✅ Ejecución rápida de tests

**Genera:**
- Score de salud (0-100)
- Estado (excellent/good/fair/poor)
- Lista de problemas y advertencias
- Recomendaciones

### 2. Backup (`backup_tests.py`)

Sistema de backup para resultados y configuración:

```bash
# Backup completo
python backup_tests.py --all
make backup

# Solo resultados
python backup_tests.py --results
make backup-results

# Solo configuración
python backup_tests.py --config
make backup-config

# Listar backups
python backup_tests.py --list
make backup-list

# Restaurar
python backup_tests.py --restore backups/results_backup_20251118_120000.tar.gz
```

**Características:**
- 📦 Backup comprimido (tar.gz, zip)
- 📅 Timestamps automáticos
- 📋 Listado de backups
- 🔄 Restauración fácil

### 3. Cleanup (`cleanup_tests.py`)

Limpieza de archivos temporales y antiguos:

```bash
# Limpieza completa
python cleanup_tests.py --all
make cleanup

# Solo cachés
python cleanup_tests.py --cache
make cleanup-cache

# Resultados antiguos (>7 días)
python cleanup_tests.py --results 7
make cleanup-results

# Solo logs
python cleanup_tests.py --logs

# Solo temporales
python cleanup_tests.py --temp
```

**Limpia:**
- 🗑️ Cachés de Python (__pycache__, .pyc)
- 🗑️ Cachés de pytest
- 🗑️ Resultados antiguos
- 🗑️ Archivos de log
- 🗑️ Archivos temporales

## Tareas de Mantenimiento Regular

### Diario
```bash
# Health check rápido
make health

# Limpiar cachés
make cleanup-cache
```

### Semanal
```bash
# Health check completo
make health

# Limpiar resultados antiguos
make cleanup-results

# Backup de configuración
make backup-config
```

### Mensual
```bash
# Health check completo
make health

# Backup completo
make backup

# Limpieza completa
make cleanup

# Revisar y actualizar dependencias
pip install --upgrade -r requirements-test.txt
```

## Workflows de Mantenimiento

### Workflow de Salud

```bash
# 1. Verificar salud
make health

# 2. Si hay problemas, revisar
python validate_structure.py

# 3. Si es necesario, reconfigurar
make setup

# 4. Verificar nuevamente
make health
```

### Workflow de Backup

```bash
# 1. Crear backup antes de cambios importantes
make backup

# 2. Realizar cambios

# 3. Si algo sale mal, restaurar
python backup_tests.py --restore backups/config_backup_*.zip
```

### Workflow de Limpieza

```bash
# 1. Verificar espacio usado
du -sh tests/

# 2. Limpiar archivos temporales
make cleanup

# 3. Verificar espacio liberado
du -sh tests/
```

## Troubleshooting

### Health Check Falla

```bash
# Ver detalles
python health_check.py --output health_report.json

# Revisar problemas específicos
cat health_report.json | jq '.issues'

# Resolver problemas
make setup
```

### Backup No Funciona

```bash
# Verificar permisos
ls -la backups/

# Verificar espacio disponible
df -h

# Crear directorio si no existe
mkdir -p backups
```

### Cleanup Muy Agresivo

```bash
# Limpiar solo cachés primero
make cleanup-cache

# Verificar qué se limpiaría
python cleanup_tests.py --results 30  # Solo >30 días

# Limpiar manualmente si es necesario
```

## Automatización

### Cron Job para Mantenimiento

```bash
# Agregar a crontab (crontab -e)

# Health check diario a las 2 AM
0 2 * * * cd /path/to/tests && make health >> logs/health.log 2>&1

# Backup semanal (domingos a las 3 AM)
0 3 * * 0 cd /path/to/tests && make backup >> logs/backup.log 2>&1

# Limpieza mensual (día 1 a las 4 AM)
0 4 1 * * cd /path/to/tests && make cleanup >> logs/cleanup.log 2>&1
```

### Script de Mantenimiento Automático

```bash
#!/bin/bash
# maintenance.sh

cd /path/to/tests

echo "🔧 Iniciando mantenimiento..."

# Health check
make health

# Backup
make backup

# Cleanup
make cleanup

echo "✅ Mantenimiento completado"
```

## Métricas de Mantenimiento

### Tamaño de Backups
- Monitorear crecimiento de backups
- Rotar backups antiguos (>30 días)
- Comprimir backups muy grandes

### Espacio en Disco
- Monitorear uso de espacio
- Limpiar regularmente
- Mover backups a almacenamiento externo

### Salud del Sistema
- Ejecutar health check regularmente
- Trackear score de salud
- Alertar si score < 75

## Recursos

- **README.md**: Documentación completa
- **TOOLS.md**: Guía de herramientas
- **EXECUTIVE_SUMMARY.md**: Resumen ejecutivo

