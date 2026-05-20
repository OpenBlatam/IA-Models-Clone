# Scripts de Utilidad - Manuales Hogar AI

Esta carpeta contiene scripts de utilidad para gestionar el servicio Manuales Hogar AI.

## 📋 Scripts Disponibles

### Gestión Básica

#### `setup.sh`
Configuración inicial del proyecto.
```bash
./scripts/setup.sh
```
- Crea archivo `.env` si no existe
- Crea directorios necesarios
- Verifica dependencias
- Configura entorno Python

#### `clean.sh`
Limpieza de contenedores y volúmenes.
```bash
./scripts/clean.sh          # Limpieza básica
./scripts/clean.sh --all    # Limpieza completa (incluye imágenes)
```

### Monitoreo y Diagnóstico

#### `check-health.sh`
Verificación detallada de salud del servicio.
```bash
./scripts/check-health.sh
```
- Verifica endpoint de salud
- Prueba múltiples endpoints
- Muestra información de diagnóstico

#### `diagnostics.sh`
Diagnósticos completos del sistema.
```bash
./scripts/diagnostics.sh
```
- Información del sistema
- Estado de Docker
- Estado de servicios
- Disponibilidad de puertos
- Conectividad de base de datos
- Resumen de problemas

#### `monitor.sh`
Monitoreo en tiempo real.
```bash
./scripts/monitor.sh
```
- Estado de contenedores
- Verificación de salud
- Uso de recursos
- Logs recientes
- Estado de base de datos
- Actualización automática cada 5 segundos

### Testing

#### `test-api.sh`
Prueba todos los endpoints de la API.
```bash
./scripts/test-api.sh
```
- Prueba endpoints principales
- Prueba endpoints autenticados (si API key está configurada)
- Muestra resumen de resultados

#### `quick-test.sh`
Validación rápida de la API.
```bash
./scripts/quick-test.sh
```
- Verificación rápida de salud
- Tiempo de respuesta
- Validación de endpoints básicos

### Backup y Restauración

#### `backup.sh`
Crea backup de la base de datos y configuración.
```bash
./scripts/backup.sh          # Backup básico (solo BD)
./scripts/backup.sh --full   # Backup completo (BD + volúmenes + config)
```
- Backup de base de datos
- Backup de volúmenes (opcional)
- Backup de configuración (opcional)
- Limpieza automática de backups antiguos (mantiene últimos 10)

#### `restore.sh`
Restaura base de datos desde backup.
```bash
./scripts/restore.sh backups/db_backup_20240101_120000.sql.gz
```
- Restaura base de datos desde backup
- Confirmación antes de restaurar
- Descompresión automática si es necesario

### Actualización

#### `update.sh`
Actualiza la aplicación y dependencias.
```bash
./scripts/update.sh              # Actualización básica
./scripts/update.sh --pull       # Actualiza código desde git
./scripts/update.sh --rebuild    # Reconstruye imágenes
./scripts/update.sh --pull --rebuild  # Actualización completa
```

### Logs

#### `export-logs.sh`
Exporta logs a un archivo.
```bash
./scripts/export-logs.sh          # Últimos 100 logs
./scripts/export-logs.sh --all    # Todos los logs
```
- Exporta logs de aplicación
- Exporta logs de base de datos
- Exporta logs de Redis
- Incluye información del sistema
- Timestamp en el nombre del archivo

### Performance y Optimización

#### `performance-test.sh`
Pruebas de rendimiento y carga.
```bash
./scripts/performance-test.sh
./scripts/performance-test.sh --concurrent 20 --requests 500
```
- Pruebas de carga con Apache Bench
- Fallback a curl si ab no está disponible
- Métricas de rendimiento
- Tiempo de respuesta promedio
- Requests por segundo

#### `optimize.sh`
Optimización de Docker y sistema.
```bash
./scripts/optimize.sh          # Optimización básica
./scripts/optimize.sh --images # Solo imágenes
./scripts/optimize.sh --system # Solo sistema
./scripts/optimize.sh --all    # Optimización completa
```
- Limpieza de imágenes no usadas
- Limpieza de caché de build
- Limpieza de contenedores detenidos
- Limpieza de volúmenes no usados
- Reporte de espacio liberado

### Seguridad

#### `security-check.sh`
Validación de seguridad básica.
```bash
./scripts/security-check.sh
```
- Verificación de archivos .env
- Validación de headers de seguridad
- Verificación de configuración de Docker
- Detección de vulnerabilidades comunes
- Recomendaciones de seguridad

#### `validate-config.sh`
Validación de configuración.
```bash
./scripts/validate-config.sh
```
- Valida archivos de configuración
- Verifica variables de entorno requeridas
- Valida sintaxis de docker-compose
- Verifica estructura de directorios
- Reporte de errores y advertencias

### Monitoreo Avanzado

#### `health-monitor.sh`
Monitoreo continuo de salud con alertas.
```bash
./scripts/health-monitor.sh
./scripts/health-monitor.sh --interval 60 --webhook https://hooks.slack.com/...
```
- Monitoreo continuo de salud
- Alertas después de múltiples fallos
- Integración con webhooks
- Logging de estado
- Recuperación automática

#### `watch.sh`
Observar cambios y reiniciar automáticamente.
```bash
./scripts/watch.sh
```
- Detecta cambios en archivos
- Reinicio automático de servicios
- Cooldown entre reinicios
- Soporte para inotifywait, fswatch, y polling

### Notificaciones

#### `notify.sh`
Enviar notificaciones sobre estado del servicio.
```bash
./scripts/notify.sh --webhook https://hooks.slack.com/...
./scripts/notify.sh --email admin@example.com
```
- Notificaciones vía webhook (Slack, Discord, etc.)
- Notificaciones por email
- Estado del servicio
- Información de salud

## 🎯 Uso con Make

Todos los scripts también están disponibles a través del Makefile:

```bash
make setup              # Configuración inicial
make clean              # Limpieza básica
make clean-all          # Limpieza completa
make check-health       # Verificación de salud
make diagnostics        # Diagnósticos
make monitor            # Monitoreo en tiempo real
make test-api           # Pruebas de API
make quick-test         # Prueba rápida
make backup             # Backup básico
make backup-full        # Backup completo
make restore BACKUP=... # Restaurar backup
make update             # Actualización
make update-full        # Actualización completa
make export-logs        # Exportar logs
```

## 📝 Notas

- Todos los scripts requieren Docker y docker-compose
- Algunos scripts requieren que los servicios estén corriendo
- Los backups se guardan en `./backups/`
- Los logs exportados se guardan en `./logs/`
- Los scripts son compatibles con Linux, Mac y Windows (con Git Bash o WSL)

## 🔧 Personalización

Puedes personalizar los scripts editándolos directamente o creando variantes para tus necesidades específicas.

## 📚 Ver También

- [README.md](../README.md) - Documentación principal
- [QUICKSTART.md](../QUICKSTART.md) - Guía rápida
- [DOCKER.md](../DOCKER.md) - Guía de Docker

