# Changelog - Manuales Hogar AI

Todos los cambios notables en este proyecto serán documentados en este archivo.

## [Unreleased]

### ✨ Nuevas Funcionalidades

#### Scripts de Utilidad
- ✅ Script de backup (`scripts/backup.sh`) - Backup de base de datos y configuración
- ✅ Script de restauración (`scripts/restore.sh`) - Restaurar desde backup
- ✅ Script de monitoreo (`scripts/monitor.sh`) - Monitoreo en tiempo real
- ✅ Script de pruebas de API (`scripts/test-api.sh`) - Pruebas automatizadas
- ✅ Script de diagnóstico (`scripts/diagnostics.sh`) - Diagnósticos completos
- ✅ Script de actualización (`scripts/update.sh`) - Actualización automática
- ✅ Script de exportación de logs (`scripts/export-logs.sh`) - Exportar logs
- ✅ Script de prueba rápida (`scripts/quick-test.sh`) - Validación rápida

#### Mejoras en Scripts Existentes
- ✅ `start.sh` - Opciones adicionales (`--no-build`, `--skip-health`, `--migrate`)
- ✅ `start.sh` - Detección automática de docker-compose
- ✅ `start.sh` - Verificación de puerto disponible
- ✅ `start.sh` - Colores y mejor feedback
- ✅ `run.py` - Argumentos de línea de comandos con argparse
- ✅ `run.py` - Mismas opciones que scripts shell
- ✅ `status.sh` / `status.ps1` - Scripts de estado completos

#### Docker
- ✅ Dockerfile optimizado para producción
- ✅ Dockerfile.dev para desarrollo
- ✅ docker-compose.yml con PostgreSQL y Redis
- ✅ docker-compose.prod.yml para producción
- ✅ Configuración de Nginx como reverse proxy
- ✅ Health checks en todos los servicios

#### AWS Deployment
- ✅ Infraestructura completa con AWS CDK
- ✅ ECS Fargate con auto-scaling
- ✅ RDS PostgreSQL
- ✅ Application Load Balancer
- ✅ CloudWatch logging y monitoring
- ✅ Secrets Manager para credenciales
- ✅ Scripts de despliegue automatizados
- ✅ CI/CD con GitHub Actions

### 🔧 Mejoras

- ✅ Validaciones mejoradas en scripts de inicio
- ✅ Mejor manejo de errores
- ✅ Compatibilidad mejorada (Linux, Mac, Windows)
- ✅ Documentación mejorada
- ✅ Makefile con más comandos
- ✅ Experiencia de usuario mejorada

### 📚 Documentación

- ✅ README.md actualizado
- ✅ QUICKSTART.md - Guía rápida
- ✅ DOCKER.md - Guía completa de Docker
- ✅ AWS_DEPLOYMENT.md - Guía de despliegue en AWS
- ✅ scripts/README.md - Documentación de scripts
- ✅ IMPROVEMENTS_SUMMARY.md - Resumen de mejoras

### 🐛 Correcciones

- ✅ Corrección en carga de variables de entorno
- ✅ Mejora en detección de docker-compose
- ✅ Corrección en manejo de puertos

## [1.0.0] - 2024-01-XX

### Características Iniciales

- ✅ Sistema de generación de manuales con IA
- ✅ Soporte para múltiples modelos de IA (OpenRouter)
- ✅ Procesamiento de imágenes
- ✅ Procesamiento de texto
- ✅ Sistema de cache
- ✅ Persistencia en base de datos
- ✅ API REST completa
- ✅ Múltiples categorías de oficios
- ✅ Historial y búsqueda
- ✅ Estadísticas de uso

---

## Formato

Este changelog sigue el formato de [Keep a Changelog](https://keepachangelog.com/es-ES/1.0.0/),
y este proyecto adhiere a [Semantic Versioning](https://semver.org/lang/es/).




