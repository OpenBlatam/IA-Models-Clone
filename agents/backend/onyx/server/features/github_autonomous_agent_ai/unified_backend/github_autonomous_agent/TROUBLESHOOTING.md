# 🔧 Troubleshooting Guide - GitHub Autonomous Agent

> Guía completa para resolver problemas comunes

## 📋 Tabla de Contenidos

- [Problemas de Instalación](#-problemas-de-instalación)
- [Problemas de Configuración](#-problemas-de-configuración)
- [Problemas de Conexión](#-problemas-de-conexión)
- [Problemas de API](#-problemas-de-api)
- [Problemas de Base de Datos](#-problemas-de-base-de-datos)
- [Problemas de Queue/Tasks](#-problemas-de-queuetasks)
- [Problemas de Performance](#-problemas-de-performance)
- [Problemas de Docker](#-problemas-de-docker)
- [Logs y Debugging](#-logs-y-debugging)

---

## 🚨 Problemas de Instalación

### Error: "Module not found"

**Síntomas:**
```
ModuleNotFoundError: No module named 'fastapi'
```

**Soluciones:**

1. **Verificar entorno virtual activado:**
```bash
which python  # Debe mostrar venv/bin/python
# Si no, activar:
source venv/bin/activate  # Linux/macOS
.\venv\Scripts\Activate.ps1  # Windows
```

2. **Reinstalar dependencias:**
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

3. **Verificar Python version:**
```bash
python --version  # Debe ser 3.10+
```

4. **Limpiar e reinstalar:**
```bash
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt -r requirements-dev.txt
```

---

### Error: "pip install failed"

**Síntomas:**
```
ERROR: Could not install packages due to an EnvironmentError
```

**Soluciones:**

1. **Actualizar pip:**
```bash
pip install --upgrade pip setuptools wheel
```

2. **Usar usuario local:**
```bash
pip install --user -r requirements.txt
```

3. **Verificar permisos:**
```bash
# Linux/macOS
sudo chown -R $USER:$USER venv/

# O reinstalar sin sudo
python3 -m venv --user venv
```

---

### Error: "Python version incompatible"

**Síntomas:**
```
ERROR: Package requires a different Python: 3.9.0 not in '>=3.10'
```

**Soluciones:**

1. **Verificar versión:**
```bash
python3 --version  # Debe ser 3.10+
```

2. **Instalar Python 3.10+:**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.10 python3.10-venv

# macOS (con Homebrew)
brew install python@3.11

# Windows
# Descargar de python.org
```

3. **Usar versión correcta:**
```bash
python3.10 -m venv venv
source venv/bin/activate
```

---

## ⚙️ Problemas de Configuración

### Error: "GitHub token invalid"

**Síntomas:**
```
HTTPException: GitHub token no configurado
# o
401 Unauthorized
```

**Soluciones:**

1. **Verificar token en .env:**
```bash
cat .env | grep GITHUB_TOKEN
```

2. **Validar token:**
```bash
curl -H "Authorization: token YOUR_TOKEN" https://api.github.com/user
```

3. **Verificar permisos del token:**
   - Ir a: https://github.com/settings/tokens
   - Verificar que el token tenga permisos:
     - `repo` (acceso completo a repositorios)
     - `workflow` (acceso a workflows)
     - `admin:repo_hook` (administrar webhooks)

4. **Generar nuevo token:**
   - https://github.com/settings/tokens/new
   - Seleccionar permisos necesarios
   - Copiar token y actualizar `.env`

5. **Validar configuración:**
```bash
python scripts/validate-env.py
```

---

### Error: "SECRET_KEY not set"

**Síntomas:**
```
ValueError: SECRET_KEY must be set
```

**Soluciones:**

1. **Generar SECRET_KEY:**
```bash
python scripts/generate-secret.py
# O manualmente:
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

2. **Agregar a .env:**
```bash
echo "SECRET_KEY=tu_clave_generada" >> .env
```

3. **Validar:**
```bash
python scripts/validate-env.py
```

---

### Error: "Database connection failed"

**Síntomas:**
```
OperationalError: unable to open database file
# o
ConnectionError: Could not connect to database
```

**Soluciones:**

1. **SQLite - Verificar permisos:**
```bash
# Verificar que el directorio existe
mkdir -p storage/

# Verificar permisos
chmod 755 storage/
```

2. **PostgreSQL - Verificar conexión:**
```bash
# Verificar que PostgreSQL está corriendo
pg_isready

# Verificar conexión
psql -h localhost -U usuario -d github_agent

# Verificar DATABASE_URL en .env
DATABASE_URL=postgresql+asyncpg://user:pass@localhost/github_agent
```

3. **Ejecutar migraciones:**
```bash
python scripts/migrate-db.py upgrade
```

---

## 🔌 Problemas de Conexión

### Error: "Redis connection failed"

**Síntomas:**
```
ConnectionError: Error connecting to Redis
# o
celery.exceptions.OperationalError: Could not connect to Redis
```

**Soluciones:**

1. **Verificar que Redis está corriendo:**
```bash
redis-cli ping  # Debe responder: PONG
```

2. **Iniciar Redis:**
```bash
# Linux
sudo systemctl start redis
# o
redis-server

# macOS (Homebrew)
brew services start redis

# Docker
docker run -d -p 6379:6379 redis:7-alpine
```

3. **Verificar REDIS_URL en .env:**
```bash
REDIS_URL=redis://localhost:6379/0
```

4. **Verificar firewall:**
```bash
# Verificar que el puerto está abierto
netstat -an | grep 6379
```

---

### Error: "GitHub API rate limit exceeded"

**Síntomas:**
```
403 Forbidden - API rate limit exceeded
```

**Soluciones:**

1. **Verificar rate limit:**
```bash
curl -H "Authorization: token YOUR_TOKEN" \
  https://api.github.com/rate_limit
```

2. **Esperar reset:**
   - Rate limit se resetea cada hora
   - Verificar `reset` timestamp en respuesta

3. **Usar token con más permisos:**
   - Tokens autenticados tienen límites más altos
   - Verificar que el token es válido

4. **Implementar retry logic:**
   - El código ya incluye retry automático
   - Verificar que está funcionando

---

## 📡 Problemas de API

### Error: "404 Not Found"

**Síntomas:**
```
404 Not Found - Endpoint no existe
```

**Soluciones:**

1. **Verificar URL:**
```bash
# Base URL correcta
http://localhost:8030/api/v1/...

# Verificar documentación
curl http://localhost:8030/docs
```

2. **Verificar que el servidor está corriendo:**
```bash
curl http://localhost:8030/health
```

3. **Verificar versión de API:**
   - Usar `/api/v1/` para endpoints
   - Verificar que el endpoint existe en la versión

---

### Error: "422 Validation Error"

**Síntomas:**
```
422 Unprocessable Entity - Validation error
```

**Soluciones:**

1. **Verificar formato de request:**
```bash
# Verificar Content-Type
-H "Content-Type: application/json"

# Verificar formato JSON válido
echo '{"key": "value"}' | jq .
```

2. **Verificar schema en documentación:**
   - Ir a http://localhost:8030/docs
   - Ver schema esperado del endpoint
   - Comparar con tu request

3. **Validar datos:**
```python
# Usar Pydantic para validar
from api.schemas import TaskCreateSchema
schema = TaskCreateSchema(**data)
```

---

### Error: "500 Internal Server Error"

**Síntomas:**
```
500 Internal Server Error
```

**Soluciones:**

1. **Revisar logs:**
```bash
tail -f storage/logs/app.log
# O con Docker:
docker-compose logs -f app
```

2. **Verificar configuración:**
```bash
python scripts/validate-env.py
python scripts/health-check.py
```

3. **Verificar dependencias:**
```bash
python scripts/check-dependencies.py
```

4. **Modo debug:**
```bash
# En .env
LOG_LEVEL=DEBUG
DEBUG=true

# Reiniciar aplicación
```

---

## 🗄️ Problemas de Base de Datos

### Error: "Database locked" (SQLite)

**Síntomas:**
```
OperationalError: database is locked
```

**Soluciones:**

1. **Verificar procesos concurrentes:**
```bash
# Ver procesos usando la BD
lsof storage/github_agent.db

# Matar procesos si es necesario
kill <PID>
```

2. **Usar PostgreSQL en producción:**
```bash
DATABASE_URL=postgresql+asyncpg://user:pass@localhost/db
```

3. **Verificar timeouts:**
```python
# En config/settings.py
SQLITE_TIMEOUT=30
```

---

### Error: "Migration failed"

**Síntomas:**
```
alembic.util.exc.CommandError: Target database is not up to date
```

**Soluciones:**

1. **Verificar estado de migraciones:**
```bash
python scripts/migrate-db.py current
```

2. **Aplicar migraciones:**
```bash
python scripts/migrate-db.py upgrade head
```

3. **Revisar migraciones pendientes:**
```bash
python scripts/migrate-db.py history
```

4. **Si hay conflictos:**
```bash
# Backup primero
cp storage/github_agent.db storage/github_agent.db.backup

# Revisar migraciones
python scripts/migrate-db.py upgrade head
```

---

## 🔄 Problemas de Queue/Tasks

### Error: "Celery worker not running"

**Síntomas:**
```
Tasks no se procesan
# o
ConnectionError: Could not connect to broker
```

**Soluciones:**

1. **Verificar que Celery está corriendo:**
```bash
# Ver procesos
ps aux | grep celery

# O con Docker
docker-compose ps worker
```

2. **Iniciar worker:**
```bash
# Manualmente
celery -A core.worker worker --loglevel=info

# O con Docker
docker-compose up -d worker
```

3. **Verificar Redis:**
```bash
redis-cli ping
```

4. **Ver logs del worker:**
```bash
# Logs de Celery
tail -f storage/logs/celery.log

# O con Docker
docker-compose logs -f worker
```

---

### Error: "Tasks stuck in queue"

**Síntomas:**
```
Tareas permanecen en estado "pending"
```

**Soluciones:**

1. **Verificar workers:**
```bash
# Con Flower (si está instalado)
# Abrir http://localhost:5555

# O verificar directamente
celery -A core.worker inspect active
```

2. **Reiniciar workers:**
```bash
# Detener workers
pkill -f celery

# Reiniciar
celery -A core.worker worker --loglevel=info
```

3. **Limpiar queue:**
```bash
# Purge tasks pendientes (¡cuidado!)
celery -A core.worker purge
```

---

## ⚡ Problemas de Performance

### Error: "Application slow"

**Síntomas:**
```
Requests tardan mucho en responder
```

**Soluciones:**

1. **Verificar recursos:**
```bash
# CPU y memoria
top
# o
htop

# Con Python
python scripts/health-check.py
```

2. **Optimizar base de datos:**
```bash
# Verificar queries lentas
# Habilitar query logging en desarrollo
```

3. **Aumentar workers:**
```bash
# En uvicorn
uvicorn main:app --workers 4

# En Celery
celery -A core.worker worker --concurrency=4
```

4. **Usar cache:**
```bash
# Verificar Redis cache
redis-cli
> KEYS *
```

---

## 🐳 Problemas de Docker

### Error: "Docker build failed"

**Síntomas:**
```
ERROR: failed to solve: process "/bin/sh -c pip install" did not complete successfully
```

**Soluciones:**

1. **Limpiar cache:**
```bash
docker builder prune
docker system prune -a
```

2. **Rebuild sin cache:**
```bash
docker build --no-cache -t github-autonomous-agent:latest .
```

3. **Verificar Dockerfile:**
```bash
# Verificar sintaxis
docker build -t test .
```

---

### Error: "Container exits immediately"

**Síntomas:**
```
Container se detiene inmediatamente después de iniciar
```

**Soluciones:**

1. **Ver logs:**
```bash
docker-compose logs app
# o
docker logs <container_id>
```

2. **Verificar .env:**
```bash
# Verificar que .env existe y tiene valores
cat .env
```

3. **Ejecutar interactivo:**
```bash
docker-compose run --rm app /bin/bash
# Luego ejecutar manualmente:
python main.py
```

---

### Error: "Port already in use"

**Síntomas:**
```
Error: bind: address already in use
```

**Soluciones:**

1. **Encontrar proceso:**
```bash
# Linux/macOS
lsof -i :8030

# Windows
netstat -ano | findstr :8030
```

2. **Matar proceso:**
```bash
# Linux/macOS
kill <PID>

# Windows
taskkill /PID <PID> /F
```

3. **Cambiar puerto:**
```bash
# En .env
API_PORT=8031

# O en docker-compose.yml
ports:
  - "8031:8030"
```

---

## 📊 Logs y Debugging

### Ver Logs

```bash
# Logs de aplicación
tail -f storage/logs/app.log

# Logs estructurados (JSON)
cat storage/logs/app.log | jq .

# Logs de Celery
tail -f storage/logs/celery.log

# Con Docker
docker-compose logs -f app
docker-compose logs -f worker
```

### Niveles de Log

```bash
# En .env
LOG_LEVEL=DEBUG  # DEBUG, INFO, WARNING, ERROR

# Reiniciar aplicación
```

### Debugging

1. **Modo debug:**
```bash
# En .env
DEBUG=true
LOG_LEVEL=DEBUG
```

2. **Python debugger:**
```python
import pdb; pdb.set_trace()
# o
breakpoint()
```

3. **Verificar variables:**
```python
import logging
logger = logging.getLogger(__name__)
logger.debug(f"Variable value: {variable}")
```

---

## 🔍 Comandos de Diagnóstico

### Health Check Completo

```bash
# Script de health check
python scripts/health-check.py

# Verificar dependencias
python scripts/check-dependencies.py

# Validar configuración
python scripts/validate-env.py

# Security check
python scripts/security-check.py
```

### Verificar Servicios

```bash
# Redis
redis-cli ping

# PostgreSQL
pg_isready

# API
curl http://localhost:8030/health

# Celery
celery -A core.worker inspect ping
```

---

## 📞 Obtener Ayuda

### Antes de Pedir Ayuda

1. ✅ Revisar esta guía de troubleshooting
2. ✅ Revisar logs detalladamente
3. ✅ Ejecutar health check
4. ✅ Buscar en issues existentes
5. ✅ Revisar documentación

### Cómo Pedir Ayuda

1. **Abrir Issue en GitHub:**
   - Usar template de bug report
   - Incluir logs relevantes
   - Incluir información del entorno
   - Describir pasos para reproducir

2. **Incluir Información:**
   - OS y versión
   - Python version
   - Versión del proyecto
   - Logs completos
   - Configuración (sin secretos)

---

## ✅ Checklist de Troubleshooting

- [ ] Revisar logs
- [ ] Ejecutar health check
- [ ] Verificar configuración (.env)
- [ ] Verificar dependencias instaladas
- [ ] Verificar servicios corriendo (Redis, DB)
- [ ] Verificar conectividad de red
- [ ] Verificar permisos de archivos
- [ ] Revisar documentación relevante
- [ ] Buscar en issues existentes

---

**¿No encuentras la solución?** Abre un issue en GitHub con toda la información relevante.



