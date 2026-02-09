# 🚀 Quick Start Guide - GitHub Autonomous Agent

> Guía rápida para empezar en **5 minutos**

## ⚡ Inicio Ultra Rápido

### Paso 1: Setup Automático (2 minutos)

#### Linux/macOS
```bash
chmod +x scripts/setup.sh
./scripts/setup.sh --dev
```

#### Windows (PowerShell)
```powershell
.\scripts\setup.ps1 -Dev
```

### Paso 2: Configurar Variables de Entorno (1 minuto)

```bash
# Copiar template
cp .env.example .env

# Generar SECRET_KEY automáticamente
python scripts/generate-secret.py

# Editar .env y agregar tu GITHUB_TOKEN
nano .env  # o usa tu editor favorito
```

**Valores mínimos requeridos:**
```env
GITHUB_TOKEN=ghp_tu_token_aqui
SECRET_KEY=generado_automaticamente
```

### Paso 3: Validar y Ejecutar (2 minutos)

```bash
# Validar configuración
python scripts/validate-env.py

# Iniciar aplicación
make run-dev
# O directamente:
python main.py
```

### ✅ Verificar que Funciona

```bash
# Health check
curl http://localhost:8030/health

# Ver documentación de API
# Abre en navegador: http://localhost:8030/docs
```

¡Listo! 🎉 Tu agente está corriendo.

---

## 📋 Instalación Manual (Alternativa)

Si prefieres hacerlo paso a paso:

### 1. Crear Entorno Virtual

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# o
.\venv\Scripts\Activate.ps1  # Windows
```

### 2. Instalar Dependencias

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 3. Configurar .env

```bash
cp .env.example .env
```

Edita `.env` con estos valores mínimos:

```env
# Obligatorias
GITHUB_TOKEN=tu_token_de_github
SECRET_KEY=genera_una_clave_segura_de_32_caracteres

# Opcionales pero recomendadas
DATABASE_URL=sqlite+aiosqlite:///./github_agent.db
REDIS_URL=redis://localhost:6379/0
```

**Generar SECRET_KEY:**
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
# O usar el script:
python scripts/generate-secret.py
```

### 4. Validar Configuración

```bash
# Validar variables de entorno
python scripts/validate-env.py

# Verificar dependencias
python scripts/check-dependencies.py

# Health check
python scripts/health-check.py
```

### 5. Iniciar Servicios (si es necesario)

```bash
# Redis (requerido para Celery)
redis-server
# O con Docker:
docker run -d -p 6379:6379 redis:7-alpine
```

### 6. Ejecutar Aplicación

```bash
# Opción 1: Con Make (recomendado)
make run-dev

# Opción 2: Directo
python main.py

# Opción 3: Con uvicorn
uvicorn main:app --reload --host 0.0.0.0 --port 8030
```

---

## 🐳 Docker (Alternativa Rápida)

### Desarrollo

```bash
# Iniciar todos los servicios
docker-compose up -d

# Ver logs
docker-compose logs -f app

# Detener
docker-compose down
```

### Producción

```bash
# Build
docker build -t github-autonomous-agent:latest .

# Run
docker run -d \
  --name github-agent \
  -p 8030:8030 \
  --env-file .env \
  github-autonomous-agent:latest
```

---

## ✅ Verificación Post-Instalación

### 1. Health Check

```bash
# Script de health check
python scripts/health-check.py

# O manualmente
curl http://localhost:8030/health
```

### 2. Verificar Dependencias

```bash
python scripts/check-dependencies.py
```

### 3. Verificar Seguridad

```bash
python scripts/security-check.py
```

### 4. Verificar API

```bash
# Abrir en navegador
http://localhost:8030/docs

# O con curl
curl http://localhost:8030/api/v1/agent/status
```

---

## 🧪 Testing Rápido

```bash
# Todos los tests
make test

# Con coverage
make test-cov

# Tests específicos
pytest tests/unit/
pytest tests/integration/
```

---

## 🎯 Primeros Pasos

### 1. Explorar API

Abre en tu navegador:
- **Swagger UI**: http://localhost:8030/docs
- **ReDoc**: http://localhost:8030/redoc

### 2. Conectar un Repositorio

```bash
curl -X POST http://localhost:8030/api/v1/github/connect \
  -H "Content-Type: application/json" \
  -d '{
    "repo_owner": "usuario",
    "repo_name": "repositorio",
    "branch": "main"
  }'
```

### 3. Crear una Tarea

```bash
curl -X POST http://localhost:8030/api/v1/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "instruction": "Analizar código y generar documentación",
    "priority": "high"
  }'
```

### 4. Ver Estado

```bash
curl http://localhost:8030/api/v1/tasks
```

---

## 🆘 Problemas Comunes

### ❌ Error: "Module not found"

**Solución:**
```bash
# Verificar entorno virtual activado
which python  # Debe mostrar venv/bin/python

# Reinstalar dependencias
pip install -r requirements.txt -r requirements-dev.txt
```

### ❌ Error: "Redis connection failed"

**Solución:**
```bash
# Verificar Redis corriendo
redis-cli ping  # Debe responder: PONG

# Iniciar Redis
redis-server
# O con Docker:
docker run -d -p 6379:6379 redis:7-alpine
```

### ❌ Error: "GitHub token invalid"

**Solución:**
1. Verifica que el token tenga los permisos correctos:
   - `repo` (acceso completo a repositorios)
   - `workflow` (acceso a workflows)
   - `admin:repo_hook` (administrar webhooks)

2. Genera nuevo token en: https://github.com/settings/tokens

3. Valida el token:
```bash
curl -H "Authorization: token YOUR_TOKEN" https://api.github.com/user
```

### ❌ Error: "Port 8030 already in use"

**Solución:**
```bash
# Opción 1: Cambiar puerto en .env
API_PORT=8031

# Opción 2: Matar proceso existente
lsof -ti:8030 | xargs kill  # Linux/macOS
netstat -ano | findstr :8030  # Windows (luego taskkill /PID <pid>)
```

### ❌ Error: "Database locked" (SQLite)

**Solución:**
```bash
# Verificar que no hay otra instancia corriendo
ps aux | grep python

# O usar PostgreSQL en lugar de SQLite
DATABASE_URL=postgresql+asyncpg://user:pass@localhost/db
```

---

## 🎓 Próximos Pasos

### Documentación Recomendada

1. **[README.md](README.md)** - Documentación completa
2. **[DEVELOPMENT.md](DEVELOPMENT.md)** - Guía de desarrollo
3. **[DEPLOYMENT.md](DEPLOYMENT.md)** - Guía de deployment
4. **[REQUIREMENTS_GUIDE.md](REQUIREMENTS_GUIDE.md)** - Dependencias

### Explorar el Código

1. **API Routes**: `api/routes/`
2. **Core Logic**: `core/`
3. **Use Cases**: `application/use_cases/`
4. **Examples**: `examples/`

### Comandos Útiles

```bash
make help              # Ver todos los comandos
make setup             # Setup completo
make check-all         # Verificaciones completas
make security-check    # Verificar seguridad
make update-deps       # Ver actualizaciones disponibles
make lint              # Ejecutar linters
make format            # Formatear código
```

---

## 📊 Checklist de Inicio

- [ ] Entorno virtual creado y activado
- [ ] Dependencias instaladas
- [ ] Archivo `.env` configurado
- [ ] `GITHUB_TOKEN` válido configurado
- [ ] `SECRET_KEY` generado
- [ ] Redis corriendo (si usas Celery)
- [ ] Configuración validada (`validate-env.py`)
- [ ] Aplicación iniciada y respondiendo
- [ ] Health check exitoso
- [ ] API accesible en `/docs`

---

## 🚀 ¿Listo para Empezar?

Una vez completado el setup:

1. ✅ Explora la API en http://localhost:8030/docs
2. ✅ Conecta tu primer repositorio
3. ✅ Crea tu primera tarea
4. ✅ Monitorea el progreso

**¿Necesitas ayuda?** Consulta la [documentación completa](README.md) o abre un issue.

---

**¡Feliz desarrollo! 🎉**
