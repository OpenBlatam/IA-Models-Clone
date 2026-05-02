# 🚀 Guía de Deployment - GitHub Autonomous Agent

> Guía completa para desplegar el proyecto en producción

## 📋 Tabla de Contenidos

- [Opciones de Deployment](#-opciones-de-deployment)
- [Preparación para Producción](#-preparación-para-producción)
- [Docker Deployment](#-docker-deployment)
- [VPS/Dedicado](#-vpsdedicado)
- [Cloud Platforms](#-cloud-platforms)
- [Seguridad](#-seguridad)
- [Monitoring](#-monitoring)
- [Backup y Restore](#-backup-y-restore)
- [Actualizaciones](#-actualizaciones)
- [Troubleshooting](#-troubleshooting)

---

## 🎯 Opciones de Deployment

### Comparación Rápida

| Opción | Complejidad | Costo | Escalabilidad | Recomendado Para |
|--------|------------|-------|---------------|------------------|
| Docker Compose | Baja | Bajo | Media | Desarrollo, Producción pequeña |
| VPS/Dedicado | Media | Medio | Alta | Producción mediana/grande |
| Heroku | Baja | Medio | Media | Prototipos, MVP |
| Railway | Baja | Medio | Media | Aplicaciones pequeñas |
| AWS/GCP/Azure | Alta | Variable | Muy Alta | Producción enterprise |

---

## 🏗️ Preparación para Producción

### Checklist Pre-Deployment

- [ ] Variables de entorno configuradas
- [ ] `SECRET_KEY` generada y segura
- [ ] `DEBUG=False` en producción
- [ ] Base de datos configurada (PostgreSQL recomendado)
- [ ] Redis configurado
- [ ] HTTPS habilitado
- [ ] CORS configurado correctamente
- [ ] Rate limiting habilitado
- [ ] Logging configurado
- [ ] Monitoring configurado
- [ ] Backup automático configurado
- [ ] Health checks configurados
- [ ] Firewall configurado
- [ ] Secrets no hardcodeados

### Variables de Entorno de Producción

```bash
# .env.production
# ==============
# Seguridad
SECRET_KEY=tu_clave_secreta_generada_de_32_caracteres
DEBUG=false
ENVIRONMENT=production

# Base de datos (PostgreSQL recomendado)
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/github_agent

# Redis
REDIS_URL=redis://localhost:6379/0

# API
API_HOST=0.0.0.0
API_PORT=8030

# GitHub
GITHUB_TOKEN=ghp_tu_token_aqui

# Logging
LOG_LEVEL=INFO
LOG_FILE=storage/logs/app.log

# Monitoring (opcional)
SENTRY_DSN=https://your-dsn@sentry.io/project-id
PROMETHEUS_ENABLED=true

# OpenRouter (opcional)
OPENROUTER_API_KEY=tu_api_key_aqui
```

### Generar SECRET_KEY

```bash
# Opción 1: Python
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Opción 2: Script incluido
python scripts/generate-secret.py

# Opción 3: OpenSSL
openssl rand -hex 32
```

---

## 🐳 Docker Deployment

### 1. Docker Compose (Recomendado)

#### Requisitos
- Docker 20.10+
- Docker Compose 2.0+

#### Configuración

```bash
# 1. Clonar repositorio
git clone <repo-url>
cd github-autonomous-agent

# 2. Configurar .env
cp .env.example .env
# Editar .env con valores de producción

# 3. Build imágenes
docker-compose build

# 4. Iniciar servicios
docker-compose up -d

# 5. Verificar logs
docker-compose logs -f app

# 6. Verificar health
curl http://localhost:8030/health
```

#### Servicios Incluidos

- **app**: Aplicación principal (FastAPI)
- **worker**: Celery worker para tareas asíncronas
- **beat**: Celery scheduler para tareas programadas
- **flower**: Monitor de Celery (puerto 5555)
- **db**: PostgreSQL
- **redis**: Redis para cache y queue
- **nginx**: Reverse proxy (perfil production)

#### Comandos Útiles

```bash
# Ver logs
docker-compose logs -f app
docker-compose logs -f worker

# Reiniciar servicio
docker-compose restart app

# Detener servicios
docker-compose down

# Detener y eliminar volúmenes
docker-compose down -v

# Rebuild después de cambios
docker-compose up -d --build

# Escalar workers
docker-compose up -d --scale worker=4
```

### 2. Docker Standalone

#### Build

```bash
# Build imagen
docker build -t github-autonomous-agent:latest .

# O con tag específico
docker build -t github-autonomous-agent:v1.0.0 .
```

#### Run

```bash
# Run con .env
docker run -d \
  --name github-autonomous-agent \
  -p 8030:8030 \
  --env-file .env \
  -v $(pwd)/storage:/app/storage \
  --restart unless-stopped \
  github-autonomous-agent:latest

# Run con variables inline
docker run -d \
  --name github-autonomous-agent \
  -p 8030:8030 \
  -e GITHUB_TOKEN=your_token \
  -e SECRET_KEY=your_secret \
  -e DATABASE_URL=postgresql+asyncpg://... \
  -v $(pwd)/storage:/app/storage \
  github-autonomous-agent:latest
```

#### Docker Compose para Producción

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8030:8030"
    env_file:
      - .env.production
    volumes:
      - ./storage:/app/storage
    restart: unless-stopped
    depends_on:
      - db
      - redis
    networks:
      - app-network

  worker:
    build: .
    command: celery -A core.worker worker --loglevel=info --concurrency=4
    env_file:
      - .env.production
    volumes:
      - ./storage:/app/storage
    restart: unless-stopped
    depends_on:
      - db
      - redis
    networks:
      - app-network

  db:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: github_agent
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped
    networks:
      - app-network

  redis:
    image: redis:7-alpine
    restart: unless-stopped
    networks:
      - app-network

volumes:
  postgres_data:

networks:
  app-network:
    driver: bridge
```

```bash
# Usar compose de producción
docker-compose -f docker-compose.prod.yml up -d
```

---

## 🖥️ VPS/Dedicado

### Requisitos Mínimos

- **OS**: Ubuntu 20.04+ / Debian 11+ / CentOS 8+
- **RAM**: 2GB mínimo (4GB recomendado)
- **CPU**: 2 cores mínimo
- **Disco**: 20GB mínimo
- **Python**: 3.10+
- **PostgreSQL**: 13+ (opcional, SQLite funciona)
- **Redis**: 6+
- **Nginx**: 1.18+ (opcional pero recomendado)

### Instalación Paso a Paso

#### 1. Preparar Sistema

```bash
# Actualizar sistema
sudo apt update && sudo apt upgrade -y

# Instalar dependencias
sudo apt install -y \
  python3.10 \
  python3.10-venv \
  python3-pip \
  git \
  postgresql \
  postgresql-contrib \
  redis-server \
  nginx \
  certbot \
  python3-certbot-nginx
```

#### 2. Clonar Repositorio

```bash
# Crear usuario para la aplicación
sudo useradd -m -s /bin/bash github-agent

# Cambiar a usuario
sudo su - github-agent

# Clonar repositorio
git clone <repo-url> /home/github-agent/app
cd /home/github-agent/app
```

#### 3. Setup de Aplicación

```bash
# Crear entorno virtual
python3.10 -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-prod.txt

# Configurar .env
cp .env.example .env
nano .env  # Editar con valores de producción
```

#### 4. Configurar PostgreSQL

```bash
# Crear base de datos
sudo -u postgres psql
CREATE DATABASE github_agent;
CREATE USER github_agent_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE github_agent TO github_agent_user;
\q

# Actualizar DATABASE_URL en .env
DATABASE_URL=postgresql+asyncpg://github_agent_user:secure_password@localhost/github_agent
```

#### 5. Ejecutar Migraciones

```bash
# Activar entorno virtual
source venv/bin/activate

# Ejecutar migraciones
python scripts/migrate-db.py upgrade head
```

#### 6. Configurar Systemd Service

```bash
# Crear service file
sudo nano /etc/systemd/system/github-agent.service
```

```ini
[Unit]
Description=GitHub Autonomous Agent
After=network.target postgresql.service redis.service

[Service]
Type=simple
User=github-agent
WorkingDirectory=/home/github-agent/app
Environment="PATH=/home/github-agent/app/venv/bin"
ExecStart=/home/github-agent/app/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8030 --workers 4
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Habilitar y iniciar servicio
sudo systemctl daemon-reload
sudo systemctl enable github-agent
sudo systemctl start github-agent

# Verificar estado
sudo systemctl status github-agent

# Ver logs
sudo journalctl -u github-agent -f
```

#### 7. Configurar Celery Worker

```bash
# Crear service file para worker
sudo nano /etc/systemd/system/github-agent-worker.service
```

```ini
[Unit]
Description=GitHub Autonomous Agent Celery Worker
After=network.target redis.service

[Service]
Type=simple
User=github-agent
WorkingDirectory=/home/github-agent/app
Environment="PATH=/home/github-agent/app/venv/bin"
ExecStart=/home/github-agent/app/venv/bin/celery -A core.worker worker --loglevel=info --concurrency=4
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Habilitar y iniciar
sudo systemctl enable github-agent-worker
sudo systemctl start github-agent-worker
```

#### 8. Configurar Nginx

```bash
# Crear configuración
sudo nano /etc/nginx/sites-available/github-agent
```

```nginx
server {
    listen 80;
    server_name your-domain.com;

    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    # Gzip compression
    gzip on;
    gzip_types text/plain text/css application/json application/javascript;

    # Proxy to FastAPI
    location / {
        proxy_pass http://127.0.0.1:8030;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

```bash
# Habilitar sitio
sudo ln -s /etc/nginx/sites-available/github-agent /etc/nginx/sites-enabled/

# Verificar configuración
sudo nginx -t

# Recargar Nginx
sudo systemctl reload nginx
```

#### 9. Configurar SSL con Let's Encrypt

```bash
# Obtener certificado
sudo certbot --nginx -d your-domain.com

# Auto-renewal (ya configurado por certbot)
sudo certbot renew --dry-run
```

---

## ☁️ Cloud Platforms

### Heroku

#### Setup

```bash
# 1. Instalar Heroku CLI
# https://devcenter.heroku.com/articles/heroku-cli

# 2. Login
heroku login

# 3. Crear app
heroku create github-autonomous-agent

# 4. Configurar variables
heroku config:set GITHUB_TOKEN=your_token
heroku config:set SECRET_KEY=your_secret
heroku config:set DATABASE_URL=postgresql://...
heroku config:set REDIS_URL=redis://...

# 5. Agregar addons
heroku addons:create heroku-postgresql:hobby-dev
heroku addons:create heroku-redis:hobby-dev

# 6. Deploy
git push heroku main

# 7. Ver logs
heroku logs --tail
```

#### Procfile

```
web: uvicorn main:app --host 0.0.0.0 --port $PORT --workers 4
worker: celery -A core.worker worker --loglevel=info
```

### Railway

1. Conectar repositorio en Railway
2. Configurar variables de entorno
3. Railway detecta automáticamente y despliega
4. Configurar servicios (PostgreSQL, Redis)

### Render

1. Conectar repositorio en Render
2. Seleccionar "Web Service"
3. Configurar:
   - **Build Command**: `pip install -r requirements.txt -r requirements-prod.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Agregar variables de entorno
5. Agregar servicios (PostgreSQL, Redis)
6. Deploy

### AWS (EC2/ECS)

Ver `deployment/aws/` para configuraciones detalladas de:
- EC2 con Auto Scaling
- ECS con Fargate
- RDS para PostgreSQL
- ElastiCache para Redis
- CloudFront para CDN
- Route 53 para DNS

---

## 🔒 Seguridad

### Checklist de Seguridad

- [ ] `SECRET_KEY` generada y segura (32+ caracteres)
- [ ] `DEBUG=False` en producción
- [ ] HTTPS habilitado (certificado SSL válido)
- [ ] CORS configurado restrictivamente
- [ ] Rate limiting habilitado
- [ ] Tokens de GitHub rotados regularmente
- [ ] PostgreSQL en lugar de SQLite
- [ ] Firewall configurado (solo puertos necesarios)
- [ ] Logging configurado (sin información sensible)
- [ ] Monitoring y alertas configurados
- [ ] Backup automático configurado
- [ ] Secrets no hardcodeados en código
- [ ] Dependencias actualizadas (sin vulnerabilidades)
- [ ] Headers de seguridad configurados

### Headers de Seguridad

```python
# En main.py o middleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["your-domain.com", "*.your-domain.com"]
)
```

### Rate Limiting

```python
# Configurar rate limiting
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.get("/api/endpoint")
@limiter.limit("10/minute")
async def endpoint():
    ...
```

### Validación de Secretos

```bash
# Verificar que no hay secretos en código
python scripts/security-check.py

# O con git-secrets
git secrets --scan
```

---

## 📊 Monitoring

### Health Checks

El endpoint `/health` está disponible:

```bash
# Health check básico
curl http://localhost:8030/health

# Health check detallado
curl http://localhost:8030/health/detailed
```

### Logging

```bash
# Ver logs en tiempo real
tail -f storage/logs/app.log

# Con systemd
sudo journalctl -u github-agent -f

# Con Docker
docker-compose logs -f app
```

### Métricas

#### Prometheus (si está habilitado)

```bash
# Endpoint de métricas
curl http://localhost:8030/metrics
```

#### Flower (Celery Monitor)

```bash
# Acceder a Flower
open http://localhost:5555
```

### Sentry (Error Tracking)

1. Crear cuenta en [Sentry](https://sentry.io)
2. Crear proyecto
3. Obtener DSN
4. Configurar en `.env`:
   ```bash
   SENTRY_DSN=https://your-dsn@sentry.io/project-id
   ```

### Alertas

Configurar alertas para:
- Errores críticos
- Alta latencia
- Alta tasa de errores
- Recursos del sistema (CPU, memoria, disco)

---

## 💾 Backup y Restore

### Backup de Base de Datos

#### PostgreSQL

```bash
# Backup manual
pg_dump -U postgres github_agent > backup_$(date +%Y%m%d_%H%M%S).sql

# Backup con compresión
pg_dump -U postgres -Fc github_agent > backup_$(date +%Y%m%d_%H%M%S).dump

# Backup remoto
pg_dump -h remote_host -U postgres github_agent > backup.sql
```

#### SQLite

```bash
# Backup simple
cp storage/github_agent.db backup_$(date +%Y%m%d_%H%M%S).db

# Con compresión
tar -czf backup_$(date +%Y%m%d_%H%M%S).tar.gz storage/github_agent.db
```

### Backup Automático

#### Script de Backup

```bash
#!/bin/bash
# scripts/backup.sh

BACKUP_DIR="/home/github-agent/backups"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/backup_$DATE.sql"

mkdir -p $BACKUP_DIR

# Backup PostgreSQL
pg_dump -U postgres github_agent > $BACKUP_FILE

# Comprimir
gzip $BACKUP_FILE

# Eliminar backups antiguos (mantener últimos 30 días)
find $BACKUP_DIR -name "backup_*.sql.gz" -mtime +30 -delete

echo "Backup completado: $BACKUP_FILE.gz"
```

#### Cron Job

```bash
# Editar crontab
crontab -e

# Backup diario a las 2 AM
0 2 * * * /home/github-agent/app/scripts/backup.sh
```

### Restore

#### PostgreSQL

```bash
# Restore desde SQL
psql -U postgres github_agent < backup_20240101.sql

# Restore desde dump comprimido
pg_restore -U postgres -d github_agent backup_20240101.dump
```

#### SQLite

```bash
# Restore
cp backup_20240101.db storage/github_agent.db
```

---

## 🔄 Actualizaciones

### Actualizar Aplicación

```bash
# 1. Backup primero
./scripts/backup.sh

# 2. Pull cambios
git pull origin main

# 3. Activar entorno virtual
source venv/bin/activate

# 4. Reinstalar dependencias (si hay cambios)
pip install -r requirements.txt -r requirements-prod.txt

# 5. Ejecutar migraciones
python scripts/migrate-db.py upgrade head

# 6. Reiniciar servicios
sudo systemctl restart github-agent
sudo systemctl restart github-agent-worker

# O con Docker:
docker-compose restart app worker
```

### Rollback

```bash
# 1. Revertir código
git checkout <previous-commit>

# 2. Revertir migraciones (si es necesario)
python scripts/migrate-db.py downgrade -1

# 3. Reiniciar servicios
sudo systemctl restart github-agent
```

---

## 🐛 Troubleshooting

### Aplicación no inicia

```bash
# Ver logs
sudo journalctl -u github-agent -f
# O con Docker:
docker-compose logs -f app

# Verificar configuración
python scripts/validate-env.py

# Verificar dependencias
python scripts/check-dependencies.py

# Health check
python scripts/health-check.py
```

### Error de conexión a base de datos

```bash
# Verificar que PostgreSQL está corriendo
sudo systemctl status postgresql

# Verificar conexión
psql -U postgres -d github_agent -c "SELECT 1;"

# Verificar DATABASE_URL en .env
cat .env | grep DATABASE_URL
```

### Error de conexión a Redis

```bash
# Verificar que Redis está corriendo
redis-cli ping

# Verificar configuración
redis-cli -u redis://localhost:6379/0 ping

# Verificar REDIS_URL en .env
cat .env | grep REDIS_URL
```

### Rate limit de GitHub

- Usar autenticación con token
- Implementar cache
- Reducir frecuencia de requests
- Usar múltiples tokens si es necesario

### Performance Issues

```bash
# Verificar recursos del sistema
htop
# o
top

# Verificar logs de errores
grep "ERROR" storage/logs/app.log

# Verificar métricas de Celery
# Acceder a Flower: http://localhost:5555
```

---

## 📈 Performance

### Optimizaciones

1. **Base de datos**: Usar PostgreSQL con connection pooling
2. **Cache**: Habilitar Redis cache
3. **Workers**: Ajustar número de workers según CPU
4. **CDN**: Usar CDN para assets estáticos
5. **Compresión**: Habilitar gzip en Nginx
6. **Database Indexing**: Agregar índices en queries frecuentes

### Monitoreo de Performance

- Usar `psutil` para métricas del sistema
- Configurar Prometheus para métricas
- Usar APM (Sentry, New Relic, Datadog)
- Monitorear latencia de endpoints
- Monitorear queue de Celery

---

## 🔗 Recursos

### Documentación

- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Nginx Configuration](https://nginx.org/en/docs/)
- [Let's Encrypt](https://letsencrypt.org/docs/)

### Herramientas

- [Sentry](https://sentry.io) - Error tracking
- [Prometheus](https://prometheus.io) - Monitoring
- [Grafana](https://grafana.com) - Visualización
- [Flower](https://flower.readthedocs.io/) - Celery monitor

---

**¿Necesitas más ayuda?** Consulta [TROUBLESHOOTING.md](TROUBLESHOOTING.md) o abre un issue en GitHub.

**Última actualización:** Diciembre 2024
