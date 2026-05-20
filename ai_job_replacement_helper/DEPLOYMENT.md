# 🚀 Deployment Guide

Guía completa para desplegar AI Job Replacement Helper.

## 📦 Opción 1: Docker Compose (Recomendado)

### Requisitos
- Docker
- Docker Compose

### Pasos

1. **Clonar y configurar**
```bash
cd ai_job_replacement_helper
cp .env.example .env
# Editar .env con tus configuraciones
```

2. **Iniciar servicios**
```bash
docker-compose up -d
```

3. **Verificar**
```bash
curl http://localhost:8030/health
```

## 📦 Opción 2: Instalación Manual

### Requisitos
- Python 3.11+
- PostgreSQL (opcional)
- Redis (opcional)

### Pasos

1. **Setup inicial**
```bash
# Linux/Mac
bash scripts/setup.sh

# Windows
scripts\setup.bat
```

2. **Configurar variables de entorno**
```bash
# Editar .env
nano .env
```

3. **Ejecutar servidor**
```bash
python main.py
```

## ☁️ Opción 3: Cloud Deployment

### Heroku

```bash
heroku create ai-job-helper
heroku addons:create heroku-postgresql
heroku addons:create heroku-redis
git push heroku main
```

### AWS (ECS/Fargate)

1. Build Docker image
2. Push to ECR
3. Deploy to ECS/Fargate
4. Configure ALB

### Google Cloud Run

```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/ai-job-helper
gcloud run deploy --image gcr.io/PROJECT_ID/ai-job-helper
```

## 🔧 Configuración de Producción

### Variables de Entorno Importantes

```env
APP_ENV=production
DEBUG=False
SECRET_KEY=your-secret-key-here
DATABASE_URL=postgresql://...
REDIS_URL=redis://...
LINKEDIN_API_KEY=...
LINKEDIN_API_SECRET=...
```

### Seguridad

- ✅ Usar HTTPS
- ✅ Configurar CORS apropiadamente
- ✅ Rate limiting habilitado
- ✅ Validación de inputs
- ✅ Sanitización de datos

### Performance

- ✅ Habilitar caché (Redis)
- ✅ Usar base de datos para persistencia
- ✅ Configurar workers (Gunicorn/Uvicorn workers)
- ✅ CDN para assets estáticos

## 📊 Monitoring

### Health Checks

```bash
curl http://localhost:8030/health
curl http://localhost:8030/health/detailed
```

### Métricas

- Prometheus metrics (si se configura)
- Logs en `/logs`
- Database monitoring

## 🔄 CI/CD

El proyecto incluye GitHub Actions workflow (`.github/workflows/ci.yml`):

- Tests automáticos
- Linting
- Build de Docker
- Deploy (configurar según necesidad)

## 📝 Checklist de Deployment

- [ ] Variables de entorno configuradas
- [ ] Base de datos migrada
- [ ] Redis configurado
- [ ] HTTPS habilitado
- [ ] CORS configurado
- [ ] Rate limiting activo
- [ ] Logs configurados
- [ ] Monitoring activo
- [ ] Backup configurado
- [ ] Documentación actualizada




