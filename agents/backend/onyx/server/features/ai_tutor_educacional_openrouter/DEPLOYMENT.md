# 🚀 Guía de Deployment - AI Tutor Educacional

## 📋 Requisitos Previos

- Python 3.11+
- Docker y Docker Compose (opcional)
- Clave API de Open Router
- 2GB RAM mínimo
- 10GB espacio en disco

## 🐳 Deployment con Docker (Recomendado)

### 1. Configuración Inicial

```bash
# Clonar o navegar al directorio
cd ai_tutor_educacional_openrouter

# Crear archivo .env
cat > .env << EOF
OPENROUTER_API_KEY=tu-api-key-aqui
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_DEFAULT_MODEL=openai/gpt-4
PORT=8000
EOF
```

### 2. Construir y Ejecutar

```bash
# Construir imagen
docker-compose build

# Ejecutar en background
docker-compose up -d

# Ver logs
docker-compose logs -f

# Verificar estado
curl http://localhost:8000/api/tutor/health
```

### 3. Verificar Deployment

```bash
# Health check
curl http://localhost:8000/api/tutor/health

# Documentación API
open http://localhost:8000/docs
```

## 🖥️ Deployment Manual

### 1. Instalación de Dependencias

```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Configuración

```bash
# Configurar variables de entorno
export OPENROUTER_API_KEY="tu-api-key-aqui"
export OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
export OPENROUTER_DEFAULT_MODEL="openai/gpt-4"
```

### 3. Ejecutar Servidor

```bash
# Desarrollo
python main.py

# Producción con uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

## ☁️ Deployment en Cloud

### AWS (EC2 + Docker)

```bash
# 1. Conectar a instancia EC2
ssh -i key.pem ubuntu@ec2-instance

# 2. Instalar Docker
sudo apt-get update
sudo apt-get install docker.io docker-compose

# 3. Clonar repositorio
git clone <repo-url>
cd ai_tutor_educacional_openrouter

# 4. Configurar .env
nano .env

# 5. Ejecutar
docker-compose up -d
```

### Google Cloud Platform

```bash
# 1. Crear instancia
gcloud compute instances create tutor-api \
  --machine-type=e2-medium \
  --image-family=ubuntu-2004-lts \
  --image-project=ubuntu-os-cloud

# 2. Instalar Docker
# (mismo proceso que AWS)

# 3. Configurar firewall
gcloud compute firewall-rules create allow-tutor-api \
  --allow tcp:8000 \
  --source-ranges 0.0.0.0/0
```

### Heroku

```bash
# 1. Instalar Heroku CLI
# 2. Login
heroku login

# 3. Crear app
heroku create ai-tutor-educacional

# 4. Configurar variables
heroku config:set OPENROUTER_API_KEY=tu-api-key

# 5. Deploy
git push heroku main
```

## 🔒 Configuración de Seguridad

### Variables de Entorno Críticas

```bash
# Nunca commitear estas variables
OPENROUTER_API_KEY=secret
DATABASE_URL=secret
SECRET_KEY=secret
```

### HTTPS con Nginx

```nginx
server {
    listen 80;
    server_name tutor.example.com;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 📊 Monitoreo

### Health Checks

```bash
# Endpoint de health
curl http://localhost:8000/api/tutor/health

# Métricas
curl http://localhost:8000/api/tutor/metrics
```

### Logs

```bash
# Docker logs
docker-compose logs -f

# Logs de aplicación
tail -f logs/tutor.log
tail -f logs/tutor_errors.log
```

## 🔄 Actualización

```bash
# Con Docker
docker-compose pull
docker-compose up -d

# Manual
git pull
pip install -r requirements.txt
systemctl restart tutor-api
```

## 🧪 Testing en Producción

```bash
# Ejecutar tests
pytest tests/

# Con cobertura
pytest --cov=core --cov=api tests/
```

## 📈 Escalabilidad

### Horizontal Scaling

```yaml
# docker-compose.yml
services:
  tutor-api:
    deploy:
      replicas: 3
    # ...
```

### Load Balancer

```nginx
upstream tutor_backend {
    server localhost:8000;
    server localhost:8001;
    server localhost:8002;
}
```

## 🆘 Troubleshooting

### Problemas Comunes

1. **Error de conexión a Open Router**
   - Verificar API key
   - Verificar conectividad de red

2. **Memoria insuficiente**
   - Aumentar límites de Docker
   - Optimizar cache

3. **Rate limiting**
   - Ajustar límites en configuración
   - Implementar queue system

## 📞 Soporte

Para problemas de deployment, consultar:
- Logs en `logs/` directory
- Health endpoint: `/api/tutor/health`
- Documentación: `/docs`






