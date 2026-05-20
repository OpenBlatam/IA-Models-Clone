# Guía de Despliegue - Music Analyzer AI

## 🚀 Opciones de Despliegue

### 1. Despliegue Local

#### Desarrollo

```bash
# Iniciar servidor de desarrollo
python main.py

# O con uvicorn
uvicorn main:app --host 0.0.0.0 --port 8010 --reload
```

#### Producción Local

```bash
# Con múltiples workers
uvicorn main:app --host 0.0.0.0 --port 8010 --workers 4
```

### 2. Docker

#### Construir Imagen

```bash
# Crear Dockerfile si no existe
cat > Dockerfile << EOF
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8010

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8010"]
EOF

# Construir imagen
docker build -t music-analyzer-ai .
```

#### Ejecutar Contenedor

```bash
docker run -d \
  -p 8010:8010 \
  -e SPOTIFY_CLIENT_ID=tu_client_id \
  -e SPOTIFY_CLIENT_SECRET=tu_client_secret \
  -e HOST=0.0.0.0 \
  -e PORT=8010 \
  --name music-analyzer-ai \
  music-analyzer-ai
```

#### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  music-analyzer:
    build: .
    ports:
      - "8010:8010"
    environment:
      - SPOTIFY_CLIENT_ID=${SPOTIFY_CLIENT_ID}
      - SPOTIFY_CLIENT_SECRET=${SPOTIFY_CLIENT_SECRET}
      - HOST=0.0.0.0
      - PORT=8010
    depends_on:
      - redis
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
```

```bash
# Iniciar
docker-compose up -d

# Detener
docker-compose down
```

### 3. AWS

#### AWS Lambda

```python
# lambda_handler.py
from mangum import Mangum
from main import app

handler = Mangum(app)
```

```bash
# Instalar mangum
pip install mangum

# Desplegar con Serverless Framework
serverless deploy
```

#### AWS ECS/Fargate

```bash
# Construir y subir imagen
docker build -t music-analyzer-ai .
docker tag music-analyzer-ai:latest 123456789.dkr.ecr.us-east-1.amazonaws.com/music-analyzer-ai:latest
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/music-analyzer-ai:latest

# Crear task definition y servicio en ECS
```

#### AWS EC2

```bash
# Conectar a instancia EC2
ssh -i key.pem ubuntu@ec2-instance

# Instalar dependencias
sudo apt-get update
sudo apt-get install python3-pip nginx

# Clonar y configurar
git clone <repo>
cd music_analyzer_ai
pip3 install -r requirements.txt

# Configurar systemd service
sudo nano /etc/systemd/system/music-analyzer.service
```

```ini
[Unit]
Description=Music Analyzer AI
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/music_analyzer_ai
Environment="PATH=/usr/bin"
ExecStart=/usr/bin/python3 main.py
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
# Iniciar servicio
sudo systemctl start music-analyzer
sudo systemctl enable music-analyzer
```

### 4. Google Cloud Platform

#### Cloud Run

```bash
# Construir y subir imagen
gcloud builds submit --tag gcr.io/PROJECT_ID/music-analyzer-ai

# Desplegar
gcloud run deploy music-analyzer-ai \
  --image gcr.io/PROJECT_ID/music-analyzer-ai \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### 5. Azure

#### Azure App Service

```bash
# Instalar Azure CLI
az login

# Crear app service
az webapp create \
  --resource-group myResourceGroup \
  --plan myAppServicePlan \
  --name music-analyzer-ai \
  --runtime "PYTHON|3.11"

# Desplegar
az webapp deployment source config-zip \
  --resource-group myResourceGroup \
  --name music-analyzer-ai \
  --src app.zip
```

### 6. Heroku

```bash
# Instalar Heroku CLI
heroku login

# Crear app
heroku create music-analyzer-ai

# Configurar variables de entorno
heroku config:set SPOTIFY_CLIENT_ID=tu_id
heroku config:set SPOTIFY_CLIENT_SECRET=tu_secret

# Desplegar
git push heroku main
```

## 🔧 Configuración de Producción

### Variables de Entorno

```env
# Producción
HOST=0.0.0.0
PORT=8010
DEBUG=False
ENVIRONMENT=production
LOG_LEVEL=WARNING

# Spotify
SPOTIFY_CLIENT_ID=tu_production_client_id
SPOTIFY_CLIENT_SECRET=tu_production_client_secret
SPOTIFY_REDIRECT_URI=https://tu-dominio.com/callback

# Redis
REDIS_URL=redis://production-redis:6379/0

# Database
DATABASE_URL=postgresql://user:pass@prod-db:5432/music_analyzer
```

### Nginx Reverse Proxy

```nginx
# /etc/nginx/sites-available/music-analyzer
server {
    listen 80;
    server_name tu-dominio.com;

    location / {
        proxy_pass http://localhost:8010;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

```bash
# Habilitar sitio
sudo ln -s /etc/nginx/sites-available/music-analyzer /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### SSL/TLS con Let's Encrypt

```bash
# Instalar certbot
sudo apt-get install certbot python3-certbot-nginx

# Obtener certificado
sudo certbot --nginx -d tu-dominio.com

# Renovación automática
sudo certbot renew --dry-run
```

## 📊 Monitoreo

### Health Checks

```bash
# Health check endpoint
curl http://localhost:8010/health

# Configurar en load balancer
# AWS ALB: Health check path: /health
# GCP: Health check path: /health
```

### Logging

```python
# Configurar logging estructurado
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_FILE=/var/log/music-analyzer/app.log
```

### Métricas

```env
# Prometheus
PROMETHEUS_ENABLED=True
PROMETHEUS_PORT=9090
```

## 🔒 Seguridad

### Rate Limiting

```env
RATE_LIMIT_ENABLED=True
RATE_LIMIT_MAX_REQUESTS=100
RATE_LIMIT_WINDOW_SECONDS=60
```

### CORS

```env
ALLOWED_ORIGINS=https://tu-dominio.com,https://www.tu-dominio.com
```

### Secrets Management

```bash
# AWS Secrets Manager
aws secretsmanager create-secret \
  --name music-analyzer/spotify-credentials \
  --secret-string file://credentials.json

# Usar en código
import boto3
secrets = boto3.client('secretsmanager')
credentials = secrets.get_secret_value(SecretId='music-analyzer/spotify-credentials')
```

## 🚀 CI/CD

### GitHub Actions

```yaml
# .github/workflows/deploy.yml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Deploy to production
        run: |
          # Comandos de despliegue
```

## 📚 Recursos Adicionales

- **Docker**: https://docs.docker.com/
- **AWS**: https://aws.amazon.com/documentation/
- **GCP**: https://cloud.google.com/docs
- **Azure**: https://docs.microsoft.com/azure/

---

**Última actualización**: 2025  
**Versión**: 2.21.0






