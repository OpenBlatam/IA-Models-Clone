# Guía de Despliegue - AI Document Processor

## 🚀 Guía Completa de Despliegue

### 📋 Tabla de Contenidos
1. [Despliegue Local](#despliegue-local)
2. [Despliegue con Docker](#despliegue-con-docker)
3. [Despliegue en la Nube](#despliegue-en-la-nube)
4. [Configuración de Producción](#configuración-de-producción)
5. [Monitoreo y Logs](#monitoreo-y-logs)
6. [Escalabilidad](#escalabilidad)

---

## 🏠 Despliegue Local

### Requisitos Previos
- Python 3.8+
- pip o pipenv
- Git

### Pasos de Instalación

1. **Clonar el repositorio**
```bash
git clone <repository-url>
cd ai_document_processor
```

2. **Configurar entorno virtual**
```bash
# Con venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate     # Windows

# Con pipenv
pipenv install
pipenv shell
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

4. **Configurar variables de entorno**
```bash
cp env.example .env
# Editar .env con tus configuraciones
```

5. **Ejecutar configuración inicial**
```bash
python scripts/setup.py
```

6. **Iniciar el servidor**
```bash
python main.py
```

7. **Verificar funcionamiento**
```bash
curl http://localhost:8001/ai-document-processor/health
```

---

## 🐳 Despliegue con Docker

### Docker Compose (Recomendado)

1. **Configurar variables de entorno**
```bash
cp env.example .env
# Editar .env con tus configuraciones
```

2. **Construir y ejecutar**
```bash
docker-compose up -d
```

3. **Verificar servicios**
```bash
docker-compose ps
docker-compose logs ai-document-processor
```

### Docker Manual

1. **Construir imagen**
```bash
docker build -t ai-document-processor .
```

2. **Ejecutar contenedor**
```bash
docker run -d \
  --name ai-document-processor \
  -p 8001:8001 \
  -e OPENAI_API_KEY=tu_clave_aqui \
  -v $(pwd)/logs:/app/logs \
  ai-document-processor
```

3. **Verificar funcionamiento**
```bash
docker logs ai-document-processor
curl http://localhost:8001/ai-document-processor/health
```

---

## ☁️ Despliegue en la Nube

### AWS

#### EC2 + Docker
```bash
# En instancia EC2
sudo yum update -y
sudo yum install -y docker
sudo systemctl start docker
sudo systemctl enable docker

# Clonar y ejecutar
git clone <repository-url>
cd ai_document_processor
sudo docker-compose up -d
```

#### ECS (Elastic Container Service)
```yaml
# task-definition.json
{
  "family": "ai-document-processor",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "ai-document-processor",
      "image": "your-account.dkr.ecr.region.amazonaws.com/ai-document-processor:latest",
      "portMappings": [
        {
          "containerPort": 8001,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "OPENAI_API_KEY",
          "value": "your-api-key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/ai-document-processor",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

### Google Cloud Platform

#### Cloud Run
```bash
# Construir y subir imagen
gcloud builds submit --tag gcr.io/PROJECT-ID/ai-document-processor

# Desplegar en Cloud Run
gcloud run deploy ai-document-processor \
  --image gcr.io/PROJECT-ID/ai-document-processor \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars OPENAI_API_KEY=tu_clave_aqui
```

### Azure

#### Container Instances
```bash
# Crear grupo de recursos
az group create --name ai-doc-processor --location eastus

# Desplegar contenedor
az container create \
  --resource-group ai-doc-processor \
  --name ai-document-processor \
  --image your-registry/ai-document-processor:latest \
  --dns-name-label ai-doc-processor \
  --ports 8001 \
  --environment-variables OPENAI_API_KEY=tu_clave_aqui
```

---

## ⚙️ Configuración de Producción

### Variables de Entorno Críticas

```bash
# Producción
DEBUG=false
HOST=0.0.0.0
PORT=8001

# OpenAI (crítico para mejores resultados)
OPENAI_API_KEY=sk-tu_clave_api_real

# Seguridad
SECRET_KEY=clave_super_secreta_y_larga
CORS_ORIGINS=https://tu-dominio.com,https://app.tu-dominio.com

# Base de datos (opcional)
DATABASE_URL=postgresql://user:pass@db-host:5432/ai_doc_processor
REDIS_URL=redis://redis-host:6379

# Límites
MAX_FILE_SIZE=52428800
MAX_CONCURRENT_REQUESTS=10
REQUEST_TIMEOUT=300

# Logging
LOG_LEVEL=INFO
```

### Configuración de Nginx (Proxy Reverso)

```nginx
server {
    listen 80;
    server_name tu-dominio.com;

    # Redirigir HTTP a HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name tu-dominio.com;

    # Certificados SSL
    ssl_certificate /path/to/certificate.crt;
    ssl_certificate_key /path/to/private.key;

    # Configuración SSL
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;

    # Límites de archivo
    client_max_body_size 50M;
    client_body_timeout 60s;
    client_header_timeout 60s;

    # Proxy al servicio
    location / {
        proxy_pass http://localhost:8001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 300s;
    }

    # Health check
    location /ai-document-processor/health {
        proxy_pass http://localhost:8001/ai-document-processor/health;
        access_log off;
    }
}
```

### Configuración de Systemd (Linux)

```ini
# /etc/systemd/system/ai-document-processor.service
[Unit]
Description=AI Document Processor
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/ai-document-processor
Environment=PATH=/opt/ai-document-processor/venv/bin
ExecStart=/opt/ai-document-processor/venv/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Habilitar y iniciar servicio
sudo systemctl daemon-reload
sudo systemctl enable ai-document-processor
sudo systemctl start ai-document-processor
sudo systemctl status ai-document-processor
```

---

## 📊 Monitoreo y Logs

### Configuración de Logs

```python
# logging.conf
[loggers]
keys=root,ai_document_processor

[handlers]
keys=consoleHandler,fileHandler,rotatingFileHandler

[formatters]
keys=simpleFormatter,detailedFormatter

[logger_root]
level=INFO
handlers=consoleHandler

[logger_ai_document_processor]
level=INFO
handlers=fileHandler,rotatingFileHandler
qualname=ai_document_processor
propagate=0

[handler_consoleHandler]
class=StreamHandler
level=INFO
formatter=simpleFormatter
args=(sys.stdout,)

[handler_fileHandler]
class=FileHandler
level=INFO
formatter=detailedFormatter
args=('logs/ai_document_processor.log',)

[handler_rotatingFileHandler]
class=handlers.RotatingFileHandler
level=INFO
formatter=detailedFormatter
args=('logs/ai_document_processor.log', 10485760, 5)

[formatter_simpleFormatter]
format=%(asctime)s - %(name)s - %(levelname)s - %(message)s

[formatter_detailedFormatter]
format=%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s
```

### Métricas con Prometheus

```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Métricas
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ACTIVE_CONNECTIONS = Gauge('active_connections', 'Active connections')
DOCUMENTS_PROCESSED = Counter('documents_processed_total', 'Total documents processed', ['format', 'status'])

# En main.py
start_http_server(8002)  # Puerto para métricas
```

### Health Checks Avanzados

```python
# health.py
import asyncio
import aiohttp
from typing import Dict, Any

class HealthChecker:
    async def check_openai(self) -> Dict[str, Any]:
        """Verifica conectividad con OpenAI"""
        try:
            # Verificar API key y conectividad
            return {"status": "healthy", "response_time": 0.1}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def check_database(self) -> Dict[str, Any]:
        """Verifica conectividad con base de datos"""
        # Implementar verificación de DB
        pass
    
    async def check_redis(self) -> Dict[str, Any]:
        """Verifica conectividad con Redis"""
        # Implementar verificación de Redis
        pass
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Obtiene estado completo del sistema"""
        checks = await asyncio.gather(
            self.check_openai(),
            self.check_database(),
            self.check_redis(),
            return_exceptions=True
        )
        
        return {
            "status": "healthy" if all(c.get("status") == "healthy" for c in checks) else "unhealthy",
            "checks": {
                "openai": checks[0],
                "database": checks[1],
                "redis": checks[2]
            }
        }
```

---

## 📈 Escalabilidad

### Horizontal Scaling con Load Balancer

```yaml
# docker-compose.production.yml
version: '3.8'

services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - ai-document-processor-1
      - ai-document-processor-2
      - ai-document-processor-3

  ai-document-processor-1:
    build: .
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - INSTANCE_ID=1
    volumes:
      - ./logs:/app/logs

  ai-document-processor-2:
    build: .
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - INSTANCE_ID=2
    volumes:
      - ./logs:/app/logs

  ai-document-processor-3:
    build: .
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - INSTANCE_ID=3
    volumes:
      - ./logs:/app/logs

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=ai_document_processor
      - POSTGRES_USER=admin
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  redis_data:
  postgres_data:
```

### Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-document-processor
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-document-processor
  template:
    metadata:
      labels:
        app: ai-document-processor
    spec:
      containers:
      - name: ai-document-processor
        image: your-registry/ai-document-processor:latest
        ports:
        - containerPort: 8001
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: ai-doc-secrets
              key: openai-api-key
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /ai-document-processor/health
            port: 8001
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ai-document-processor/health
            port: 8001
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: ai-document-processor-service
spec:
  selector:
    app: ai-document-processor
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8001
  type: LoadBalancer
```

---

## 🔧 Comandos de Mantenimiento

### Backup y Restauración

```bash
# Backup de base de datos
pg_dump ai_document_processor > backup_$(date +%Y%m%d_%H%M%S).sql

# Backup de logs
tar -czf logs_backup_$(date +%Y%m%d_%H%M%S).tar.gz logs/

# Restauración
psql ai_document_processor < backup_20251015_120000.sql
```

### Actualización de Servicio

```bash
# Con Docker Compose
docker-compose pull
docker-compose up -d

# Con systemd
sudo systemctl stop ai-document-processor
git pull
pip install -r requirements.txt
sudo systemctl start ai-document-processor

# Verificar estado
sudo systemctl status ai-document-processor
```

### Limpieza de Archivos Temporales

```bash
# Limpiar archivos temporales
find /tmp -name "*ai_document_processor*" -mtime +1 -delete

# Limpiar logs antiguos
find logs/ -name "*.log.*" -mtime +30 -delete
```

---

## 🚨 Troubleshooting

### Problemas Comunes

1. **Error de memoria insuficiente**
```bash
# Aumentar límites de memoria
docker run -m 2g ai-document-processor
```

2. **Timeout en procesamiento**
```bash
# Aumentar timeout
export REQUEST_TIMEOUT=600
```

3. **Error de OpenAI API**
```bash
# Verificar API key
curl -H "Authorization: Bearer $OPENAI_API_KEY" https://api.openai.com/v1/models
```

4. **Archivos demasiado grandes**
```bash
# Aumentar límite de archivo
export MAX_FILE_SIZE=104857600  # 100MB
```

### Logs de Debug

```bash
# Habilitar debug
export DEBUG=true
export LOG_LEVEL=DEBUG

# Ver logs en tiempo real
docker-compose logs -f ai-document-processor

# Ver logs específicos
tail -f logs/ai_document_processor.log
```

---

## 📞 Soporte

Para soporte técnico:
- Revisar logs del sistema
- Verificar configuración de variables de entorno
- Consultar documentación de la API
- Contactar al equipo de desarrollo

**¡Sistema listo para producción!** 🚀


