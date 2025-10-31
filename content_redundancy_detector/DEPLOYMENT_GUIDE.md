# 🚀 Guía de Despliegue - Advanced Content Redundancy Detector

## 📋 Requisitos del Sistema

### Requisitos Mínimos
- **Python**: 3.8 o superior
- **RAM**: 4GB mínimo (8GB recomendado para AI/ML)
- **CPU**: 2 cores mínimo (4+ cores recomendado)
- **Almacenamiento**: 10GB espacio libre
- **GPU**: Opcional, pero recomendado para mejor rendimiento AI/ML

### Requisitos Recomendados para Producción
- **Python**: 3.9 o superior
- **RAM**: 16GB o más
- **CPU**: 8+ cores
- **GPU**: NVIDIA GPU con CUDA support
- **Almacenamiento**: SSD con 50GB+ espacio libre

## 🛠️ Instalación

### 1. Clonar el Repositorio
```bash
git clone <repository-url>
cd content_redundancy_detector
```

### 2. Crear Entorno Virtual
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate  # Windows
```

### 3. Instalar Dependencias
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Instalar Modelos de spaCy
```bash
python -m spacy download en_core_web_sm
```

### 5. Configurar Variables de Entorno
```bash
cp env.example .env
# Editar .env con tus configuraciones
```

## ⚙️ Configuración

### Variables de Entorno Principales

```bash
# Configuración de la aplicación
APP_NAME="Advanced Content Redundancy Detector"
APP_VERSION="2.0.0"
DEBUG=false
ENVIRONMENT=production

# Configuración del servidor
HOST=0.0.0.0
PORT=8000
WORKERS=4

# Configuración de AI/ML
MODEL_CACHE_SIZE=20
ENABLE_GPU=true
MODEL_TIMEOUT=60
EMBEDDING_MODEL=all-MiniLM-L6-v2
LANGUAGE_MODEL=distilbert-base-uncased

# Base de datos
DATABASE_URL=postgresql://user:password@localhost:5432/content_detector
REDIS_URL=redis://localhost:6379
CACHE_TTL=3600
MAX_CACHE_SIZE=10000

# Seguridad
SECRET_KEY=your-super-secret-key-here
ACCESS_TOKEN_EXPIRE_MINUTES=30
CORS_ORIGINS=["https://yourdomain.com"]

# Rate Limiting
RATE_LIMIT_REQUESTS=1000
RATE_LIMIT_WINDOW=60
RATE_LIMIT_BURST=100

# Monitoreo
LOG_LEVEL=INFO
ENABLE_METRICS=true
METRICS_PORT=9090
SENTRY_DSN=https://your-sentry-dsn@sentry.io/project-id
```

## 🐳 Despliegue con Docker

### 1. Crear Dockerfile
```dockerfile
FROM python:3.9-slim

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Establecer directorio de trabajo
WORKDIR /app

# Copiar requirements y instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Instalar modelo de spaCy
RUN python -m spacy download en_core_web_sm

# Copiar código de la aplicación
COPY . .

# Exponer puerto
EXPOSE 8000

# Comando de inicio
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### 2. Crear docker-compose.yml
```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/content_detector
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis
    volumes:
      - ./models:/app/models
    restart: unless-stopped

  db:
    image: postgres:13
    environment:
      - POSTGRES_DB=content_detector
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  redis:
    image: redis:6-alpine
    volumes:
      - redis_data:/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - app
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

### 3. Desplegar con Docker Compose
```bash
docker-compose up -d
```

## ☸️ Despliegue con Kubernetes

### 1. Crear ConfigMap
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: content-detector-config
data:
  APP_NAME: "Advanced Content Redundancy Detector"
  APP_VERSION: "2.0.0"
  DEBUG: "false"
  ENVIRONMENT: "production"
  HOST: "0.0.0.0"
  PORT: "8000"
  WORKERS: "4"
  LOG_LEVEL: "INFO"
  ENABLE_METRICS: "true"
  METRICS_PORT: "9090"
```

### 2. Crear Secret
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: content-detector-secrets
type: Opaque
data:
  SECRET_KEY: <base64-encoded-secret-key>
  DATABASE_URL: <base64-encoded-database-url>
  REDIS_URL: <base64-encoded-redis-url>
  SENTRY_DSN: <base64-encoded-sentry-dsn>
```

### 3. Crear Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: content-detector
spec:
  replicas: 3
  selector:
    matchLabels:
      app: content-detector
  template:
    metadata:
      labels:
        app: content-detector
    spec:
      containers:
      - name: content-detector
        image: your-registry/content-detector:latest
        ports:
        - containerPort: 8000
        - containerPort: 9090
        envFrom:
        - configMapRef:
            name: content-detector-config
        - secretRef:
            name: content-detector-secrets
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

### 4. Crear Service
```yaml
apiVersion: v1
kind: Service
metadata:
  name: content-detector-service
spec:
  selector:
    app: content-detector
  ports:
  - name: http
    port: 80
    targetPort: 8000
  - name: metrics
    port: 9090
    targetPort: 9090
  type: LoadBalancer
```

## 🔧 Configuración de Nginx

### nginx.conf
```nginx
events {
    worker_connections 1024;
}

http {
    upstream content_detector {
        server app:8000;
    }

    server {
        listen 80;
        server_name your-domain.com;

        location / {
            proxy_pass http://content_detector;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        location /metrics {
            proxy_pass http://content_detector:9090;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }
    }
}
```

## 📊 Monitoreo y Observabilidad

### 1. Prometheus Configuration
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'content-detector'
    static_configs:
      - targets: ['localhost:9090']
    metrics_path: /metrics
    scrape_interval: 5s
```

### 2. Grafana Dashboard
Importar dashboard con métricas:
- Request rate
- Response time
- Error rate
- AI/ML model performance
- Cache hit rate
- Memory usage

### 3. Logging Configuration
```python
# En config.py
import structlog

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)
```

## 🔒 Seguridad

### 1. Configuración de Firewall
```bash
# Permitir solo puertos necesarios
ufw allow 22    # SSH
ufw allow 80    # HTTP
ufw allow 443   # HTTPS
ufw allow 8000  # App (solo si es necesario)
ufw enable
```

### 2. SSL/TLS Configuration
```bash
# Usar Let's Encrypt
certbot --nginx -d your-domain.com
```

### 3. Rate Limiting
```python
# En middleware.py
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/ai/sentiment")
@limiter.limit("100/minute")
async def analyze_sentiment_endpoint(request: Request, input_data: ContentInput):
    # Endpoint implementation
    pass
```

## 🚀 Optimización de Rendimiento

### 1. Configuración de Workers
```bash
# Para producción
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4 --worker-class uvicorn.workers.UvicornWorker
```

### 2. Configuración de Redis
```bash
# redis.conf
maxmemory 2gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
```

### 3. Configuración de Base de Datos
```sql
-- PostgreSQL optimizations
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;
```

## 🔄 CI/CD Pipeline

### GitHub Actions
```yaml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest
    - name: Run tests
      run: pytest tests_ai_ml.py -v

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Deploy to server
      run: |
        # Deploy commands
        docker-compose up -d --build
```

## 📈 Escalabilidad

### 1. Horizontal Scaling
- Usar load balancer (nginx, HAProxy)
- Múltiples instancias de la aplicación
- Base de datos con réplicas de lectura

### 2. Vertical Scaling
- Aumentar RAM para modelos AI/ML
- Usar GPU para inferencia más rápida
- Optimizar configuración de base de datos

### 3. Caching Strategy
- Redis para caché de resultados
- CDN para archivos estáticos
- Caché de modelos AI/ML en memoria

## 🆘 Troubleshooting

### Problemas Comunes

1. **Error de memoria insuficiente**
   ```bash
   # Aumentar límites de memoria
   docker run --memory=8g your-image
   ```

2. **Modelos AI/ML no cargan**
   ```bash
   # Verificar instalación de spaCy
   python -m spacy download en_core_web_sm
   ```

3. **Conexión a base de datos falla**
   ```bash
   # Verificar configuración de red
   docker network ls
   docker network inspect bridge
   ```

4. **Rate limiting muy restrictivo**
   ```bash
   # Ajustar configuración
   RATE_LIMIT_REQUESTS=1000
   RATE_LIMIT_WINDOW=60
   ```

### Logs y Debugging
```bash
# Ver logs de la aplicación
docker-compose logs -f app

# Ver logs de base de datos
docker-compose logs -f db

# Ver logs de Redis
docker-compose logs -f redis

# Debug mode
DEBUG=true python app.py
```

## 📞 Soporte

Para soporte técnico:
- Crear issue en GitHub
- Contactar al equipo de desarrollo
- Revisar documentación oficial

---

**Nota**: Esta guía asume un entorno Linux/Unix. Para Windows, algunos comandos pueden variar.
















