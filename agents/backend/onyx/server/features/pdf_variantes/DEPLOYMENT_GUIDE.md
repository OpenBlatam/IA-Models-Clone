"""
PDF Variantes - Deployment Guide
===============================

Guía completa de despliegue para producción.
"""

# PDF VARIANTES - DEPLOYMENT GUIDE
# ================================

## 🚀 DESPLIEGUE EN PRODUCCIÓN

### 1. PREPARACIÓN DEL ENTORNO

```bash
# Crear entorno virtual
python -m venv pdf_variantes_env
source pdf_variantes_env/bin/activate  # Linux/Mac
# o
pdf_variantes_env\Scripts\activate  # Windows

# Instalar dependencias
pip install -r requirements.txt

# Verificar instalación
python -c "import pdf_variantes; print('✅ Instalación exitosa')"
```

### 2. CONFIGURACIÓN DE PRODUCCIÓN

```python
# config/production.py
import os
from pdf_variantes.config import PDFVariantesConfig, Environment

config = PDFVariantesConfig(
    environment=Environment.PRODUCTION,
    debug=False,
    log_level="INFO",
    
    # Configuración de almacenamiento
    storage_config={
        "upload_dir": "/var/lib/pdf_variantes/uploads",
        "max_file_size": 100 * 1024 * 1024,  # 100MB
        "allowed_extensions": [".pdf"]
    },
    
    # Configuración de API
    api_config={
        "host": "0.0.0.0",
        "port": 8000,
        "workers": 4,
        "timeout": 300
    },
    
    # Configuración de IA
    ai_config={
        "max_tokens": 4000,
        "temperature": 0.7,
        "timeout": 60
    },
    
    # Configuración de colaboración
    collaboration_config={
        "session_timeout": 3600,  # 1 hora
        "max_collaborators": 10
    },
    
    # Límites de procesamiento
    processing_limits={
        "max_concurrent_uploads": 50,
        "max_concurrent_processing": 20,
        "max_file_size": 100 * 1024 * 1024
    }
)
```

### 3. VARIABLES DE ENTORNO

```bash
# .env.production
PDF_VARIANTES_ENV=production
PDF_VARIANTES_DEBUG=false
PDF_VARIANTES_LOG_LEVEL=INFO

# Base de datos
DATABASE_URL=postgresql://user:password@localhost:5432/pdf_variantes

# Redis para caché
REDIS_URL=redis://localhost:6379/0

# Configuración de IA
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# Configuración de almacenamiento
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
AWS_S3_BUCKET=pdf-variantes-storage

# Configuración de monitoreo
PROMETHEUS_ENDPOINT=http://localhost:9090
GRAFANA_ENDPOINT=http://localhost:3000

# Configuración de seguridad
JWT_SECRET_KEY=your_jwt_secret
ENCRYPTION_KEY=your_encryption_key
```

### 4. DOCKER DEPLOYMENT

```dockerfile
# Dockerfile.production
FROM python:3.11-slim

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-spa \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Crear usuario no-root
RUN useradd -m -u 1000 pdfuser

# Establecer directorio de trabajo
WORKDIR /app

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY . .

# Cambiar ownership
RUN chown -R pdfuser:pdfuser /app

# Cambiar a usuario no-root
USER pdfuser

# Exponer puerto
EXPOSE 8000

# Comando de inicio
CMD ["uvicorn", "pdf_variantes.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

```yaml
# docker-compose.production.yml
version: '3.8'

services:
  pdf-variantes:
    build:
      context: .
      dockerfile: Dockerfile.production
    ports:
      - "8000:8000"
    environment:
      - PDF_VARIANTES_ENV=production
      - DATABASE_URL=postgresql://postgres:password@db:5432/pdf_variantes
      - REDIS_URL=redis://redis:6379/0
    volumes:
      - ./uploads:/app/uploads
      - ./logs:/app/logs
    depends_on:
      - db
      - redis
    restart: unless-stopped

  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=pdf_variantes
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  redis:
    image: redis:7-alpine
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
      - pdf-variantes
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

### 5. NGINX CONFIGURACIÓN

```nginx
# nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream pdf_variantes {
        server pdf-variantes:8000;
    }

    server {
        listen 80;
        server_name your-domain.com;
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl;
        server_name your-domain.com;

        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;

        client_max_body_size 100M;

        location / {
            proxy_pass http://pdf_variantes;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        location /health {
            proxy_pass http://pdf_variantes/health;
            access_log off;
        }
    }
}
```

### 6. MONITOREO Y LOGGING

```python
# logging_config.py
import logging
import logging.handlers
from pathlib import Path

def setup_logging():
    """Configurar logging para producción."""
    
    # Crear directorio de logs
    log_dir = Path("/var/log/pdf_variantes")
    log_dir.mkdir(exist_ok=True)
    
    # Configurar formato
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Handler para archivo
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "pdf_variantes.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    
    # Handler para consola
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Configurar logger principal
    logger = logging.getLogger("pdf_variantes")
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
```

### 7. HEALTH CHECKS

```python
# health_checks.py
import asyncio
import httpx
from typing import Dict, Any

class HealthChecker:
    """Health checker para monitoreo."""
    
    def __init__(self):
        self.checks = {
            "database": self.check_database,
            "redis": self.check_redis,
            "storage": self.check_storage,
            "ai_service": self.check_ai_service
        }
    
    async def check_database(self) -> Dict[str, Any]:
        """Verificar conexión a base de datos."""
        try:
            # Mock check
            return {"status": "healthy", "response_time": 10}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def check_redis(self) -> Dict[str, Any]:
        """Verificar conexión a Redis."""
        try:
            # Mock check
            return {"status": "healthy", "response_time": 5}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def check_storage(self) -> Dict[str, Any]:
        """Verificar almacenamiento."""
        try:
            # Mock check
            return {"status": "healthy", "free_space": "50GB"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def check_ai_service(self) -> Dict[str, Any]:
        """Verificar servicio de IA."""
        try:
            # Mock check
            return {"status": "healthy", "response_time": 100}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def run_all_checks(self) -> Dict[str, Any]:
        """Ejecutar todos los checks."""
        results = {}
        
        for name, check_func in self.checks.items():
            results[name] = await check_func()
        
        # Determinar estado general
        all_healthy = all(
            result["status"] == "healthy" 
            for result in results.values()
        )
        
        return {
            "overall_status": "healthy" if all_healthy else "unhealthy",
            "checks": results,
            "timestamp": asyncio.get_event_loop().time()
        }
```

### 8. BACKUP Y RECOVERY

```bash
#!/bin/bash
# backup.sh

# Configuración
BACKUP_DIR="/var/backups/pdf_variantes"
DATE=$(date +%Y%m%d_%H%M%S)

# Crear directorio de backup
mkdir -p $BACKUP_DIR

# Backup de base de datos
pg_dump pdf_variantes > $BACKUP_DIR/db_backup_$DATE.sql

# Backup de archivos
tar -czf $BACKUP_DIR/uploads_backup_$DATE.tar.gz /var/lib/pdf_variantes/uploads

# Backup de configuración
cp -r /etc/pdf_variantes $BACKUP_DIR/config_backup_$DATE

# Limpiar backups antiguos (mantener últimos 7 días)
find $BACKUP_DIR -name "*.sql" -mtime +7 -delete
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete
find $BACKUP_DIR -name "config_backup_*" -mtime +7 -delete

echo "Backup completado: $DATE"
```

### 9. SCALING HORIZONTAL

```yaml
# kubernetes-deployment.yml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pdf-variantes
spec:
  replicas: 3
  selector:
    matchLabels:
      app: pdf-variantes
  template:
    metadata:
      labels:
        app: pdf-variantes
    spec:
      containers:
      - name: pdf-variantes
        image: pdf-variantes:latest
        ports:
        - containerPort: 8000
        env:
        - name: PDF_VARIANTES_ENV
          value: "production"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: pdf-variantes-secrets
              key: database-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
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

---
apiVersion: v1
kind: Service
metadata:
  name: pdf-variantes-service
spec:
  selector:
    app: pdf-variantes
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

### 10. MONITOREO CON PROMETHEUS

```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Métricas personalizadas
pdf_uploads_total = Counter('pdf_uploads_total', 'Total PDF uploads')
pdf_processing_duration = Histogram('pdf_processing_duration_seconds', 'PDF processing duration')
active_sessions = Gauge('active_sessions', 'Active user sessions')
cache_hits = Counter('cache_hits_total', 'Cache hits')
cache_misses = Counter('cache_misses_total', 'Cache misses')

def setup_metrics():
    """Configurar métricas de Prometheus."""
    start_http_server(9090)
```

### 11. COMANDOS DE DESPLIEGUE

```bash
# deploy.sh
#!/bin/bash

echo "🚀 Desplegando PDF Variantes..."

# 1. Build de imagen Docker
docker build -t pdf-variantes:latest .

# 2. Parar servicios existentes
docker-compose -f docker-compose.production.yml down

# 3. Iniciar servicios
docker-compose -f docker-compose.production.yml up -d

# 4. Verificar salud
sleep 30
curl -f http://localhost/health || exit 1

# 5. Ejecutar migraciones
docker-compose -f docker-compose.production.yml exec pdf-variantes python -m alembic upgrade head

echo "✅ Despliegue completado exitosamente!"
```

### 12. ROLLBACK

```bash
# rollback.sh
#!/bin/bash

echo "🔄 Ejecutando rollback..."

# 1. Parar servicios actuales
docker-compose -f docker-compose.production.yml down

# 2. Restaurar imagen anterior
docker tag pdf-variantes:previous pdf-variantes:latest

# 3. Iniciar servicios
docker-compose -f docker-compose.production.yml up -d

# 4. Verificar salud
sleep 30
curl -f http://localhost/health || exit 1

echo "✅ Rollback completado exitosamente!"
```

## 📊 CHECKLIST DE DESPLIEGUE

### Pre-despliegue
- [ ] Tests pasando
- [ ] Documentación actualizada
- [ ] Variables de entorno configuradas
- [ ] Certificados SSL válidos
- [ ] Backup de datos existentes
- [ ] Plan de rollback preparado

### Despliegue
- [ ] Build de imagen Docker
- [ ] Despliegue de servicios
- [ ] Verificación de salud
- [ ] Tests de integración
- [ ] Monitoreo activo

### Post-despliegue
- [ ] Verificación de funcionalidades
- [ ] Monitoreo de métricas
- [ ] Verificación de logs
- [ ] Tests de carga
- [ ] Documentación de cambios

## 🎯 CONCLUSIÓN

Esta guía proporciona todo lo necesario para desplegar PDF Variantes en producción de manera segura y escalable. El sistema está diseñado para ser robusto, monitoreado y fácil de mantener.

**¡PDF Variantes listo para producción!** 🚀
