# Deployment Guide

## 🚀 Guía de Despliegue

### Requisitos Previos

- Python 3.8+
- ComfyUI corriendo y accesible
- (Opcional) OpenRouter API key
- (Opcional) TruthGPT endpoint

### Configuración de Producción

#### 1. Variables de Entorno

Crear archivo `.env`:
```env
# ComfyUI
COMFYUI_API_URL=http://comfyui:8188
COMFYUI_WORKFLOW_PATH=workflows/flux_fill_clothing_changer.json

# OpenRouter (opcional)
OPENROUTER_ENABLED=true
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=openai/gpt-4

# TruthGPT (opcional)
TRUTHGPT_ENABLED=true
TRUTHGPT_ENDPOINT=http://truthgpt:8000

# Logging
LOG_LEVEL=INFO
LOG_FILE=/var/log/clothing_changer.log

# Cache
CACHE_ENABLED=true
CACHE_MAX_SIZE=1000

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_DEFAULT=100

# Security
ENABLE_CORS=true
ENABLE_API_KEY_AUTH=false
```

#### 2. Docker (Recomendado)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### 3. Docker Compose

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - COMFYUI_API_URL=http://comfyui:8188
      - OPENROUTER_ENABLED=true
    depends_on:
      - comfyui
  
  comfyui:
    image: comfyui/comfyui
    ports:
      - "8188:8188"
```

### Configuración de Servidor

#### Nginx Reverse Proxy

```nginx
server {
    listen 80;
    server_name api.example.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

#### Systemd Service

```ini
[Unit]
Description=Character Clothing Changer API
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/clothing_changer
Environment="PATH=/opt/clothing_changer/venv/bin"
ExecStart=/opt/clothing_changer/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

### Monitoreo

#### Health Checks

Configurar health check endpoint:
```bash
# Kubernetes
livenessProbe:
  httpGet:
    path: /api/v1/health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10

# Docker
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s \
  CMD curl -f http://localhost:8000/api/v1/health || exit 1
```

#### Logging

Configurar logging rotativo:
```python
from logging.handlers import RotatingFileHandler

handler = RotatingFileHandler(
    '/var/log/clothing_changer.log',
    maxBytes=10485760,  # 10MB
    backupCount=5
)
```

#### Métricas

Habilitar Prometheus (opcional):
```env
ENABLE_PROMETHEUS=true
PROMETHEUS_PORT=9090
```

### Seguridad

#### API Keys

Habilitar autenticación por API key:
```env
ENABLE_API_KEY_AUTH=true
API_KEY_HEADER=X-API-Key
```

#### CORS

Configurar CORS apropiadamente:
```env
ENABLE_CORS=true
CORS_ORIGINS=https://example.com,https://app.example.com
```

#### Rate Limiting

Ajustar rate limits según necesidad:
```env
RATE_LIMIT_DEFAULT=100
RATE_LIMIT_WINDOW=60.0
```

### Escalabilidad

#### Horizontal Scaling

- Usar load balancer (Nginx, HAProxy)
- Múltiples instancias de la API
- Cache compartido (Redis opcional)
- Rate limiting compartido

#### Vertical Scaling

- Aumentar recursos del servidor
- Ajustar `batch_max_concurrent_default`
- Aumentar `cache_max_size`
- Optimizar ComfyUI

### Backup y Recuperación

#### Datos Importantes

- Configuración (`.env`)
- Workflows (`workflows/`)
- Logs (`/var/log/`)

#### Estrategia de Backup

```bash
# Backup diario
tar -czf backup-$(date +%Y%m%d).tar.gz \
  .env workflows/ /var/log/clothing_changer.log
```

### Troubleshooting

#### Ver Logs

```bash
# Systemd
journalctl -u clothing_changer -f

# Docker
docker logs -f clothing_changer

# Archivo
tail -f /var/log/clothing_changer.log
```

#### Verificar Salud

```bash
curl http://localhost:8000/api/v1/health/components
```

#### Verificar Métricas

```bash
curl http://localhost:8000/api/v1/metrics
```

### Checklist de Despliegue

- [ ] Variables de entorno configuradas
- [ ] ComfyUI accesible
- [ ] OpenRouter/TruthGPT configurados (opcional)
- [ ] Logging configurado
- [ ] Health checks funcionando
- [ ] Rate limiting configurado
- [ ] CORS configurado
- [ ] Monitoreo configurado
- [ ] Backup configurado
- [ ] Documentación actualizada

¡Sistema listo para producción!

