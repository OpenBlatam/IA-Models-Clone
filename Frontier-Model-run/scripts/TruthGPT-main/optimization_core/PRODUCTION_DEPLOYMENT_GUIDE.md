# 🎯 TRUTHGPT - PRODUCTION DEPLOYMENT GUIDE

## ⚡ Guía de Deployment en Producción

### 🚀 Sistema de Producción Completo

#### 1. **API REST con FastAPI**
- Endpoints optimizados para inferencia
- Rate limiting y autenticación
- Batch processing
- Health checks
- Documentación automática

#### 2. **Interfaz Gradio**
- UI interactiva para testing
- Configuración de parámetros
- Visualización en tiempo real
- Compartir fácilmente

#### 3. **Docker & Docker Compose**
- Containerización completa
- GPU support con NVIDIA Docker
- Nginx para load balancing
- Volumes para persistencia

#### 4. **Monitoring & Logging**
- Prometheus metrics
- WandB integration
- System monitoring
- Request logging

### 📦 Archivos de Producción

#### Core Files
- `production_system.py` - Sistema principal
- `Dockerfile` - Containerización
- `docker-compose.yml` - Orquestación
- `nginx.conf` - Load balancer
- `deploy.sh` - Script de deployment

#### Configuration
- `.env` - Variables de entorno
- `requirements.txt` - Dependencias
- `monitoring.py` - Sistema de monitoreo

### 🎯 Configuración de Producción

#### Variables de Entorno
```bash
# Modelo
MODEL_NAME=gpt2
DEVICE=auto
PRECISION=fp16

# API
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1

# Gradio
GRADIO_PORT=7860
GRADIO_SHARE=false

# Monitoreo
ENABLE_WANDB=false
WANDB_PROJECT=truthgpt-production

# Seguridad
API_KEY=your-secret-key
RATE_LIMIT=100
```

#### Configuración Óptima
```python
PRODUCTION_CONFIG = {
    # Modelo
    'model_name': 'gpt2',
    'device': 'auto',
    'precision': 'fp16',
    'max_length': 512,
    'temperature': 0.7,
    'top_p': 0.9,
    
    # API
    'api_host': '0.0.0.0',
    'api_port': 8000,
    'api_workers': 1,
    
    # Gradio
    'gradio_port': 7860,
    'gradio_share': False,
    
    # Monitoreo
    'enable_wandb': False,
    'wandb_project': 'truthgpt-production',
    
    # Seguridad
    'api_key': None,
    'rate_limit': 100,
}
```

### 🐳 Docker Deployment

#### Dockerfile
```dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu20.04

RUN apt-get update && apt-get install -y \
    python3 python3-pip git curl \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python
RUN pip install --upgrade pip

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY . /app
WORKDIR /app

ENV MODEL_NAME=gpt2
ENV DEVICE=auto
ENV PRECISION=fp16
ENV API_HOST=0.0.0.0
ENV API_PORT=8000
ENV GRADIO_PORT=7860

EXPOSE 8000 7860

CMD ["python", "production_system.py"]
```

#### Docker Compose
```yaml
version: '3.8'

services:
  truthgpt-api:
    build: .
    ports:
      - "8000:8000"
      - "7860:7860"
    environment:
      - MODEL_NAME=gpt2
      - DEVICE=auto
      - PRECISION=fp16
    volumes:
      - ./models:/app/models
      - ./cache:/app/cache
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
```

### 🚀 Deployment Rápido

#### 1. Preparar Entorno
```bash
# Clonar repositorio
git clone <repository>
cd truthgpt-production

# Configurar variables
cp .env.example .env
# Editar .env con tus configuraciones
```

#### 2. Construir y Desplegar
```bash
# Construir imagen
docker build -t truthgpt-production .

# Desplegar con Docker Compose
docker-compose up -d

# Verificar estado
docker-compose ps
```

#### 3. Verificar Servicios
```bash
# API Health Check
curl http://localhost:8000/health

# Gradio Interface
open http://localhost:7860

# Prometheus Metrics
open http://localhost:9090
```

### 📊 API Endpoints

#### Generar Texto
```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-key" \
  -d '{
    "prompt": "Hello, how are you?",
    "max_length": 100,
    "temperature": 0.7,
    "top_p": 0.9
  }'
```

#### Batch Generation
```bash
curl -X POST "http://localhost:8000/batch-generate" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-key" \
  -d '{
    "prompts": ["Hello", "How are you?", "What is AI?"],
    "max_length": 100
  }'
```

#### Model Info
```bash
curl http://localhost:8000/model-info
```

### 📈 Monitoring

#### Métricas Prometheus
- `truthgpt_requests_total` - Total requests
- `truthgpt_request_duration_seconds` - Request duration
- `truthgpt_model_memory_bytes` - Model memory usage
- `truthgpt_gpu_utilization_percent` - GPU utilization

#### Logs
```bash
# Ver logs en tiempo real
docker-compose logs -f truthgpt-api

# Ver logs específicos
docker-compose logs truthgpt-api | grep ERROR
```

### 🔧 Troubleshooting

#### Problemas Comunes

1. **GPU no disponible**
```bash
# Verificar NVIDIA Docker
docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi
```

2. **Memoria insuficiente**
```bash
# Reducir batch size
export BATCH_SIZE=1

# Usar gradient checkpointing
export GRADIENT_CHECKPOINTING=true
```

3. **API no responde**
```bash
# Verificar logs
docker-compose logs truthgpt-api

# Reiniciar servicio
docker-compose restart truthgpt-api
```

### 🎯 Optimizaciones de Producción

#### 1. **Caching**
```python
# Cache de respuestas frecuentes
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_generate(prompt_hash):
    # Generar texto
    pass
```

#### 2. **Load Balancing**
```nginx
# nginx.conf
upstream truthgpt_backend {
    server truthgpt-api:8000;
}

server {
    listen 80;
    location / {
        proxy_pass http://truthgpt_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

#### 3. **Auto-scaling**
```yaml
# docker-compose.yml con replicas
services:
  truthgpt-api:
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 8G
        reservations:
          memory: 4G
```

### 📊 Performance en Producción

#### Benchmarks Esperados
- **Latencia**: < 100ms para prompts cortos
- **Throughput**: 100+ requests/minuto
- **Memoria**: < 8GB por instancia
- **GPU**: 80-90% utilización

#### Métricas de Monitoreo
- Request rate
- Response time
- Error rate
- GPU utilization
- Memory usage
- Model accuracy

### ✅ Checklist de Producción

#### Pre-deployment
- [ ] Modelo optimizado cargado
- [ ] Variables de entorno configuradas
- [ ] Docker image construida
- [ ] Tests de integración pasados
- [ ] Monitoring configurado

#### Post-deployment
- [ ] Health checks pasando
- [ ] API endpoints funcionando
- [ ] Gradio interface accesible
- [ ] Métricas siendo recopiladas
- [ ] Logs siendo generados

#### Monitoring
- [ ] Prometheus metrics activos
- [ ] WandB logging configurado
- [ ] Alertas configuradas
- [ ] Dashboards creados
- [ ] Logs siendo analizados

---

**¡Sistema de producción completo y optimizado!** 🚀⚡🎯

