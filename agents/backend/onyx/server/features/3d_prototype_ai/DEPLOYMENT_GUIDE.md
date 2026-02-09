# 🚀 Guía de Deployment - 3D Prototype AI

## 📋 Opciones de Deployment

### 1. Deployment Local

#### Usando Scripts

**Linux/Mac:**
```bash
chmod +x scripts/deploy.sh
./scripts/deploy.sh development
```

**Windows:**
```powershell
.\scripts\deploy.ps1 -Environment development
```

#### Manual
```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
.\venv\Scripts\Activate.ps1  # Windows

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar servidor
uvicorn main:app --host 0.0.0.0 --port 8030 --reload
```

### 2. Deployment con Docker

#### Build
```bash
docker build -t 3d-prototype-ai:latest .
```

#### Run
```bash
docker run -d \
  -p 8030:8030 \
  -v $(pwd)/storage:/app/storage \
  -v $(pwd)/logs:/app/logs \
  --name 3d-prototype-ai \
  3d-prototype-ai:latest
```

### 3. Deployment con Docker Compose

```bash
# Iniciar todos los servicios
docker-compose up -d

# Ver logs
docker-compose logs -f

# Detener servicios
docker-compose down
```

**Servicios incluidos:**
- API (puerto 8030)
- Redis (puerto 6379)
- Prometheus (puerto 9090)
- Grafana (puerto 3000)

### 4. Deployment en Producción

#### Con Gunicorn
```bash
gunicorn main:app \
  -w 4 \
  -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8030 \
  --access-logfile logs/access.log \
  --error-logfile logs/error.log
```

#### Con Nginx (Reverse Proxy)
```nginx
server {
    listen 80;
    server_name api.3dprototype.ai;

    location / {
        proxy_pass http://127.0.0.1:8030;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 5. Deployment en Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: 3d-prototype-ai
spec:
  replicas: 3
  selector:
    matchLabels:
      app: 3d-prototype-ai
  template:
    metadata:
      labels:
        app: 3d-prototype-ai
    spec:
      containers:
      - name: api
        image: 3d-prototype-ai:latest
        ports:
        - containerPort: 8030
        env:
        - name: REDIS_URL
          value: "redis://redis-service:6379"
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
            port: 8030
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8030
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: 3d-prototype-ai-service
spec:
  selector:
    app: 3d-prototype-ai
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8030
  type: LoadBalancer
```

## 🔧 Configuración

### Variables de Entorno

Crear archivo `.env`:
```env
# Server
HOST=0.0.0.0
PORT=8030
DEBUG=false

# Redis
REDIS_ENABLED=true
REDIS_URL=redis://localhost:6379

# LLM (opcional)
LLM_PROVIDER=openai
LLM_API_KEY=your_key_here

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_DEFAULT=100

# Security
SECRET_KEY=your_secret_key_here
JWT_SECRET=your_jwt_secret_here
```

## 📊 Monitoreo

### Health Checks

```bash
# Health básico
curl http://localhost:8030/health

# Health detallado
curl http://localhost:8030/health/detailed

# Health del sistema
curl http://localhost:8030/api/v1/system/health-check
```

### Métricas

- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` (admin/admin)
- API Metrics: `http://localhost:8030/metrics`

## 🧪 Testing

### Ejecutar Tests

**Linux/Mac:**
```bash
chmod +x scripts/test.sh
./scripts/test.sh
```

**Windows:**
```powershell
.\scripts\test.ps1
```

### Con Coverage
```bash
./scripts/test.sh true
```

## 📝 Logs

Los logs se guardan en:
- `logs/app.log` - Logs de aplicación
- `logs/access.log` - Logs de acceso (si usas Gunicorn)
- `logs/error.log` - Logs de errores (si usas Gunicorn)

## 🔄 CI/CD

El proyecto incluye configuración de GitHub Actions (`.github/workflows/ci.yml`):

- Tests automáticos en cada push
- Build de Docker image
- Deploy automático a producción (si está en main)

## 🚨 Troubleshooting

### Puerto en uso
```bash
# Encontrar proceso usando puerto 8030
lsof -i :8030  # Linux/Mac
netstat -ano | findstr :8030  # Windows

# Matar proceso
kill -9 <PID>
```

### Problemas con Redis
```bash
# Verificar Redis
redis-cli ping

# Reiniciar Redis
docker-compose restart redis
```

### Problemas con Docker
```bash
# Limpiar contenedores
docker-compose down -v

# Rebuild
docker-compose build --no-cache
docker-compose up -d
```

## ✅ Checklist de Deployment

- [ ] Variables de entorno configuradas
- [ ] Dependencias instaladas
- [ ] Tests pasando
- [ ] Health checks funcionando
- [ ] Logs configurados
- [ ] Monitoreo configurado
- [ ] Backup configurado
- [ ] SSL/TLS configurado (producción)
- [ ] Rate limiting configurado
- [ ] Autenticación configurada




