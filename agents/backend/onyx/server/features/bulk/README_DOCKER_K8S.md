# 🐳 Docker y Kubernetes - API BUL

## 🐳 Docker

### Construcción

```bash
# Construir imagen
docker build -t bul-api:latest .

# O con docker-compose
docker-compose build
```

### Ejecución

```bash
# Ejecutar con Docker
docker run -p 8000:8000 bul-api:latest

# O con docker-compose
docker-compose up -d

# Ver logs
docker-compose logs -f bul-api
```

### Docker Compose

El archivo `docker-compose.yml` incluye:

- **bul-api**: Servicio principal de la API
- **prometheus** (opcional): Monitoreo de métricas
- **grafana** (opcional): Dashboards de visualización

**Iniciar con monitoreo:**
```bash
docker-compose --profile monitoring up -d
```

### Variables de Entorno

```yaml
environment:
  - PYTHONUNBUFFERED=1
  - LOG_LEVEL=INFO
```

## ☸️ Kubernetes

### Requisitos

- Kubernetes cluster
- kubectl configurado
- Docker registry (opcional)

### Despliegue

```bash
# Aplicar manifests
kubectl apply -f k8s/

# Verificar deployment
kubectl get deployments
kubectl get services
kubectl get pods

# Ver logs
kubectl logs -f deployment/bul-api
```

### Archivos Kubernetes

1. **deployment.yaml**: Deployment con 3 réplicas
   - Resource limits configurados
   - Health checks (liveness/readiness)
   - Auto-scaling ready

2. **service.yaml**: Service tipo LoadBalancer
   - Expone puerto 80
   - Target port 8000

3. **ingress.yaml**: Ingress con TLS
   - Configurado para nginx
   - Cert-manager para SSL
   - Dominio: api.bul.example.com

### Configuración

**Recursos:**
- Requests: 512Mi RAM, 250m CPU
- Limits: 2Gi RAM, 1000m CPU

**Health Checks:**
- Liveness: `/api/health` cada 10s
- Readiness: `/api/health` cada 5s

### Escalado

```bash
# Escalar manualmente
kubectl scale deployment bul-api --replicas=5

# Auto-scaling (requiere metrics-server)
kubectl autoscale deployment bul-api --min=3 --max=10 --cpu-percent=70
```

### Actualización

```bash
# Actualizar imagen
kubectl set image deployment/bul-api bul-api=bul-api:v1.1.0

# Rollout status
kubectl rollout status deployment/bul-api

# Rollback si es necesario
kubectl rollout undo deployment/bul-api
```

## 📊 Monitoreo

### Prometheus

**Configuración:**
- Archivo `prometheus.yml` incluido
- Scrape interval: 10s
- Endpoint: `/metrics`

**Con Docker Compose:**
```bash
docker-compose --profile monitoring up -d
```

**Acceder:**
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

### Métricas Disponibles

- `bul_requests_total`
- `bul_request_duration_seconds`
- `bul_active_tasks`
- `bul_document_generation_seconds`
- `bul_errors_total`

## 🔧 Troubleshooting

### Docker

```bash
# Ver logs
docker-compose logs bul-api

# Inspeccionar contenedor
docker exec -it bul-api bash

# Verificar salud
curl http://localhost:8000/api/health
```

### Kubernetes

```bash
# Ver eventos
kubectl get events

# Describir pod
kubectl describe pod <pod-name>

# Port-forward para debugging
kubectl port-forward deployment/bul-api 8000:8000
```

## 📝 Variables de Entorno

### Producción

```yaml
env:
  - name: LOG_LEVEL
    value: "INFO"
  - name: PYTHONUNBUFFERED
    value: "1"
  - name: DATABASE_URL
    valueFrom:
      secretKeyRef:
        name: bul-secrets
        key: database-url
```

## 🚀 CI/CD

### GitHub Actions (ejemplo)

```yaml
- name: Build and push Docker image
  run: |
    docker build -t bul-api:${{ github.sha }} .
    docker push bul-api:${{ github.sha }}

- name: Deploy to Kubernetes
  run: |
    kubectl set image deployment/bul-api bul-api=bul-api:${{ github.sha }}
```

---

**Estado**: ✅ **Listo para Producción**
































