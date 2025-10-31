# 🚀 Instagram Captions API v4.0 - PRODUCTION DEPLOYMENT GUIDE

## 📋 **RESUMEN EJECUTIVO**

Esta guía detalla el despliegue en producción de la **Instagram Captions API v4.0**, optimizada para entornos enterprise con seguridad, escalabilidad y observabilidad completas.

## 🎯 **CARACTERÍSTICAS DE PRODUCCIÓN**

### **🔒 SEGURIDAD ENTERPRISE**
- ✅ Autenticación por API Key
- ✅ Rate limiting por cliente
- ✅ Validación robusta de inputs
- ✅ Headers de seguridad (HTTPS, CSP, HSTS)
- ✅ Usuario no-root en contenedores
- ✅ Scanning automático de vulnerabilidades

### **📊 OBSERVABILIDAD COMPLETA**
- ✅ Métricas Prometheus integradas
- ✅ Dashboards Grafana preconfigurados
- ✅ Logging estructurado JSON
- ✅ Health checks y readiness probes
- ✅ Alertas automáticas
- ✅ Request tracing con IDs únicos

### **⚡ PERFORMANCE OPTIMIZADA**
- ✅ Cache Redis multi-nivel
- ✅ Compresión de respuestas
- ✅ Procesamiento asíncrono
- ✅ Connection pooling
- ✅ Load balancing con Nginx
- ✅ Auto-scaling horizontal

## 🏗️ **ARQUITECTURA DE PRODUCCIÓN**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │────│  Nginx Proxy    │────│   Instagram     │
│   (CloudFlare)  │    │   (SSL Term.)   │    │   Captions API  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                       ┌─────────────────┐             │
                       │  Redis Cluster  │─────────────┘
                       │   (Cache/Rate)  │
                       └─────────────────┘
                                │
                       ┌─────────────────┐
                       │   Monitoring    │
                       │ Prometheus +    │
                       │    Grafana      │
                       └─────────────────┘
```

## 🚀 **REQUISITOS DE SISTEMA**

### **Mínimos (Desarrollo)**
- **CPU**: 2 cores
- **RAM**: 4GB
- **Disco**: 20GB SSD
- **Ancho de banda**: 100 Mbps

### **Recomendados (Producción)**
- **CPU**: 4+ cores
- **RAM**: 8GB+
- **Disco**: 50GB+ SSD NVMe
- **Ancho de banda**: 1 Gbps+

### **Enterprise (Alta disponibilidad)**
- **CPU**: 8+ cores
- **RAM**: 16GB+
- **Disco**: 100GB+ SSD NVMe
- **Ancho de banda**: 10 Gbps+
- **Multi-zona**: 3+ availability zones

## 📦 **DESPLIEGUE CON DOCKER**

### **1. Configuración de Variables**

Crear archivo `.env.production`:

```bash
# API Configuration
ENVIRONMENT=production
API_VERSION=4.0.0
HOST=0.0.0.0
PORT=8080
WORKERS=4

# Security
SECRET_KEY=your-super-secret-key-here
VALID_API_KEYS=key1,key2,key3
CORS_ORIGINS=https://yourapp.com,https://api.yourapp.com
TRUSTED_HOSTS=localhost,yourapp.com,api.yourapp.com

# AI Provider
OPENAI_API_KEY=your-openai-api-key
OPENAI_MODEL=gpt-4
OPENAI_MAX_TOKENS=500

# Redis
REDIS_URL=redis://redis:6379

# Monitoring
SENTRY_DSN=your-sentry-dsn
GRAFANA_PASSWORD=secure-grafana-password

# SSL (para HTTPS)
SSL_CERTFILE=/etc/ssl/certs/yourapp.crt
SSL_KEYFILE=/etc/ssl/private/yourapp.key
```

### **2. Construir y Desplegar**

```bash
# Clonar repositorio
git clone https://github.com/company/instagram-captions-api.git
cd instagram-captions-api

# Configurar variables de entorno
cp .env.example .env.production
nano .env.production

# Construir contenedores
docker-compose -f docker-compose.production.yml build

# Desplegar en producción
docker-compose -f docker-compose.production.yml up -d

# Verificar estado
docker-compose -f docker-compose.production.yml ps
```

### **3. Verificación del Despliegue**

```bash
# Health check
curl -f http://localhost:8080/health

# Métricas
curl http://localhost:8080/metrics

# Test de API (requiere API key)
curl -X POST "http://localhost:8080/api/v4/generate" \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "content_description": "Beautiful sunset at the beach",
    "style": "casual",
    "audience": "general",
    "client_id": "test-client"
  }'
```

## ☸️ **DESPLIEGUE EN KUBERNETES**

### **1. Namespace y Configuración**

```yaml
# kubernetes/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: instagram-captions
  labels:
    name: instagram-captions
    environment: production
```

### **2. ConfigMap y Secrets**

```yaml
# kubernetes/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: instagram-captions-config
  namespace: instagram-captions
data:
  ENVIRONMENT: "production"
  HOST: "0.0.0.0"
  PORT: "8080"
  WORKERS: "4"
  LOG_LEVEL: "INFO"

---
# kubernetes/secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: instagram-captions-secrets
  namespace: instagram-captions
type: Opaque
stringData:
  SECRET_KEY: "your-super-secret-key"
  OPENAI_API_KEY: "your-openai-api-key"
  VALID_API_KEYS: "key1,key2,key3"
  SENTRY_DSN: "your-sentry-dsn"
```

### **3. Deployment**

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: instagram-captions-api
  namespace: instagram-captions
spec:
  replicas: 3
  selector:
    matchLabels:
      app: instagram-captions-api
  template:
    metadata:
      labels:
        app: instagram-captions-api
    spec:
      containers:
      - name: api
        image: instagram-captions-api:4.0.0-production
        ports:
        - containerPort: 8080
        envFrom:
        - configMapRef:
            name: instagram-captions-config
        - secretRef:
            name: instagram-captions-secrets
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
```

### **4. Service y Ingress**

```yaml
# kubernetes/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: instagram-captions-service
  namespace: instagram-captions
spec:
  selector:
    app: instagram-captions-api
  ports:
  - port: 80
    targetPort: 8080
  type: ClusterIP

---
# kubernetes/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: instagram-captions-ingress
  namespace: instagram-captions
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
spec:
  tls:
  - hosts:
    - api.yourapp.com
    secretName: instagram-captions-tls
  rules:
  - host: api.yourapp.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: instagram-captions-service
            port:
              number: 80
```

## 🔧 **CONFIGURACIÓN DE NGINX**

```nginx
# nginx/nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream instagram_captions_api {
        server instagram-captions-api:8080;
        # Add more servers for load balancing
        # server instagram-captions-api-2:8080;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    
    # Security headers
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    server {
        listen 80;
        server_name api.yourapp.com;
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name api.yourapp.com;

        ssl_certificate /etc/nginx/ssl/yourapp.crt;
        ssl_certificate_key /etc/nginx/ssl/yourapp.key;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;

        # Gzip compression
        gzip on;
        gzip_types text/plain application/json;

        location / {
            limit_req zone=api burst=20 nodelay;
            
            proxy_pass http://instagram_captions_api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Timeouts
            proxy_connect_timeout 30s;
            proxy_send_timeout 30s;
            proxy_read_timeout 30s;
        }

        location /health {
            access_log off;
            proxy_pass http://instagram_captions_api/health;
        }
    }
}
```

## 📊 **MONITOREO Y ALERTAS**

### **1. Métricas Clave**

- **Latencia**: p50, p95, p99 de respuesta
- **Throughput**: Requests por segundo
- **Error Rate**: % de errores 4xx/5xx
- **Cache Hit Rate**: % de hits de cache
- **Queue Depth**: Requests en cola
- **Resource Usage**: CPU, Memory, Disk

### **2. Alertas Recomendadas**

```yaml
# monitoring/alerts.yml
groups:
- name: instagram-captions-api
  rules:
  - alert: HighErrorRate
    expr: rate(instagram_captions_requests_total{status=~"5.."}[5m]) > 0.1
    for: 5m
    annotations:
      summary: "High error rate detected"
      
  - alert: HighLatency
    expr: histogram_quantile(0.95, rate(instagram_captions_request_duration_seconds_bucket[5m])) > 2
    for: 10m
    annotations:
      summary: "High latency detected"
      
  - alert: LowCacheHitRate
    expr: rate(instagram_captions_cache_hits_total[10m]) / rate(instagram_captions_cache_misses_total[10m]) < 0.8
    for: 15m
    annotations:
      summary: "Cache hit rate below 80%"
```

## 🔐 **CONFIGURACIÓN DE SEGURIDAD**

### **1. API Keys Management**

```python
# Rotar API keys regularmente
import secrets

def generate_api_key():
    return secrets.token_urlsafe(32)

# Implementar en base de datos
CREATE TABLE api_keys (
    id UUID PRIMARY KEY,
    key_hash VARCHAR(255) NOT NULL,
    client_name VARCHAR(100) NOT NULL,
    rate_limit_requests INTEGER DEFAULT 1000,
    rate_limit_window INTEGER DEFAULT 3600,
    created_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);
```

### **2. Firewall Rules**

```bash
# UFW (Ubuntu)
ufw deny all
ufw allow ssh
ufw allow 80/tcp
ufw allow 443/tcp
ufw allow from 10.0.0.0/8 to any port 8080  # Internal only
ufw enable
```

## 📈 **ESCALABILIDAD**

### **1. Horizontal Pod Autoscaler (K8s)**

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: instagram-captions-hpa
  namespace: instagram-captions
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: instagram-captions-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### **2. Cluster Autoscaler**

```yaml
# Para AWS EKS
apiVersion: v1
kind: ConfigMap
metadata:
  name: cluster-autoscaler-status
  namespace: kube-system
data:
  nodes.max: "100"
  nodes.min: "3"
  scale-down-delay-after-add: "10m"
  scale-down-unneeded-time: "10m"
```

## 🔄 **CI/CD PIPELINE**

### **1. GitHub Actions**

```yaml
# .github/workflows/production.yml
name: Production Deployment

on:
  push:
    branches: [main]
    tags: ['v*']

jobs:
  security-scan:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Run security scan
      run: |
        pip install bandit safety
        bandit -r . -f json -o bandit-report.json
        safety check --json --output safety-report.json

  build-and-deploy:
    needs: security-scan
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Build Docker image
      run: |
        docker build -t instagram-captions-api:${{ github.sha }} .
        
    - name: Push to registry
      run: |
        docker tag instagram-captions-api:${{ github.sha }} ${{ secrets.REGISTRY }}/instagram-captions-api:${{ github.sha }}
        docker push ${{ secrets.REGISTRY }}/instagram-captions-api:${{ github.sha }}
        
    - name: Deploy to Kubernetes
      run: |
        kubectl set image deployment/instagram-captions-api api=${{ secrets.REGISTRY }}/instagram-captions-api:${{ github.sha }}
        kubectl rollout status deployment/instagram-captions-api
```

## 🔧 **MANTENIMIENTO**

### **1. Backups**

```bash
# Backup Redis
redis-cli --rdb /backup/redis-$(date +%Y%m%d).rdb

# Backup configuración
tar -czf config-backup-$(date +%Y%m%d).tar.gz kubernetes/ nginx/ monitoring/
```

### **2. Logs Management**

```yaml
# Fluentd o Fluent Bit para agregación de logs
apiVersion: v1
kind: ConfigMap
metadata:
  name: fluent-bit-config
data:
  fluent-bit.conf: |
    [SERVICE]
        Flush         1
        Log_Level     info
        Daemon        off
        
    [INPUT]
        Name              tail
        Path              /var/log/containers/*instagram-captions*.log
        Parser            docker
        Tag               kube.*
        
    [OUTPUT]
        Name              es
        Match             *
        Host              elasticsearch.logging.svc.cluster.local
        Port              9200
        Index             instagram-captions-logs
```

## 🎯 **CHECKLIST DE PRODUCCIÓN**

### **Pre-despliegue**
- [ ] Variables de entorno configuradas
- [ ] Certificados SSL válidos
- [ ] API keys generadas y configuradas
- [ ] Base de datos inicializada
- [ ] Monitoreo configurado
- [ ] Backups programados
- [ ] Firewall configurado
- [ ] Load testing completado

### **Post-despliegue**
- [ ] Health checks respondiendo
- [ ] Métricas fluyendo a Prometheus
- [ ] Dashboards de Grafana activos
- [ ] Alertas configuradas
- [ ] Logs agregándose correctamente
- [ ] Performance dentro de SLAs
- [ ] Security scanning pasando
- [ ] Documentación actualizada

## 📞 **SOPORTE Y ESCALACIÓN**

### **Contactos de Emergencia**
- **On-call Engineer**: +1-555-0123
- **DevOps Team**: devops@company.com
- **Security Team**: security@company.com

### **Runbooks**
- **High Latency**: [runbook-latency.md](./runbooks/latency.md)
- **High Error Rate**: [runbook-errors.md](./runbooks/errors.md)
- **Cache Issues**: [runbook-cache.md](./runbooks/cache.md)
- **Database Issues**: [runbook-db.md](./runbooks/database.md)

---

## 🎉 **RESUMEN**

La **Instagram Captions API v4.0** está lista para producción con:

✅ **Seguridad Enterprise** - Autenticación, rate limiting, validación robusta  
✅ **Observabilidad Completa** - Métricas, logs, alertas, dashboards  
✅ **Alta Disponibilidad** - Load balancing, auto-scaling, health checks  
✅ **Performance Optimizada** - Cache, compresión, async processing  
✅ **Mantenimiento Simplificado** - Backups, CI/CD, runbooks  

**¡Lista para escalar y servir millones de requests!** 🚀 