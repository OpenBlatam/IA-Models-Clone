# 🎼 Orquestación - Cursor Agent 24/7

Guía completa para orquestar y desplegar el sistema completo en diferentes entornos.

## 📋 Tabla de Contenidos

- [Docker Compose](#docker-compose)
- [Kubernetes](#kubernetes)
- [AWS ECS](#aws-ecs)
- [CI/CD](#cicd)
- [Configuración de Servicios](#configuración-de-servicios)
- [Monitoreo y Logging](#monitoreo-y-logging)
- [Escalabilidad](#escalabilidad)
- [High Availability](#high-availability)

## 🐳 Docker Compose

### Stack Completo

```bash
# Desarrollo
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

# Producción
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Con monitoreo
docker-compose --profile monitoring up -d
```

### Orquestación de Servicios

```yaml
# Orden de inicio automático
services:
  redis:
    # Primero: Redis debe estar listo
    healthcheck: ...
  
  api:
    depends_on:
      redis:
        condition: service_healthy
  
  worker:
    depends_on:
      redis:
        condition: service_healthy
      api:
        condition: service_started
```

### Escalado Horizontal

```bash
# Escalar workers
docker-compose up -d --scale worker=5

# Escalar API
docker-compose up -d --scale api=3

# Con límites de recursos
docker-compose -f docker-compose.prod.yml up -d --scale worker=5
```

## ☸️ Kubernetes

### Deployment Manifests

#### Namespace

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: cursor-agent
```

#### ConfigMap

```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: cursor-agent-config
  namespace: cursor-agent
data:
  API_PORT: "8024"
  REDIS_URL: "redis://redis-service:6379/0"
  CELERY_BROKER_URL: "redis://redis-service:6379/0"
  CELERY_RESULT_BACKEND: "redis://redis-service:6379/0"
  JWT_SECRET_KEY: "change-me"
  AWS_REGION: "us-east-1"
```

#### Secrets

```yaml
# k8s/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: cursor-agent-secrets
  namespace: cursor-agent
type: Opaque
stringData:
  jwt-secret-key: "your-secret-key"
  redis-password: "your-redis-password"
  aws-access-key-id: "your-access-key"
  aws-secret-access-key: "your-secret-key"
```

#### Redis Deployment

```yaml
# k8s/redis.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: cursor-agent
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: redis-service
  namespace: cursor-agent
spec:
  selector:
    app: redis
  ports:
  - port: 6379
    targetPort: 6379
```

#### API Deployment

```yaml
# k8s/api.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cursor-agent-api
  namespace: cursor-agent
spec:
  replicas: 3
  selector:
    matchLabels:
      app: cursor-agent-api
  template:
    metadata:
      labels:
        app: cursor-agent-api
    spec:
      containers:
      - name: api
        image: cursor-agent-24-7:latest
        ports:
        - containerPort: 8024
        envFrom:
        - configMapRef:
            name: cursor-agent-config
        - secretRef:
            name: cursor-agent-secrets
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /api/health
            port: 8024
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /api/health
            port: 8024
          initialDelaySeconds: 10
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: cursor-agent-api
  namespace: cursor-agent
spec:
  selector:
    app: cursor-agent-api
  ports:
  - port: 80
    targetPort: 8024
  type: LoadBalancer
```

#### Worker Deployment

```yaml
# k8s/worker.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cursor-agent-worker
  namespace: cursor-agent
spec:
  replicas: 5
  selector:
    matchLabels:
      app: cursor-agent-worker
  template:
    metadata:
      labels:
        app: cursor-agent-worker
    spec:
      containers:
      - name: worker
        image: cursor-agent-24-7:latest
        command: ["celery", "-A", "core.celery_worker.celery_app", "worker", "--loglevel=info", "--concurrency=4"]
        envFrom:
        - configMapRef:
            name: cursor-agent-config
        - secretRef:
            name: cursor-agent-secrets
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
```

#### Ingress

```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: cursor-agent-ingress
  namespace: cursor-agent
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
spec:
  tls:
  - hosts:
    - api.cursor-agent.example.com
    secretName: cursor-agent-tls
  rules:
  - host: api.cursor-agent.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: cursor-agent-api
            port:
              number: 80
```

#### Horizontal Pod Autoscaler

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: cursor-agent-api-hpa
  namespace: cursor-agent
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: cursor-agent-api
  minReplicas: 3
  maxReplicas: 10
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

### Despliegue en Kubernetes

```bash
# Aplicar todos los recursos
kubectl apply -f k8s/

# Ver estado
kubectl get pods -n cursor-agent
kubectl get services -n cursor-agent

# Logs
kubectl logs -f deployment/cursor-agent-api -n cursor-agent
kubectl logs -f deployment/cursor-agent-worker -n cursor-agent

# Escalar
kubectl scale deployment cursor-agent-api --replicas=5 -n cursor-agent
kubectl scale deployment cursor-agent-worker --replicas=10 -n cursor-agent
```

## ☁️ AWS ECS

### Task Definitions

#### API Task Definition

```json
{
  "family": "cursor-agent-api",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "containerDefinitions": [
    {
      "name": "api",
      "image": "your-ecr-repo/cursor-agent-24-7:latest",
      "portMappings": [
        {
          "containerPort": 8024,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "API_PORT", "value": "8024"},
        {"name": "REDIS_URL", "value": "redis://redis-service:6379/0"}
      ],
      "secrets": [
        {
          "name": "JWT_SECRET_KEY",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:jwt-secret"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/cursor-agent-api",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8024/api/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
```

#### Worker Task Definition

```json
{
  "family": "cursor-agent-worker",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "containerDefinitions": [
    {
      "name": "worker",
      "image": "your-ecr-repo/cursor-agent-24-7:latest",
      "command": [
        "celery",
        "-A",
        "core.celery_worker.celery_app",
        "worker",
        "--loglevel=info",
        "--concurrency=8"
      ],
      "environment": [
        {"name": "CELERY_BROKER_URL", "value": "redis://redis-service:6379/0"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/cursor-agent-worker",
          "awslogs-region": "us-east-1"
        }
      }
    }
  ]
}
```

### ECS Services

```bash
# Crear servicio API
aws ecs create-service \
  --cluster cursor-agent-cluster \
  --service-name cursor-agent-api \
  --task-definition cursor-agent-api \
  --desired-count 3 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}" \
  --load-balancers "targetGroupArn=arn:aws:elasticloadbalancing:region:account:targetgroup/xxx,containerName=api,containerPort=8024"

# Crear servicio Worker
aws ecs create-service \
  --cluster cursor-agent-cluster \
  --service-name cursor-agent-worker \
  --task-definition cursor-agent-worker \
  --desired-count 5 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx]}"

# Auto-scaling
aws application-autoscaling register-scalable-target \
  --service-namespace ecs \
  --scalable-dimension ecs:service:DesiredCount \
  --resource-id service/cursor-agent-cluster/cursor-agent-api \
  --min-capacity 3 \
  --max-capacity 10

aws application-autoscaling put-scaling-policy \
  --service-namespace ecs \
  --scalable-dimension ecs:service:DesiredCount \
  --resource-id service/cursor-agent-cluster/cursor-agent-api \
  --policy-name cpu-scaling \
  --policy-type TargetTrackingScaling \
  --target-tracking-scaling-policy-configuration '{
    "TargetValue": 70.0,
    "PredefinedMetricSpecification": {
      "PredefinedMetricType": "ECSServiceAverageCPUUtilization"
    }
  }'
```

## 🔄 CI/CD

### GitHub Actions

```yaml
# .github/workflows/deploy.yml
name: Deploy Cursor Agent 24/7

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      - name: Run tests
        run: pytest

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1
      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v1
      - name: Build and push
        env:
          ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
          ECR_REPOSITORY: cursor-agent-24-7
          IMAGE_TAG: ${{ github.sha }}
        run: |
          docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
          docker tag $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG $ECR_REGISTRY/$ECR_REPOSITORY:latest
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:latest

  deploy-ecs:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1
      - name: Update ECS service
        run: |
          aws ecs update-service \
            --cluster cursor-agent-cluster \
            --service cursor-agent-api \
            --force-new-deployment
```

### GitLab CI/CD

```yaml
# .gitlab-ci.yml
stages:
  - test
  - build
  - deploy

test:
  stage: test
  image: python:3.11
  script:
    - pip install -r requirements.txt
    - pip install -r requirements-dev.txt
    - pytest

build:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA

deploy:
  stage: deploy
  image: amazon/aws-cli:latest
  script:
    - aws ecs update-service --cluster cursor-agent-cluster --service cursor-agent-api --force-new-deployment
  only:
    - main
```

## ⚙️ Configuración de Servicios

### Orden de Inicio

1. **Redis** - Debe estar disponible primero
2. **API** - Depende de Redis
3. **Worker** - Depende de Redis y API
4. **Beat** - Depende de Redis
5. **Monitoreo** - Depende de API y Worker

### Health Checks

```yaml
# Health checks configurados
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8024/api/health"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s
```

### Dependencies

```yaml
depends_on:
  redis:
    condition: service_healthy
  api:
    condition: service_started
```

## 📊 Monitoreo y Logging

### Stack de Monitoreo

```bash
# Iniciar stack completo
docker-compose --profile monitoring up -d

# Servicios:
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000
# - Flower: http://localhost:5555
```

### Logging Centralizado

```yaml
# Configuración de logging
logging:
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"
```

### Alertas

```yaml
# Prometheus alerts
groups:
  - name: cursor_agent
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 5m
        annotations:
          summary: "High error rate detected"
      
      - alert: HighLatency
        expr: histogram_quantile(0.99, http_request_duration_seconds) > 1
        for: 5m
        annotations:
          summary: "High latency detected"
```

## 📈 Escalabilidad

### Auto-scaling

#### Docker Compose

```bash
# Escalar manualmente
docker-compose up -d --scale worker=10

# Con límites
docker-compose -f docker-compose.prod.yml up -d --scale worker=10
```

#### Kubernetes

```yaml
# HPA automático
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
spec:
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        averageUtilization: 70
```

#### AWS ECS

```bash
# Auto-scaling basado en CPU
aws application-autoscaling put-scaling-policy \
  --target-tracking-scaling-policy-configuration '{
    "TargetValue": 70.0,
    "PredefinedMetricType": "ECSServiceAverageCPUUtilization"
  }'
```

### Load Balancing

```yaml
# Nginx load balancing
upstream api_backend {
    least_conn;
    server api1:8024;
    server api2:8024;
    server api3:8024;
}
```

## 🛡️ High Availability

### Multi-AZ Deployment

```yaml
# Kubernetes
spec:
  affinity:
    podAntiAffinity:
      preferredDuringSchedulingIgnoredDuringExecution:
      - weight: 100
        podAffinityTerm:
          labelSelector:
            matchExpressions:
            - key: app
              operator: In
              values:
              - cursor-agent-api
          topologyKey: kubernetes.io/hostname
```

### Database Replication

```yaml
# Redis Sentinel para HA
redis-sentinel:
  image: redis:7-alpine
  command: redis-sentinel /etc/redis/sentinel.conf
```

### Backup y Recovery

```bash
# Backup de estado
docker-compose exec api python scripts/backup.py

# Restore
docker-compose exec api python scripts/restore.py backup-2024-01-01.json
```

## 🚀 Quick Start

### Desarrollo Local

```bash
# Stack completo
make dev

# Solo servicios básicos
make quick-start
```

### Producción

```bash
# Con Terraform
cd aws/terraform
terraform apply

# Con Docker Compose
make prod

# Con Kubernetes
kubectl apply -f k8s/
```

## 📚 Más Información

- [DOCKER.md](DOCKER.md) - Docker Compose
- [ARCHITECTURE.md](ARCHITECTURE.md) - Arquitectura
- [AWS_DEPLOYMENT.md](aws/AWS_DEPLOYMENT.md) - AWS




