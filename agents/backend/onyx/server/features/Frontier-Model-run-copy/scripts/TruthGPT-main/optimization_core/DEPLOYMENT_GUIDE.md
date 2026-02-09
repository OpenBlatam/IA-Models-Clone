# 🚀 TruthGPT Optimization Core - Deployment Guide

## 📦 Quick Start

### 1. Local Deployment
```python
from utils.modules.deployment import (
    create_deployer, create_deployment_config, DeploymentTarget
)

# Create configuration
config = create_deployment_config(
    target=DeploymentTarget.LOCAL,
    model_name="truthgpt-model",
    model_version="1.0.0",
    port=8000,
    use_mixed_precision=True
)

# Create deployer
deployer = create_deployer(config, model)

# Deploy
result = deployer.deploy()
print(f"Deployed: {result}")
```

### 2. Docker Deployment
```python
config = create_deployment_config(
    target=DeploymentTarget.DOCKER,
    model_name="truthgpt-model",
    model_version="1.0.0",
    cpu_request="2",
    cpu_limit="4",
    memory_request="4Gi",
    memory_limit="8Gi"
)

deployer = create_deployer(config, model)
result = deployer.deploy()
```

### 3. Kubernetes Deployment
```python
config = create_deployment_config(
    target=DeploymentTarget.KUBERNETES,
    model_name="truthgpt-model",
    min_replicas=3,
    max_replicas=10,
    scaling_policy=ScalingPolicy.HPA
)

deployer = create_deployer(config, model)
result = deployer.deploy()
```

## 🐳 Docker Deployment

### Build Image
```bash
docker build -t truthgpt-model:1.0.0 .
```

### Run Container
```bash
docker run -p 8000:8000 \
  --gpus all \
  -e CUDA_VISIBLE_DEVICES=0 \
  truthgpt-model:1.0.0
```

### Docker Compose
```yaml
version: '3.8'
services:
  truthgpt:
    image: truthgpt-model:1.0.0
    ports:
      - "8000:8000"
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - MODEL_NAME=truthgpt-model
      - MODEL_VERSION=1.0.0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## ☸️ Kubernetes Deployment

### Apply Deployment
```bash
kubectl apply -f deployment.yaml
```

### Deployment Manifest
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: truthgpt-model
spec:
  replicas: 3
  selector:
    matchLabels:
      app: truthgpt-model
  template:
    metadata:
      labels:
        app: truthgpt-model
    spec:
      containers:
      - name: truthgpt
        image: truthgpt-model:1.0.0
        ports:
        - containerPort: 8000
        resources:
          requests:
            cpu: "2"
            memory: "4Gi"
          limits:
            cpu: "4"
            memory: "8Gi"
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
---
apiVersion: v1
kind: Service
metadata:
  name: truthgpt-service
spec:
  selector:
    app: truthgpt-model
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Horizontal Pod Autoscaler
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: truthgpt-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: truthgpt-model
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

## ☁️ Cloud Deployment

### AWS SageMaker
```python
config = create_deployment_config(
    target=DeploymentTarget.AWS,
    model_name="truthgpt-model",
    cpu_request="4",
    memory_request="8Gi"
)

deployer = create_deployer(config, model)
result = deployer.deploy()
```

### Azure ML
```python
config = create_deployment_config(
    target=DeploymentTarget.AZURE,
    model_name="truthgpt-model"
)

deployer = create_deployer(config, model)
result = deployer.deploy()
```

### Google Cloud AI Platform
```python
config = create_deployment_config(
    target=DeploymentTarget.GCP,
    model_name="truthgpt-model"
)

deployer = create_deployer(config, model)
result = deployer.deploy()
```

## 🎯 Model Serving

### FastAPI Server
```python
from utils.modules.deployment import create_fastapi_server

# Create server
server = create_fastapi_server(model, config)

# Run server
server.run()
```

### Ray Serve
```python
config = create_deployment_config(
    target=DeploymentTarget.RAY_SERVE,
    min_replicas=3,
    max_replicas=10
)

deployer = create_deployer(config, model)
result = deployer.deploy()
```

### Triton Inference Server
```python
config = create_deployment_config(
    target=DeploymentTarget.TRITON,
    batch_size=32,
    use_mixed_precision=True
)

deployer = create_deployer(config, model)
result = deployer.deploy()
```

## 📊 Monitoring

### Prometheus Metrics
```python
from prometheus_client import start_http_server

start_http_server(9090)
```

### Grafana Dashboards
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: grafana-dashboards
data:
  truthgpt.json: |
    {
      "dashboard": {
        "title": "TruthGPT Metrics"
      }
    }
```

### Health Checks
```python
# Health check endpoint
GET /health

# Metrics endpoint
GET /metrics
```

## 🔒 Security

### TLS Configuration
```yaml
env:
- name: TLS_CERT_PATH
  value: "/etc/ssl/certs/tls.crt"
- name: TLS_KEY_PATH
  value: "/etc/ssl/certs/tls.key"
```

### Authentication
```python
config = create_deployment_config(
    enable_auth=True,
    auth_token="your-token-here"
)
```

## 🚀 Production Best Practices

### 1. Resource Limits
```yaml
resources:
  requests:
    cpu: "2"
    memory: "4Gi"
    nvidia.com/gpu: 1
  limits:
    cpu: "4"
    memory: "8Gi"
    nvidia.com/gpu: 1
```

### 2. Health Checks
```yaml
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

### 3. Auto-scaling
```yaml
autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80
```

### 4. Monitoring
```yaml
metrics:
  - prometheus:
      port: 9090
      path: /metrics
```

## 📈 Performance Optimization

### Mixed Precision
```python
config = create_deployment_config(
    use_mixed_precision=True,
    use_tensor_cores=True
)
```

### GPU Optimization
```python
import torch

# Enable CUDA optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
```

## 🎯 Complete Deployment Example

```python
from utils.modules.deployment import (
    create_deployer, create_deployment_config, 
    create_server, create_fastapi_server,
    DeploymentTarget, ScalingPolicy
)

# 1. Create configuration
config = create_deployment_config(
    target=DeploymentTarget.KUBERNETES,
    model_name="truthgpt-model",
    model_version="1.0.0",
    cpu_request="2",
    cpu_limit="4",
    memory_request="4Gi",
    memory_limit="8Gi",
    min_replicas=3,
    max_replicas=10,
    scaling_policy=ScalingPolicy.HPA,
    use_mixed_precision=True,
    enable_prometheus=True,
    enable_grafana=True
)

# 2. Deploy
deployer = create_deployer(config, model)
result = deployer.deploy()

# 3. Create server
server = create_server(model, config)

# 4. Run API server
api_server = create_fastapi_server(model, config)
api_server.run()

print(f"✅ Deployment complete: {result}")
```

## 🎉 Success!

Your model is now deployed and ready to serve predictions! 🚀
