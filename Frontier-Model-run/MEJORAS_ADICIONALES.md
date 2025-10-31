# 🚀 Mejoras Adicionales para ULTIMATE_PLATFORM_FINAL_COMPLETE.md

## 📚 Integración con Herramientas Cloud

### AWS SageMaker Integration

```python
# scripts/aws_sagemaker_deploy.py
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.predictor import Predictor

# Configurar SageMaker
sagemaker_session = sagemaker.Session()
role = "arn:aws:iam::ACCOUNT:role/SageMakerRole"

# Entrenamiento en SageMaker
estimator = PyTorch(
    entry_point="train_llm.py",
    role=role,
    instance_type="ml.p3.2xlarge",
    instance_count=1,
    framework_version="2.0",
    py_version="py310",
    hyperparameters={
        "epochs": 5,
        "batch-size": 8,
        "lr": 3e-5
    }
)

# Iniciar entrenamiento
estimator.fit({"training": "s3://bucket/train", "validation": "s3://bucket/val"})

# Desplegar modelo
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type="ml.g4dn.xlarge"
)

# Inferencia
response = predictor.predict({"prompt": "Hello", "max_length": 100})
```

### Google Cloud AI Platform

```python
# scripts/gcp_ai_platform_deploy.py
from google.cloud import aiplatform
from google.cloud.aiplatform import training_jobs

# Configurar proyecto
aiplatform.init(project="your-project", location="us-central1")

# Entrenamiento
job = training_jobs.CustomTrainingJob(
    display_name="llm-training",
    script_path="train_llm.py",
    container_uri="gcr.io/cloud-aiplatform/training/pytorch-gpu.2-0:latest",
    requirements=["torch", "transformers"],
    model_serving_container_image_uri="gcr.io/cloud-aiplatform/prediction/pytorch-gpu.2-0:latest"
)

# Ejecutar
model = job.run(
    args=["--epochs", "5", "--batch-size", "8"],
    replica_count=1,
    machine_type="n1-standard-4",
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1
)

# Desplegar endpoint
endpoint = model.deploy(
    machine_type="n1-standard-4",
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1
)

# Inferencia
prediction = endpoint.predict({"prompt": "Hello", "max_length": 100})
```

### Azure ML Integration

```python
# scripts/azure_ml_deploy.py
from azure.ai.ml import MLClient
from azure.ai.ml import command, Input
from azure.ai.ml.constants import AssetTypes

# Conectar a workspace
ml_client = MLClient.from_config()

# Crear job de entrenamiento
job = command(
    code="./scripts",
    command="python train_llm.py --epochs 5 --batch-size 8",
    environment="pytorch-2.0:latest",
    compute="gpu-cluster",
    inputs={
        "data": Input(type=AssetTypes.URI_FOLDER, path="azureml://datastores/workspaceblobstore/paths/data")
    }
)

# Ejecutar
returned_job = ml_client.jobs.create_or_update(job)

# Registrar modelo
model = ml_client.models.create_or_update(
    name="llm-model",
    version="1",
    path="./outputs"
)

# Desplegar como endpoint
endpoint = ml_client.online_endpoints.begin_create_or_update(
    name="llm-endpoint",
    endpoint_type="managed"
)
```

---

## 🔐 Seguridad Avanzada

### Implementación de Rate Limiting Avanzado

```python
# scripts/advanced_rate_limiter.py
from collections import defaultdict
import time
from typing import Dict, Tuple
from dataclasses import dataclass
from enum import Enum

class RateLimitStrategy(Enum):
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"

@dataclass
class RateLimitConfig:
    max_requests: int
    window_seconds: int
    strategy: RateLimitStrategy

class AdvancedRateLimiter:
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.requests: Dict[str, list] = defaultdict(list)
        self.tokens: Dict[str, Tuple[float, int]] = defaultdict(lambda: (time.time(), config.max_requests))
    
    def is_allowed(self, key: str) -> Tuple[bool, float]:
        """Retorna (permitido, tiempo_restante)"""
        if self.config.strategy == RateLimitStrategy.FIXED_WINDOW:
            return self._fixed_window_check(key)
        elif self.config.strategy == RateLimitStrategy.SLIDING_WINDOW:
            return self._sliding_window_check(key)
        else:
            return self._token_bucket_check(key)
    
    def _fixed_window_check(self, key: str) -> Tuple[bool, float]:
        now = time.time()
        window_start = now - self.config.window_seconds
        
        # Limpiar requests fuera de la ventana
        self.requests[key] = [t for t in self.requests[key] if t > window_start]
        
        if len(self.requests[key]) >= self.config.max_requests:
            next_reset = window_start + self.config.window_seconds
            return False, next_reset - now
        
        self.requests[key].append(now)
        return True, self.config.window_seconds
    
    def _sliding_window_check(self, key: str) -> Tuple[bool, float]:
        now = time.time()
        window_start = now - self.config.window_seconds
        
        # Mantener solo requests en la ventana
        self.requests[key] = [t for t in self.requests[key] if t > window_start]
        
        if len(self.requests[key]) >= self.config.max_requests:
            oldest_request = min(self.requests[key])
            next_allowed = oldest_request + self.config.window_seconds
            return False, next_allowed - now
        
        self.requests[key].append(now)
        return True, self.config.window_seconds
    
    def _token_bucket_check(self, key: str) -> Tuple[bool, float]:
        now = time.time()
        last_update, tokens = self.tokens[key]
        
        # Reponer tokens
        elapsed = now - last_update
        tokens_to_add = (elapsed / self.config.window_seconds) * self.config.max_requests
        tokens = min(self.config.max_requests, tokens + tokens_to_add)
        
        if tokens >= 1:
            tokens -= 1
            self.tokens[key] = (now, tokens)
            
            # Calcular tiempo hasta próximo token disponible
            if tokens < self.config.max_requests:
                time_to_next = (1 - tokens) * (self.config.window_seconds / self.config.max_requests)
            else:
                time_to_next = self.config.window_seconds
            
            return True, time_to_next
        else:
            # Calcular tiempo hasta próximo token disponible
            time_to_next = (1 - tokens) * (self.config.window_seconds / self.config.max_requests)
            return False, time_to_next

# Uso en FastAPI
from fastapi import FastAPI, Request, HTTPException, status

app = FastAPI()
rate_limiter = AdvancedRateLimiter(
    RateLimitConfig(
        max_requests=100,
        window_seconds=60,
        strategy=RateLimitStrategy.TOKEN_BUCKET
    )
)

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host
    allowed, retry_after = rate_limiter.is_allowed(client_ip)
    
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={"Retry-After": str(int(retry_after))}
        )
    
    response = await call_next(request)
    response.headers["X-RateLimit-Remaining"] = str(int(retry_after))
    return response
```

### Encriptación de Modelos y Datos

```python
# scripts/model_encryption.py
from cryptography.fernet import Fernet
import pickle
import os

class ModelEncryption:
    def __init__(self, key_path: str = None):
        if key_path and os.path.exists(key_path):
            with open(key_path, 'rb') as f:
                self.key = f.read()
        else:
            self.key = Fernet.generate_key()
            if key_path:
                with open(key_path, 'wb') as f:
                    f.write(self.key)
        
        self.cipher = Fernet(self.key)
    
    def encrypt_model(self, model_path: str, encrypted_path: str):
        """Encriptar checkpoint de modelo"""
        with open(model_path, 'rb') as f:
            model_data = f.read()
        
        encrypted_data = self.cipher.encrypt(model_data)
        
        with open(encrypted_path, 'wb') as f:
            f.write(encrypted_data)
    
    def decrypt_model(self, encrypted_path: str, decrypted_path: str):
        """Desencriptar checkpoint de modelo"""
        with open(encrypted_path, 'rb') as f:
            encrypted_data = f.read()
        
        decrypted_data = self.cipher.decrypt(encrypted_data)
        
        with open(decrypted_path, 'wb') as f:
            f.write(decrypted_data)
    
    def encrypt_data(self, data: dict) -> bytes:
        """Encriptar datos (p.ej., prompts sensibles)"""
        serialized = pickle.dumps(data)
        return self.cipher.encrypt(serialized)
    
    def decrypt_data(self, encrypted_data: bytes) -> dict:
        """Desencriptar datos"""
        decrypted = self.cipher.decrypt(encrypted_data)
        return pickle.loads(decrypted)

# Uso
encryption = ModelEncryption(key_path="keys/model_key.key")

# Encriptar modelo
encryption.encrypt_model("models/model.pt", "models/model.encrypted")

# Desencriptar para uso
encryption.decrypt_model("models/model.encrypted", "models/model.decrypted")

# Encriptar datos sensibles
sensitive_data = {"prompt": "Secret information", "user_id": "12345"}
encrypted = encryption.encrypt_data(sensitive_data)
```

---

## 🔄 CI/CD Pipelines Completos

### GitHub Actions - Pipeline Completo

```yaml
# .github/workflows/ci-cd-complete.yml
name: Complete CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  PYTHON_VERSION: '3.11'
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      
      - name: Cache dependencies
        uses: actions/cache@v3
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements*.txt') }}
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements_advanced.txt
          pip install pytest pytest-cov ruff mypy
      
      - name: Lint code
        run: |
          ruff check .
          mypy .
      
      - name: Run tests
        run: |
          pytest --cov=optimization_core --cov-report=xml --cov-report=html
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml

  build:
    needs: test
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
      
      - name: Log in to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=ref,event=branch
            type=ref,event=pr
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=sha,prefix={{branch}}-
      
      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          file: ./Dockerfile
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  deploy-staging:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/develop'
    environment:
      name: staging
      url: https://staging.example.com
    steps:
      - name: Deploy to staging
        run: |
          echo "Deploying to staging..."
          # Agregar comandos de despliegue aquí

  deploy-production:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    environment:
      name: production
      url: https://api.example.com
    steps:
      - name: Deploy to production
        run: |
          echo "Deploying to production..."
          # Agregar comandos de despliegue aquí
```

### GitLab CI/CD Pipeline

```yaml
# .gitlab-ci.yml
stages:
  - test
  - build
  - deploy

variables:
  PYTHON_VERSION: "3.11"
  DOCKER_IMAGE: $CI_REGISTRY_IMAGE:$CI_COMMIT_REF_SLUG

before_script:
  - python --version
  - pip install --upgrade pip
  - pip install -r requirements_advanced.txt

test:
  stage: test
  image: python:$PYTHON_VERSION
  script:
    - pip install pytest pytest-cov ruff
    - ruff check .
    - pytest --cov=optimization_core --cov-report=html
  coverage: '/TOTAL.*\s+(\d+%)$/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml
    paths:
      - htmlcov/

build:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  before_script:
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
  script:
    - docker build -t $DOCKER_IMAGE .
    - docker push $DOCKER_IMAGE
  only:
    - main
    - develop

deploy-staging:
  stage: deploy
  image: alpine:latest
  before_script:
    - apk add --no-cache curl
  script:
    - echo "Deploying to staging..."
    - curl -X POST $STAGING_WEBHOOK_URL
  environment:
    name: staging
    url: https://staging.example.com
  only:
    - develop

deploy-production:
  stage: deploy
  image: alpine:latest
  before_script:
    - apk add --no-cache curl
  script:
    - echo "Deploying to production..."
    - curl -X POST $PRODUCTION_WEBHOOK_URL
  environment:
    name: production
    url: https://api.example.com
  when: manual
  only:
    - main
```

---

## 📊 Monitoreo y Alertas Avanzados

### Dashboard Grafana Completo

```json
{
  "dashboard": {
    "title": "LLM Inference Platform Dashboard",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [{
          "expr": "rate(http_requests_total[5m])",
          "legendFormat": "{{method}} {{status}}"
        }]
      },
      {
        "title": "Latency Percentiles",
        "targets": [{
          "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
          "legendFormat": "p95"
        }, {
          "expr": "histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))",
          "legendFormat": "p99"
        }]
      },
      {
        "title": "Error Rate",
        "targets": [{
          "expr": "rate(http_requests_total{status=~\"5..\"}[5m]) / rate(http_requests_total[5m])",
          "legendFormat": "Error Rate"
        }]
      },
      {
        "title": "GPU Utilization",
        "targets": [{
          "expr": "DCGM_FI_DEV_GPU_UTIL",
          "legendFormat": "GPU {{gpu}}"
        }]
      },
      {
        "title": "Memory Usage",
        "targets": [{
          "expr": "DCGM_FI_DEV_FB_USED / DCGM_FI_DEV_FB_TOTAL",
          "legendFormat": "GPU {{gpu}}"
        }]
      },
      {
        "title": "Cache Hit Rate",
        "targets": [{
          "expr": "cache_hits_total / (cache_hits_total + cache_misses_total)",
          "legendFormat": "Hit Rate"
        }]
      }
    ]
  }
}
```

### Sistema de Alertas con PagerDuty

```python
# scripts/alerting_pagerduty.py
import requests
import json
from typing import Dict, List

class PagerDutyAlerter:
    def __init__(self, api_key: str, service_id: str):
        self.api_key = api_key
        self.service_id = service_id
        self.base_url = "https://api.pagerduty.com"
    
    def trigger_incident(self, title: str, severity: str, details: Dict):
        """Trigger a PagerDuty incident"""
        payload = {
            "incident": {
                "type": "incident",
                "title": title,
                "service": {
                    "id": self.service_id,
                    "type": "service_reference"
                },
                "severity": severity,  # "critical", "error", "warning", "info"
                "body": {
                    "type": "incident_body",
                    "details": json.dumps(details)
                }
            }
        }
        
        headers = {
            "Authorization": f"Token token={self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/vnd.pagerduty+json;version=2"
        }
        
        response = requests.post(
            f"{self.base_url}/incidents",
            headers=headers,
            json=payload
        )
        
        return response.json()
    
    def check_and_alert(self, metrics: Dict):
        """Verificar métricas y enviar alertas si es necesario"""
        alerts = []
        
        # Latencia alta
        if metrics.get("latency_p95", 0) > 500:
            alerts.append({
                "title": "High Latency Alert",
                "severity": "error",
                "details": {
                    "metric": "latency_p95",
                    "value": metrics["latency_p95"],
                    "threshold": 500
                }
            })
        
        # Error rate alto
        if metrics.get("error_rate", 0) > 0.05:
            alerts.append({
                "title": "High Error Rate Alert",
                "severity": "critical",
                "details": {
                    "metric": "error_rate",
                    "value": metrics["error_rate"],
                    "threshold": 0.05
                }
            })
        
        # GPU sobrecarga
        if metrics.get("gpu_utilization", 0) > 0.95:
            alerts.append({
                "title": "GPU Overload Alert",
                "severity": "warning",
                "details": {
                    "metric": "gpu_utilization",
                    "value": metrics["gpu_utilization"],
                    "threshold": 0.95
                }
            })
        
        # Enviar alertas
        for alert in alerts:
            self.trigger_incident(
                title=alert["title"],
                severity=alert["severity"],
                details=alert["details"]
            )

# Uso
alerter = PagerDutyAlerter(
    api_key="your-api-key",
    service_id="your-service-id"
)

# Monitorear y alertar
metrics = {
    "latency_p95": 600,
    "error_rate": 0.06,
    "gpu_utilization": 0.98
}

alerter.check_and_alert(metrics)
```

---

*Contenido adicional para ULTIMATE_PLATFORM_FINAL_COMPLETE.md*
*Versión 2.3 - Mejoras Adicionales*


