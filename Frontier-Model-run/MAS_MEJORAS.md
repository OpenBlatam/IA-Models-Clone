---

## 🚀 Despliegue Multi-Cloud

### Estrategia Multi-Cloud con Terraform

```hcl
# terraform/multi-cloud/main.tf
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }
}

# Configuración AWS
provider "aws" {
  region = var.aws_region
}

# Configuración GCP
provider "google" {
  project = var.gcp_project
  region  = var.gcp_region
}

# Configuración Azure
provider "azurerm" {
  features {}
}

# Recursos comunes
module "aws_inference" {
  source = "./modules/aws"
  
  instance_type = "ml.p3.2xlarge"
  min_size     = 1
  max_size     = 5
}

module "gcp_inference" {
  source = "./modules/gcp"
  
  machine_type = "n1-standard-4"
  min_replicas = 1
  max_replicas = 5
}

module "azure_inference" {
  source = "./modules/azure"
  
  vm_size = "Standard_NC6s_v3"
  min_instances = 1
  max_instances = 5
}

# Load balancer multi-cloud
resource "aws_lb" "global" {
  name               = "inference-global-lb"
  internal           = false
  load_balancer_type = "application"
  
  subnets = var.subnet_ids
}

resource "google_compute_backend_service" "inference" {
  name = "inference-backend"
}

resource "azurerm_lb" "inference" {
  name                = "inference-lb"
  location            = var.azure_location
  resource_group_name = var.azure_resource_group
}
```

---

## 🔄 Blue-Green Deployment Completo

### Script de Blue-Green Deployment

```python
# scripts/blue_green_deployment.py
import subprocess
import time
import requests
from typing import Dict, Optional

class BlueGreenDeployment:
    def __init__(self, service_name: str, namespace: str = "default"):
        self.service_name = service_name
        self.namespace = namespace
        self.blue_version = None
        self.green_version = None
        self.active_color = "blue"
    
    def deploy_new_version(self, image_tag: str, replicas: int = 3) -> str:
        """Desplegar nueva versión en el entorno inactivo"""
        target_color = "green" if self.active_color == "blue" else "blue"
        
        # Crear deployment para color inactivo
        deployment_name = f"{self.service_name}-{target_color}"
        
        subprocess.run([
            "kubectl", "apply", "-f", "-"
        ], input=self._generate_deployment_yaml(deployment_name, image_tag, replicas).encode())
        
        # Esperar a que esté listo
        self._wait_for_deployment(deployment_name)
        
        # Guardar versión
        if target_color == "green":
            self.green_version = image_tag
        else:
            self.blue_version = image_tag
        
        return target_color
    
    def switch_traffic(self, percentage: int = 100) -> bool:
        """Cambiar tráfico gradualmente al nuevo entorno"""
        target_color = "green" if self.active_color == "blue" else "blue"
        deployment_name = f"{self.service_name}-{target_color}"
        
        # Verificar salud del nuevo deployment
        if not self._check_health(deployment_name):
            print(f"Health check failed for {deployment_name}")
            return False
        
        # Actualizar service para apuntar al nuevo deployment
        self._update_service(target_color, percentage)
        
        # Monitorear por un período
        time.sleep(60)  # Monitorear por 1 minuto
        
        # Verificar métricas
        if self._check_metrics(target_color):
            self.active_color = target_color
            print(f"Successfully switched to {target_color}")
            return True
        else:
            print(f"Metrics check failed, rolling back")
            self._rollback()
            return False
    
    def rollback(self):
        """Hacer rollback al entorno anterior"""
        self._rollback()
        target_color = "green" if self.active_color == "blue" else "blue"
        self.active_color = target_color
    
    def _generate_deployment_yaml(self, name: str, image: str, replicas: int) -> str:
        return f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {name}
  namespace: {self.namespace}
spec:
  replicas: {replicas}
  selector:
    matchLabels:
      app: {self.service_name}
      version: {name.split('-')[-1]}
  template:
    metadata:
      labels:
        app: {self.service_name}
        version: {name.split('-')[-1]}
    spec:
      containers:
      - name: inference
        image: {image}
        ports:
        - containerPort: 8080
        resources:
          requests:
            nvidia.com/gpu: 1
          limits:
            nvidia.com/gpu: 1
"""
    
    def _wait_for_deployment(self, name: str, timeout: int = 300):
        """Esperar a que deployment esté listo"""
        start = time.time()
        while time.time() - start < timeout:
            result = subprocess.run(
                ["kubectl", "get", "deployment", name, "-o", "json"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                # Verificar que todas las réplicas estén listas
                import json
                data = json.loads(result.stdout)
                ready = data["status"].get("readyReplicas", 0)
                desired = data["spec"]["replicas"]
                if ready == desired:
                    return True
            time.sleep(5)
        raise TimeoutError(f"Deployment {name} not ready in {timeout}s")
    
    def _check_health(self, deployment_name: str) -> bool:
        """Verificar salud del deployment"""
        # Obtener pods del deployment
        result = subprocess.run(
            ["kubectl", "get", "pods", "-l", f"app={self.service_name}", "-o", "json"],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            return False
        
        import json
        data = json.loads(result.stdout)
        
        for pod in data.get("items", []):
            pod_name = pod["metadata"]["name"]
            # Ejecutar health check
            health_result = subprocess.run(
                ["kubectl", "exec", pod_name, "--", "curl", "-f", "http://localhost:8080/health"],
                capture_output=True
            )
            if health_result.returncode != 0:
                return False
        
        return True
    
    def _update_service(self, target_color: str, percentage: int):
        """Actualizar service para enrutar tráfico"""
        # Crear o actualizar VirtualService para Istio
        # O actualizar labels del Service en Kubernetes vanilla
        subprocess.run([
            "kubectl", "patch", "service", self.service_name,
            "-p", f'{{"spec": {{"selector": {{"version": "{target_color}"}}}}}}'
        ])
    
    def _check_metrics(self, color: str) -> bool:
        """Verificar métricas del deployment"""
        # Consultar Prometheus para métricas
        # Verificar que error rate esté bajo y latencia aceptable
        return True  # Placeholder
    
    def _rollback(self):
        """Rollback al deployment anterior"""
        target_color = "green" if self.active_color == "blue" else "blue"
        self._update_service(target_color, 100)

# Uso
deployment = BlueGreenDeployment("inference-api", namespace="production")

# Desplegar nueva versión
new_color = deployment.deploy_new_version("inference-api:v2.0.0", replicas=3)

# Cambiar tráfico gradualmente
if deployment.switch_traffic(percentage=10):
    # Si funciona, aumentar a 50%
    if deployment.switch_traffic(percentage=50):
        # Si sigue bien, cambiar al 100%
        deployment.switch_traffic(percentage=100)
else:
    # Si falla, hacer rollback
    deployment.rollback()
```

---

## 📊 Análisis de Costos y Optimización

### Calculadora de Costos Avanzada

```python
# scripts/cost_calculator.py
from dataclasses import dataclass
from typing import Dict, List
from datetime import datetime, timedelta
import json

@dataclass
class InstanceConfig:
    provider: str
    instance_type: str
    cost_per_hour: float
    gpu_count: int
    memory_gb: int

@dataclass
class UsageMetrics:
    requests_per_second: float
    avg_latency_ms: float
    avg_tokens_per_request: int
    uptime_percentage: float

class CostCalculator:
    def __init__(self):
        self.instance_configs = {
            "aws_ml_p3_2xlarge": InstanceConfig(
                provider="AWS",
                instance_type="ml.p3.2xlarge",
                cost_per_hour=3.06,
                gpu_count=1,
                memory_gb=61
            ),
            "gcp_n1_standard_4": InstanceConfig(
                provider="GCP",
                instance_type="n1-standard-4",
                cost_per_hour=0.19,
                gpu_count=0,
                memory_gb=15
            ),
            "azure_nc6s_v3": InstanceConfig(
                provider="Azure",
                instance_type="Standard_NC6s_v3",
                cost_per_hour=3.67,
                gpu_count=1,
                memory_gb=112
            )
        }
    
    def calculate_monthly_cost(
        self,
        instance_config: InstanceConfig,
        usage: UsageMetrics,
        num_instances: int = 1
    ) -> Dict:
        """Calcular costos mensuales"""
        hours_per_month = 730
        
        # Costo base de instancias
        instance_cost = (
            instance_config.cost_per_hour *
            hours_per_month *
            num_instances *
            usage.uptime_percentage
        )
        
        # Costo de datos (egress)
        data_transfer_cost = self._calculate_data_transfer_cost(
            usage.requests_per_second,
            usage.avg_tokens_per_request,
            hours_per_month
        )
        
        # Costo de almacenamiento (modelos, checkpoints)
        storage_cost = self._calculate_storage_cost()
        
        # Costo total
        total_cost = instance_cost + data_transfer_cost + storage_cost
        
        # Costo por request
        requests_per_month = (
            usage.requests_per_second *
            3600 *
            hours_per_month *
            usage.uptime_percentage
        )
        cost_per_request = total_cost / requests_per_month if requests_per_month > 0 else 0
        
        # Costo por 1K tokens
        tokens_per_month = requests_per_month * usage.avg_tokens_per_request
        cost_per_1k_tokens = (total_cost / tokens_per_month * 1000) if tokens_per_month > 0 else 0
        
        return {
            "instance_cost": instance_cost,
            "data_transfer_cost": data_transfer_cost,
            "storage_cost": storage_cost,
            "total_cost": total_cost,
            "cost_per_request": cost_per_request,
            "cost_per_1k_tokens": cost_per_1k_tokens,
            "requests_per_month": requests_per_month,
            "tokens_per_month": tokens_per_month
        }
    
    def compare_providers(
        self,
        usage: UsageMetrics,
        num_instances: int = 1
    ) -> Dict[str, Dict]:
        """Comparar costos entre proveedores"""
        results = {}
        
        for config_name, config in self.instance_configs.items():
            results[config_name] = self.calculate_monthly_cost(
                config,
                usage,
                num_instances
            )
        
        # Encontrar más económico
        best_provider = min(
            results.items(),
            key=lambda x: x[1]["total_cost"]
        )
        
        return {
            "comparison": results,
            "best_provider": {
                "name": best_provider[0],
                "cost": best_provider[1]["total_cost"],
                "savings_vs_avg": self._calculate_savings(results, best_provider[1]["total_cost"])
            }
        }
    
    def optimize_cost(
        self,
        target_throughput: float,
        max_latency_ms: float,
        budget: float
    ) -> Dict:
        """Encontrar configuración óptima dentro del presupuesto"""
        recommendations = []
        
        for config_name, config in self.instance_configs.items():
            # Estimar número de instancias necesarias
            estimated_instances = self._estimate_instances_needed(
                config,
                target_throughput,
                max_latency_ms
            )
            
            usage = UsageMetrics(
                requests_per_second=target_throughput,
                avg_latency_ms=max_latency_ms,
                avg_tokens_per_request=100,
                uptime_percentage=0.99
            )
            
            cost = self.calculate_monthly_cost(config, usage, estimated_instances)
            
            if cost["total_cost"] <= budget:
                recommendations.append({
                    "config": config_name,
                    "instances": estimated_instances,
                    "cost": cost["total_cost"],
                    "meets_latency": True,
                    "meets_throughput": True
                })
        
        if not recommendations:
            return {
                "error": "No configuration meets budget and requirements",
                "suggestions": [
                    "Increase budget",
                    "Reduce target throughput",
                    "Accept higher latency"
                ]
            }
        
        # Ordenar por costo
        recommendations.sort(key=lambda x: x["cost"])
        
        return {
            "recommendations": recommendations,
            "best_option": recommendations[0]
        }
    
    def _calculate_data_transfer_cost(
        self,
        rps: float,
        tokens_per_request: int,
        hours: float
    ) -> float:
        # Estimación: ~4 bytes por token, $0.09 por GB de egress (AWS)
        bytes_per_request = tokens_per_request * 4
        gb_per_month = (rps * bytes_per_request * 3600 * hours) / (1024 ** 3)
        return gb_per_month * 0.09
    
    def _calculate_storage_cost(self) -> float:
        # Estimación: 50GB de modelos, $0.023 por GB-mes (S3 standard)
        storage_gb = 50
        return storage_gb * 0.023
    
    def _estimate_instances_needed(
        self,
        config: InstanceConfig,
        target_throughput: float,
        max_latency_ms: float
    ) -> int:
        # Estimación simplificada
        # En producción, usar métricas reales
        if config.gpu_count > 0:
            # GPU: ~100 req/s por GPU
            throughput_per_instance = 100
        else:
            # CPU: ~20 req/s
            throughput_per_instance = 20
        
        instances = int(target_throughput / throughput_per_instance) + 1
        return max(1, instances)
    
    def _calculate_savings(self, results: Dict, best_cost: float) -> float:
        """Calcular ahorro vs promedio"""
        avg_cost = sum(r["total_cost"] for r in results.values()) / len(results)
        return ((avg_cost - best_cost) / avg_cost) * 100

# Uso
calculator = CostCalculator()

usage = UsageMetrics(
    requests_per_second=50,
    avg_latency_ms=250,
    avg_tokens_per_request=100,
    uptime_percentage=0.99
)

# Comparar proveedores
comparison = calculator.compare_providers(usage, num_instances=2)
print(json.dumps(comparison, indent=2))

# Optimizar con presupuesto
optimization = calculator.optimize_cost(
    target_throughput=100,
    max_latency_ms=300,
    budget=5000
)
print(json.dumps(optimization, indent=2))
```

---

## 🧠 Model Serving Avanzado con Triton

### Configuración Triton Inference Server

```python
# scripts/triton_setup.py
"""
Configuración para NVIDIA Triton Inference Server
Permite servir múltiples modelos con optimizaciones automáticas
"""

# config.pbtxt - Configuración del modelo
triton_config = """
name: "gpt2_model"
platform: "pytorch_libtorch"
max_batch_size: 32
input [
  {
    name: "input_ids"
    data_type: TYPE_INT64
    dims: [ -1 ]
  }
]
output [
  {
    name: "output_ids"
    data_type: TYPE_INT64
    dims: [ -1 ]
  }
]
instance_group [
  {
    count: 2
    kind: KIND_GPU
    gpus: [ 0, 1 ]
  }
]
dynamic_batching {
  max_queue_delay_microseconds: 20000
  preferred_batch_size: [ 8, 16 ]
}
"""

# Script de despliegue
deployment_script = """
# Desplegar modelo en Triton
tritonserver --model-repository=/models \
  --gpu-memory-fraction=0.8 \
  --http-port=8000 \
  --grpc-port=8001 \
  --metrics-port=8002
"""

# Cliente Python para Triton
triton_client_code = """
import tritonclient.http as tritonhttpclient
import numpy as np

class TritonInferenceClient:
    def __init__(self, url: str = "localhost:8000"):
        self.client = tritonhttpclient.InferenceServerClient(url)
    
    def infer(self, prompt: str, model_name: str = "gpt2_model"):
        # Tokenizar prompt
        input_ids = self._tokenize(prompt)
        
        # Preparar inputs
        inputs = [
            tritonhttpclient.InferInput("input_ids", input_ids.shape, "INT64")
        ]
        inputs[0].set_data_from_numpy(input_ids)
        
        # Preparar outputs
        outputs = [
            tritonhttpclient.InferRequestedOutput("output_ids")
        ]
        
        # Ejecutar inferencia
        response = self.client.infer(model_name, inputs, outputs=outputs)
        
        # Obtener resultados
        output_ids = response.as_numpy("output_ids")
        
        return self._detokenize(output_ids)
    
    def _tokenize(self, text: str) -> np.ndarray:
        # Implementar tokenización
        pass
    
    def _detokenize(self, ids: np.ndarray) -> str:
        # Implementar detokenización
        pass
"""
```

---

## 🎯 Feature Flags y Experimentación

### Sistema de Feature Flags

```python
# scripts/feature_flags.py
from typing import Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum
import redis
import json

class FeatureFlagStatus(Enum):
    ENABLED = "enabled"
    DISABLED = "disabled"
    ROLLOUT = "rollout"  # Activado para % de usuarios

@dataclass
class FeatureFlag:
    name: str
    status: FeatureFlagStatus
    rollout_percentage: float = 0.0
    conditions: Dict[str, Any] = None
    metadata: Dict[str, Any] = None

class FeatureFlagManager:
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis = redis_client or redis.Redis(host='localhost', port=6379, db=0)
        self.flags: Dict[str, FeatureFlag] = {}
        self._load_flags()
    
    def _load_flags(self):
        """Cargar flags desde Redis"""
        keys = self.redis.keys("feature_flag:*")
        for key in keys:
            data = json.loads(self.redis.get(key))
            self.flags[data["name"]] = FeatureFlag(**data)
    
    def is_enabled(self, flag_name: str, user_id: Optional[str] = None) -> bool:
        """Verificar si un flag está habilitado"""
        flag = self.flags.get(flag_name)
        
        if not flag:
            return False
        
        if flag.status == FeatureFlagStatus.DISABLED:
            return False
        
        if flag.status == FeatureFlagStatus.ENABLED:
            return True
        
        if flag.status == FeatureFlagStatus.ROLLOUT:
            return self._check_rollout(flag, user_id)
        
        return False
    
    def _check_rollout(self, flag: FeatureFlag, user_id: Optional[str]) -> bool:
        """Verificar si usuario está en rollout"""
        if not user_id:
            return False
        
        # Hash del user_id para distribución consistente
        user_hash = hash(f"{flag.name}:{user_id}") % 100
        return user_hash < flag.rollout_percentage
    
    def set_flag(self, flag: FeatureFlag):
        """Establecer o actualizar un flag"""
        self.flags[flag.name] = flag
        
        # Persistir en Redis
        key = f"feature_flag:{flag.name}"
        self.redis.set(key, json.dumps({
            "name": flag.name,
            "status": flag.status.value,
            "rollout_percentage": flag.rollout_percentage,
            "conditions": flag.conditions or {},
            "metadata": flag.metadata or {}
        }))
    
    def increment_rollout(self, flag_name: str, percentage: float):
        """Incrementar porcentaje de rollout gradualmente"""
        flag = self.flags.get(flag_name)
        if flag:
            flag.rollout_percentage = min(100.0, flag.rollout_percentage + percentage)
            self.set_flag(flag)

# Uso en aplicación
flag_manager = FeatureFlagManager()

# Crear flag
new_model_flag = FeatureFlag(
    name="new_model_v2",
    status=FeatureFlagStatus.ROLLOUT,
    rollout_percentage=10.0,
    metadata={"description": "Nuevo modelo GPT-2 mejorado"}
)
flag_manager.set_flag(new_model_flag)

# Verificar en endpoint
@app.post("/v1/infer")
async def infer(request: InferRequest):
    # Verificar feature flag
    use_new_model = flag_manager.is_enabled("new_model_v2", user_id=request.user_id)
    
    if use_new_model:
        model = load_model("models/gpt2-v2.pt")
    else:
        model = load_model("models/gpt2-v1.pt")
    
    return await model.generate(request.prompt)
```

---

*Más mejoras agregadas - Versión 2.4*
*Total de contenido: Más de 6,000 líneas de documentación práctica*


