"""
Kubernetes Generator
====================

Generador de configuraciones Kubernetes para deployment de modelos.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class KubernetesConfig:
    """Configuración de Kubernetes."""
    app_name: str
    namespace: str = "default"
    replicas: int = 1
    cpu_request: str = "500m"
    cpu_limit: str = "2000m"
    memory_request: str = "1Gi"
    memory_limit: str = "4Gi"
    gpu_enabled: bool = False
    gpu_count: int = 1
    image: str = ""
    port: int = 8000
    service_type: str = "LoadBalancer"


class KubernetesGenerator:
    """
    Generador de configuraciones Kubernetes.
    """
    
    def __init__(self):
        """Inicializar generador."""
        pass
    
    def generate_deployment(
        self,
        project_dir: Path,
        config: Optional[KubernetesConfig] = None
    ) -> str:
        """
        Generar Deployment de Kubernetes.
        
        Args:
            project_dir: Directorio del proyecto
            config: Configuración (opcional)
            
        Returns:
            Contenido del Deployment YAML
        """
        if config is None:
            config = KubernetesConfig(
                app_name=project_dir.name,
                image=f"{project_dir.name}:latest"
            )
        
        deployment_content = f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: {config.app_name}
  namespace: {config.namespace}
  labels:
    app: {config.app_name}
spec:
  replicas: {config.replicas}
  selector:
    matchLabels:
      app: {config.app_name}
  template:
    metadata:
      labels:
        app: {config.app_name}
    spec:
      containers:
      - name: {config.app_name}
        image: {config.image}
        ports:
        - containerPort: {config.port}
          name: http
        resources:
          requests:
            cpu: {config.cpu_request}
            memory: {config.memory_request}
          limits:
            cpu: {config.cpu_limit}
            memory: {config.memory_limit}
"""
        
        if config.gpu_enabled:
            deployment_content += f"""        resources:
          limits:
            nvidia.com/gpu: {config.gpu_count}
"""
        
        deployment_content += """        env:
        - name: PYTHONUNBUFFERED
          value: "1"
        livenessProbe:
          httpGet:
            path: /health
            port: http
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: http
          initialDelaySeconds: 10
          periodSeconds: 5
"""
        
        return deployment_content
    
    def generate_service(
        self,
        project_dir: Path,
        config: Optional[KubernetesConfig] = None
    ) -> str:
        """
        Generar Service de Kubernetes.
        
        Args:
            project_dir: Directorio del proyecto
            config: Configuración (opcional)
            
        Returns:
            Contenido del Service YAML
        """
        if config is None:
            config = KubernetesConfig(app_name=project_dir.name)
        
        service_content = f"""apiVersion: v1
kind: Service
metadata:
  name: {config.app_name}
  namespace: {config.namespace}
  labels:
    app: {config.app_name}
spec:
  type: {config.service_type}
  ports:
  - port: 80
    targetPort: {config.port}
    protocol: TCP
    name: http
  selector:
    app: {config.app_name}
"""
        
        return service_content
    
    def generate_hpa(
        self,
        project_dir: Path,
        config: Optional[KubernetesConfig] = None,
        min_replicas: int = 1,
        max_replicas: int = 10,
        target_cpu: int = 70
    ) -> str:
        """
        Generar HorizontalPodAutoscaler.
        
        Args:
            project_dir: Directorio del proyecto
            config: Configuración (opcional)
            min_replicas: Mínimo de réplicas
            max_replicas: Máximo de réplicas
            target_cpu: CPU objetivo (%)
            
        Returns:
            Contenido del HPA YAML
        """
        if config is None:
            config = KubernetesConfig(app_name=project_dir.name)
        
        hpa_content = f"""apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {config.app_name}
  namespace: {config.namespace}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {config.app_name}
  minReplicas: {min_replicas}
  maxReplicas: {max_replicas}
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: {target_cpu}
"""
        
        return hpa_content
    
    def generate_ingress(
        self,
        project_dir: Path,
        config: Optional[KubernetesConfig] = None,
        host: str = ""
    ) -> str:
        """
        Generar Ingress.
        
        Args:
            project_dir: Directorio del proyecto
            config: Configuración (opcional)
            host: Hostname (opcional)
            
        Returns:
            Contenido del Ingress YAML
        """
        if config is None:
            config = KubernetesConfig(app_name=project_dir.name)
        
        if not host:
            host = f"{config.app_name}.example.com"
        
        ingress_content = f"""apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: {config.app_name}
  namespace: {config.namespace}
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  rules:
  - host: {host}
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: {config.app_name}
            port:
              number: 80
"""
        
        return ingress_content
    
    def generate_all(
        self,
        project_dir: Path,
        config: Optional[KubernetesConfig] = None
    ) -> Dict[str, str]:
        """
        Generar todas las configuraciones Kubernetes.
        
        Args:
            project_dir: Directorio del proyecto
            config: Configuración (opcional)
            
        Returns:
            Diccionario con archivos generados
        """
        if config is None:
            config = KubernetesConfig(
                app_name=project_dir.name,
                image=f"{project_dir.name}:latest"
            )
        
        files = {}
        k8s_dir = project_dir / "k8s"
        k8s_dir.mkdir(parents=True, exist_ok=True)
        
        # Deployment
        deployment_content = self.generate_deployment(project_dir, config)
        deployment_path = k8s_dir / "deployment.yaml"
        deployment_path.write_text(deployment_content, encoding='utf-8')
        files['k8s/deployment.yaml'] = deployment_content
        
        # Service
        service_content = self.generate_service(project_dir, config)
        service_path = k8s_dir / "service.yaml"
        service_path.write_text(service_content, encoding='utf-8')
        files['k8s/service.yaml'] = service_content
        
        # HPA
        hpa_content = self.generate_hpa(project_dir, config)
        hpa_path = k8s_dir / "hpa.yaml"
        hpa_path.write_text(hpa_content, encoding='utf-8')
        files['k8s/hpa.yaml'] = hpa_content
        
        # Ingress
        ingress_content = self.generate_ingress(project_dir, config)
        ingress_path = k8s_dir / "ingress.yaml"
        ingress_path.write_text(ingress_content, encoding='utf-8')
        files['k8s/ingress.yaml'] = ingress_content
        
        logger.info(f"Configuraciones Kubernetes generadas en {k8s_dir}")
        
        return files


# Instancia global
_global_k8s_generator: Optional[KubernetesGenerator] = None


def get_k8s_generator() -> KubernetesGenerator:
    """
    Obtener instancia global del generador de Kubernetes.
    
    Returns:
        Instancia del generador
    """
    global _global_k8s_generator
    
    if _global_k8s_generator is None:
        _global_k8s_generator = KubernetesGenerator()
    
    return _global_k8s_generator

