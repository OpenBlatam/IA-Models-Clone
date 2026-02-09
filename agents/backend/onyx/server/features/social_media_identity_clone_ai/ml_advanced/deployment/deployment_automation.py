"""
Automatización de deployment
"""

import logging
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
import subprocess
import shutil

logger = logging.getLogger(__name__)


class DeploymentAutomation:
    """Automatización de deployment"""
    
    def __init__(self, deployment_dir: str = "./deployments"):
        self.deployment_dir = Path(deployment_dir)
        self.deployment_dir.mkdir(parents=True, exist_ok=True)
    
    def create_docker_image(
        self,
        model_path: str,
        model_name: str,
        tag: str = "latest"
    ) -> Dict[str, Any]:
        """
        Crea imagen Docker con modelo
        
        Args:
            model_path: Path al modelo
            model_name: Nombre del modelo
            tag: Tag de la imagen
            
        Returns:
            Información del deployment
        """
        try:
            # Crear Dockerfile
            dockerfile_content = f"""
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Copiar modelo
COPY {model_path} ./models/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
"""
            
            dockerfile_path = self.deployment_dir / "Dockerfile"
            with open(dockerfile_path, 'w') as f:
                f.write(dockerfile_content)
            
            # Build image
            image_name = f"{model_name}:{tag}"
            build_cmd = [
                "docker", "build",
                "-t", image_name,
                "-f", str(dockerfile_path),
                "."
            ]
            
            result = subprocess.run(build_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Imagen Docker creada: {image_name}")
                return {
                    "success": True,
                    "image_name": image_name,
                    "dockerfile_path": str(dockerfile_path)
                }
            else:
                logger.error(f"Error creando imagen: {result.stderr}")
                return {
                    "success": False,
                    "error": result.stderr
                }
                
        except Exception as e:
            logger.error(f"Error en deployment: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    def create_kubernetes_deployment(
        self,
        model_name: str,
        replicas: int = 3,
        image_name: str = None
    ) -> Dict[str, Any]:
        """
        Crea deployment de Kubernetes
        
        Args:
            model_name: Nombre del modelo
            replicas: Número de réplicas
            image_name: Nombre de la imagen
            
        Returns:
            Configuración de Kubernetes
        """
        if image_name is None:
            image_name = f"{model_name}:latest"
        
        k8s_config = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": f"{model_name}-deployment"
            },
            "spec": {
                "replicas": replicas,
                "selector": {
                    "matchLabels": {
                        "app": model_name
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": model_name
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": model_name,
                            "image": image_name,
                            "ports": [{
                                "containerPort": 8000
                            }],
                            "resources": {
                                "requests": {
                                    "memory": "2Gi",
                                    "cpu": "1"
                                },
                                "limits": {
                                    "memory": "4Gi",
                                    "cpu": "2"
                                }
                            }
                        }]
                    }
                }
            }
        }
        
        # Guardar configuración
        k8s_path = self.deployment_dir / f"{model_name}-deployment.yaml"
        with open(k8s_path, 'w') as f:
            json.dump(k8s_config, f, indent=2)
        
        logger.info(f"Configuración Kubernetes creada: {k8s_path}")
        
        return {
            "success": True,
            "k8s_config": k8s_config,
            "k8s_path": str(k8s_path)
        }
    
    def create_health_check_script(self, model_name: str) -> str:
        """Crea script de health check"""
        script_content = f"""#!/bin/bash
# Health check para {model_name}

HEALTH_URL="http://localhost:8000/health"
MAX_RETRIES=5
RETRY_DELAY=5

for i in $(seq 1 $MAX_RETRIES); do
    if curl -f $HEALTH_URL > /dev/null 2>&1; then
        echo "Health check passed"
        exit 0
    fi
    echo "Health check failed, retrying in $RETRY_DELAY seconds..."
    sleep $RETRY_DELAY
done

echo "Health check failed after $MAX_RETRIES attempts"
exit 1
"""
        
        script_path = self.deployment_dir / f"{model_name}-health-check.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Hacer ejecutable
        script_path.chmod(0o755)
        
        return str(script_path)




