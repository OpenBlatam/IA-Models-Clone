"""
Deployment Service - Servicio de despliegue
============================================

Servicio independiente para despliegue de proyectos.
"""

import logging
from typing import Dict, Any

from ..utils.deployment_generator import DeploymentGenerator

logger = logging.getLogger(__name__)


class DeploymentService:
    """Servicio para despliegue de proyectos"""
    
    def __init__(self, deployment_generator: DeploymentGenerator = None):
        self.deployment_generator = deployment_generator or DeploymentGenerator()
    
    async def deploy(
        self,
        project_path: str,
        platform: str
    ) -> Dict[str, Any]:
        """
        Despliega un proyecto.
        
        Args:
            project_path: Ruta del proyecto
            platform: Plataforma de despliegue
        
        Returns:
            Información de despliegue
        """
        try:
            result = self.deployment_generator.generate_deployment_config(
                project_path,
                platform
            )
            return result
        except Exception as e:
            logger.error(f"Error deploying project: {e}", exc_info=True)
            raise















