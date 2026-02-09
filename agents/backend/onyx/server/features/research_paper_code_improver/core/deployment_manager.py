"""
Model Deployment Manager - Gestor de despliegue de modelos
===========================================================
"""

import logging
import os
import shutil
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class DeploymentStatus(Enum):
    """Estado de despliegue"""
    PENDING = "pending"
    DEPLOYING = "deploying"
    ACTIVE = "active"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"


@dataclass
class Deployment:
    """Despliegue de modelo"""
    deployment_id: str
    model_version: str
    model_path: str
    status: DeploymentStatus
    created_at: datetime = field(default_factory=datetime.now)
    deployed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    health_check_url: Optional[str] = None


class ModelDeploymentManager:
    """Gestor de despliegue de modelos"""
    
    def __init__(self, deployment_dir: str = "./deployments"):
        self.deployment_dir = Path(deployment_dir)
        self.deployment_dir.mkdir(parents=True, exist_ok=True)
        self.deployments: Dict[str, Deployment] = {}
        self.active_deployment: Optional[str] = None
    
    def create_deployment(
        self,
        model_version: str,
        model_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Deployment:
        """Crea un nuevo despliegue"""
        deployment_id = f"deploy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Copiar modelo a directorio de despliegues
        deployment_path = self.deployment_dir / f"{deployment_id}.pt"
        shutil.copy(model_path, deployment_path)
        
        deployment = Deployment(
            deployment_id=deployment_id,
            model_version=model_version,
            model_path=str(deployment_path),
            status=DeploymentStatus.PENDING,
            metadata=metadata or {}
        )
        
        self.deployments[deployment_id] = deployment
        logger.info(f"Despliegue creado: {deployment_id}")
        
        return deployment
    
    def deploy(
        self,
        deployment_id: str,
        health_check_url: Optional[str] = None
    ) -> bool:
        """Despliega un modelo"""
        if deployment_id not in self.deployments:
            logger.error(f"Despliegue no encontrado: {deployment_id}")
            return False
        
        deployment = self.deployments[deployment_id]
        deployment.status = DeploymentStatus.DEPLOYING
        
        try:
            # Aquí iría la lógica real de despliegue
            # Por ejemplo, copiar a servidor, actualizar configuración, etc.
            
            # Simular despliegue
            deployment.status = DeploymentStatus.ACTIVE
            deployment.deployed_at = datetime.now()
            deployment.health_check_url = health_check_url
            
            # Desactivar despliegue anterior
            if self.active_deployment:
                old_deployment = self.deployments.get(self.active_deployment)
                if old_deployment:
                    old_deployment.status = DeploymentStatus.PENDING
            
            self.active_deployment = deployment_id
            
            logger.info(f"Despliegue exitoso: {deployment_id}")
            return True
        
        except Exception as e:
            deployment.status = DeploymentStatus.FAILED
            logger.error(f"Error en despliegue: {e}")
            return False
    
    def rollback_deployment(self, deployment_id: str) -> bool:
        """Hace rollback de un despliegue"""
        if deployment_id not in self.deployments:
            return False
        
        deployment = self.deployments[deployment_id]
        deployment.status = DeploymentStatus.ROLLING_BACK
        
        try:
            # Lógica de rollback
            deployment.status = DeploymentStatus.PENDING
            self.active_deployment = None
            
            logger.info(f"Rollback exitoso: {deployment_id}")
            return True
        
        except Exception as e:
            deployment.status = DeploymentStatus.FAILED
            logger.error(f"Error en rollback: {e}")
            return False
    
    def get_active_deployment(self) -> Optional[Deployment]:
        """Obtiene despliegue activo"""
        if self.active_deployment:
            return self.deployments.get(self.active_deployment)
        return None
    
    def list_deployments(
        self,
        status: Optional[DeploymentStatus] = None
    ) -> List[Deployment]:
        """Lista despliegues"""
        deployments = list(self.deployments.values())
        
        if status:
            deployments = [d for d in deployments if d.status == status]
        
        return sorted(deployments, key=lambda d: d.created_at, reverse=True)
    
    def delete_deployment(self, deployment_id: str) -> bool:
        """Elimina un despliegue"""
        if deployment_id not in self.deployments:
            return False
        
        deployment = self.deployments[deployment_id]
        
        # No eliminar si está activo
        if deployment.status == DeploymentStatus.ACTIVE:
            logger.warning("No se puede eliminar despliegue activo")
            return False
        
        # Eliminar archivo
        if os.path.exists(deployment.model_path):
            os.remove(deployment.model_path)
        
        # Eliminar de registro
        del self.deployments[deployment_id]
        
        if self.active_deployment == deployment_id:
            self.active_deployment = None
        
        logger.info(f"Despliegue eliminado: {deployment_id}")
        return True
