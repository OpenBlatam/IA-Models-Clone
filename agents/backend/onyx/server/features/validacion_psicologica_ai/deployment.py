"""
Model Deployment
================
Production deployment utilities
"""

from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
from pathlib import Path
import structlog
import json
import yaml
from datetime import datetime

logger = structlog.get_logger()


class ModelDeployment:
    """
    Model deployment manager
    """
    
    def __init__(self, deployment_dir: str = "./deployments"):
        """
        Initialize deployment manager
        
        Args:
            deployment_dir: Directory for deployments
        """
        self.deployment_dir = Path(deployment_dir)
        self.deployment_dir.mkdir(parents=True, exist_ok=True)
        logger.info("ModelDeployment initialized", deployment_dir=str(self.deployment_dir))
    
    def create_deployment_package(
        self,
        model: nn.Module,
        model_name: str,
        version: str,
        metadata: Optional[Dict[str, Any]] = None,
        dependencies: Optional[List[str]] = None
    ) -> str:
        """
        Create deployment package
        
        Args:
            model: Model to deploy
            model_name: Model name
            version: Version
            metadata: Additional metadata
            dependencies: Dependencies list
            
        Returns:
            Deployment package path
        """
        try:
            deployment_path = self.deployment_dir / f"{model_name}_v{version}"
            deployment_path.mkdir(exist_ok=True)
            
            # Save model
            model_path = deployment_path / "model.pt"
            torch.save(model.state_dict(), model_path)
            
            # Save metadata
            metadata_dict = {
                "model_name": model_name,
                "version": version,
                "created_at": datetime.utcnow().isoformat(),
                "pytorch_version": torch.__version__,
                **(metadata or {})
            }
            
            metadata_path = deployment_path / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata_dict, f, indent=2)
            
            # Save dependencies
            if dependencies:
                deps_path = deployment_path / "requirements.txt"
                with open(deps_path, 'w') as f:
                    f.write('\n'.join(dependencies))
            
            # Save deployment config
            config = {
                "model_name": model_name,
                "version": version,
                "model_path": str(model_path.relative_to(self.deployment_dir)),
                "metadata_path": str(metadata_path.relative_to(self.deployment_dir))
            }
            
            config_path = deployment_path / "deployment.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            
            logger.info("Deployment package created", path=str(deployment_path))
            return str(deployment_path)
        except Exception as e:
            logger.error("Error creating deployment package", error=str(e))
            raise
    
    def load_deployment(
        self,
        model_name: str,
        version: str
    ) -> Dict[str, Any]:
        """
        Load deployment package
        
        Args:
            model_name: Model name
            version: Version
            
        Returns:
            Deployment info
        """
        try:
            deployment_path = self.deployment_dir / f"{model_name}_v{version}"
            
            if not deployment_path.exists():
                raise ValueError(f"Deployment {model_name}_v{version} not found")
            
            # Load config
            config_path = deployment_path / "deployment.yaml"
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Load metadata
            metadata_path = deployment_path / "metadata.json"
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            return {
                "config": config,
                "metadata": metadata,
                "deployment_path": str(deployment_path)
            }
        except Exception as e:
            logger.error("Error loading deployment", error=str(e))
            raise


class ModelVersioning:
    """
    Model versioning system
    """
    
    def __init__(self, versions_dir: str = "./model_versions"):
        """
        Initialize versioning system
        
        Args:
            versions_dir: Directory for versions
        """
        self.versions_dir = Path(versions_dir)
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        logger.info("ModelVersioning initialized", versions_dir=str(self.versions_dir))
    
    def create_version(
        self,
        model: nn.Module,
        model_name: str,
        version: str,
        metrics: Optional[Dict[str, float]] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Create model version
        
        Args:
            model: Model
            model_name: Model name
            version: Version string
            metrics: Performance metrics
            tags: Tags
            
        Returns:
            Version path
        """
        try:
            version_path = self.versions_dir / model_name / version
            version_path.mkdir(parents=True, exist_ok=True)
            
            # Save model
            model_path = version_path / "model.pt"
            torch.save(model.state_dict(), model_path)
            
            # Save version info
            version_info = {
                "model_name": model_name,
                "version": version,
                "created_at": datetime.utcnow().isoformat(),
                "metrics": metrics or {},
                "tags": tags or []
            }
            
            info_path = version_path / "version_info.json"
            with open(info_path, 'w') as f:
                json.dump(version_info, f, indent=2)
            
            logger.info("Model version created", model_name=model_name, version=version)
            return str(version_path)
        except Exception as e:
            logger.error("Error creating model version", error=str(e))
            raise
    
    def list_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """
        List all versions of a model
        
        Args:
            model_name: Model name
            
        Returns:
            List of versions
        """
        try:
            model_dir = self.versions_dir / model_name
            if not model_dir.exists():
                return []
            
            versions = []
            for version_dir in model_dir.iterdir():
                if version_dir.is_dir():
                    info_path = version_dir / "version_info.json"
                    if info_path.exists():
                        with open(info_path, 'r') as f:
                            version_info = json.load(f)
                        versions.append(version_info)
            
            return sorted(versions, key=lambda x: x["created_at"], reverse=True)
        except Exception as e:
            logger.error("Error listing versions", error=str(e))
            raise


# Global instances
model_deployment = ModelDeployment()
model_versioning = ModelVersioning()




