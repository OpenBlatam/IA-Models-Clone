"""
Deployment Utilities
Utilities for deploying models and services.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ModelDeployer:
    """Deploy models to production."""
    
    def __init__(self, deployment_dir: str = "./deployments"):
        self.deployment_dir = Path(deployment_dir)
        self.deployment_dir.mkdir(parents=True, exist_ok=True)
    
    def deploy_model(
        self,
        model_path: str,
        model_name: str,
        version: str = "1.0.0",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Deploy a model."""
        source = Path(model_path)
        if not source.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Create deployment directory
        deploy_path = self.deployment_dir / model_name / version
        deploy_path.mkdir(parents=True, exist_ok=True)
        
        # Copy model
        if source.is_file():
            shutil.copy2(source, deploy_path / source.name)
        else:
            shutil.copytree(source, deploy_path, dirs_exist_ok=True)
        
        # Save metadata
        deploy_metadata = {
            "model_name": model_name,
            "version": version,
            "deployed_at": datetime.now().isoformat(),
            "source_path": str(source),
            **(metadata or {}),
        }
        
        metadata_path = deploy_path / "deployment_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(deploy_metadata, f, indent=2)
        
        logger.info(f"Model deployed: {deploy_path}")
        return str(deploy_path)
    
    def list_deployments(self) -> List[Dict[str, Any]]:
        """List all deployments."""
        deployments = []
        
        for model_dir in self.deployment_dir.iterdir():
            if not model_dir.is_dir():
                continue
            
            for version_dir in model_dir.iterdir():
                if not version_dir.is_dir():
                    continue
                
                metadata_path = version_dir / "deployment_metadata.json"
                if metadata_path.exists():
                    try:
                        with open(metadata_path, "r") as f:
                            metadata = json.load(f)
                        deployments.append(metadata)
                    except Exception:
                        pass
        
        return sorted(deployments, key=lambda x: x.get("deployed_at", ""), reverse=True)
    
    def get_deployment(self, model_name: str, version: str) -> Optional[Dict[str, Any]]:
        """Get deployment information."""
        deploy_path = self.deployment_dir / model_name / version
        metadata_path = deploy_path / "deployment_metadata.json"
        
        if not metadata_path.exists():
            return None
        
        with open(metadata_path, "r") as f:
            return json.load(f)
    
    def rollback(self, model_name: str, target_version: str) -> bool:
        """Rollback to a previous version."""
        deploy_path = self.deployment_dir / model_name / target_version
        if not deploy_path.exists():
            return False
        
        # Get current version
        current = self.get_current_version(model_name)
        if current:
            # Mark current as previous
            current_path = self.deployment_dir / model_name / current
            if current_path.exists():
                prev_path = current_path.parent / f"{current}.previous"
                if prev_path.exists():
                    shutil.rmtree(prev_path)
                current_path.rename(prev_path)
        
        # Activate target version
        target_path = self.deployment_dir / model_name / target_version
        active_path = self.deployment_dir / model_name / "active"
        if active_path.exists():
            active_path.unlink()
        active_path.symlink_to(target_path)
        
        logger.info(f"Rolled back {model_name} to version {target_version}")
        return True
    
    def get_current_version(self, model_name: str) -> Optional[str]:
        """Get current active version."""
        active_path = self.deployment_dir / model_name / "active"
        if active_path.exists() and active_path.is_symlink():
            return active_path.readlink().name
        return None


class ServiceDeployer:
    """Deploy services."""
    
    def __init__(self, config_dir: str = "./service_configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    def create_service_config(
        self,
        service_name: str,
        config: Dict[str, Any],
        environment: str = "production",
    ) -> str:
        """Create service configuration."""
        config_path = self.config_dir / f"{service_name}_{environment}.json"
        
        config_data = {
            "service_name": service_name,
            "environment": environment,
            "created_at": datetime.now().isoformat(),
            **config,
        }
        
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)
        
        logger.info(f"Service config created: {config_path}")
        return str(config_path)
    
    def get_service_config(
        self,
        service_name: str,
        environment: str = "production",
    ) -> Optional[Dict[str, Any]]:
        """Get service configuration."""
        config_path = self.config_dir / f"{service_name}_{environment}.json"
        
        if not config_path.exists():
            return None
        
        with open(config_path, "r") as f:
            return json.load(f)



