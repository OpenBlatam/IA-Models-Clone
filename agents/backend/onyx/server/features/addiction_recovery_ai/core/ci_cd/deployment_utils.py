"""
Deployment Utilities
CI/CD and deployment helpers
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class DeploymentConfig:
    """
    Deployment configuration manager
    """
    
    @staticmethod
    def load_deployment_config(path: str = "deployment_config.json") -> Dict[str, Any]:
        """
        Load deployment configuration
        
        Args:
            path: Config file path
            
        Returns:
            Configuration dictionary
        """
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return {}
    
    @staticmethod
    def save_deployment_config(config: Dict[str, Any], path: str = "deployment_config.json"):
        """
        Save deployment configuration
        
        Args:
            config: Configuration dictionary
            path: Config file path
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Deployment config saved to {path}")
    
    @staticmethod
    def get_environment_config() -> Dict[str, Any]:
        """
        Get configuration from environment variables
        
        Returns:
            Configuration dictionary
        """
        config = {
            "device": os.getenv("DEVICE", "cuda" if os.getenv("CUDA_AVAILABLE") == "true" else "cpu"),
            "batch_size": int(os.getenv("BATCH_SIZE", "32")),
            "num_workers": int(os.getenv("NUM_WORKERS", "4")),
            "use_mixed_precision": os.getenv("USE_MIXED_PRECISION", "true").lower() == "true",
            "model_path": os.getenv("MODEL_PATH", "models/"),
            "checkpoint_path": os.getenv("CHECKPOINT_PATH", "checkpoints/"),
            "log_level": os.getenv("LOG_LEVEL", "INFO")
        }
        return config


class HealthCheck:
    """
    Health check for deployment
    """
    
    @staticmethod
    def check_system_health() -> Dict[str, Any]:
        """
        Check system health
        
        Returns:
            Health status
        """
        import torch
        
        health = {
            "status": "healthy",
            "pytorch_available": True,
            "cuda_available": torch.cuda.is_available(),
            "issues": []
        }
        
        if not torch.cuda.is_available():
            health["issues"].append("CUDA not available")
        
        try:
            import transformers
            health["transformers_available"] = True
        except ImportError:
            health["transformers_available"] = False
            health["issues"].append("Transformers not available")
            health["status"] = "unhealthy"
        
        return health
    
    @staticmethod
    def check_model_health(model_path: str) -> Dict[str, Any]:
        """
        Check model health
        
        Args:
            model_path: Path to model
            
        Returns:
            Health status
        """
        health = {
            "status": "healthy",
            "model_exists": os.path.exists(model_path),
            "issues": []
        }
        
        if not health["model_exists"]:
            health["status"] = "unhealthy"
            health["issues"].append(f"Model not found: {model_path}")
        
        return health


def get_deployment_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Get deployment configuration"""
    if path:
        return DeploymentConfig.load_deployment_config(path)
    return DeploymentConfig.get_environment_config()


def check_system_health() -> Dict[str, Any]:
    """Check system health"""
    return HealthCheck.check_system_health()


def check_model_health(model_path: str) -> Dict[str, Any]:
    """Check model health"""
    return HealthCheck.check_model_health(model_path)








