import logging
import time
import json
import os
import shutil
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from pathlib import Path
import hashlib
import yaml
import mlflow
import mlflow.pytorch
import mlflow.sklearn
from mlflow.tracking import MlflowClient

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import sklearn
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Enums
class ModelStatus(Enum):
    """Model lifecycle status."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    FAILED = "failed"

class DeploymentStatus(Enum):
    """Deployment pipeline status."""
    PENDING = "pending"
    BUILDING = "building"
    TESTING = "testing"
    DEPLOYED = "deployed"
    FAILED = "failed"
    ROLLBACK = "rollback"

class ModelType(Enum):
    """Supported model types."""
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    SKLEARN = "sklearn"
    ONNX = "onnx"
    CUSTOM = "custom"

# Configurations
@dataclass
class MLOpsConfig:
    """Configuration for Advanced MLOps System."""
    # Core Settings
    enable_mlflow: bool = True
    enable_model_registry: bool = True
    enable_deployment_pipelines: bool = True
    enable_lifecycle_management: bool = True
    
    # Model Registry
    registry_path: str = "mlops_registry"
    max_model_versions: int = 10
    enable_auto_archiving: bool = True
    archive_threshold_days: int = 90
    
    # Deployment
    enable_auto_deployment: bool = True
    enable_rollback: bool = True
    deployment_timeout_minutes: int = 30
    health_check_interval: int = 60
    
    # Monitoring
    enable_production_monitoring: bool = True
    enable_drift_detection: bool = True
    enable_auto_retraining: bool = True
    performance_threshold: float = 0.8
    
    # A/B Testing
    enable_ab_testing: bool = True
    ab_test_traffic_split: float = 0.1
    ab_test_duration_days: int = 7
    
    # Logging
    log_level: str = "INFO"
    enable_audit_logging: bool = True

# Core Classes
class ModelRegistry:
    """Advanced model registry with versioning and lifecycle management."""
    
    def __init__(self, config: MLOpsConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.registry")
        self.registry_path = Path(config.registry_path)
        self.registry_path.mkdir(exist_ok=True)
        self.models = {}
        self._load_registry()
        
        if config.enable_mlflow:
            mlflow.set_tracking_uri(f"file://{self.registry_path}/mlflow")
    
    def _load_registry(self):
        """Load existing registry from disk."""
        registry_file = self.registry_path / "registry.json"
        if registry_file.exists():
            try:
                with open(registry_file, 'r') as f:
                    self.models = json.load(f)
                self.logger.info(f"Loaded {len(self.models)} models from registry")
            except Exception as e:
                self.logger.error(f"Failed to load registry: {e}")
    
    def _save_registry(self):
        """Save registry to disk."""
        registry_file = self.registry_path / "registry.json"
        try:
            with open(registry_file, 'w') as f:
                json.dump(self.models, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save registry: {e}")
    
    def register_model(self, model_name: str, model_path: str, 
                      model_type: ModelType, metadata: Dict[str, Any]) -> str:
        """Register a new model version."""
        if model_name not in self.models:
            self.models[model_name] = []
        
        # Generate version hash
        version_hash = hashlib.md5(f"{model_name}_{time.time()}".encode()).hexdigest()[:8]
        
        model_info = {
            "version": version_hash,
            "path": model_path,
            "type": model_type.value,
            "status": ModelStatus.DEVELOPMENT.value,
            "created_at": datetime.now().isoformat(),
            "metadata": metadata,
            "performance_metrics": {},
            "deployment_history": []
        }
        
        self.models[model_name].append(model_info)
        
        # Enforce max versions
        if len(self.models[model_name]) > self.config.max_model_versions:
            oldest_version = self.models[model_name].pop(0)
            self._archive_model(model_name, oldest_version)
        
        self._save_registry()
        self.logger.info(f"Registered model {model_name} version {version_hash}")
        return version_hash
    
    def _archive_model(self, model_name: str, model_info: Dict[str, Any]):
        """Archive old model version."""
        archive_path = self.registry_path / "archived" / model_name
        archive_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Move model files
            old_path = Path(model_info["path"])
            if old_path.exists():
                new_path = archive_path / f"{model_info['version']}_{old_path.name}"
                shutil.move(str(old_path), str(new_path))
                model_info["path"] = str(new_path)
                model_info["status"] = ModelStatus.ARCHIVED.value
                model_info["archived_at"] = datetime.now().isoformat()
        except Exception as e:
            self.logger.error(f"Failed to archive model {model_name}: {e}")
    
    def get_model_info(self, model_name: str, version: Optional[str] = None) -> Dict[str, Any]:
        """Get model information."""
        if model_name not in self.models:
            return {}
        
        if version is None:
            # Return latest version
            return self.models[model_name][-1] if self.models[model_name] else {}
        
        for model_info in self.models[model_name]:
            if model_info["version"] == version:
                return model_info
        return {}
    
    def update_model_status(self, model_name: str, version: str, status: ModelStatus):
        """Update model status."""
        model_info = self.get_model_info(model_name, version)
        if model_info:
            model_info["status"] = status.value
            model_info["updated_at"] = datetime.now().isoformat()
            self._save_registry()
            self.logger.info(f"Updated {model_name} version {version} status to {status.value}")

class DeploymentPipeline:
    """Automated deployment pipeline for ML models."""
    
    def __init__(self, config: MLOpsConfig, registry: ModelRegistry):
        self.config = config
        self.registry = registry
        self.logger = logging.getLogger(f"{__name__}.pipeline")
        self.deployments = {}
        self.deployment_history = []
    
    def deploy_model(self, model_name: str, version: str, 
                    target_environment: str = "staging") -> str:
        """Deploy a model to target environment."""
        deployment_id = f"{model_name}_{version}_{target_environment}_{int(time.time())}"
        
        deployment_info = {
            "id": deployment_id,
            "model_name": model_name,
            "version": version,
            "target_environment": target_environment,
            "status": DeploymentStatus.PENDING.value,
            "created_at": datetime.now().isoformat(),
            "steps": [],
            "logs": []
        }
        
        self.deployments[deployment_id] = deployment_info
        self.logger.info(f"Starting deployment {deployment_id}")
        
        try:
            # Execute deployment steps
            self._execute_deployment_steps(deployment_id)
            return deployment_id
        except Exception as e:
            self.logger.error(f"Deployment {deployment_id} failed: {e}")
            self._update_deployment_status(deployment_id, DeploymentStatus.FAILED)
            raise
    
    def _execute_deployment_steps(self, deployment_id: str):
        """Execute deployment pipeline steps."""
        deployment = self.deployments[deployment_id]
        
        # Step 1: Build
        self._update_deployment_status(deployment_id, DeploymentStatus.BUILDING)
        self._add_deployment_step(deployment_id, "build", "Building model container...")
        time.sleep(2)  # Simulate build time
        
        # Step 2: Test
        self._update_deployment_status(deployment_id, DeploymentStatus.TESTING)
        self._add_deployment_step(deployment_id, "test", "Running model tests...")
        time.sleep(3)  # Simulate testing time
        
        # Step 3: Deploy
        self._update_deployment_status(deployment_id, DeploymentStatus.DEPLOYED)
        self._add_deployment_step(deployment_id, "deploy", "Model deployed successfully")
        
        # Update registry
        self.registry.update_model_status(
            deployment["model_name"], 
            deployment["version"], 
            ModelStatus.PRODUCTION if deployment["target_environment"] == "production" else ModelStatus.STAGING
        )
        
        self.logger.info(f"Deployment {deployment_id} completed successfully")
    
    def _update_deployment_status(self, deployment_id: str, status: DeploymentStatus):
        """Update deployment status."""
        if deployment_id in self.deployments:
            self.deployments[deployment_id]["status"] = status.value
            self.deployments[deployment_id]["updated_at"] = datetime.now().isoformat()
    
    def _add_deployment_step(self, deployment_id: str, step_name: str, description: str):
        """Add deployment step."""
        if deployment_id in self.deployments:
            step = {
                "name": step_name,
                "description": description,
                "timestamp": datetime.now().isoformat(),
                "status": "completed"
            }
            self.deployments[deployment_id]["steps"].append(step)
    
    def rollback_deployment(self, deployment_id: str) -> bool:
        """Rollback a deployment."""
        if deployment_id not in self.deployments:
            return False
        
        deployment = self.deployments[deployment_id]
        self.logger.info(f"Rolling back deployment {deployment_id}")
        
        try:
            # Simulate rollback
            self._update_deployment_status(deployment_id, DeploymentStatus.ROLLBACK)
            time.sleep(2)
            
            # Update registry status
            self.registry.update_model_status(
                deployment["model_name"],
                deployment["version"],
                ModelStatus.ARCHIVED
            )
            
            self.logger.info(f"Rollback completed for {deployment_id}")
            return True
        except Exception as e:
            self.logger.error(f"Rollback failed for {deployment_id}: {e}")
            return False

class ModelLifecycleManager:
    """Comprehensive model lifecycle management."""
    
    def __init__(self, config: MLOpsConfig, registry: ModelRegistry):
        self.config = config
        self.registry = registry
        self.logger = logging.getLogger(f"{__name__}.lifecycle")
        self.lifecycle_rules = {}
        self._load_lifecycle_rules()
    
    def _load_lifecycle_rules(self):
        """Load lifecycle management rules."""
        rules_file = Path(self.config.registry_path) / "lifecycle_rules.yaml"
        if rules_file.exists():
            try:
                with open(rules_file, 'r') as f:
                    self.lifecycle_rules = yaml.safe_load(f)
            except Exception as e:
                self.logger.error(f"Failed to load lifecycle rules: {e}")
    
    def create_lifecycle_rule(self, model_name: str, rules: Dict[str, Any]):
        """Create lifecycle management rules for a model."""
        self.lifecycle_rules[model_name] = rules
        self._save_lifecycle_rules()
        self.logger.info(f"Created lifecycle rules for {model_name}")
    
    def _save_lifecycle_rules(self):
        """Save lifecycle rules to disk."""
        rules_file = Path(self.config.registry_path) / "lifecycle_rules.yaml"
        try:
            with open(rules_file, 'w') as f:
                yaml.dump(self.lifecycle_rules, f, default_flow_style=False)
        except Exception as e:
            self.logger.error(f"Failed to save lifecycle rules: {e}")
    
    def apply_lifecycle_policies(self):
        """Apply lifecycle policies to all models."""
        for model_name, model_versions in self.registry.models.items():
            if model_name in self.lifecycle_rules:
                rules = self.lifecycle_rules[model_name]
                self._apply_model_policies(model_name, model_versions, rules)
    
    def _apply_model_policies(self, model_name: str, model_versions: List[Dict], rules: Dict):
        """Apply policies to a specific model."""
        current_time = datetime.now()
        
        for model_info in model_versions:
            if model_info["status"] == ModelStatus.PRODUCTION.value:
                # Check performance thresholds
                if self._should_retrain(model_info, rules):
                    self.logger.info(f"Triggering retraining for {model_name} version {model_info['version']}")
                    self._schedule_retraining(model_name, model_info)
                
                # Check archiving thresholds
                if self._should_archive(model_info, rules):
                    self.logger.info(f"Archiving {model_name} version {model_info['version']}")
                    self.registry.update_model_status(model_name, model_info['version'], ModelStatus.ARCHIVED)
    
    def _should_retrain(self, model_info: Dict, rules: Dict) -> bool:
        """Check if model should be retrained."""
        if "retraining" not in rules:
            return False
        
        retraining_rules = rules["retraining"]
        
        # Check performance threshold
        if "performance_threshold" in retraining_rules:
            current_performance = model_info.get("performance_metrics", {}).get("accuracy", 1.0)
            if current_performance < retraining_rules["performance_threshold"]:
                return True
        
        # Check time-based retraining
        if "retrain_interval_days" in retraining_rules:
            created_at = datetime.fromisoformat(model_info["created_at"])
            days_since_creation = (datetime.now() - created_at).days
            if days_since_creation >= retraining_rules["retrain_interval_days"]:
                return True
        
        return False
    
    def _should_archive(self, model_info: Dict, rules: Dict) -> bool:
        """Check if model should be archived."""
        if "archiving" not in rules:
            return False
        
        archiving_rules = rules["archiving"]
        
        # Check age threshold
        if "max_age_days" in archiving_rules:
            created_at = datetime.fromisoformat(model_info["created_at"])
            days_since_creation = (datetime.now() - created_at).days
            if days_since_creation >= archiving_rules["max_age_days"]:
                return True
        
        return False
    
    def _schedule_retraining(self, model_name: str, model_info: Dict):
        """Schedule model retraining."""
        # This would integrate with training systems
        self.logger.info(f"Scheduled retraining for {model_name} version {model_info['version']}")

class AdvancedMLOpsSystem:
    """Main system for Advanced MLOps capabilities."""
    
    def __init__(self, config: MLOpsConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.main_system")
        self.initialized = False
        
        # Initialize components
        self.registry = ModelRegistry(config)
        self.deployment_pipeline = DeploymentPipeline(config, self.registry)
        self.lifecycle_manager = ModelLifecycleManager(config, self.registry)
        
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize the MLOps system."""
        try:
            # Setup MLflow if enabled
            if self.config.enable_mlflow:
                mlflow.set_experiment("HeyGen_AI_Enterprise_MLOps")
                self.logger.info("MLflow integration initialized")
            
            # Create necessary directories
            Path(self.config.registry_path).mkdir(exist_ok=True)
            Path(self.config.registry_path / "archived").mkdir(exist_ok=True)
            Path(self.config.registry_path / "deployments").mkdir(exist_ok=True)
            
            self.initialized = True
            self.logger.info("Advanced MLOps System initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize MLOps system: {e}")
            raise
    
    def register_model(self, model_name: str, model_path: str, 
                      model_type: ModelType, metadata: Dict[str, Any]) -> str:
        """Register a new model in the registry."""
        if not self.initialized:
            raise RuntimeError("MLOps system not initialized")
        
        return self.registry.register_model(model_name, model_path, model_type, metadata)
    
    def deploy_model(self, model_name: str, version: str, 
                    target_environment: str = "staging") -> str:
        """Deploy a model using the deployment pipeline."""
        if not self.initialized:
            raise RuntimeError("MLOps system not initialized")
        
        return self.deployment_pipeline.deploy_model(model_name, version, target_environment)
    
    def get_model_info(self, model_name: str, version: Optional[str] = None) -> Dict[str, Any]:
        """Get model information from registry."""
        return self.registry.get_model_info(model_name, version)
    
    def update_model_status(self, model_name: str, version: str, status: ModelStatus):
        """Update model status in registry."""
        self.registry.update_model_status(model_name, version, status)
    
    def create_lifecycle_rule(self, model_name: str, rules: Dict[str, Any]):
        """Create lifecycle management rules for a model."""
        self.lifecycle_manager.create_lifecycle_rule(model_name, rules)
    
    def apply_lifecycle_policies(self):
        """Apply lifecycle policies to all models."""
        self.lifecycle_manager.apply_lifecycle_policies()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        return {
            "initialized": self.initialized,
            "total_models": len(self.registry.models),
            "total_deployments": len(self.deployment_pipeline.deployments),
            "mlflow_enabled": self.config.enable_mlflow,
            "registry_path": str(self.config.registry_path),
            "config": {
                "enable_model_registry": self.config.enable_model_registry,
                "enable_deployment_pipelines": self.config.enable_deployment_pipelines,
                "enable_lifecycle_management": self.config.enable_lifecycle_management
            }
        }

# Factory functions
def create_mlops_config(
    enable_mlflow: bool = True,
    enable_model_registry: bool = True,
    enable_deployment_pipelines: bool = True,
    enable_lifecycle_management: bool = True
) -> MLOpsConfig:
    """Create MLOps configuration."""
    return MLOpsConfig(
        enable_mlflow=enable_mlflow,
        enable_model_registry=enable_model_registry,
        enable_deployment_pipelines=enable_deployment_pipelines,
        enable_lifecycle_management=enable_lifecycle_management
    )

def create_advanced_mlops_system(config: Optional[MLOpsConfig] = None) -> AdvancedMLOpsSystem:
    """Create Advanced MLOps System instance."""
    if config is None:
        config = create_mlops_config()
    return AdvancedMLOpsSystem(config)

def create_minimal_mlops_config() -> MLOpsConfig:
    """Create minimal MLOps configuration."""
    return MLOpsConfig(
        enable_mlflow=False,
        enable_model_registry=True,
        enable_deployment_pipelines=True,
        enable_lifecycle_management=False
    )

def create_maximum_mlops_config() -> MLOpsConfig:
    """Create maximum MLOps configuration."""
    return MLOpsConfig(
        enable_mlflow=True,
        enable_model_registry=True,
        enable_deployment_pipelines=True,
        enable_lifecycle_management=True,
        enable_auto_deployment=True,
        enable_rollback=True,
        enable_production_monitoring=True,
        enable_drift_detection=True,
        enable_auto_retraining=True,
        enable_ab_testing=True
    )
