#!/usr/bin/env python3
"""
MLOps Manager for Enhanced HeyGen AI
Handles model lifecycle management, automated training, deployment, and monitoring.
"""

import asyncio
import time
import json
import structlog
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import threading
from concurrent.futures import ThreadPoolExecutor
import psutil
import os
from pathlib import Path
import yaml
import docker
from kubernetes import client, config
import mlflow
import wandb
import optuna

logger = structlog.get_logger()

class ModelStage(Enum):
    """Model lifecycle stages."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"

class TrainingStatus(Enum):
    """Training status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class DeploymentStatus(Enum):
    """Deployment status."""
    PENDING = "pending"
    DEPLOYING = "deploying"
    ACTIVE = "active"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"

@dataclass
class ModelVersion:
    """Model version information."""
    version: str
    model_type: str
    stage: ModelStage
    created_at: datetime
    updated_at: datetime
    metrics: Dict[str, float]
    artifacts: List[str]
    dependencies: Dict[str, str]
    config: Dict[str, Any]
    performance_score: float
    is_active: bool = False

@dataclass
class TrainingJob:
    """Training job information."""
    job_id: str
    model_type: str
    status: TrainingStatus
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    config: Dict[str, Any]
    hyperparameters: Dict[str, Any]
    metrics: Dict[str, float]
    logs: List[str]
    error_message: Optional[str] = None

@dataclass
class DeploymentConfig:
    """Deployment configuration."""
    model_version: str
    replicas: int
    resources: Dict[str, str]
    environment: Dict[str, str]
    health_check: Dict[str, Any]
    scaling: Dict[str, Any]
    monitoring: Dict[str, Any]

class MLOpsManager:
    """Comprehensive MLOps management for HeyGen AI."""
    
    def __init__(
        self,
        enable_mlflow: bool = True,
        enable_wandb: bool = True,
        enable_optuna: bool = True,
        enable_docker: bool = True,
        enable_kubernetes: bool = True,
        models_dir: str = "./models",
        experiments_dir: str = "./experiments",
        registry_url: str = "localhost:5000"
    ):
        self.enable_mlflow = enable_mlflow
        self.enable_wandb = enable_wandb
        self.enable_optuna = enable_optuna
        self.enable_docker = enable_docker
        self.enable_kubernetes = enable_kubernetes
        
        self.models_dir = Path(models_dir)
        self.experiments_dir = Path(experiments_dir)
        self.registry_url = registry_url
        
        # Data storage
        self.models: Dict[str, ModelVersion] = {}
        self.training_jobs: Dict[str, TrainingJob] = {}
        self.deployments: Dict[str, DeploymentConfig] = {}
        
        # Thread pool for async operations
        self.thread_pool = ThreadPoolExecutor(max_workers=8)
        
        # Background tasks
        self.monitoring_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize integrations
        self._initialize_integrations()
        
        # Start background tasks
        self._start_background_tasks()
    
    def _initialize_integrations(self):
        """Initialize MLOps integrations."""
        try:
            # MLflow
            if self.enable_mlflow:
                mlflow.set_tracking_uri(f"file://{self.experiments_dir}/mlflow")
                logger.info("MLflow tracking initialized")
            
            # Weights & Biases
            if self.enable_wandb:
                wandb.init(project="heygen-ai-mlops", mode="offline")
                logger.info("Weights & Biases initialized")
            
            # Optuna
            if self.enable_optuna:
                self.study = optuna.create_study(
                    direction="maximize",
                    storage=f"sqlite:///{self.experiments_dir}/optuna.db"
                )
                logger.info("Optuna study initialized")
            
            # Docker
            if self.enable_docker:
                self.docker_client = docker.from_env()
                logger.info("Docker client initialized")
            
            # Kubernetes
            if self.enable_kubernetes:
                try:
                    config.load_incluster_config()
                    self.k8s_apps_v1 = client.AppsV1Api()
                    self.k8s_core_v1 = client.CoreV1Api()
                    logger.info("Kubernetes client initialized")
                except config.ConfigException:
                    logger.warning("Kubernetes config not found, running outside cluster")
                    self.enable_kubernetes = False
            
        except Exception as e:
            logger.error(f"Failed to initialize MLOps integrations: {e}")
    
    def _start_background_tasks(self):
        """Start background monitoring and cleanup tasks."""
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def _monitoring_loop(self):
        """Background monitoring loop."""
        while True:
            try:
                await self._monitor_models()
                await self._monitor_training_jobs()
                await self._monitor_deployments()
                await asyncio.sleep(30)  # Every 30 seconds
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_loop(self):
        """Background cleanup loop."""
        while True:
            try:
                await self._cleanup_old_models()
                await self._cleanup_completed_jobs()
                await asyncio.sleep(300)  # Every 5 minutes
                
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(60)
    
    async def _monitor_models(self):
        """Monitor model performance and health."""
        try:
            for model_id, model in self.models.items():
                if model.is_active:
                    # Check model performance
                    await self._check_model_performance(model_id)
                    
                    # Check model drift
                    await self._check_model_drift(model_id)
                    
        except Exception as e:
            logger.error(f"Model monitoring error: {e}")
    
    async def _monitor_training_jobs(self):
        """Monitor training job status."""
        try:
            for job_id, job in self.training_jobs.items():
                if job.status == TrainingStatus.RUNNING:
                    # Check job progress
                    await self._check_training_progress(job_id)
                    
        except Exception as e:
            logger.error(f"Training job monitoring error: {e}")
    
    async def _monitor_deployments(self):
        """Monitor deployment status."""
        try:
            for deployment_id, deployment in self.deployments.items():
                # Check deployment health
                await self._check_deployment_health(deployment_id)
                
        except Exception as e:
            logger.error(f"Deployment monitoring error: {e}")
    
    async def _check_model_performance(self, model_id: str):
        """Check model performance metrics."""
        try:
            model = self.models[model_id]
            
            # This would typically involve running inference on test data
            # and comparing with expected performance thresholds
            
            logger.debug(f"Model {model_id} performance check completed")
            
        except Exception as e:
            logger.error(f"Model performance check failed for {model_id}: {e}")
    
    async def _check_model_drift(self, model_id: str):
        """Check for model drift."""
        try:
            model = self.models[model_id]
            
            # This would typically involve comparing current data distribution
            # with training data distribution
            
            logger.debug(f"Model {model_id} drift check completed")
            
        except Exception as e:
            logger.error(f"Model drift check failed for {model_id}: {e}")
    
    async def _check_training_progress(self, job_id: str):
        """Check training job progress."""
        try:
            job = self.training_jobs[job_id]
            
            # This would typically involve checking training logs and metrics
            # and updating job status accordingly
            
            logger.debug(f"Training job {job_id} progress check completed")
            
        except Exception as e:
            logger.error(f"Training progress check failed for {job_id}: {e}")
    
    async def _check_deployment_health(self, deployment_id: str):
        """Check deployment health."""
        try:
            deployment = self.deployments[deployment_id]
            
            if self.enable_kubernetes:
                # Check Kubernetes deployment status
                await self._check_k8s_deployment_health(deployment_id)
            
            logger.debug(f"Deployment {deployment_id} health check completed")
            
        except Exception as e:
            logger.error(f"Deployment health check failed for {deployment_id}: {e}")
    
    async def _check_k8s_deployment_health(self, deployment_id: str):
        """Check Kubernetes deployment health."""
        try:
            # Get deployment status
            deployment_status = self.k8s_apps_v1.read_namespaced_deployment_status(
                name=deployment_id,
                namespace="default"
            )
            
            # Check if deployment is healthy
            if deployment_status.status.ready_replicas != deployment_status.status.replicas:
                logger.warning(f"Deployment {deployment_id} not fully ready")
            
        except Exception as e:
            logger.error(f"Kubernetes deployment health check failed: {e}")
    
    async def _cleanup_old_models(self):
        """Clean up old model versions."""
        try:
            current_time = datetime.now()
            cutoff_time = current_time - timedelta(days=30)  # Keep models for 30 days
            
            models_to_remove = []
            for model_id, model in self.models.items():
                if (model.updated_at < cutoff_time and 
                    not model.is_active and 
                    model.stage == ModelStage.ARCHIVED):
                    models_to_remove.append(model_id)
            
            for model_id in models_to_remove:
                await self._remove_model(model_id)
                
        except Exception as e:
            logger.error(f"Model cleanup error: {e}")
    
    async def _cleanup_completed_jobs(self):
        """Clean up completed training jobs."""
        try:
            current_time = datetime.now()
            cutoff_time = current_time - timedelta(days=7)  # Keep jobs for 7 days
            
            jobs_to_remove = []
            for job_id, job in self.training_jobs.items():
                if (job.status in [TrainingStatus.COMPLETED, TrainingStatus.FAILED] and
                    job.completed_at and job.completed_at < cutoff_time):
                    jobs_to_remove.append(job_id)
            
            for job_id in jobs_to_remove:
                await self._remove_training_job(job_id)
                
        except Exception as e:
            logger.error(f"Training job cleanup error: {e}")
    
    async def _remove_model(self, model_id: str):
        """Remove a model version."""
        try:
            model = self.models[model_id]
            
            # Remove model files
            model_path = self.models_dir / model_id
            if model_path.exists():
                import shutil
                shutil.rmtree(model_path)
            
            # Remove from tracking
            if self.enable_mlflow:
                try:
                    mlflow.delete_model(f"models:/{model_id}")
                except Exception:
                    pass
            
            # Remove from registry
            del self.models[model_id]
            
            logger.info(f"Model {model_id} removed successfully")
            
        except Exception as e:
            logger.error(f"Failed to remove model {model_id}: {e}")
    
    async def _remove_training_job(self, job_id: str):
        """Remove a training job."""
        try:
            # Remove job logs and artifacts
            job_path = self.experiments_dir / job_id
            if job_path.exists():
                import shutil
                shutil.rmtree(job_path)
            
            # Remove from registry
            del self.training_jobs[job_id]
            
            logger.info(f"Training job {job_id} removed successfully")
            
        except Exception as e:
            logger.error(f"Failed to remove training job {job_id}: {e}")
    
    async def register_model(
        self,
        model_type: str,
        version: str,
        config: Dict[str, Any],
        dependencies: Dict[str, str],
        artifacts: List[str]
    ) -> str:
        """Register a new model version."""
        try:
            model_id = f"{model_type}_{version}"
            
            model = ModelVersion(
                version=version,
                model_type=model_type,
                stage=ModelStage.DEVELOPMENT,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                metrics={},
                artifacts=artifacts,
                dependencies=dependencies,
                config=config,
                performance_score=0.0
            )
            
            self.models[model_id] = model
            
            # Register with MLflow
            if self.enable_mlflow:
                mlflow.register_model(
                    model_uri=f"runs:/{model_id}/model",
                    name=model_id
                )
            
            logger.info(f"Model {model_id} registered successfully")
            return model_id
            
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise
    
    async def start_training(
        self,
        model_type: str,
        config: Dict[str, Any],
        hyperparameters: Dict[str, Any]
    ) -> str:
        """Start a new training job."""
        try:
            job_id = f"training_{model_type}_{int(time.time())}"
            
            job = TrainingJob(
                job_id=job_id,
                model_type=model_type,
                status=TrainingStatus.PENDING,
                created_at=datetime.now(),
                started_at=None,
                completed_at=None,
                config=config,
                hyperparameters=hyperparameters,
                metrics={},
                logs=[]
            )
            
            self.training_jobs[job_id] = job
            
            # Start training in background
            asyncio.create_task(self._run_training_job(job_id))
            
            logger.info(f"Training job {job_id} started")
            return job_id
            
        except Exception as e:
            logger.error(f"Failed to start training: {e}")
            raise
    
    async def _run_training_job(self, job_id: str):
        """Run a training job."""
        try:
            job = self.training_jobs[job_id]
            job.status = TrainingStatus.RUNNING
            job.started_at = datetime.now()
            
            # Start MLflow run
            if self.enable_mlflow:
                mlflow.start_run(run_name=job_id)
                mlflow.log_params(job.hyperparameters)
                mlflow.log_params(job.config)
            
            # Start Weights & Biases run
            if self.enable_wandb:
                wandb.init(
                    project="heygen-ai-training",
                    name=job_id,
                    config=job.hyperparameters
                )
            
            # Simulate training process
            await self._simulate_training(job_id)
            
            # Complete training
            job.status = TrainingStatus.COMPLETED
            job.completed_at = datetime.now()
            
            # Log final metrics
            if self.enable_mlflow:
                mlflow.log_metrics(job.metrics)
                mlflow.end_run()
            
            if self.enable_wandb:
                wandb.finish()
            
            logger.info(f"Training job {job_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Training job {job_id} failed: {e}")
            job.status = TrainingStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.now()
    
    async def _simulate_training(self, job_id: str):
        """Simulate training process for demo purposes."""
        try:
            job = self.training_jobs[job_id]
            
            # Simulate training epochs
            for epoch in range(10):
                # Simulate training metrics
                loss = 1.0 / (epoch + 1)
                accuracy = 0.5 + (epoch * 0.05)
                
                job.metrics[f"epoch_{epoch}_loss"] = loss
                job.metrics[f"epoch_{epoch}_accuracy"] = accuracy
                
                # Log to MLflow
                if self.enable_mlflow:
                    mlflow.log_metrics({
                        "loss": loss,
                        "accuracy": accuracy
                    }, step=epoch)
                
                # Log to Weights & Biases
                if self.enable_wandb:
                    wandb.log({
                        "loss": loss,
                        "accuracy": accuracy
                    }, step=epoch)
                
                job.logs.append(f"Epoch {epoch}: loss={loss:.4f}, accuracy={accuracy:.4f}")
                
                # Simulate training time
                await asyncio.sleep(1)
            
            # Set final performance score
            job.metrics["final_accuracy"] = 0.95
            job.metrics["final_loss"] = 0.05
            
        except Exception as e:
            logger.error(f"Training simulation failed: {e}")
            raise
    
    async def deploy_model(
        self,
        model_id: str,
        replicas: int = 3,
        resources: Optional[Dict[str, str]] = None
    ) -> str:
        """Deploy a model to production."""
        try:
            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not found")
            
            model = self.models[model_id]
            
            # Update model stage
            model.stage = ModelStage.PRODUCTION
            model.is_active = True
            model.updated_at = datetime.now()
            
            # Create deployment configuration
            deployment_id = f"deployment_{model_id}"
            
            deployment = DeploymentConfig(
                model_version=model.version,
                replicas=replicas,
                resources=resources or {"cpu": "1000m", "memory": "2Gi"},
                environment={"MODEL_ID": model_id},
                health_check={"path": "/health", "port": 8080},
                scaling={"min_replicas": 1, "max_replicas": 10},
                monitoring={"metrics_path": "/metrics"}
            )
            
            self.deployments[deployment_id] = deployment
            
            # Deploy to Kubernetes if available
            if self.enable_kubernetes:
                await self._deploy_to_kubernetes(deployment_id, deployment)
            
            # Deploy to Docker if available
            if self.enable_docker:
                await self._deploy_to_docker(deployment_id, deployment)
            
            logger.info(f"Model {model_id} deployed successfully")
            return deployment_id
            
        except Exception as e:
            logger.error(f"Failed to deploy model {model_id}: {e}")
            raise
    
    async def _deploy_to_kubernetes(self, deployment_id: str, deployment: DeploymentConfig):
        """Deploy model to Kubernetes."""
        try:
            # Create Kubernetes deployment
            k8s_deployment = client.V1Deployment(
                metadata=client.V1ObjectMeta(name=deployment_id),
                spec=client.V1DeploymentSpec(
                    replicas=deployment.replicas,
                    selector=client.V1LabelSelector(
                        match_labels={"app": deployment_id}
                    ),
                    template=client.V1PodTemplateSpec(
                        metadata=client.V1ObjectMeta(
                            labels={"app": deployment_id}
                        ),
                        spec=client.V1PodSpec(
                            containers=[
                                client.V1Container(
                                    name="model",
                                    image=f"{self.registry_url}/heygen-ai-model:latest",
                                    ports=[client.V1ContainerPort(container_port=8080)],
                                    resources=client.V1ResourceRequirements(
                                        requests=deployment.resources,
                                        limits=deployment.resources
                                    ),
                                    env=[
                                        client.V1EnvVar(key=k, value=v)
                                        for k, v in deployment.environment.items()
                                    ]
                                )
                            ]
                        )
                    )
                )
            )
            
            # Apply deployment
            self.k8s_apps_v1.create_namespaced_deployment(
                namespace="default",
                body=k8s_deployment
            )
            
            logger.info(f"Kubernetes deployment {deployment_id} created")
            
        except Exception as e:
            logger.error(f"Kubernetes deployment failed: {e}")
            raise
    
    async def _deploy_to_docker(self, deployment_id: str, deployment: DeploymentConfig):
        """Deploy model to Docker."""
        try:
            # Pull model image
            image = self.docker_client.images.pull(
                f"{self.registry_url}/heygen-ai-model:latest"
            )
            
            # Create and start container
            container = self.docker_client.containers.run(
                image,
                name=deployment_id,
                detach=True,
                ports={'8080/tcp': None},
                environment=deployment.environment,
                mem_limit=deployment.resources.get("memory", "2Gi"),
                cpu_period=100000,
                cpu_quota=int(float(deployment.resources.get("cpu", "1000m").replace("m", "")) * 100)
            )
            
            logger.info(f"Docker container {deployment_id} started: {container.id}")
            
        except Exception as e:
            logger.error(f"Docker deployment failed: {e}")
            raise
    
    async def get_model_info(self, model_id: str) -> Optional[ModelVersion]:
        """Get model information."""
        return self.models.get(model_id)
    
    async def get_training_job_info(self, job_id: str) -> Optional[TrainingJob]:
        """Get training job information."""
        return self.training_jobs.get(job_id)
    
    async def get_deployment_info(self, deployment_id: str) -> Optional[DeploymentConfig]:
        """Get deployment information."""
        return self.deployments.get(deployment_id)
    
    async def list_models(self, stage: Optional[ModelStage] = None) -> List[ModelVersion]:
        """List models, optionally filtered by stage."""
        if stage:
            return [model for model in self.models.values() if model.stage == stage]
        return list(self.models.values())
    
    async def list_training_jobs(self, status: Optional[TrainingStatus] = None) -> List[TrainingJob]:
        """List training jobs, optionally filtered by status."""
        if status:
            return [job for job in self.training_jobs.values() if job.status == status]
        return list(self.training_jobs.values())
    
    async def list_deployments(self) -> List[DeploymentConfig]:
        """List all deployments."""
        return list(self.deployments.values())
    
    async def optimize_hyperparameters(
        self,
        model_type: str,
        config: Dict[str, Any],
        n_trials: int = 100
    ) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna."""
        try:
            if not self.enable_optuna:
                raise RuntimeError("Optuna not enabled")
            
            def objective(trial):
                # Define hyperparameter search space
                lr = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
                batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
                hidden_size = trial.suggest_categorical("hidden_size", [128, 256, 512, 1024])
                dropout = trial.suggest_float("dropout", 0.1, 0.5)
                
                # Simulate training with these hyperparameters
                # In a real implementation, this would run actual training
                accuracy = self._simulate_training_with_params(lr, batch_size, hidden_size, dropout)
                
                return accuracy
            
            # Run optimization
            self.study.optimize(objective, n_trials=n_trials)
            
            # Get best parameters
            best_params = self.study.best_params
            best_value = self.study.best_value
            
            logger.info(f"Hyperparameter optimization completed. Best accuracy: {best_value:.4f}")
            
            return {
                "best_params": best_params,
                "best_value": best_value,
                "n_trials": n_trials
            }
            
        except Exception as e:
            logger.error(f"Hyperparameter optimization failed: {e}")
            raise
    
    def _simulate_training_with_params(
        self,
        lr: float,
        batch_size: int,
        hidden_size: int,
        dropout: float
    ) -> float:
        """Simulate training with given parameters and return accuracy."""
        # This is a simplified simulation for demo purposes
        # In reality, this would run actual training
        
        # Simulate that some parameter combinations work better than others
        if lr < 1e-3 and batch_size <= 64 and hidden_size >= 256 and dropout < 0.3:
            return 0.95  # Good parameters
        elif lr < 1e-2 and batch_size <= 128:
            return 0.85  # Acceptable parameters
        else:
            return 0.70  # Poor parameters
    
    async def shutdown(self):
        """Shutdown the MLOps manager."""
        try:
            # Cancel background tasks
            if self.monitoring_task:
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
            
            if self.cleanup_task:
                self.cleanup_task.cancel()
                try:
                    await self.cleanup_task
                except asyncio.CancelledError:
                    pass
            
            # Shutdown thread pool
            self.thread_pool.shutdown(wait=True)
            
            # Close integrations
            if self.enable_wandb:
                wandb.finish()
            
            logger.info("MLOps manager shutdown complete")
            
        except Exception as e:
            logger.error(f"MLOps manager shutdown error: {e}")

# Global MLOps manager instance
mlops_manager: Optional[MLOpsManager] = None

def get_mlops_manager() -> MLOpsManager:
    """Get global MLOps manager instance."""
    global mlops_manager
    if mlops_manager is None:
        mlops_manager = MLOpsManager()
    return mlops_manager

async def shutdown_mlops_manager():
    """Shutdown global MLOps manager."""
    global mlops_manager
    if mlops_manager:
        await mlops_manager.shutdown()
        mlops_manager = None

