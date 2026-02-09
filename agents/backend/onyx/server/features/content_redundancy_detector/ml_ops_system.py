"""
ML Ops System for Machine Learning Operations
Sistema ML Ops para operaciones de Machine Learning ultra-optimizado
"""

import asyncio
import logging
import time
import json
import threading
import hashlib
import pickle
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.routing import APIRouter
from collections import defaultdict, deque
import uuid
import os
import shutil
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
import torch
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import mlflow.pytorch

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Tipos de modelos ML"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"
    NEURAL_NETWORK = "neural_network"
    DEEP_LEARNING = "deep_learning"
    TRANSFORMER = "transformer"
    REINFORCEMENT_LEARNING = "reinforcement_learning"


class ModelStatus(Enum):
    """Estados de modelos ML"""
    TRAINING = "training"
    TRAINED = "trained"
    VALIDATING = "validating"
    VALIDATED = "validated"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    SERVING = "serving"
    RETIRED = "retired"
    FAILED = "failed"


class ExperimentStatus(Enum):
    """Estados de experimentos ML"""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DeploymentType(Enum):
    """Tipos de despliegue"""
    REST_API = "rest_api"
    BATCH = "batch"
    STREAMING = "streaming"
    EDGE = "edge"
    CONTAINER = "container"
    SERVERLESS = "serverless"


@dataclass
class MLModel:
    """Modelo ML"""
    id: str
    name: str
    version: str
    type: ModelType
    status: ModelStatus
    framework: str
    algorithm: str
    hyperparameters: Dict[str, Any]
    metrics: Dict[str, float]
    training_data_size: int
    validation_data_size: int
    created_at: float
    last_updated: float
    model_path: str
    metadata: Dict[str, Any]


@dataclass
class Experiment:
    """Experimento ML"""
    id: str
    name: str
    description: str
    status: ExperimentStatus
    model_type: ModelType
    hyperparameters: Dict[str, Any]
    metrics: Dict[str, float]
    created_at: float
    started_at: Optional[float]
    completed_at: Optional[float]
    model_id: Optional[str]
    metadata: Dict[str, Any]


@dataclass
class Dataset:
    """Dataset ML"""
    id: str
    name: str
    description: str
    size: int
    features: List[str]
    target_column: Optional[str]
    data_types: Dict[str, str]
    created_at: float
    last_updated: float
    file_path: str
    metadata: Dict[str, Any]


@dataclass
class Deployment:
    """Despliegue de modelo"""
    id: str
    model_id: str
    name: str
    type: DeploymentType
    status: str
    endpoint_url: Optional[str]
    replicas: int
    resources: Dict[str, Any]
    created_at: float
    last_updated: float
    metadata: Dict[str, Any]


class ModelRegistry:
    """Registry de modelos ML"""
    
    def __init__(self):
        self.models: Dict[str, MLModel] = {}
        self.experiments: Dict[str, Experiment] = {}
        self.datasets: Dict[str, Dataset] = {}
        self.deployments: Dict[str, Deployment] = {}
        self._lock = threading.Lock()
        self.model_storage_path = "/tmp/ml_models"
        self._ensure_storage_path()
    
    def _ensure_storage_path(self):
        """Asegurar que el directorio de almacenamiento existe"""
        os.makedirs(self.model_storage_path, exist_ok=True)
    
    async def register_model(self, model_info: Dict[str, Any]) -> str:
        """Registrar modelo ML"""
        model_id = f"model_{uuid.uuid4().hex[:8]}"
        
        model = MLModel(
            id=model_id,
            name=model_info["name"],
            version=model_info.get("version", "1.0.0"),
            type=ModelType(model_info["type"]),
            status=ModelStatus.TRAINING,
            framework=model_info["framework"],
            algorithm=model_info["algorithm"],
            hyperparameters=model_info.get("hyperparameters", {}),
            metrics={},
            training_data_size=0,
            validation_data_size=0,
            created_at=time.time(),
            last_updated=time.time(),
            model_path="",
            metadata=model_info.get("metadata", {})
        )
        
        async with self._lock:
            self.models[model_id] = model
        
        logger.info(f"ML model registered: {model_id} ({model.name})")
        return model_id
    
    async def create_experiment(self, experiment_info: Dict[str, Any]) -> str:
        """Crear experimento ML"""
        experiment_id = f"exp_{uuid.uuid4().hex[:8]}"
        
        experiment = Experiment(
            id=experiment_id,
            name=experiment_info["name"],
            description=experiment_info.get("description", ""),
            status=ExperimentStatus.RUNNING,
            model_type=ModelType(experiment_info["model_type"]),
            hyperparameters=experiment_info.get("hyperparameters", {}),
            metrics={},
            created_at=time.time(),
            started_at=time.time(),
            completed_at=None,
            model_id=None,
            metadata=experiment_info.get("metadata", {})
        )
        
        async with self._lock:
            self.experiments[experiment_id] = experiment
        
        logger.info(f"ML experiment created: {experiment_id} ({experiment.name})")
        return experiment_id
    
    async def register_dataset(self, dataset_info: Dict[str, Any]) -> str:
        """Registrar dataset"""
        dataset_id = f"dataset_{uuid.uuid4().hex[:8]}"
        
        dataset = Dataset(
            id=dataset_id,
            name=dataset_info["name"],
            description=dataset_info.get("description", ""),
            size=dataset_info.get("size", 0),
            features=dataset_info.get("features", []),
            target_column=dataset_info.get("target_column"),
            data_types=dataset_info.get("data_types", {}),
            created_at=time.time(),
            last_updated=time.time(),
            file_path=dataset_info.get("file_path", ""),
            metadata=dataset_info.get("metadata", {})
        )
        
        async with self._lock:
            self.datasets[dataset_id] = dataset
        
        logger.info(f"Dataset registered: {dataset_id} ({dataset.name})")
        return dataset_id
    
    async def create_deployment(self, deployment_info: Dict[str, Any]) -> str:
        """Crear despliegue de modelo"""
        deployment_id = f"deploy_{uuid.uuid4().hex[:8]}"
        
        deployment = Deployment(
            id=deployment_id,
            model_id=deployment_info["model_id"],
            name=deployment_info["name"],
            type=DeploymentType(deployment_info["type"]),
            status="creating",
            endpoint_url=None,
            replicas=deployment_info.get("replicas", 1),
            resources=deployment_info.get("resources", {}),
            created_at=time.time(),
            last_updated=time.time(),
            metadata=deployment_info.get("metadata", {})
        )
        
        async with self._lock:
            self.deployments[deployment_id] = deployment
        
        logger.info(f"Model deployment created: {deployment_id} ({deployment.name})")
        return deployment_id


class ModelTrainer:
    """Entrenador de modelos ML"""
    
    def __init__(self):
        self.training_jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
    
    async def train_model(self, model_id: str, dataset_id: str, 
                         hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        """Entrenar modelo ML"""
        try:
            # Simular entrenamiento
            training_job_id = f"job_{uuid.uuid4().hex[:8]}"
            
            async with self._lock:
                self.training_jobs[training_job_id] = {
                    "model_id": model_id,
                    "dataset_id": dataset_id,
                    "status": "training",
                    "progress": 0,
                    "started_at": time.time(),
                    "hyperparameters": hyperparameters
                }
            
            # Simular progreso de entrenamiento
            for progress in range(0, 101, 10):
                await asyncio.sleep(0.1)  # Simular tiempo de entrenamiento
                
                async with self._lock:
                    if training_job_id in self.training_jobs:
                        self.training_jobs[training_job_id]["progress"] = progress
            
            # Generar métricas simuladas
            metrics = {
                "accuracy": np.random.uniform(0.7, 0.95),
                "precision": np.random.uniform(0.7, 0.95),
                "recall": np.random.uniform(0.7, 0.95),
                "f1_score": np.random.uniform(0.7, 0.95),
                "training_time": time.time() - self.training_jobs[training_job_id]["started_at"]
            }
            
            async with self._lock:
                self.training_jobs[training_job_id]["status"] = "completed"
                self.training_jobs[training_job_id]["metrics"] = metrics
            
            return {
                "training_job_id": training_job_id,
                "model_id": model_id,
                "status": "completed",
                "metrics": metrics,
                "training_time": metrics["training_time"]
            }
            
        except Exception as e:
            logger.error(f"Error training model {model_id}: {e}")
            raise
    
    async def get_training_status(self, training_job_id: str) -> Optional[Dict[str, Any]]:
        """Obtener estado de entrenamiento"""
        async with self._lock:
            return self.training_jobs.get(training_job_id)


class ModelValidator:
    """Validador de modelos ML"""
    
    def __init__(self):
        self.validation_jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
    
    async def validate_model(self, model_id: str, validation_dataset_id: str) -> Dict[str, Any]:
        """Validar modelo ML"""
        try:
            validation_job_id = f"val_{uuid.uuid4().hex[:8]}"
            
            async with self._lock:
                self.validation_jobs[validation_job_id] = {
                    "model_id": model_id,
                    "validation_dataset_id": validation_dataset_id,
                    "status": "validating",
                    "progress": 0,
                    "started_at": time.time()
                }
            
            # Simular validación
            for progress in range(0, 101, 20):
                await asyncio.sleep(0.05)
                
                async with self._lock:
                    if validation_job_id in self.validation_jobs:
                        self.validation_jobs[validation_job_id]["progress"] = progress
            
            # Generar métricas de validación simuladas
            validation_metrics = {
                "validation_accuracy": np.random.uniform(0.6, 0.9),
                "validation_precision": np.random.uniform(0.6, 0.9),
                "validation_recall": np.random.uniform(0.6, 0.9),
                "validation_f1_score": np.random.uniform(0.6, 0.9),
                "validation_time": time.time() - self.validation_jobs[validation_job_id]["started_at"]
            }
            
            async with self._lock:
                self.validation_jobs[validation_job_id]["status"] = "completed"
                self.validation_jobs[validation_job_id]["metrics"] = validation_metrics
            
            return {
                "validation_job_id": validation_job_id,
                "model_id": model_id,
                "status": "completed",
                "metrics": validation_metrics,
                "validation_time": validation_metrics["validation_time"]
            }
            
        except Exception as e:
            logger.error(f"Error validating model {model_id}: {e}")
            raise
    
    async def get_validation_status(self, validation_job_id: str) -> Optional[Dict[str, Any]]:
        """Obtener estado de validación"""
        async with self._lock:
            return self.validation_jobs.get(validation_job_id)


class ModelDeployer:
    """Desplegador de modelos ML"""
    
    def __init__(self):
        self.deployment_jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
    
    async def deploy_model(self, model_id: str, deployment_config: Dict[str, Any]) -> Dict[str, Any]:
        """Desplegar modelo ML"""
        try:
            deployment_job_id = f"deploy_{uuid.uuid4().hex[:8]}"
            
            async with self._lock:
                self.deployment_jobs[deployment_job_id] = {
                    "model_id": model_id,
                    "deployment_config": deployment_config,
                    "status": "deploying",
                    "progress": 0,
                    "started_at": time.time()
                }
            
            # Simular despliegue
            for progress in range(0, 101, 25):
                await asyncio.sleep(0.1)
                
                async with self._lock:
                    if deployment_job_id in self.deployment_jobs:
                        self.deployment_jobs[deployment_job_id]["progress"] = progress
            
            # Generar endpoint URL simulado
            endpoint_url = f"https://api.example.com/models/{model_id}/predict"
            
            async with self._lock:
                self.deployment_jobs[deployment_job_id]["status"] = "deployed"
                self.deployment_jobs[deployment_job_id]["endpoint_url"] = endpoint_url
            
            return {
                "deployment_job_id": deployment_job_id,
                "model_id": model_id,
                "status": "deployed",
                "endpoint_url": endpoint_url,
                "deployment_time": time.time() - self.deployment_jobs[deployment_job_id]["started_at"]
            }
            
        except Exception as e:
            logger.error(f"Error deploying model {model_id}: {e}")
            raise
    
    async def get_deployment_status(self, deployment_job_id: str) -> Optional[Dict[str, Any]]:
        """Obtener estado de despliegue"""
        async with self._lock:
            return self.deployment_jobs.get(deployment_job_id)


class ModelMonitor:
    """Monitor de modelos ML"""
    
    def __init__(self):
        self.model_metrics: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.alerts: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._lock = threading.Lock()
    
    async def collect_metrics(self, model_id: str, metrics: Dict[str, Any]):
        """Recolectar métricas del modelo"""
        async with self._lock:
            self.model_metrics[model_id].append({
                "timestamp": time.time(),
                "metrics": metrics
            })
            
            # Mantener solo las últimas 1000 métricas
            if len(self.model_metrics[model_id]) > 1000:
                self.model_metrics[model_id] = self.model_metrics[model_id][-1000:]
    
    async def check_model_drift(self, model_id: str) -> Dict[str, Any]:
        """Verificar drift del modelo"""
        try:
            if model_id not in self.model_metrics or len(self.model_metrics[model_id]) < 10:
                return {"drift_detected": False, "reason": "insufficient_data"}
            
            # Obtener métricas recientes y antiguas
            recent_metrics = self.model_metrics[model_id][-5:]
            older_metrics = self.model_metrics[model_id][-10:-5]
            
            if not older_metrics:
                return {"drift_detected": False, "reason": "insufficient_data"}
            
            # Calcular drift simple basado en accuracy
            recent_accuracy = np.mean([m["metrics"].get("accuracy", 0) for m in recent_metrics])
            older_accuracy = np.mean([m["metrics"].get("accuracy", 0) for m in older_metrics])
            
            drift_threshold = 0.1  # 10% de cambio
            drift_detected = abs(recent_accuracy - older_accuracy) > drift_threshold
            
            return {
                "drift_detected": drift_detected,
                "recent_accuracy": recent_accuracy,
                "older_accuracy": older_accuracy,
                "drift_amount": abs(recent_accuracy - older_accuracy),
                "threshold": drift_threshold
            }
            
        except Exception as e:
            logger.error(f"Error checking model drift for {model_id}: {e}")
            return {"drift_detected": False, "reason": "error", "error": str(e)}
    
    async def create_alert(self, model_id: str, alert_type: str, message: str, severity: str = "medium"):
        """Crear alerta"""
        alert = {
            "id": f"alert_{uuid.uuid4().hex[:8]}",
            "model_id": model_id,
            "type": alert_type,
            "message": message,
            "severity": severity,
            "timestamp": time.time(),
            "is_resolved": False
        }
        
        async with self._lock:
            self.alerts[model_id].append(alert)
    
    async def get_model_metrics(self, model_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Obtener métricas del modelo"""
        async with self._lock:
            if model_id not in self.model_metrics:
                return []
            
            return self.model_metrics[model_id][-limit:]
    
    async def get_alerts(self, model_id: str, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """Obtener alertas del modelo"""
        async with self._lock:
            if model_id not in self.alerts:
                return []
            
            alerts = self.alerts[model_id]
            if severity:
                alerts = [alert for alert in alerts if alert["severity"] == severity]
            
            return alerts


class MLOpsSystem:
    """Sistema principal ML Ops"""
    
    def __init__(self):
        self.registry = ModelRegistry()
        self.trainer = ModelTrainer()
        self.validator = ModelValidator()
        self.deployer = ModelDeployer()
        self.monitor = ModelMonitor()
        self.is_running = False
        self._monitoring_task = None
        self._cleanup_task = None
        self._lock = threading.Lock()
    
    async def start(self):
        """Iniciar sistema ML Ops"""
        try:
            self.is_running = True
            
            # Iniciar tareas de monitoreo y limpieza
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            logger.info("ML Ops system started")
            
        except Exception as e:
            logger.error(f"Error starting ML Ops system: {e}")
            raise
    
    async def stop(self):
        """Detener sistema ML Ops"""
        try:
            self.is_running = False
            
            # Detener tareas
            if self._monitoring_task:
                self._monitoring_task.cancel()
                try:
                    await self._monitoring_task
                except asyncio.CancelledError:
                    pass
            
            if self._cleanup_task:
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("ML Ops system stopped")
            
        except Exception as e:
            logger.error(f"Error stopping ML Ops system: {e}")
    
    async def _monitoring_loop(self):
        """Loop de monitoreo"""
        while self.is_running:
            try:
                # Monitorear modelos desplegados
                await self._monitor_deployed_models()
                
                # Verificar drift de modelos
                await self._check_model_drift()
                
                await asyncio.sleep(60)  # Monitorear cada minuto
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_loop(self):
        """Loop de limpieza"""
        while self.is_running:
            try:
                # Limpiar datos antiguos
                await self._cleanup_old_data()
                
                await asyncio.sleep(3600)  # Limpiar cada hora
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(3600)
    
    async def _monitor_deployed_models(self):
        """Monitorear modelos desplegados"""
        # Implementar monitoreo de modelos desplegados
        pass
    
    async def _check_model_drift(self):
        """Verificar drift de modelos"""
        for model_id in self.registry.models:
            drift_result = await self.monitor.check_model_drift(model_id)
            if drift_result.get("drift_detected"):
                await self.monitor.create_alert(
                    model_id, "model_drift", 
                    f"Model drift detected: {drift_result['drift_amount']:.3f}", 
                    "high"
                )
    
    async def _cleanup_old_data(self):
        """Limpiar datos antiguos"""
        # Implementar limpieza de datos antiguos
        pass
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del sistema"""
        return {
            "is_running": self.is_running,
            "models": {
                "total": len(self.registry.models),
                "by_status": {
                    status.value: sum(1 for m in self.registry.models.values() if m.status == status)
                    for status in ModelStatus
                },
                "by_type": {
                    model_type.value: sum(1 for m in self.registry.models.values() if m.type == model_type)
                    for model_type in ModelType
                }
            },
            "experiments": {
                "total": len(self.registry.experiments),
                "by_status": {
                    status.value: sum(1 for e in self.registry.experiments.values() if e.status == status)
                    for status in ExperimentStatus
                }
            },
            "datasets": len(self.registry.datasets),
            "deployments": {
                "total": len(self.registry.deployments),
                "active": sum(1 for d in self.registry.deployments.values() if d.status == "deployed")
            },
            "training_jobs": len(self.trainer.training_jobs),
            "validation_jobs": len(self.validator.validation_jobs),
            "deployment_jobs": len(self.deployer.deployment_jobs)
        }


# Instancia global del sistema ML Ops
ml_ops_system = MLOpsSystem()


# Router para endpoints ML Ops
ml_ops_router = APIRouter()


@ml_ops_router.post("/ml-ops/models/register")
async def register_ml_model_endpoint(model_data: dict):
    """Registrar modelo ML"""
    try:
        model_id = await ml_ops_system.registry.register_model(model_data)
        
        return {
            "message": "ML model registered successfully",
            "model_id": model_id,
            "name": model_data["name"],
            "type": model_data["type"],
            "framework": model_data["framework"]
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid model type: {e}")
    except Exception as e:
        logger.error(f"Error registering ML model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to register ML model: {str(e)}")


@ml_ops_router.get("/ml-ops/models")
async def get_ml_models_endpoint():
    """Obtener modelos ML"""
    try:
        models = ml_ops_system.registry.models
        return {
            "models": [
                {
                    "id": model.id,
                    "name": model.name,
                    "version": model.version,
                    "type": model.type.value,
                    "status": model.status.value,
                    "framework": model.framework,
                    "algorithm": model.algorithm,
                    "metrics": model.metrics,
                    "created_at": model.created_at,
                    "last_updated": model.last_updated
                }
                for model in models.values()
            ]
        }
    except Exception as e:
        logger.error(f"Error getting ML models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get ML models: {str(e)}")


@ml_ops_router.get("/ml-ops/models/{model_id}")
async def get_ml_model_endpoint(model_id: str):
    """Obtener modelo ML específico"""
    try:
        if model_id not in ml_ops_system.registry.models:
            raise HTTPException(status_code=404, detail="ML model not found")
        
        model = ml_ops_system.registry.models[model_id]
        
        return {
            "id": model.id,
            "name": model.name,
            "version": model.version,
            "type": model.type.value,
            "status": model.status.value,
            "framework": model.framework,
            "algorithm": model.algorithm,
            "hyperparameters": model.hyperparameters,
            "metrics": model.metrics,
            "training_data_size": model.training_data_size,
            "validation_data_size": model.validation_data_size,
            "created_at": model.created_at,
            "last_updated": model.last_updated,
            "model_path": model.model_path,
            "metadata": model.metadata
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting ML model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get ML model: {str(e)}")


@ml_ops_router.post("/ml-ops/experiments")
async def create_ml_experiment_endpoint(experiment_data: dict):
    """Crear experimento ML"""
    try:
        experiment_id = await ml_ops_system.registry.create_experiment(experiment_data)
        
        return {
            "message": "ML experiment created successfully",
            "experiment_id": experiment_id,
            "name": experiment_data["name"],
            "model_type": experiment_data["model_type"]
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid model type: {e}")
    except Exception as e:
        logger.error(f"Error creating ML experiment: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create ML experiment: {str(e)}")


@ml_ops_router.get("/ml-ops/experiments")
async def get_ml_experiments_endpoint():
    """Obtener experimentos ML"""
    try:
        experiments = ml_ops_system.registry.experiments
        return {
            "experiments": [
                {
                    "id": exp.id,
                    "name": exp.name,
                    "description": exp.description,
                    "status": exp.status.value,
                    "model_type": exp.model_type.value,
                    "metrics": exp.metrics,
                    "created_at": exp.created_at,
                    "started_at": exp.started_at,
                    "completed_at": exp.completed_at,
                    "model_id": exp.model_id
                }
                for exp in experiments.values()
            ]
        }
    except Exception as e:
        logger.error(f"Error getting ML experiments: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get ML experiments: {str(e)}")


@ml_ops_router.post("/ml-ops/datasets/register")
async def register_ml_dataset_endpoint(dataset_data: dict):
    """Registrar dataset ML"""
    try:
        dataset_id = await ml_ops_system.registry.register_dataset(dataset_data)
        
        return {
            "message": "ML dataset registered successfully",
            "dataset_id": dataset_id,
            "name": dataset_data["name"],
            "size": dataset_data.get("size", 0)
        }
        
    except Exception as e:
        logger.error(f"Error registering ML dataset: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to register ML dataset: {str(e)}")


@ml_ops_router.get("/ml-ops/datasets")
async def get_ml_datasets_endpoint():
    """Obtener datasets ML"""
    try:
        datasets = ml_ops_system.registry.datasets
        return {
            "datasets": [
                {
                    "id": dataset.id,
                    "name": dataset.name,
                    "description": dataset.description,
                    "size": dataset.size,
                    "features": dataset.features,
                    "target_column": dataset.target_column,
                    "created_at": dataset.created_at,
                    "last_updated": dataset.last_updated
                }
                for dataset in datasets.values()
            ]
        }
    except Exception as e:
        logger.error(f"Error getting ML datasets: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get ML datasets: {str(e)}")


@ml_ops_router.post("/ml-ops/models/{model_id}/train")
async def train_ml_model_endpoint(model_id: str, training_data: dict):
    """Entrenar modelo ML"""
    try:
        if model_id not in ml_ops_system.registry.models:
            raise HTTPException(status_code=404, detail="ML model not found")
        
        dataset_id = training_data["dataset_id"]
        hyperparameters = training_data.get("hyperparameters", {})
        
        result = await ml_ops_system.trainer.train_model(model_id, dataset_id, hyperparameters)
        
        return {
            "message": "ML model training started successfully",
            "result": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error training ML model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to train ML model: {str(e)}")


@ml_ops_router.get("/ml-ops/training/{training_job_id}")
async def get_training_status_endpoint(training_job_id: str):
    """Obtener estado de entrenamiento"""
    try:
        status = await ml_ops_system.trainer.get_training_status(training_job_id)
        
        if status:
            return status
        else:
            raise HTTPException(status_code=404, detail="Training job not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting training status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get training status: {str(e)}")


@ml_ops_router.post("/ml-ops/models/{model_id}/validate")
async def validate_ml_model_endpoint(model_id: str, validation_data: dict):
    """Validar modelo ML"""
    try:
        if model_id not in ml_ops_system.registry.models:
            raise HTTPException(status_code=404, detail="ML model not found")
        
        validation_dataset_id = validation_data["validation_dataset_id"]
        
        result = await ml_ops_system.validator.validate_model(model_id, validation_dataset_id)
        
        return {
            "message": "ML model validation started successfully",
            "result": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating ML model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to validate ML model: {str(e)}")


@ml_ops_router.post("/ml-ops/deployments")
async def create_ml_deployment_endpoint(deployment_data: dict):
    """Crear despliegue de modelo ML"""
    try:
        model_id = deployment_data["model_id"]
        if model_id not in ml_ops_system.registry.models:
            raise HTTPException(status_code=404, detail="ML model not found")
        
        deployment_id = await ml_ops_system.registry.create_deployment(deployment_data)
        
        # Iniciar despliegue
        deployment_config = {
            "type": deployment_data["type"],
            "replicas": deployment_data.get("replicas", 1),
            "resources": deployment_data.get("resources", {})
        }
        
        result = await ml_ops_system.deployer.deploy_model(model_id, deployment_config)
        
        return {
            "message": "ML model deployment started successfully",
            "deployment_id": deployment_id,
            "result": result
        }
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid deployment type: {e}")
    except Exception as e:
        logger.error(f"Error creating ML deployment: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create ML deployment: {str(e)}")


@ml_ops_router.get("/ml-ops/deployments")
async def get_ml_deployments_endpoint():
    """Obtener despliegues ML"""
    try:
        deployments = ml_ops_system.registry.deployments
        return {
            "deployments": [
                {
                    "id": deploy.id,
                    "model_id": deploy.model_id,
                    "name": deploy.name,
                    "type": deploy.type.value,
                    "status": deploy.status,
                    "endpoint_url": deploy.endpoint_url,
                    "replicas": deploy.replicas,
                    "created_at": deploy.created_at,
                    "last_updated": deploy.last_updated
                }
                for deploy in deployments.values()
            ]
        }
    except Exception as e:
        logger.error(f"Error getting ML deployments: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get ML deployments: {str(e)}")


@ml_ops_router.get("/ml-ops/models/{model_id}/metrics")
async def get_model_metrics_endpoint(model_id: str, limit: int = 100):
    """Obtener métricas del modelo"""
    try:
        if model_id not in ml_ops_system.registry.models:
            raise HTTPException(status_code=404, detail="ML model not found")
        
        metrics = await ml_ops_system.monitor.get_model_metrics(model_id, limit)
        
        return {
            "model_id": model_id,
            "metrics": metrics,
            "count": len(metrics)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model metrics: {str(e)}")


@ml_ops_router.get("/ml-ops/models/{model_id}/drift")
async def check_model_drift_endpoint(model_id: str):
    """Verificar drift del modelo"""
    try:
        if model_id not in ml_ops_system.registry.models:
            raise HTTPException(status_code=404, detail="ML model not found")
        
        drift_result = await ml_ops_system.monitor.check_model_drift(model_id)
        
        return {
            "model_id": model_id,
            "drift_result": drift_result
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking model drift: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to check model drift: {str(e)}")


@ml_ops_router.get("/ml-ops/models/{model_id}/alerts")
async def get_model_alerts_endpoint(model_id: str, severity: Optional[str] = None):
    """Obtener alertas del modelo"""
    try:
        if model_id not in ml_ops_system.registry.models:
            raise HTTPException(status_code=404, detail="ML model not found")
        
        alerts = await ml_ops_system.monitor.get_alerts(model_id, severity)
        
        return {
            "model_id": model_id,
            "alerts": alerts,
            "count": len(alerts)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model alerts: {str(e)}")


@ml_ops_router.get("/ml-ops/stats")
async def get_ml_ops_stats_endpoint():
    """Obtener estadísticas del sistema ML Ops"""
    try:
        stats = await ml_ops_system.get_system_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting ML Ops stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get ML Ops stats: {str(e)}")


# Funciones de utilidad para integración
async def start_ml_ops_system():
    """Iniciar sistema ML Ops"""
    await ml_ops_system.start()


async def stop_ml_ops_system():
    """Detener sistema ML Ops"""
    await ml_ops_system.stop()


async def register_ml_model(model_info: Dict[str, Any]) -> str:
    """Registrar modelo ML"""
    return await ml_ops_system.registry.register_model(model_info)


async def create_ml_experiment(experiment_info: Dict[str, Any]) -> str:
    """Crear experimento ML"""
    return await ml_ops_system.registry.create_experiment(experiment_info)


async def train_ml_model(model_id: str, dataset_id: str, hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
    """Entrenar modelo ML"""
    return await ml_ops_system.trainer.train_model(model_id, dataset_id, hyperparameters)


async def get_ml_ops_system_stats() -> Dict[str, Any]:
    """Obtener estadísticas del sistema ML Ops"""
    return await ml_ops_system.get_system_stats()


logger.info("ML Ops system module loaded successfully")

