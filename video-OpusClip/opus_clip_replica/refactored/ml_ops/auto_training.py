"""
ML Ops and Auto-Training for Opus Clip

Advanced ML Ops capabilities with:
- Automated model training
- Hyperparameter optimization
- Model versioning and management
- A/B testing for models
- Continuous learning
- Model performance monitoring
- Automated deployment
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Union, Tuple
import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import structlog
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.pytorch
from pathlib import Path
import joblib
import pickle
import shutil
from collections import defaultdict
import threading
import queue

logger = structlog.get_logger("ml_ops_auto_training")

class ModelType(Enum):
    """Model type enumeration."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    DEEP_LEARNING = "deep_learning"
    TRANSFORMER = "transformer"
    CUSTOM = "custom"

class TrainingStatus(Enum):
    """Training status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ModelStatus(Enum):
    """Model status enumeration."""
    TRAINING = "training"
    STAGING = "staging"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    FAILED = "failed"

@dataclass
class TrainingJob:
    """Training job information."""
    job_id: str
    model_type: ModelType
    dataset_path: str
    target_column: str
    features: List[str]
    hyperparameters: Dict[str, Any]
    status: TrainingStatus = TrainingStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    model_path: Optional[str] = None
    error_message: Optional[str] = None
    version: str = "1.0.0"

@dataclass
class ModelVersion:
    """Model version information."""
    model_id: str
    version: str
    model_type: ModelType
    model_path: str
    metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    created_at: datetime
    status: ModelStatus = ModelStatus.STAGING
    deployment_info: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    name: str
    description: str
    model_type: ModelType
    dataset_path: str
    target_column: str
    features: List[str]
    hyperparameter_space: Dict[str, Any]
    optimization_metric: str = "accuracy"
    max_trials: int = 100
    timeout: int = 3600  # seconds
    early_stopping_patience: int = 10

class AutoTrainingManager:
    """
    Automated model training and ML Ops manager.
    
    Features:
    - Automated model training
    - Hyperparameter optimization
    - Model versioning
    - A/B testing
    - Continuous learning
    - Performance monitoring
    """
    
    def __init__(self, mlflow_uri: str = "http://localhost:5000"):
        self.logger = structlog.get_logger("auto_training_manager")
        self.mlflow_uri = mlflow_uri
        self.training_jobs: Dict[str, TrainingJob] = {}
        self.model_versions: Dict[str, List[ModelVersion]] = defaultdict(list)
        self.experiments: Dict[str, ExperimentConfig] = {}
        self.training_queue = queue.Queue()
        self.training_thread = None
        self.is_running = False
        
        # Initialize MLflow
        mlflow.set_tracking_uri(mlflow_uri)
        
        # Model performance tracking
        self.performance_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
    async def initialize(self) -> bool:
        """Initialize auto-training manager."""
        try:
            # Start training thread
            self.training_thread = threading.Thread(target=self._training_worker)
            self.training_thread.daemon = True
            self.training_thread.start()
            
            self.is_running = True
            self.logger.info("Auto-training manager initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Auto-training manager initialization failed: {e}")
            return False
    
    def _training_worker(self):
        """Background training worker thread."""
        while self.is_running:
            try:
                if not self.training_queue.empty():
                    job_id = self.training_queue.get()
                    asyncio.run(self._execute_training_job(job_id))
                else:
                    time.sleep(1)
                    
            except Exception as e:
                self.logger.error(f"Training worker error: {e}")
                time.sleep(5)
    
    async def create_experiment(self, config: ExperimentConfig) -> str:
        """Create a new ML experiment."""
        try:
            experiment_id = str(uuid.uuid4())
            self.experiments[experiment_id] = config
            
            # Create MLflow experiment
            mlflow.create_experiment(
                name=config.name,
                tags={
                    "description": config.description,
                    "model_type": config.model_type.value,
                    "target_column": config.target_column
                }
            )
            
            self.logger.info(f"Created experiment: {config.name} ({experiment_id})")
            return experiment_id
            
        except Exception as e:
            self.logger.error(f"Experiment creation failed: {e}")
            raise
    
    async def start_hyperparameter_optimization(self, experiment_id: str) -> str:
        """Start hyperparameter optimization for an experiment."""
        try:
            if experiment_id not in self.experiments:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            config = self.experiments[experiment_id]
            
            # Create Optuna study
            study = optuna.create_study(
                direction="maximize" if config.optimization_metric in ["accuracy", "f1_score"] else "minimize",
                study_name=f"{config.name}_optimization"
            )
            
            # Define objective function
            def objective(trial):
                return asyncio.run(self._optimize_hyperparameters(trial, config))
            
            # Run optimization
            study.optimize(
                objective,
                n_trials=config.max_trials,
                timeout=config.timeout
            )
            
            # Get best parameters
            best_params = study.best_params
            best_value = study.best_value
            
            self.logger.info(f"Hyperparameter optimization completed. Best value: {best_value}")
            
            return json.dumps({
                "best_parameters": best_params,
                "best_value": best_value,
                "n_trials": len(study.trials)
            })
            
        except Exception as e:
            self.logger.error(f"Hyperparameter optimization failed: {e}")
            raise
    
    async def _optimize_hyperparameters(self, trial, config: ExperimentConfig) -> float:
        """Optimize hyperparameters for a single trial."""
        try:
            # Sample hyperparameters
            hyperparameters = {}
            for param_name, param_config in config.hyperparameter_space.items():
                if param_config["type"] == "float":
                    hyperparameters[param_name] = trial.suggest_float(
                        param_name, 
                        param_config["low"], 
                        param_config["high"]
                    )
                elif param_config["type"] == "int":
                    hyperparameters[param_name] = trial.suggest_int(
                        param_name, 
                        param_config["low"], 
                        param_config["high"]
                    )
                elif param_config["type"] == "categorical":
                    hyperparameters[param_name] = trial.suggest_categorical(
                        param_name, 
                        param_config["choices"]
                    )
            
            # Create training job
            job = TrainingJob(
                job_id=str(uuid.uuid4()),
                model_type=config.model_type,
                dataset_path=config.dataset_path,
                target_column=config.target_column,
                features=config.features,
                hyperparameters=hyperparameters
            )
            
            # Train model
            await self._train_model(job)
            
            # Return metric value
            return job.metrics.get(config.optimization_metric, 0.0)
            
        except Exception as e:
            self.logger.error(f"Hyperparameter optimization trial failed: {e}")
            return 0.0
    
    async def submit_training_job(self, job: TrainingJob) -> str:
        """Submit a training job."""
        try:
            self.training_jobs[job.job_id] = job
            self.training_queue.put(job.job_id)
            
            self.logger.info(f"Submitted training job: {job.job_id}")
            return job.job_id
            
        except Exception as e:
            self.logger.error(f"Training job submission failed: {e}")
            raise
    
    async def _execute_training_job(self, job_id: str):
        """Execute a training job."""
        try:
            job = self.training_jobs[job_id]
            job.status = TrainingStatus.RUNNING
            job.started_at = datetime.now()
            
            self.logger.info(f"Starting training job: {job_id}")
            
            # Train model
            await self._train_model(job)
            
            # Update job status
            job.status = TrainingStatus.COMPLETED
            job.completed_at = datetime.now()
            
            # Create model version
            model_version = ModelVersion(
                model_id=f"model_{job_id}",
                version=job.version,
                model_type=job.model_type,
                model_path=job.model_path,
                metrics=job.metrics,
                hyperparameters=job.hyperparameters,
                created_at=datetime.now()
            )
            
            self.model_versions[model_version.model_id].append(model_version)
            
            self.logger.info(f"Training job completed: {job_id}")
            
        except Exception as e:
            self.logger.error(f"Training job execution failed: {e}")
            job = self.training_jobs[job_id]
            job.status = TrainingStatus.FAILED
            job.error_message = str(e)
    
    async def _train_model(self, job: TrainingJob):
        """Train a model based on job configuration."""
        try:
            # Load dataset
            dataset = await self._load_dataset(job.dataset_path)
            
            # Prepare data
            X, y = await self._prepare_data(dataset, job.features, job.target_column)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train model based on type
            if job.model_type == ModelType.CLASSIFICATION:
                model = await self._train_classification_model(X_train, y_train, job.hyperparameters)
            elif job.model_type == ModelType.REGRESSION:
                model = await self._train_regression_model(X_train, y_train, job.hyperparameters)
            elif job.model_type == ModelType.DEEP_LEARNING:
                model = await self._train_deep_learning_model(X_train, y_train, job.hyperparameters)
            else:
                raise ValueError(f"Unsupported model type: {job.model_type}")
            
            # Evaluate model
            metrics = await self._evaluate_model(model, X_test, y_test, job.model_type)
            job.metrics = metrics
            
            # Save model
            model_path = await self._save_model(model, job.job_id)
            job.model_path = model_path
            
            # Log to MLflow
            await self._log_to_mlflow(job, metrics)
            
        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            raise
    
    async def _load_dataset(self, dataset_path: str) -> pd.DataFrame:
        """Load dataset from file."""
        try:
            if dataset_path.endswith('.csv'):
                return pd.read_csv(dataset_path)
            elif dataset_path.endswith('.json'):
                return pd.read_json(dataset_path)
            elif dataset_path.endswith('.parquet'):
                return pd.read_parquet(dataset_path)
            else:
                raise ValueError(f"Unsupported dataset format: {dataset_path}")
                
        except Exception as e:
            self.logger.error(f"Dataset loading failed: {e}")
            raise
    
    async def _prepare_data(self, dataset: pd.DataFrame, features: List[str], target_column: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training."""
        try:
            # Select features
            X = dataset[features].values
            
            # Handle missing values
            X = np.nan_to_num(X)
            
            # Get target
            y = dataset[target_column].values
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Data preparation failed: {e}")
            raise
    
    async def _train_classification_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                                        hyperparameters: Dict[str, Any]) -> Any:
        """Train classification model."""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.svm import SVC
            
            model_type = hyperparameters.get("model_type", "random_forest")
            
            if model_type == "random_forest":
                model = RandomForestClassifier(
                    n_estimators=hyperparameters.get("n_estimators", 100),
                    max_depth=hyperparameters.get("max_depth", None),
                    random_state=42
                )
            elif model_type == "logistic_regression":
                model = LogisticRegression(
                    C=hyperparameters.get("C", 1.0),
                    random_state=42
                )
            elif model_type == "svm":
                model = SVC(
                    C=hyperparameters.get("C", 1.0),
                    kernel=hyperparameters.get("kernel", "rbf"),
                    random_state=42
                )
            else:
                raise ValueError(f"Unsupported classification model: {model_type}")
            
            model.fit(X_train, y_train)
            return model
            
        except Exception as e:
            self.logger.error(f"Classification model training failed: {e}")
            raise
    
    async def _train_regression_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                                    hyperparameters: Dict[str, Any]) -> Any:
        """Train regression model."""
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.linear_model import LinearRegression
            from sklearn.svm import SVR
            
            model_type = hyperparameters.get("model_type", "random_forest")
            
            if model_type == "random_forest":
                model = RandomForestRegressor(
                    n_estimators=hyperparameters.get("n_estimators", 100),
                    max_depth=hyperparameters.get("max_depth", None),
                    random_state=42
                )
            elif model_type == "linear_regression":
                model = LinearRegression()
            elif model_type == "svr":
                model = SVR(
                    C=hyperparameters.get("C", 1.0),
                    kernel=hyperparameters.get("kernel", "rbf")
                )
            else:
                raise ValueError(f"Unsupported regression model: {model_type}")
            
            model.fit(X_train, y_train)
            return model
            
        except Exception as e:
            self.logger.error(f"Regression model training failed: {e}")
            raise
    
    async def _train_deep_learning_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                                       hyperparameters: Dict[str, Any]) -> Any:
        """Train deep learning model."""
        try:
            # Convert to PyTorch tensors
            X_tensor = torch.FloatTensor(X_train)
            y_tensor = torch.LongTensor(y_train) if hyperparameters.get("task") == "classification" else torch.FloatTensor(y_train)
            
            # Create dataset
            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=hyperparameters.get("batch_size", 32), shuffle=True)
            
            # Create model
            input_size = X_train.shape[1]
            hidden_size = hyperparameters.get("hidden_size", 64)
            output_size = len(np.unique(y_train)) if hyperparameters.get("task") == "classification" else 1
            
            model = SimpleNeuralNetwork(input_size, hidden_size, output_size)
            
            # Training setup
            criterion = nn.CrossEntropyLoss() if hyperparameters.get("task") == "classification" else nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=hyperparameters.get("learning_rate", 0.001))
            
            # Training loop
            epochs = hyperparameters.get("epochs", 100)
            for epoch in range(epochs):
                for batch_X, batch_y in dataloader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
            
            return model
            
        except Exception as e:
            self.logger.error(f"Deep learning model training failed: {e}")
            raise
    
    async def _evaluate_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray, 
                            model_type: ModelType) -> Dict[str, float]:
        """Evaluate model performance."""
        try:
            if model_type == ModelType.CLASSIFICATION:
                y_pred = model.predict(X_test)
                metrics = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred, average="weighted"),
                    "recall": recall_score(y_test, y_pred, average="weighted"),
                    "f1_score": f1_score(y_test, y_pred, average="weighted")
                }
            elif model_type == ModelType.REGRESSION:
                y_pred = model.predict(X_test)
                mse = np.mean((y_test - y_pred) ** 2)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(y_test - y_pred))
                r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
                metrics = {
                    "mse": mse,
                    "rmse": rmse,
                    "mae": mae,
                    "r2": r2
                }
            else:
                metrics = {"score": 0.0}
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Model evaluation failed: {e}")
            return {"error": str(e)}
    
    async def _save_model(self, model: Any, job_id: str) -> str:
        """Save trained model."""
        try:
            model_dir = Path(f"models/{job_id}")
            model_dir.mkdir(parents=True, exist_ok=True)
            
            model_path = model_dir / "model.pkl"
            
            if hasattr(model, 'state_dict'):  # PyTorch model
                torch.save(model.state_dict(), model_path)
            else:  # Scikit-learn model
                joblib.dump(model, model_path)
            
            return str(model_path)
            
        except Exception as e:
            self.logger.error(f"Model saving failed: {e}")
            raise
    
    async def _log_to_mlflow(self, job: TrainingJob, metrics: Dict[str, float]):
        """Log training results to MLflow."""
        try:
            with mlflow.start_run():
                # Log parameters
                mlflow.log_params(job.hyperparameters)
                
                # Log metrics
                mlflow.log_metrics(metrics)
                
                # Log model
                if job.model_path:
                    mlflow.log_artifact(job.model_path)
                
                # Log tags
                mlflow.set_tags({
                    "model_type": job.model_type.value,
                    "job_id": job.job_id,
                    "version": job.version
                })
                
        except Exception as e:
            self.logger.error(f"MLflow logging failed: {e}")
    
    async def deploy_model(self, model_id: str, version: str, environment: str = "staging") -> bool:
        """Deploy model to environment."""
        try:
            if model_id not in self.model_versions:
                raise ValueError(f"Model {model_id} not found")
            
            # Find model version
            model_version = None
            for v in self.model_versions[model_id]:
                if v.version == version:
                    model_version = v
                    break
            
            if not model_version:
                raise ValueError(f"Model version {version} not found")
            
            # Update model status
            model_version.status = ModelStatus.PRODUCTION if environment == "production" else ModelStatus.STAGING
            model_version.deployment_info = {
                "environment": environment,
                "deployed_at": datetime.now().isoformat(),
                "deployment_id": str(uuid.uuid4())
            }
            
            self.logger.info(f"Deployed model {model_id} version {version} to {environment}")
            return True
            
        except Exception as e:
            self.logger.error(f"Model deployment failed: {e}")
            return False
    
    async def get_model_performance(self, model_id: str) -> Dict[str, Any]:
        """Get model performance metrics."""
        try:
            if model_id not in self.model_versions:
                return {"error": "Model not found"}
            
            versions = self.model_versions[model_id]
            latest_version = max(versions, key=lambda v: v.created_at)
            
            return {
                "model_id": model_id,
                "latest_version": latest_version.version,
                "metrics": latest_version.metrics,
                "status": latest_version.status.value,
                "created_at": latest_version.created_at.isoformat(),
                "deployment_info": latest_version.deployment_info
            }
            
        except Exception as e:
            self.logger.error(f"Performance retrieval failed: {e}")
            return {"error": str(e)}
    
    async def get_training_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get training job status."""
        try:
            if job_id not in self.training_jobs:
                return {"error": "Job not found"}
            
            job = self.training_jobs[job_id]
            return {
                "job_id": job_id,
                "status": job.status.value,
                "created_at": job.created_at.isoformat(),
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                "metrics": job.metrics,
                "error_message": job.error_message
            }
            
        except Exception as e:
            self.logger.error(f"Job status retrieval failed: {e}")
            return {"error": str(e)}
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get ML Ops system status."""
        try:
            return {
                "total_jobs": len(self.training_jobs),
                "running_jobs": len([j for j in self.training_jobs.values() if j.status == TrainingStatus.RUNNING]),
                "completed_jobs": len([j for j in self.training_jobs.values() if j.status == TrainingStatus.COMPLETED]),
                "failed_jobs": len([j for j in self.training_jobs.values() if j.status == TrainingStatus.FAILED]),
                "total_models": len(self.model_versions),
                "experiments": len(self.experiments),
                "queue_size": self.training_queue.qsize(),
                "is_running": self.is_running
            }
            
        except Exception as e:
            self.logger.error(f"System status retrieval failed: {e}")
            return {"error": str(e)}
    
    async def cleanup(self):
        """Cleanup ML Ops manager."""
        try:
            self.is_running = False
            if self.training_thread:
                self.training_thread.join()
            
            self.logger.info("ML Ops manager cleaned up")
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")

class SimpleNeuralNetwork(nn.Module):
    """Simple neural network for demonstration."""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(SimpleNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Example usage
async def main():
    """Example usage of ML Ops auto-training."""
    manager = AutoTrainingManager()
    
    # Initialize manager
    success = await manager.initialize()
    if not success:
        print("Failed to initialize ML Ops manager")
        return
    
    # Create experiment
    config = ExperimentConfig(
        name="video_classification_experiment",
        description="Classify video content types",
        model_type=ModelType.CLASSIFICATION,
        dataset_path="/path/to/dataset.csv",
        target_column="label",
        features=["feature1", "feature2", "feature3"],
        hyperparameter_space={
            "model_type": {"type": "categorical", "choices": ["random_forest", "logistic_regression", "svm"]},
            "n_estimators": {"type": "int", "low": 50, "high": 200},
            "max_depth": {"type": "int", "low": 3, "high": 20}
        }
    )
    
    experiment_id = await manager.create_experiment(config)
    print(f"Created experiment: {experiment_id}")
    
    # Start hyperparameter optimization
    result = await manager.start_hyperparameter_optimization(experiment_id)
    print(f"Optimization result: {result}")
    
    # Get system status
    status = await manager.get_system_status()
    print(f"System status: {status}")

if __name__ == "__main__":
    asyncio.run(main())


