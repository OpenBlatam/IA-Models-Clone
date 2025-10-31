"""
Machine Learning Engine - Motor de Machine Learning avanzado
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import pickle
import joblib
from pathlib import Path
import uuid
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Tipos de modelos ML."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    DEEP_LEARNING = "deep_learning"
    NLP_MODEL = "nlp_model"
    TIME_SERIES = "time_series"


class ModelStatus(Enum):
    """Estados de modelos."""
    TRAINING = "training"
    TRAINED = "trained"
    DEPLOYED = "deployed"
    FAILED = "failed"
    RETIRED = "retired"


@dataclass
class ModelMetadata:
    """Metadatos de modelo."""
    model_id: str
    name: str
    model_type: ModelType
    algorithm: str
    version: str
    created_at: datetime
    updated_at: datetime
    status: ModelStatus
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    mse: Optional[float] = None
    r2_score: Optional[float] = None
    training_samples: int = 0
    features: List[str] = field(default_factory=list)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingJob:
    """Trabajo de entrenamiento."""
    job_id: str
    model_id: str
    status: str
    progress: float
    started_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)


class MachineLearningEngine:
    """
    Motor de Machine Learning avanzado.
    """
    
    def __init__(self, models_directory: str = "models"):
        """Inicializar motor de ML."""
        self.models_directory = Path(models_directory)
        self.models_directory.mkdir(exist_ok=True)
        
        # Almacenamiento de modelos
        self.models: Dict[str, ModelMetadata] = {}
        self.trained_models: Dict[str, Any] = {}
        self.training_jobs: Dict[str, TrainingJob] = {}
        
        # Configuración
        self.max_models = 100
        self.auto_retrain_threshold = 0.1  # 10% de degradación
        self.model_retention_days = 90
        
        # Estadísticas
        self.stats = {
            "total_models": 0,
            "trained_models": 0,
            "deployed_models": 0,
            "failed_models": 0,
            "total_predictions": 0,
            "start_time": datetime.now()
        }
        
        # Cargar modelos existentes
        self._load_existing_models()
        
        logger.info("MachineLearningEngine inicializado")
    
    async def initialize(self):
        """Inicializar el motor de ML."""
        try:
            # Iniciar limpieza automática
            asyncio.create_task(self._cleanup_loop())
            
            logger.info("MachineLearningEngine inicializado exitosamente")
            
        except Exception as e:
            logger.error(f"Error al inicializar MachineLearningEngine: {e}")
            raise
    
    async def shutdown(self):
        """Cerrar el motor de ML."""
        try:
            # Guardar modelos
            await self._save_all_models()
            
            logger.info("MachineLearningEngine cerrado")
            
        except Exception as e:
            logger.error(f"Error al cerrar MachineLearningEngine: {e}")
    
    def _load_existing_models(self):
        """Cargar modelos existentes."""
        try:
            metadata_file = self.models_directory / "models_metadata.pkl"
            if metadata_file.exists():
                with open(metadata_file, 'rb') as f:
                    self.models = pickle.load(f)
                
                # Cargar modelos entrenados
                for model_id, metadata in self.models.items():
                    model_file = self.models_directory / f"{model_id}.pkl"
                    if model_file.exists() and metadata.status == ModelStatus.TRAINED:
                        try:
                            with open(model_file, 'rb') as f:
                                self.trained_models[model_id] = pickle.load(f)
                        except Exception as e:
                            logger.warning(f"No se pudo cargar modelo {model_id}: {e}")
                            metadata.status = ModelStatus.FAILED
                
                logger.info(f"Cargados {len(self.models)} modelos existentes")
                
        except Exception as e:
            logger.error(f"Error al cargar modelos existentes: {e}")
    
    async def create_model(
        self,
        name: str,
        model_type: ModelType,
        algorithm: str,
        features: List[str],
        hyperparameters: Dict[str, Any] = None
    ) -> str:
        """Crear nuevo modelo."""
        try:
            model_id = str(uuid.uuid4())
            now = datetime.now()
            
            metadata = ModelMetadata(
                model_id=model_id,
                name=name,
                model_type=model_type,
                algorithm=algorithm,
                version="1.0.0",
                created_at=now,
                updated_at=now,
                status=ModelStatus.TRAINING,
                features=features,
                hyperparameters=hyperparameters or {}
            )
            
            self.models[model_id] = metadata
            self.stats["total_models"] += 1
            
            logger.info(f"Modelo creado: {name} ({model_id})")
            return model_id
            
        except Exception as e:
            logger.error(f"Error al crear modelo: {e}")
            raise
    
    async def train_model(
        self,
        model_id: str,
        training_data: pd.DataFrame,
        target_column: str,
        test_size: float = 0.2,
        validation_split: float = 0.1
    ) -> str:
        """Entrenar modelo."""
        try:
            if model_id not in self.models:
                raise ValueError(f"Modelo {model_id} no encontrado")
            
            metadata = self.models[model_id]
            job_id = str(uuid.uuid4())
            
            # Crear trabajo de entrenamiento
            training_job = TrainingJob(
                job_id=job_id,
                model_id=model_id,
                status="started",
                progress=0.0,
                started_at=datetime.now()
            )
            self.training_jobs[job_id] = training_job
            
            # Ejecutar entrenamiento
            asyncio.create_task(self._train_model_async(
                model_id, training_data, target_column, test_size, validation_split, job_id
            ))
            
            logger.info(f"Entrenamiento iniciado para modelo {model_id} (job: {job_id})")
            return job_id
            
        except Exception as e:
            logger.error(f"Error al iniciar entrenamiento: {e}")
            raise
    
    async def _train_model_async(
        self,
        model_id: str,
        training_data: pd.DataFrame,
        target_column: str,
        test_size: float,
        validation_split: float,
        job_id: str
    ):
        """Entrenar modelo de forma asíncrona."""
        try:
            metadata = self.models[model_id]
            job = self.training_jobs[job_id]
            
            # Preparar datos
            job.status = "preparing_data"
            job.progress = 0.1
            
            X = training_data.drop(columns=[target_column])
            y = training_data[target_column]
            
            # Dividir datos
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            job.status = "training"
            job.progress = 0.2
            
            # Crear y entrenar modelo
            model = self._create_model_instance(metadata)
            
            if metadata.model_type == ModelType.CLASSIFICATION:
                model.fit(X_train, y_train)
                
                # Evaluar modelo
                y_pred = model.predict(X_test)
                job.metrics = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred, average='weighted'),
                    "recall": recall_score(y_test, y_pred, average='weighted'),
                    "f1_score": f1_score(y_test, y_pred, average='weighted')
                }
                
                metadata.accuracy = job.metrics["accuracy"]
                metadata.precision = job.metrics["precision"]
                metadata.recall = job.metrics["recall"]
                metadata.f1_score = job.metrics["f1_score"]
                
            elif metadata.model_type == ModelType.REGRESSION:
                model.fit(X_train, y_train)
                
                # Evaluar modelo
                y_pred = model.predict(X_test)
                job.metrics = {
                    "mse": mean_squared_error(y_test, y_pred),
                    "r2_score": r2_score(y_test, y_pred)
                }
                
                metadata.mse = job.metrics["mse"]
                metadata.r2_score = job.metrics["r2_score"]
                
            elif metadata.model_type == ModelType.CLUSTERING:
                model.fit(X_train)
                
                # Evaluar clustering
                labels = model.labels_
                job.metrics = {
                    "n_clusters": len(set(labels)) - (1 if -1 in labels else 0),
                    "silhouette_score": self._calculate_silhouette_score(X_train, labels)
                }
            
            job.progress = 0.8
            
            # Guardar modelo
            self.trained_models[model_id] = model
            metadata.status = ModelStatus.TRAINED
            metadata.training_samples = len(training_data)
            metadata.updated_at = datetime.now()
            
            # Guardar modelo en disco
            await self._save_model(model_id)
            
            job.status = "completed"
            job.progress = 1.0
            job.completed_at = datetime.now()
            
            self.stats["trained_models"] += 1
            
            logger.info(f"Modelo {model_id} entrenado exitosamente")
            
        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)
            job.completed_at = datetime.now()
            
            metadata.status = ModelStatus.FAILED
            self.stats["failed_models"] += 1
            
            logger.error(f"Error al entrenar modelo {model_id}: {e}")
    
    def _create_model_instance(self, metadata: ModelMetadata):
        """Crear instancia de modelo."""
        algorithm = metadata.algorithm.lower()
        hyperparams = metadata.hyperparameters
        
        if metadata.model_type == ModelType.CLASSIFICATION:
            if algorithm == "random_forest":
                return RandomForestClassifier(**hyperparams)
            elif algorithm == "logistic_regression":
                return LogisticRegression(**hyperparams)
            else:
                return RandomForestClassifier()
                
        elif metadata.model_type == ModelType.REGRESSION:
            if algorithm == "gradient_boosting":
                return GradientBoostingRegressor(**hyperparams)
            elif algorithm == "linear_regression":
                return LinearRegression(**hyperparams)
            else:
                return GradientBoostingRegressor()
                
        elif metadata.model_type == ModelType.CLUSTERING:
            if algorithm == "kmeans":
                return KMeans(**hyperparams)
            elif algorithm == "dbscan":
                return DBSCAN(**hyperparams)
            else:
                return KMeans()
        
        raise ValueError(f"Algoritmo no soportado: {algorithm}")
    
    def _calculate_silhouette_score(self, X, labels):
        """Calcular silhouette score."""
        try:
            from sklearn.metrics import silhouette_score
            return silhouette_score(X, labels)
        except:
            return 0.0
    
    async def predict(
        self,
        model_id: str,
        data: Union[pd.DataFrame, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Realizar predicción."""
        try:
            if model_id not in self.trained_models:
                raise ValueError(f"Modelo {model_id} no está entrenado")
            
            model = self.trained_models[model_id]
            metadata = self.models[model_id]
            
            # Convertir datos si es necesario
            if isinstance(data, list):
                data = pd.DataFrame(data)
            
            # Asegurar que las columnas coincidan
            expected_features = metadata.features
            if expected_features:
                missing_features = set(expected_features) - set(data.columns)
                if missing_features:
                    raise ValueError(f"Faltan características: {missing_features}")
                data = data[expected_features]
            
            # Realizar predicción
            predictions = model.predict(data)
            
            # Calcular probabilidades si es posible
            probabilities = None
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(data).tolist()
            
            self.stats["total_predictions"] += 1
            
            return {
                "model_id": model_id,
                "predictions": predictions.tolist() if hasattr(predictions, 'tolist') else predictions,
                "probabilities": probabilities,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error al realizar predicción: {e}")
            raise
    
    async def deploy_model(self, model_id: str) -> bool:
        """Desplegar modelo."""
        try:
            if model_id not in self.models:
                return False
            
            metadata = self.models[model_id]
            if metadata.status != ModelStatus.TRAINED:
                return False
            
            metadata.status = ModelStatus.DEPLOYED
            metadata.updated_at = datetime.now()
            
            self.stats["deployed_models"] += 1
            
            logger.info(f"Modelo {model_id} desplegado")
            return True
            
        except Exception as e:
            logger.error(f"Error al desplegar modelo: {e}")
            return False
    
    async def retrain_model(
        self,
        model_id: str,
        new_data: pd.DataFrame,
        target_column: str,
        incremental: bool = True
    ) -> str:
        """Reentrenar modelo."""
        try:
            if model_id not in self.models:
                raise ValueError(f"Modelo {model_id} no encontrado")
            
            metadata = self.models[model_id]
            
            if incremental and metadata.training_samples > 0:
                # Entrenamiento incremental
                # Combinar datos existentes con nuevos datos
                # (En una implementación real, se cargarían los datos originales)
                combined_data = new_data
            else:
                combined_data = new_data
            
            # Iniciar reentrenamiento
            return await self.train_model(model_id, combined_data, target_column)
            
        except Exception as e:
            logger.error(f"Error al reentrenar modelo: {e}")
            raise
    
    async def get_model_performance(self, model_id: str) -> Dict[str, Any]:
        """Obtener rendimiento del modelo."""
        try:
            if model_id not in self.models:
                raise ValueError(f"Modelo {model_id} no encontrado")
            
            metadata = self.models[model_id]
            
            return {
                "model_id": model_id,
                "name": metadata.name,
                "status": metadata.status.value,
                "accuracy": metadata.accuracy,
                "precision": metadata.precision,
                "recall": metadata.recall,
                "f1_score": metadata.f1_score,
                "mse": metadata.mse,
                "r2_score": metadata.r2_score,
                "training_samples": metadata.training_samples,
                "created_at": metadata.created_at.isoformat(),
                "updated_at": metadata.updated_at.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error al obtener rendimiento del modelo: {e}")
            raise
    
    async def get_training_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Obtener estado del trabajo de entrenamiento."""
        if job_id not in self.training_jobs:
            return None
        
        job = self.training_jobs[job_id]
        
        return {
            "job_id": job.job_id,
            "model_id": job.model_id,
            "status": job.status,
            "progress": job.progress,
            "started_at": job.started_at.isoformat(),
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "error_message": job.error_message,
            "metrics": job.metrics
        }
    
    async def _save_model(self, model_id: str):
        """Guardar modelo en disco."""
        try:
            if model_id not in self.trained_models:
                return
            
            model_file = self.models_directory / f"{model_id}.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump(self.trained_models[model_id], f)
            
            # Guardar metadatos
            await self._save_metadata()
            
        except Exception as e:
            logger.error(f"Error al guardar modelo {model_id}: {e}")
    
    async def _save_all_models(self):
        """Guardar todos los modelos."""
        try:
            for model_id in self.trained_models:
                await self._save_model(model_id)
            
            await self._save_metadata()
            
        except Exception as e:
            logger.error(f"Error al guardar modelos: {e}")
    
    async def _save_metadata(self):
        """Guardar metadatos."""
        try:
            metadata_file = self.models_directory / "models_metadata.pkl"
            with open(metadata_file, 'wb') as f:
                pickle.dump(self.models, f)
                
        except Exception as e:
            logger.error(f"Error al guardar metadatos: {e}")
    
    async def _cleanup_loop(self):
        """Bucle de limpieza automática."""
        while True:
            try:
                await asyncio.sleep(3600)  # Cada hora
                await self._cleanup_old_models()
                await self._cleanup_old_jobs()
            except Exception as e:
                logger.error(f"Error en limpieza automática: {e}")
                await asyncio.sleep(3600)
    
    async def _cleanup_old_models(self):
        """Limpiar modelos antiguos."""
        try:
            cutoff_time = datetime.now() - timedelta(days=self.model_retention_days)
            
            models_to_remove = []
            for model_id, metadata in self.models.items():
                if (metadata.status == ModelStatus.RETIRED and 
                    metadata.updated_at < cutoff_time):
                    models_to_remove.append(model_id)
            
            for model_id in models_to_remove:
                # Eliminar archivos
                model_file = self.models_directory / f"{model_id}.pkl"
                if model_file.exists():
                    model_file.unlink()
                
                # Eliminar de memoria
                del self.models[model_id]
                if model_id in self.trained_models:
                    del self.trained_models[model_id]
            
            if models_to_remove:
                await self._save_metadata()
                logger.info(f"Limpiados {len(models_to_remove)} modelos antiguos")
                
        except Exception as e:
            logger.error(f"Error en limpieza de modelos: {e}")
    
    async def _cleanup_old_jobs(self):
        """Limpiar trabajos antiguos."""
        try:
            cutoff_time = datetime.now() - timedelta(days=7)
            
            jobs_to_remove = []
            for job_id, job in self.training_jobs.items():
                if job.completed_at and job.completed_at < cutoff_time:
                    jobs_to_remove.append(job_id)
            
            for job_id in jobs_to_remove:
                del self.training_jobs[job_id]
            
            if jobs_to_remove:
                logger.info(f"Limpiados {len(jobs_to_remove)} trabajos antiguos")
                
        except Exception as e:
            logger.error(f"Error en limpieza de trabajos: {e}")
    
    async def get_ml_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de ML."""
        return {
            **self.stats,
            "uptime_seconds": (datetime.now() - self.stats["start_time"]).total_seconds(),
            "models_count": len(self.models),
            "trained_models_count": len(self.trained_models),
            "active_jobs": len([j for j in self.training_jobs.values() if j.status == "started"]),
            "models_directory": str(self.models_directory),
            "timestamp": datetime.now().isoformat()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Verificar salud del motor de ML."""
        try:
            return {
                "status": "healthy",
                "models_count": len(self.models),
                "trained_models_count": len(self.trained_models),
                "active_jobs": len([j for j in self.training_jobs.values() if j.status == "started"]),
                "stats": self.stats,
                "uptime_seconds": (datetime.now() - self.stats["start_time"]).total_seconds(),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error en health check de ML: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }




