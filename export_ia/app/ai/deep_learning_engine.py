"""
Deep Learning Engine - Motor de Deep Learning avanzado
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
import json
from pathlib import Path
import uuid
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class DeepLearningModelType(Enum):
    """Tipos de modelos de Deep Learning."""
    NEURAL_NETWORK = "neural_network"
    CONVOLUTIONAL_NN = "convolutional_nn"
    RECURRENT_NN = "recurrent_nn"
    LSTM = "lstm"
    GRU = "gru"
    TRANSFORMER = "transformer"
    AUTOENCODER = "autoencoder"
    GAN = "gan"
    VAE = "vae"


class ModelArchitecture(Enum):
    """Arquitecturas de modelos."""
    SEQUENTIAL = "sequential"
    FUNCTIONAL = "functional"
    SUBCLASSING = "subclassing"


@dataclass
class DeepLearningModel:
    """Modelo de Deep Learning."""
    model_id: str
    name: str
    model_type: DeepLearningModelType
    architecture: ModelArchitecture
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    layers_config: List[Dict[str, Any]]
    optimizer: str
    loss_function: str
    metrics: List[str]
    created_at: datetime
    updated_at: datetime
    status: str
    training_history: Dict[str, List[float]] = field(default_factory=dict)
    best_accuracy: float = 0.0
    best_loss: float = float('inf')
    epochs_trained: int = 0
    total_params: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """Configuración de entrenamiento."""
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    save_best_only: bool = True
    monitor: str = "val_loss"
    mode: str = "min"


class DeepLearningEngine:
    """
    Motor de Deep Learning avanzado.
    """
    
    def __init__(self, models_directory: str = "deep_learning_models"):
        """Inicializar motor de Deep Learning."""
        self.models_directory = Path(models_directory)
        self.models_directory.mkdir(exist_ok=True)
        
        # Almacenamiento de modelos
        self.models: Dict[str, DeepLearningModel] = {}
        self.trained_models: Dict[str, keras.Model] = {}
        self.training_jobs: Dict[str, Dict[str, Any]] = {}
        
        # Configuración
        self.max_models = 50
        self.model_retention_days = 90
        self.gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
        
        # Configurar GPU si está disponible
        if self.gpu_available:
            self._configure_gpu()
        
        # Estadísticas
        self.stats = {
            "total_models": 0,
            "trained_models": 0,
            "training_jobs": 0,
            "total_predictions": 0,
            "gpu_available": self.gpu_available,
            "start_time": datetime.now()
        }
        
        # Cargar modelos existentes
        self._load_existing_models()
        
        logger.info(f"DeepLearningEngine inicializado (GPU: {self.gpu_available})")
    
    async def initialize(self):
        """Inicializar el motor de Deep Learning."""
        try:
            # Iniciar limpieza automática
            asyncio.create_task(self._cleanup_loop())
            
            logger.info("DeepLearningEngine inicializado exitosamente")
            
        except Exception as e:
            logger.error(f"Error al inicializar DeepLearningEngine: {e}")
            raise
    
    async def shutdown(self):
        """Cerrar el motor de Deep Learning."""
        try:
            # Guardar modelos
            await self._save_all_models()
            
            # Limpiar memoria
            tf.keras.backend.clear_session()
            
            logger.info("DeepLearningEngine cerrado")
            
        except Exception as e:
            logger.error(f"Error al cerrar DeepLearningEngine: {e}")
    
    def _configure_gpu(self):
        """Configurar GPU."""
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"GPU configurada: {len(gpus)} dispositivos")
        except Exception as e:
            logger.warning(f"Error al configurar GPU: {e}")
    
    def _load_existing_models(self):
        """Cargar modelos existentes."""
        try:
            metadata_file = self.models_directory / "models_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    models_data = json.load(f)
                
                for model_id, model_data in models_data.items():
                    # Convertir fechas
                    model_data['created_at'] = datetime.fromisoformat(model_data['created_at'])
                    model_data['updated_at'] = datetime.fromisoformat(model_data['updated_at'])
                    
                    # Crear objeto del modelo
                    model = DeepLearningModel(**model_data)
                    self.models[model_id] = model
                    
                    # Cargar modelo entrenado si existe
                    model_file = self.models_directory / f"{model_id}.h5"
                    if model_file.exists() and model.status == "trained":
                        try:
                            self.trained_models[model_id] = keras.models.load_model(model_file)
                        except Exception as e:
                            logger.warning(f"No se pudo cargar modelo {model_id}: {e}")
                            model.status = "failed"
                
                logger.info(f"Cargados {len(self.models)} modelos existentes")
                
        except Exception as e:
            logger.error(f"Error al cargar modelos existentes: {e}")
    
    async def create_model(
        self,
        name: str,
        model_type: DeepLearningModelType,
        input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
        layers_config: List[Dict[str, Any]],
        optimizer: str = "adam",
        loss_function: str = "categorical_crossentropy",
        metrics: List[str] = None
    ) -> str:
        """Crear nuevo modelo de Deep Learning."""
        try:
            model_id = str(uuid.uuid4())
            now = datetime.now()
            
            model = DeepLearningModel(
                model_id=model_id,
                name=name,
                model_type=model_type,
                architecture=ModelArchitecture.SEQUENTIAL,
                input_shape=input_shape,
                output_shape=output_shape,
                layers_config=layers_config,
                optimizer=optimizer,
                loss_function=loss_function,
                metrics=metrics or ["accuracy"],
                created_at=now,
                updated_at=now,
                status="created"
            )
            
            self.models[model_id] = model
            self.stats["total_models"] += 1
            
            logger.info(f"Modelo de Deep Learning creado: {name} ({model_id})")
            return model_id
            
        except Exception as e:
            logger.error(f"Error al crear modelo: {e}")
            raise
    
    async def build_model(self, model_id: str) -> keras.Model:
        """Construir modelo de Keras."""
        try:
            if model_id not in self.models:
                raise ValueError(f"Modelo {model_id} no encontrado")
            
            model_config = self.models[model_id]
            
            # Crear modelo secuencial
            model = keras.Sequential()
            
            # Agregar capas según configuración
            for layer_config in model_config.layers_config:
                layer_type = layer_config.get("type")
                layer_params = layer_config.get("params", {})
                
                if layer_type == "dense":
                    model.add(layers.Dense(**layer_params))
                elif layer_type == "conv2d":
                    model.add(layers.Conv2D(**layer_params))
                elif layer_type == "maxpool2d":
                    model.add(layers.MaxPooling2D(**layer_params))
                elif layer_type == "lstm":
                    model.add(layers.LSTM(**layer_params))
                elif layer_type == "gru":
                    model.add(layers.GRU(**layer_params))
                elif layer_type == "dropout":
                    model.add(layers.Dropout(**layer_params))
                elif layer_type == "batch_normalization":
                    model.add(layers.BatchNormalization(**layer_params))
                elif layer_type == "flatten":
                    model.add(layers.Flatten(**layer_params))
                elif layer_type == "global_average_pooling2d":
                    model.add(layers.GlobalAveragePooling2D(**layer_params))
                else:
                    logger.warning(f"Tipo de capa no soportado: {layer_type}")
            
            # Compilar modelo
            optimizer = self._get_optimizer(model_config.optimizer)
            model.compile(
                optimizer=optimizer,
                loss=model_config.loss_function,
                metrics=model_config.metrics
            )
            
            # Contar parámetros
            model_config.total_params = model.count_params()
            
            return model
            
        except Exception as e:
            logger.error(f"Error al construir modelo: {e}")
            raise
    
    def _get_optimizer(self, optimizer_name: str):
        """Obtener optimizador."""
        if optimizer_name.lower() == "adam":
            return optimizers.Adam()
        elif optimizer_name.lower() == "sgd":
            return optimizers.SGD()
        elif optimizer_name.lower() == "rmsprop":
            return optimizers.RMSprop()
        elif optimizer_name.lower() == "adagrad":
            return optimizers.Adagrad()
        else:
            return optimizers.Adam()
    
    async def train_model(
        self,
        model_id: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        config: TrainingConfig = None
    ) -> str:
        """Entrenar modelo de Deep Learning."""
        try:
            if model_id not in self.models:
                raise ValueError(f"Modelo {model_id} no encontrado")
            
            config = config or TrainingConfig()
            job_id = str(uuid.uuid4())
            
            # Crear trabajo de entrenamiento
            training_job = {
                "job_id": job_id,
                "model_id": model_id,
                "status": "started",
                "progress": 0.0,
                "started_at": datetime.now(),
                "config": config.__dict__
            }
            self.training_jobs[job_id] = training_job
            
            # Ejecutar entrenamiento
            asyncio.create_task(self._train_model_async(
                model_id, X_train, y_train, X_val, y_val, config, job_id
            ))
            
            logger.info(f"Entrenamiento iniciado para modelo {model_id} (job: {job_id})")
            return job_id
            
        except Exception as e:
            logger.error(f"Error al iniciar entrenamiento: {e}")
            raise
    
    async def _train_model_async(
        self,
        model_id: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        config: TrainingConfig,
        job_id: str
    ):
        """Entrenar modelo de forma asíncrona."""
        try:
            model_config = self.models[model_id]
            job = self.training_jobs[job_id]
            
            # Construir modelo
            job["status"] = "building_model"
            job["progress"] = 0.1
            
            model = await self.build_model(model_id)
            
            # Preparar datos de validación
            if X_val is None or y_val is None:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train, y_train, test_size=config.validation_split, random_state=42
                )
            
            # Configurar callbacks
            callbacks_list = self._create_callbacks(model_id, config)
            
            job["status"] = "training"
            job["progress"] = 0.2
            
            # Entrenar modelo
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=config.epochs,
                batch_size=config.batch_size,
                callbacks=callbacks_list,
                verbose=0
            )
            
            job["progress"] = 0.8
            
            # Guardar modelo
            self.trained_models[model_id] = model
            model_config.status = "trained"
            model_config.training_history = history.history
            model_config.epochs_trained = len(history.history['loss'])
            model_config.updated_at = datetime.now()
            
            # Calcular mejores métricas
            if 'val_accuracy' in history.history:
                model_config.best_accuracy = max(history.history['val_accuracy'])
            if 'val_loss' in history.history:
                model_config.best_loss = min(history.history['val_loss'])
            
            # Guardar modelo en disco
            await self._save_model(model_id)
            
            job["status"] = "completed"
            job["progress"] = 1.0
            job["completed_at"] = datetime.now()
            job["metrics"] = {
                "best_accuracy": model_config.best_accuracy,
                "best_loss": model_config.best_loss,
                "epochs_trained": model_config.epochs_trained
            }
            
            self.stats["trained_models"] += 1
            
            logger.info(f"Modelo {model_id} entrenado exitosamente")
            
        except Exception as e:
            job["status"] = "failed"
            job["error_message"] = str(e)
            job["completed_at"] = datetime.now()
            
            model_config.status = "failed"
            
            logger.error(f"Error al entrenar modelo {model_id}: {e}")
    
    def _create_callbacks(self, model_id: str, config: TrainingConfig) -> List[keras.callbacks.Callback]:
        """Crear callbacks para entrenamiento."""
        callbacks_list = []
        
        # Early stopping
        if config.early_stopping_patience > 0:
            early_stopping = callbacks.EarlyStopping(
                monitor=config.monitor,
                patience=config.early_stopping_patience,
                restore_best_weights=True
            )
            callbacks_list.append(early_stopping)
        
        # Reduce learning rate
        if config.reduce_lr_patience > 0:
            reduce_lr = callbacks.ReduceLROnPlateau(
                monitor=config.monitor,
                patience=config.reduce_lr_patience,
                factor=0.5,
                min_lr=1e-7
            )
            callbacks_list.append(reduce_lr)
        
        # Model checkpoint
        checkpoint_path = self.models_directory / f"{model_id}_checkpoint.h5"
        checkpoint = callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor=config.monitor,
            save_best_only=config.save_best_only,
            save_weights_only=False
        )
        callbacks_list.append(checkpoint)
        
        return callbacks_list
    
    async def predict(
        self,
        model_id: str,
        data: np.ndarray,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """Realizar predicción."""
        try:
            if model_id not in self.trained_models:
                raise ValueError(f"Modelo {model_id} no está entrenado")
            
            model = self.trained_models[model_id]
            model_config = self.models[model_id]
            
            # Realizar predicción
            predictions = model.predict(data, batch_size=batch_size, verbose=0)
            
            # Calcular probabilidades si es clasificación
            probabilities = None
            if model_config.model_type in [DeepLearningModelType.NEURAL_NETWORK, DeepLearningModelType.CONVOLUTIONAL_NN]:
                if len(predictions.shape) > 1 and predictions.shape[1] > 1:
                    probabilities = predictions.tolist()
                    predictions = np.argmax(predictions, axis=1).tolist()
                else:
                    predictions = predictions.flatten().tolist()
            else:
                predictions = predictions.tolist()
            
            self.stats["total_predictions"] += 1
            
            return {
                "model_id": model_id,
                "predictions": predictions,
                "probabilities": probabilities,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error al realizar predicción: {e}")
            raise
    
    async def evaluate_model(
        self,
        model_id: str,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """Evaluar modelo."""
        try:
            if model_id not in self.trained_models:
                raise ValueError(f"Modelo {model_id} no está entrenado")
            
            model = self.trained_models[model_id]
            
            # Evaluar modelo
            results = model.evaluate(X_test, y_test, verbose=0)
            
            # Crear diccionario de métricas
            metrics = {}
            model_config = self.models[model_id]
            
            # Loss siempre es la primera métrica
            metrics["loss"] = results[0]
            
            # Otras métricas
            for i, metric_name in enumerate(model_config.metrics, 1):
                if i < len(results):
                    metrics[metric_name] = results[i]
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error al evaluar modelo: {e}")
            raise
    
    async def get_model_summary(self, model_id: str) -> Dict[str, Any]:
        """Obtener resumen del modelo."""
        try:
            if model_id not in self.models:
                raise ValueError(f"Modelo {model_id} no encontrado")
            
            model_config = self.models[model_id]
            
            summary = {
                "model_id": model_id,
                "name": model_config.name,
                "model_type": model_config.model_type.value,
                "architecture": model_config.architecture.value,
                "input_shape": model_config.input_shape,
                "output_shape": model_config.output_shape,
                "total_params": model_config.total_params,
                "status": model_config.status,
                "best_accuracy": model_config.best_accuracy,
                "best_loss": model_config.best_loss,
                "epochs_trained": model_config.epochs_trained,
                "created_at": model_config.created_at.isoformat(),
                "updated_at": model_config.updated_at.isoformat(),
                "layers_count": len(model_config.layers_config)
            }
            
            # Agregar resumen de capas si el modelo está entrenado
            if model_id in self.trained_models:
                model = self.trained_models[model_id]
                summary["model_summary"] = self._get_model_summary_string(model)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error al obtener resumen del modelo: {e}")
            raise
    
    def _get_model_summary_string(self, model: keras.Model) -> str:
        """Obtener resumen del modelo como string."""
        try:
            import io
            import sys
            
            # Capturar output del summary
            old_stdout = sys.stdout
            sys.stdout = buffer = io.StringIO()
            model.summary()
            sys.stdout = old_stdout
            
            return buffer.getvalue()
            
        except Exception as e:
            return f"Error al obtener resumen: {e}"
    
    async def _save_model(self, model_id: str):
        """Guardar modelo en disco."""
        try:
            if model_id not in self.trained_models:
                return
            
            model_file = self.models_directory / f"{model_id}.h5"
            self.trained_models[model_id].save(str(model_file))
            
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
            metadata_file = self.models_directory / "models_metadata.json"
            
            # Convertir modelos a diccionario serializable
            models_data = {}
            for model_id, model in self.models.items():
                model_dict = model.__dict__.copy()
                model_dict['created_at'] = model_dict['created_at'].isoformat()
                model_dict['updated_at'] = model_dict['updated_at'].isoformat()
                models_data[model_id] = model_dict
            
            with open(metadata_file, 'w') as f:
                json.dump(models_data, f, indent=2)
                
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
            for model_id, model in self.models.items():
                if (model.status == "retired" and 
                    model.updated_at < cutoff_time):
                    models_to_remove.append(model_id)
            
            for model_id in models_to_remove:
                # Eliminar archivos
                model_file = self.models_directory / f"{model_id}.h5"
                if model_file.exists():
                    model_file.unlink()
                
                checkpoint_file = self.models_directory / f"{model_id}_checkpoint.h5"
                if checkpoint_file.exists():
                    checkpoint_file.unlink()
                
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
                if (job.get("completed_at") and 
                    job["completed_at"] < cutoff_time):
                    jobs_to_remove.append(job_id)
            
            for job_id in jobs_to_remove:
                del self.training_jobs[job_id]
            
            if jobs_to_remove:
                logger.info(f"Limpiados {len(jobs_to_remove)} trabajos antiguos")
                
        except Exception as e:
            logger.error(f"Error en limpieza de trabajos: {e}")
    
    async def get_dl_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de Deep Learning."""
        return {
            **self.stats,
            "uptime_seconds": (datetime.now() - self.stats["start_time"]).total_seconds(),
            "models_count": len(self.models),
            "trained_models_count": len(self.trained_models),
            "active_jobs": len([j for j in self.training_jobs.values() if j.get("status") == "started"]),
            "models_directory": str(self.models_directory),
            "gpu_available": self.gpu_available,
            "tensorflow_version": tf.__version__,
            "timestamp": datetime.now().isoformat()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Verificar salud del motor de Deep Learning."""
        try:
            return {
                "status": "healthy",
                "models_count": len(self.models),
                "trained_models_count": len(self.trained_models),
                "active_jobs": len([j for j in self.training_jobs.values() if j.get("status") == "started"]),
                "gpu_available": self.gpu_available,
                "tensorflow_version": tf.__version__,
                "stats": self.stats,
                "uptime_seconds": (datetime.now() - self.stats["start_time"]).total_seconds(),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error en health check de Deep Learning: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }




