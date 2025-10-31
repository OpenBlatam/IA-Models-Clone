"""
Motor de Aprendizaje AI
======================

Motor para aprendizaje continuo, adaptación de modelos y mejora automática del sistema.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
from pathlib import Path
import hashlib
import numpy as np
from collections import defaultdict, deque
import pickle
import threading

logger = logging.getLogger(__name__)

class LearningType(str, Enum):
    """Tipos de aprendizaje"""
    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"
    REINFORCEMENT = "reinforcement"
    TRANSFER = "transfer"
    CONTINUOUS = "continuous"
    FEDERATED = "federated"

class ModelType(str, Enum):
    """Tipos de modelos"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    NLP = "nlp"
    COMPUTER_VISION = "computer_vision"
    RECOMMENDATION = "recommendation"
    ANOMALY_DETECTION = "anomaly_detection"

class LearningStatus(str, Enum):
    """Estados de aprendizaje"""
    IDLE = "idle"
    TRAINING = "training"
    EVALUATING = "evaluating"
    DEPLOYING = "deploying"
    ACTIVE = "active"
    ERROR = "error"
    PAUSED = "paused"

@dataclass
class LearningTask:
    """Tarea de aprendizaje"""
    id: str
    name: str
    description: str
    learning_type: LearningType
    model_type: ModelType
    dataset_id: str
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    status: LearningStatus = LearningStatus.IDLE
    progress: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

@dataclass
class Model:
    """Modelo de aprendizaje"""
    id: str
    name: str
    model_type: ModelType
    version: str
    learning_task_id: str
    model_data: Any = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    training_data_size: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    is_active: bool = False
    deployment_info: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Dataset:
    """Dataset de aprendizaje"""
    id: str
    name: str
    description: str
    data_type: str
    size: int
    features: List[str] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LearningFeedback:
    """Feedback de aprendizaje"""
    id: str
    model_id: str
    user_id: str
    feedback_type: str
    feedback_data: Dict[str, Any] = field(default_factory=dict)
    rating: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class LearningMetrics:
    """Métricas de aprendizaje"""
    model_id: str
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    loss: float = 0.0
    training_time: float = 0.0
    inference_time: float = 0.0
    data_quality_score: float = 0.0
    model_complexity: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

class AILearningEngine:
    """Motor de aprendizaje AI"""
    
    def __init__(self):
        self.learning_tasks: Dict[str, LearningTask] = {}
        self.models: Dict[str, Model] = {}
        self.datasets: Dict[str, Dataset] = {}
        self.learning_feedback: List[LearningFeedback] = []
        self.learning_metrics: Dict[str, List[LearningMetrics]] = defaultdict(list)
        
        # Configuración de aprendizaje
        self.max_concurrent_tasks = 3
        self.model_retention_days = 30
        self.feedback_retention_days = 90
        self.auto_retrain_threshold = 0.1  # 10% de degradación
        
        # Workers de aprendizaje
        self.learning_workers: Dict[str, asyncio.Task] = {}
        self.continuous_learning_active = False
        
        # Almacenamiento de modelos
        self.model_storage_path = Path("data/models")
        self.model_storage_path.mkdir(parents=True, exist_ok=True)
        
    async def initialize(self):
        """Inicializa el motor de aprendizaje"""
        logger.info("Inicializando motor de aprendizaje AI...")
        
        # Cargar datos existentes
        await self._load_learning_data()
        
        # Iniciar workers de aprendizaje
        await self._start_learning_workers()
        
        # Iniciar aprendizaje continuo
        await self._start_continuous_learning()
        
        logger.info("Motor de aprendizaje AI inicializado")
    
    async def _load_learning_data(self):
        """Carga datos de aprendizaje"""
        try:
            # Cargar tareas de aprendizaje
            tasks_file = Path("data/learning_tasks.json")
            if tasks_file.exists():
                with open(tasks_file, 'r', encoding='utf-8') as f:
                    tasks_data = json.load(f)
                
                for task_data in tasks_data:
                    task = LearningTask(
                        id=task_data["id"],
                        name=task_data["name"],
                        description=task_data["description"],
                        learning_type=LearningType(task_data["learning_type"]),
                        model_type=ModelType(task_data["model_type"]),
                        dataset_id=task_data["dataset_id"],
                        hyperparameters=task_data["hyperparameters"],
                        status=LearningStatus(task_data["status"]),
                        progress=task_data["progress"],
                        metrics=task_data["metrics"],
                        created_at=datetime.fromisoformat(task_data["created_at"]),
                        started_at=datetime.fromisoformat(task_data["started_at"]) if task_data.get("started_at") else None,
                        completed_at=datetime.fromisoformat(task_data["completed_at"]) if task_data.get("completed_at") else None,
                        error_message=task_data.get("error_message")
                    )
                    self.learning_tasks[task.id] = task
                
                logger.info(f"Cargadas {len(self.learning_tasks)} tareas de aprendizaje")
            
            # Cargar modelos
            models_file = Path("data/learning_models.json")
            if models_file.exists():
                with open(models_file, 'r', encoding='utf-8') as f:
                    models_data = json.load(f)
                
                for model_data in models_data:
                    model = Model(
                        id=model_data["id"],
                        name=model_data["name"],
                        model_type=ModelType(model_data["model_type"]),
                        version=model_data["version"],
                        learning_task_id=model_data["learning_task_id"],
                        performance_metrics=model_data["performance_metrics"],
                        training_data_size=model_data["training_data_size"],
                        created_at=datetime.fromisoformat(model_data["created_at"]),
                        is_active=model_data["is_active"],
                        deployment_info=model_data["deployment_info"]
                    )
                    self.models[model.id] = model
                
                logger.info(f"Cargados {len(self.models)} modelos")
            
            # Cargar datasets
            datasets_file = Path("data/learning_datasets.json")
            if datasets_file.exists():
                with open(datasets_file, 'r', encoding='utf-8') as f:
                    datasets_data = json.load(f)
                
                for dataset_data in datasets_data:
                    dataset = Dataset(
                        id=dataset_data["id"],
                        name=dataset_data["name"],
                        description=dataset_data["description"],
                        data_type=dataset_data["data_type"],
                        size=dataset_data["size"],
                        features=dataset_data["features"],
                        labels=dataset_data["labels"],
                        created_at=datetime.fromisoformat(dataset_data["created_at"]),
                        updated_at=datetime.fromisoformat(dataset_data["updated_at"]),
                        metadata=dataset_data["metadata"]
                    )
                    self.datasets[dataset.id] = dataset
                
                logger.info(f"Cargados {len(self.datasets)} datasets")
            
        except Exception as e:
            logger.error(f"Error cargando datos de aprendizaje: {e}")
    
    async def _start_learning_workers(self):
        """Inicia workers de aprendizaje"""
        try:
            # Worker de monitoreo de tareas
            asyncio.create_task(self._task_monitoring_worker())
            
            # Worker de evaluación de modelos
            asyncio.create_task(self._model_evaluation_worker())
            
            # Worker de limpieza de datos
            asyncio.create_task(self._data_cleanup_worker())
            
            logger.info("Workers de aprendizaje iniciados")
            
        except Exception as e:
            logger.error(f"Error iniciando workers de aprendizaje: {e}")
    
    async def _start_continuous_learning(self):
        """Inicia aprendizaje continuo"""
        try:
            self.continuous_learning_active = True
            asyncio.create_task(self._continuous_learning_worker())
            logger.info("Aprendizaje continuo iniciado")
            
        except Exception as e:
            logger.error(f"Error iniciando aprendizaje continuo: {e}")
    
    async def _task_monitoring_worker(self):
        """Worker de monitoreo de tareas"""
        while True:
            try:
                await asyncio.sleep(30)  # Cada 30 segundos
                
                # Monitorear tareas en progreso
                for task in self.learning_tasks.values():
                    if task.status == LearningStatus.TRAINING:
                        # Simular progreso de entrenamiento
                        task.progress = min(100.0, task.progress + np.random.uniform(1, 5))
                        
                        if task.progress >= 100.0:
                            await self._complete_learning_task(task.id)
                
            except Exception as e:
                logger.error(f"Error en worker de monitoreo de tareas: {e}")
                await asyncio.sleep(60)
    
    async def _model_evaluation_worker(self):
        """Worker de evaluación de modelos"""
        while True:
            try:
                await asyncio.sleep(300)  # Cada 5 minutos
                
                # Evaluar modelos activos
                for model in self.models.values():
                    if model.is_active:
                        await self._evaluate_model_performance(model.id)
                
            except Exception as e:
                logger.error(f"Error en worker de evaluación de modelos: {e}")
                await asyncio.sleep(60)
    
    async def _data_cleanup_worker(self):
        """Worker de limpieza de datos"""
        while True:
            try:
                await asyncio.sleep(3600)  # Cada hora
                
                # Limpiar feedback antiguo
                cutoff_date = datetime.now() - timedelta(days=self.feedback_retention_days)
                self.learning_feedback = [
                    feedback for feedback in self.learning_feedback
                    if feedback.timestamp > cutoff_date
                ]
                
                # Limpiar métricas antiguas
                for model_id in list(self.learning_metrics.keys()):
                    self.learning_metrics[model_id] = [
                        metrics for metrics in self.learning_metrics[model_id]
                        if metrics.timestamp > cutoff_date
                    ]
                
                logger.info("Limpieza de datos completada")
                
            except Exception as e:
                logger.error(f"Error en worker de limpieza de datos: {e}")
                await asyncio.sleep(300)
    
    async def _continuous_learning_worker(self):
        """Worker de aprendizaje continuo"""
        while self.continuous_learning_active:
            try:
                await asyncio.sleep(1800)  # Cada 30 minutos
                
                # Verificar si se necesita reentrenamiento
                for model in self.models.values():
                    if model.is_active:
                        performance_degradation = await self._check_performance_degradation(model.id)
                        
                        if performance_degradation > self.auto_retrain_threshold:
                            await self._schedule_retraining(model.id)
                
            except Exception as e:
                logger.error(f"Error en worker de aprendizaje continuo: {e}")
                await asyncio.sleep(300)
    
    async def create_learning_task(
        self,
        name: str,
        description: str,
        learning_type: LearningType,
        model_type: ModelType,
        dataset_id: str,
        hyperparameters: Dict[str, Any] = None
    ) -> str:
        """Crea nueva tarea de aprendizaje"""
        try:
            if dataset_id not in self.datasets:
                raise ValueError(f"Dataset no encontrado: {dataset_id}")
            
            task_id = f"task_{uuid.uuid4().hex[:8]}"
            
            task = LearningTask(
                id=task_id,
                name=name,
                description=description,
                learning_type=learning_type,
                model_type=model_type,
                dataset_id=dataset_id,
                hyperparameters=hyperparameters or {}
            )
            
            self.learning_tasks[task_id] = task
            
            # Guardar datos
            await self._save_learning_data()
            
            logger.info(f"Tarea de aprendizaje creada: {task_id}")
            return task_id
            
        except Exception as e:
            logger.error(f"Error creando tarea de aprendizaje: {e}")
            raise
    
    async def start_learning_task(self, task_id: str) -> bool:
        """Inicia tarea de aprendizaje"""
        try:
            if task_id not in self.learning_tasks:
                return False
            
            task = self.learning_tasks[task_id]
            
            if task.status != LearningStatus.IDLE:
                return False
            
            # Verificar límite de tareas concurrentes
            active_tasks = len([t for t in self.learning_tasks.values() if t.status == LearningStatus.TRAINING])
            if active_tasks >= self.max_concurrent_tasks:
                return False
            
            # Iniciar tarea
            task.status = LearningStatus.TRAINING
            task.started_at = datetime.now()
            task.progress = 0.0
            
            # Crear worker de entrenamiento
            worker = asyncio.create_task(self._train_model_worker(task_id))
            self.learning_workers[task_id] = worker
            
            # Guardar datos
            await self._save_learning_data()
            
            logger.info(f"Tarea de aprendizaje iniciada: {task_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error iniciando tarea de aprendizaje: {e}")
            return False
    
    async def _train_model_worker(self, task_id: str):
        """Worker de entrenamiento de modelo"""
        try:
            task = self.learning_tasks[task_id]
            dataset = self.datasets[task.dataset_id]
            
            # Simular entrenamiento
            training_steps = 100
            for step in range(training_steps):
                await asyncio.sleep(0.1)  # Simular tiempo de entrenamiento
                
                # Actualizar progreso
                task.progress = (step + 1) / training_steps * 100
                
                # Simular métricas de entrenamiento
                if step % 10 == 0:
                    task.metrics = {
                        "accuracy": min(0.95, 0.5 + (step / training_steps) * 0.45),
                        "loss": max(0.1, 1.0 - (step / training_steps) * 0.9),
                        "precision": min(0.93, 0.4 + (step / training_steps) * 0.53),
                        "recall": min(0.91, 0.3 + (step / training_steps) * 0.61)
                    }
            
            # Completar entrenamiento
            await self._complete_learning_task(task_id)
            
        except Exception as e:
            logger.error(f"Error en worker de entrenamiento: {e}")
            task = self.learning_tasks[task_id]
            task.status = LearningStatus.ERROR
            task.error_message = str(e)
            await self._save_learning_data()
    
    async def _complete_learning_task(self, task_id: str):
        """Completa tarea de aprendizaje"""
        try:
            task = self.learning_tasks[task_id]
            
            # Crear modelo
            model_id = await self._create_model_from_task(task)
            
            # Completar tarea
            task.status = LearningStatus.COMPLETED
            task.completed_at = datetime.now()
            task.progress = 100.0
            
            # Limpiar worker
            if task_id in self.learning_workers:
                self.learning_workers[task_id].cancel()
                del self.learning_workers[task_id]
            
            # Guardar datos
            await self._save_learning_data()
            
            logger.info(f"Tarea de aprendizaje completada: {task_id}")
            
        except Exception as e:
            logger.error(f"Error completando tarea de aprendizaje: {e}")
    
    async def _create_model_from_task(self, task: LearningTask) -> str:
        """Crea modelo desde tarea"""
        try:
            model_id = f"model_{uuid.uuid4().hex[:8]}"
            dataset = self.datasets[task.dataset_id]
            
            model = Model(
                id=model_id,
                name=f"{task.name}_v1",
                model_type=task.model_type,
                version="1.0.0",
                learning_task_id=task.id,
                performance_metrics=task.metrics.copy(),
                training_data_size=dataset.size,
                is_active=False
            )
            
            self.models[model_id] = model
            
            # Guardar modelo
            await self._save_model(model_id)
            
            logger.info(f"Modelo creado: {model_id}")
            return model_id
            
        except Exception as e:
            logger.error(f"Error creando modelo: {e}")
            raise
    
    async def _save_model(self, model_id: str):
        """Guarda modelo"""
        try:
            model = self.models[model_id]
            
            # En implementación real, guardar modelo entrenado
            model_file = self.model_storage_path / f"{model_id}.pkl"
            
            # Simular guardado de modelo
            model_data = {
                "id": model.id,
                "name": model.name,
                "model_type": model.model_type.value,
                "version": model.version,
                "performance_metrics": model.performance_metrics,
                "training_data_size": model.training_data_size,
                "created_at": model.created_at.isoformat()
            }
            
            with open(model_file, 'wb') as f:
                pickle.dump(model_data, f)
            
        except Exception as e:
            logger.error(f"Error guardando modelo: {e}")
    
    async def deploy_model(self, model_id: str) -> bool:
        """Despliega modelo"""
        try:
            if model_id not in self.models:
                return False
            
            model = self.models[model_id]
            
            # Desactivar modelos activos del mismo tipo
            for other_model in self.models.values():
                if (other_model.model_type == model.model_type and 
                    other_model.is_active and other_model.id != model_id):
                    other_model.is_active = False
            
            # Activar nuevo modelo
            model.is_active = True
            model.deployment_info = {
                "deployed_at": datetime.now().isoformat(),
                "deployed_by": "system",
                "endpoint": f"/api/models/{model_id}/predict"
            }
            
            # Guardar datos
            await self._save_learning_data()
            
            logger.info(f"Modelo desplegado: {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error desplegando modelo: {e}")
            return False
    
    async def add_learning_feedback(
        self,
        model_id: str,
        user_id: str,
        feedback_type: str,
        feedback_data: Dict[str, Any],
        rating: float = 0.0
    ) -> str:
        """Agrega feedback de aprendizaje"""
        try:
            if model_id not in self.models:
                raise ValueError(f"Modelo no encontrado: {model_id}")
            
            feedback_id = f"feedback_{uuid.uuid4().hex[:8]}"
            
            feedback = LearningFeedback(
                id=feedback_id,
                model_id=model_id,
                user_id=user_id,
                feedback_type=feedback_type,
                feedback_data=feedback_data,
                rating=rating
            )
            
            self.learning_feedback.append(feedback)
            
            # Analizar feedback para mejora
            await self._analyze_feedback(feedback)
            
            logger.info(f"Feedback agregado: {feedback_id}")
            return feedback_id
            
        except Exception as e:
            logger.error(f"Error agregando feedback: {e}")
            raise
    
    async def _analyze_feedback(self, feedback: LearningFeedback):
        """Analiza feedback para mejora"""
        try:
            model = self.models[feedback.model_id]
            
            # Analizar rating
            if feedback.rating < 3.0:  # Rating bajo
                # Incrementar contador de feedback negativo
                if "negative_feedback_count" not in model.deployment_info:
                    model.deployment_info["negative_feedback_count"] = 0
                model.deployment_info["negative_feedback_count"] += 1
                
                # Si hay mucho feedback negativo, programar reentrenamiento
                if model.deployment_info["negative_feedback_count"] > 10:
                    await self._schedule_retraining(model.id)
            
            # Analizar tipo de feedback
            if feedback.feedback_type == "prediction_error":
                # Registrar error de predicción
                if "prediction_errors" not in model.deployment_info:
                    model.deployment_info["prediction_errors"] = []
                model.deployment_info["prediction_errors"].append({
                    "timestamp": feedback.timestamp.isoformat(),
                    "error_data": feedback.feedback_data
                })
            
        except Exception as e:
            logger.error(f"Error analizando feedback: {e}")
    
    async def _check_performance_degradation(self, model_id: str) -> float:
        """Verifica degradación de rendimiento"""
        try:
            if model_id not in self.learning_metrics:
                return 0.0
            
            metrics_history = self.learning_metrics[model_id]
            if len(metrics_history) < 2:
                return 0.0
            
            # Comparar métricas recientes con históricas
            recent_metrics = metrics_history[-1]
            historical_avg = np.mean([m.accuracy for m in metrics_history[:-1]])
            
            degradation = max(0.0, historical_avg - recent_metrics.accuracy)
            return degradation
            
        except Exception as e:
            logger.error(f"Error verificando degradación de rendimiento: {e}")
            return 0.0
    
    async def _schedule_retraining(self, model_id: str):
        """Programa reentrenamiento"""
        try:
            model = self.models[model_id]
            
            # Crear nueva tarea de reentrenamiento
            task_id = await self.create_learning_task(
                name=f"Retraining_{model.name}",
                description=f"Reentrenamiento automático de {model.name}",
                learning_type=LearningType.CONTINUOUS,
                model_type=model.model_type,
                dataset_id=model.learning_task_id,  # Usar mismo dataset
                hyperparameters={"retraining": True, "base_model": model_id}
            )
            
            # Iniciar reentrenamiento
            await self.start_learning_task(task_id)
            
            logger.info(f"Reentrenamiento programado para modelo: {model_id}")
            
        except Exception as e:
            logger.error(f"Error programando reentrenamiento: {e}")
    
    async def _evaluate_model_performance(self, model_id: str):
        """Evalúa rendimiento del modelo"""
        try:
            model = self.models[model_id]
            
            # Simular evaluación
            metrics = LearningMetrics(
                model_id=model_id,
                accuracy=np.random.uniform(0.8, 0.95),
                precision=np.random.uniform(0.75, 0.92),
                recall=np.random.uniform(0.78, 0.90),
                f1_score=np.random.uniform(0.76, 0.91),
                loss=np.random.uniform(0.05, 0.25),
                training_time=np.random.uniform(100, 500),
                inference_time=np.random.uniform(0.01, 0.1),
                data_quality_score=np.random.uniform(0.7, 0.95),
                model_complexity=np.random.uniform(0.3, 0.8)
            )
            
            self.learning_metrics[model_id].append(metrics)
            
            # Mantener solo las últimas métricas
            if len(self.learning_metrics[model_id]) > 100:
                self.learning_metrics[model_id] = self.learning_metrics[model_id][-50:]
            
        except Exception as e:
            logger.error(f"Error evaluando rendimiento del modelo: {e}")
    
    async def _save_learning_data(self):
        """Guarda datos de aprendizaje"""
        try:
            # Crear directorio de datos
            data_dir = Path("data")
            data_dir.mkdir(parents=True, exist_ok=True)
            
            # Guardar tareas de aprendizaje
            tasks_data = []
            for task in self.learning_tasks.values():
                tasks_data.append({
                    "id": task.id,
                    "name": task.name,
                    "description": task.description,
                    "learning_type": task.learning_type.value,
                    "model_type": task.model_type.value,
                    "dataset_id": task.dataset_id,
                    "hyperparameters": task.hyperparameters,
                    "status": task.status.value,
                    "progress": task.progress,
                    "metrics": task.metrics,
                    "created_at": task.created_at.isoformat(),
                    "started_at": task.started_at.isoformat() if task.started_at else None,
                    "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                    "error_message": task.error_message
                })
            
            tasks_file = data_dir / "learning_tasks.json"
            with open(tasks_file, 'w', encoding='utf-8') as f:
                json.dump(tasks_data, f, indent=2, ensure_ascii=False, default=str)
            
            # Guardar modelos
            models_data = []
            for model in self.models.values():
                models_data.append({
                    "id": model.id,
                    "name": model.name,
                    "model_type": model.model_type.value,
                    "version": model.version,
                    "learning_task_id": model.learning_task_id,
                    "performance_metrics": model.performance_metrics,
                    "training_data_size": model.training_data_size,
                    "created_at": model.created_at.isoformat(),
                    "is_active": model.is_active,
                    "deployment_info": model.deployment_info
                })
            
            models_file = data_dir / "learning_models.json"
            with open(models_file, 'w', encoding='utf-8') as f:
                json.dump(models_data, f, indent=2, ensure_ascii=False, default=str)
            
            # Guardar datasets
            datasets_data = []
            for dataset in self.datasets.values():
                datasets_data.append({
                    "id": dataset.id,
                    "name": dataset.name,
                    "description": dataset.description,
                    "data_type": dataset.data_type,
                    "size": dataset.size,
                    "features": dataset.features,
                    "labels": dataset.labels,
                    "created_at": dataset.created_at.isoformat(),
                    "updated_at": dataset.updated_at.isoformat(),
                    "metadata": dataset.metadata
                })
            
            datasets_file = data_dir / "learning_datasets.json"
            with open(datasets_file, 'w', encoding='utf-8') as f:
                json.dump(datasets_data, f, indent=2, ensure_ascii=False, default=str)
            
        except Exception as e:
            logger.error(f"Error guardando datos de aprendizaje: {e}")
    
    async def get_learning_dashboard_data(self) -> Dict[str, Any]:
        """Obtiene datos para dashboard de aprendizaje"""
        try:
            # Estadísticas generales
            total_tasks = len(self.learning_tasks)
            active_tasks = len([t for t in self.learning_tasks.values() if t.status == LearningStatus.TRAINING])
            completed_tasks = len([t for t in self.learning_tasks.values() if t.status == LearningStatus.COMPLETED])
            failed_tasks = len([t for t in self.learning_tasks.values() if t.status == LearningStatus.ERROR])
            
            total_models = len(self.models)
            active_models = len([m for m in self.models.values() if m.is_active])
            
            total_datasets = len(self.datasets)
            total_feedback = len(self.learning_feedback)
            
            # Distribución por tipo de aprendizaje
            learning_type_distribution = {}
            for task in self.learning_tasks.values():
                learning_type = task.learning_type.value
                learning_type_distribution[learning_type] = learning_type_distribution.get(learning_type, 0) + 1
            
            # Distribución por tipo de modelo
            model_type_distribution = {}
            for model in self.models.values():
                model_type = model.model_type.value
                model_type_distribution[model_type] = model_type_distribution.get(model_type, 0) + 1
            
            # Tareas recientes
            recent_tasks = sorted(
                self.learning_tasks.values(),
                key=lambda x: x.created_at,
                reverse=True
            )[:5]
            
            # Modelos con mejor rendimiento
            best_models = sorted(
                [m for m in self.models.values() if m.performance_metrics],
                key=lambda x: x.performance_metrics.get("accuracy", 0),
                reverse=True
            )[:5]
            
            return {
                "total_tasks": total_tasks,
                "active_tasks": active_tasks,
                "completed_tasks": completed_tasks,
                "failed_tasks": failed_tasks,
                "total_models": total_models,
                "active_models": active_models,
                "total_datasets": total_datasets,
                "total_feedback": total_feedback,
                "learning_type_distribution": learning_type_distribution,
                "model_type_distribution": model_type_distribution,
                "recent_tasks": [
                    {
                        "id": task.id,
                        "name": task.name,
                        "status": task.status.value,
                        "progress": task.progress,
                        "created_at": task.created_at.isoformat()
                    }
                    for task in recent_tasks
                ],
                "best_models": [
                    {
                        "id": model.id,
                        "name": model.name,
                        "model_type": model.model_type.value,
                        "accuracy": model.performance_metrics.get("accuracy", 0),
                        "is_active": model.is_active
                    }
                    for model in best_models
                ],
                "continuous_learning_active": self.continuous_learning_active,
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo datos del dashboard de aprendizaje: {e}")
            return {"error": str(e)}

