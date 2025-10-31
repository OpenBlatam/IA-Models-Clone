"""
Motor de Optimización AI
========================

Motor para optimización automática de modelos de IA y mejora de rendimiento.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import numpy as np
from pathlib import Path
import pickle
import hashlib

logger = logging.getLogger(__name__)

class OptimizationType(str, Enum):
    """Tipos de optimización"""
    HYPERPARAMETER = "hyperparameter"
    MODEL_ARCHITECTURE = "model_architecture"
    FEATURE_SELECTION = "feature_selection"
    ENSEMBLE = "ensemble"
    QUANTIZATION = "quantization"
    PRUNING = "pruning"

class OptimizationStatus(str, Enum):
    """Estados de optimización"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class OptimizationResult:
    """Resultado de optimización"""
    optimization_id: str
    optimization_type: OptimizationType
    status: OptimizationStatus
    best_score: float
    best_parameters: Dict[str, Any]
    optimization_history: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    error: Optional[str] = None

@dataclass
class ModelPerformance:
    """Rendimiento de modelo"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    inference_time: float
    memory_usage: float
    model_size: float

@dataclass
class OptimizationConfig:
    """Configuración de optimización"""
    optimization_type: OptimizationType
    target_metric: str
    max_iterations: int
    timeout_seconds: int
    early_stopping_patience: int
    parameters_space: Dict[str, Any]
    validation_split: float = 0.2
    cross_validation_folds: int = 5

class AIOptimizationEngine:
    """Motor de optimización de IA"""
    
    def __init__(self):
        self.optimization_results: Dict[str, OptimizationResult] = {}
        self.running_optimizations: Dict[str, asyncio.Task] = {}
        self.model_cache: Dict[str, Any] = {}
        self.performance_history: List[Dict[str, Any]] = []
        
        # Configuraciones por defecto
        self.default_configs = {
            OptimizationType.HYPERPARAMETER: {
                "max_iterations": 100,
                "timeout_seconds": 3600,
                "early_stopping_patience": 10
            },
            OptimizationType.MODEL_ARCHITECTURE: {
                "max_iterations": 50,
                "timeout_seconds": 7200,
                "early_stopping_patience": 5
            },
            OptimizationType.FEATURE_SELECTION: {
                "max_iterations": 200,
                "timeout_seconds": 1800,
                "early_stopping_patience": 15
            }
        }
        
    async def initialize(self):
        """Inicializa el motor de optimización"""
        logger.info("Inicializando motor de optimización AI...")
        
        # Cargar resultados previos
        await self._load_optimization_history()
        
        # Inicializar optimizadores
        await self._initialize_optimizers()
        
        logger.info("Motor de optimización AI inicializado")
    
    async def _load_optimization_history(self):
        """Carga historial de optimizaciones"""
        try:
            history_file = Path("data/optimization_history.json")
            if history_file.exists():
                with open(history_file, 'r', encoding='utf-8') as f:
                    history_data = json.load(f)
                
                for result_data in history_data:
                    result = OptimizationResult(
                        optimization_id=result_data["optimization_id"],
                        optimization_type=OptimizationType(result_data["optimization_type"]),
                        status=OptimizationStatus(result_data["status"]),
                        best_score=result_data["best_score"],
                        best_parameters=result_data["best_parameters"],
                        optimization_history=result_data.get("optimization_history", []),
                        created_at=datetime.fromisoformat(result_data["created_at"]),
                        completed_at=datetime.fromisoformat(result_data["completed_at"]) if result_data.get("completed_at") else None,
                        error=result_data.get("error")
                    )
                    self.optimization_results[result.optimization_id] = result
                
                logger.info(f"Cargado historial de {len(self.optimization_results)} optimizaciones")
                
        except Exception as e:
            logger.error(f"Error cargando historial de optimizaciones: {e}")
    
    async def _initialize_optimizers(self):
        """Inicializa optimizadores"""
        try:
            # Importar librerías de optimización
            try:
                import optuna
                self.optuna_available = True
                logger.info("✅ Optuna disponible para optimización")
            except ImportError:
                self.optuna_available = False
                logger.warning("⚠️ Optuna no disponible")
            
            try:
                from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
                self.sklearn_available = True
                logger.info("✅ Scikit-learn disponible para optimización")
            except ImportError:
                self.sklearn_available = False
                logger.warning("⚠️ Scikit-learn no disponible")
            
        except Exception as e:
            logger.error(f"Error inicializando optimizadores: {e}")
    
    async def optimize_classification_model(
        self,
        model_type: str,
        training_data: List[Dict[str, Any]],
        config: Optional[OptimizationConfig] = None
    ) -> str:
        """Optimiza modelo de clasificación"""
        try:
            optimization_id = f"classification_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            if not config:
                config = OptimizationConfig(
                    optimization_type=OptimizationType.HYPERPARAMETER,
                    target_metric="f1_score",
                    max_iterations=100,
                    timeout_seconds=3600,
                    early_stopping_patience=10,
                    parameters_space=self._get_default_parameters_space(model_type)
                )
            
            # Crear resultado de optimización
            result = OptimizationResult(
                optimization_id=optimization_id,
                optimization_type=config.optimization_type,
                status=OptimizationStatus.PENDING
            )
            
            self.optimization_results[optimization_id] = result
            
            # Ejecutar optimización
            task = asyncio.create_task(
                self._run_classification_optimization(optimization_id, model_type, training_data, config)
            )
            self.running_optimizations[optimization_id] = task
            
            logger.info(f"Optimización de clasificación iniciada: {optimization_id}")
            return optimization_id
            
        except Exception as e:
            logger.error(f"Error iniciando optimización de clasificación: {e}")
            raise
    
    async def _run_classification_optimization(
        self,
        optimization_id: str,
        model_type: str,
        training_data: List[Dict[str, Any]],
        config: OptimizationConfig
    ):
        """Ejecuta optimización de clasificación"""
        try:
            result = self.optimization_results[optimization_id]
            result.status = OptimizationStatus.RUNNING
            
            # Preparar datos
            X, y = self._prepare_classification_data(training_data)
            
            # Ejecutar optimización según tipo
            if config.optimization_type == OptimizationType.HYPERPARAMETER:
                best_score, best_params, history = await self._optimize_hyperparameters(
                    model_type, X, y, config
                )
            elif config.optimization_type == OptimizationType.FEATURE_SELECTION:
                best_score, best_params, history = await self._optimize_feature_selection(
                    model_type, X, y, config
                )
            else:
                raise ValueError(f"Tipo de optimización no soportado: {config.optimization_type}")
            
            # Actualizar resultado
            result.best_score = best_score
            result.best_parameters = best_params
            result.optimization_history = history
            result.status = OptimizationStatus.COMPLETED
            result.completed_at = datetime.now()
            
            # Guardar historial
            await self._save_optimization_history()
            
            logger.info(f"Optimización completada: {optimization_id} (score: {best_score:.4f})")
            
        except Exception as e:
            logger.error(f"Error en optimización {optimization_id}: {e}")
            result = self.optimization_results[optimization_id]
            result.status = OptimizationStatus.FAILED
            result.error = str(e)
            result.completed_at = datetime.now()
    
    def _prepare_classification_data(self, training_data: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepara datos para clasificación"""
        try:
            # Extraer características y etiquetas
            X = []
            y = []
            
            for item in training_data:
                # Extraer características del texto
                text = item.get("text", "")
                features = self._extract_text_features(text)
                X.append(features)
                
                # Extraer etiqueta
                label = item.get("label", item.get("category", "unknown"))
                y.append(label)
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            logger.error(f"Error preparando datos de clasificación: {e}")
            raise
    
    def _extract_text_features(self, text: str) -> List[float]:
        """Extrae características del texto"""
        try:
            features = []
            
            # Características básicas
            features.append(len(text))  # Longitud del texto
            features.append(len(text.split()))  # Número de palabras
            features.append(len(text.split('.')))  # Número de oraciones
            
            # Características de complejidad
            avg_word_length = sum(len(word) for word in text.split()) / max(len(text.split()), 1)
            features.append(avg_word_length)
            
            # Características de puntuación
            features.append(text.count('.'))
            features.append(text.count(','))
            features.append(text.count('!'))
            features.append(text.count('?'))
            
            # Características de formato
            features.append(text.count('\n'))
            features.append(text.count('\t'))
            
            # Características de mayúsculas
            features.append(sum(1 for c in text if c.isupper()))
            features.append(sum(1 for c in text if c.islower()))
            
            return features
            
        except Exception as e:
            logger.error(f"Error extrayendo características: {e}")
            return [0.0] * 12  # Retornar características por defecto
    
    async def _optimize_hyperparameters(
        self,
        model_type: str,
        X: np.ndarray,
        y: np.ndarray,
        config: OptimizationConfig
    ) -> Tuple[float, Dict[str, Any], List[Dict[str, Any]]]:
        """Optimiza hiperparámetros"""
        try:
            if self.optuna_available:
                return await self._optimize_with_optuna(model_type, X, y, config)
            elif self.sklearn_available:
                return await self._optimize_with_sklearn(model_type, X, y, config)
            else:
                return await self._optimize_basic(model_type, X, y, config)
                
        except Exception as e:
            logger.error(f"Error optimizando hiperparámetros: {e}")
            raise
    
    async def _optimize_with_optuna(
        self,
        model_type: str,
        X: np.ndarray,
        y: np.ndarray,
        config: OptimizationConfig
    ) -> Tuple[float, Dict[str, Any], List[Dict[str, Any]]]:
        """Optimiza usando Optuna"""
        try:
            import optuna
            from sklearn.model_selection import cross_val_score
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.svm import SVC
            from sklearn.linear_model import LogisticRegression
            
            def objective(trial):
                # Definir espacio de búsqueda según tipo de modelo
                if model_type == "random_forest":
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 10, 200),
                        'max_depth': trial.suggest_int('max_depth', 3, 20),
                        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
                    }
                    model = RandomForestClassifier(**params)
                elif model_type == "svm":
                    params = {
                        'C': trial.suggest_float('C', 0.1, 10.0),
                        'gamma': trial.suggest_float('gamma', 0.001, 1.0),
                        'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly'])
                    }
                    model = SVC(**params)
                elif model_type == "logistic_regression":
                    params = {
                        'C': trial.suggest_float('C', 0.1, 10.0),
                        'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
                        'solver': trial.suggest_categorical('solver', ['liblinear', 'saga'])
                    }
                    model = LogisticRegression(**params)
                else:
                    raise ValueError(f"Tipo de modelo no soportado: {model_type}")
                
                # Evaluar modelo
                scores = cross_val_score(model, X, y, cv=config.cross_validation_folds, scoring='f1_macro')
                return scores.mean()
            
            # Crear estudio
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=config.max_iterations)
            
            best_score = study.best_value
            best_params = study.best_params
            
            # Crear historial
            history = []
            for trial in study.trials:
                history.append({
                    "trial": trial.number,
                    "value": trial.value,
                    "params": trial.params,
                    "state": trial.state.name
                })
            
            return best_score, best_params, history
            
        except Exception as e:
            logger.error(f"Error en optimización con Optuna: {e}")
            raise
    
    async def _optimize_with_sklearn(
        self,
        model_type: str,
        X: np.ndarray,
        y: np.ndarray,
        config: OptimizationConfig
    ) -> Tuple[float, Dict[str, Any], List[Dict[str, Any]]]:
        """Optimiza usando scikit-learn"""
        try:
            from sklearn.model_selection import RandomizedSearchCV
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.svm import SVC
            from sklearn.linear_model import LogisticRegression
            
            # Definir modelo base
            if model_type == "random_forest":
                base_model = RandomForestClassifier()
                param_distributions = {
                    'n_estimators': [10, 50, 100, 200],
                    'max_depth': [3, 5, 10, 20, None],
                    'min_samples_split': [2, 5, 10, 20],
                    'min_samples_leaf': [1, 2, 4, 8]
                }
            elif model_type == "svm":
                base_model = SVC()
                param_distributions = {
                    'C': [0.1, 1, 10, 100],
                    'gamma': [0.001, 0.01, 0.1, 1],
                    'kernel': ['rbf', 'linear', 'poly']
                }
            elif model_type == "logistic_regression":
                base_model = LogisticRegression()
                param_distributions = {
                    'C': [0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            else:
                raise ValueError(f"Tipo de modelo no soportado: {model_type}")
            
            # Ejecutar búsqueda aleatoria
            random_search = RandomizedSearchCV(
                base_model,
                param_distributions,
                n_iter=config.max_iterations,
                cv=config.cross_validation_folds,
                scoring='f1_macro',
                random_state=42
            )
            
            random_search.fit(X, y)
            
            best_score = random_search.best_score_
            best_params = random_search.best_params_
            
            # Crear historial
            history = []
            for i, (params, score) in enumerate(zip(random_search.cv_results_['params'], random_search.cv_results_['mean_test_score'])):
                history.append({
                    "trial": i,
                    "value": score,
                    "params": params,
                    "state": "COMPLETE"
                })
            
            return best_score, best_params, history
            
        except Exception as e:
            logger.error(f"Error en optimización con scikit-learn: {e}")
            raise
    
    async def _optimize_basic(
        self,
        model_type: str,
        X: np.ndarray,
        y: np.ndarray,
        config: OptimizationConfig
    ) -> Tuple[float, Dict[str, Any], List[Dict[str, Any]]]:
        """Optimización básica sin librerías externas"""
        try:
            # Implementación básica de búsqueda en cuadrícula
            best_score = 0.0
            best_params = {}
            history = []
            
            # Parámetros básicos para probar
            if model_type == "random_forest":
                param_grid = {
                    'n_estimators': [10, 50, 100],
                    'max_depth': [3, 5, 10],
                    'min_samples_split': [2, 5, 10]
                }
            else:
                param_grid = {
                    'C': [0.1, 1, 10],
                    'penalty': ['l1', 'l2']
                }
            
            # Búsqueda básica
            trial_count = 0
            for params in self._generate_param_combinations(param_grid):
                if trial_count >= config.max_iterations:
                    break
                
                # Simular evaluación (en implementación real, entrenar modelo)
                score = np.random.random()  # Placeholder
                
                history.append({
                    "trial": trial_count,
                    "value": score,
                    "params": params,
                    "state": "COMPLETE"
                })
                
                if score > best_score:
                    best_score = score
                    best_params = params
                
                trial_count += 1
            
            return best_score, best_params, history
            
        except Exception as e:
            logger.error(f"Error en optimización básica: {e}")
            raise
    
    def _generate_param_combinations(self, param_grid: Dict[str, List]) -> List[Dict[str, Any]]:
        """Genera combinaciones de parámetros"""
        import itertools
        
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        combinations = []
        for combination in itertools.product(*values):
            param_dict = dict(zip(keys, combination))
            combinations.append(param_dict)
        
        return combinations
    
    async def _optimize_feature_selection(
        self,
        model_type: str,
        X: np.ndarray,
        y: np.ndarray,
        config: OptimizationConfig
    ) -> Tuple[float, Dict[str, Any], List[Dict[str, Any]]]:
        """Optimiza selección de características"""
        try:
            # Implementación básica de selección de características
            best_score = 0.0
            best_params = {}
            history = []
            
            # Probar diferentes números de características
            n_features = X.shape[1]
            for n_select in range(1, min(n_features + 1, 20)):
                # Simular selección de características
                selected_features = list(range(n_select))
                
                # Simular evaluación
                score = np.random.random()  # Placeholder
                
                history.append({
                    "trial": len(history),
                    "value": score,
                    "params": {"n_features": n_select, "selected_features": selected_features},
                    "state": "COMPLETE"
                })
                
                if score > best_score:
                    best_score = score
                    best_params = {
                        "n_features": n_select,
                        "selected_features": selected_features
                    }
            
            return best_score, best_params, history
            
        except Exception as e:
            logger.error(f"Error en optimización de características: {e}")
            raise
    
    def _get_default_parameters_space(self, model_type: str) -> Dict[str, Any]:
        """Obtiene espacio de parámetros por defecto"""
        spaces = {
            "random_forest": {
                "n_estimators": [10, 50, 100, 200],
                "max_depth": [3, 5, 10, 20, None],
                "min_samples_split": [2, 5, 10, 20],
                "min_samples_leaf": [1, 2, 4, 8]
            },
            "svm": {
                "C": [0.1, 1, 10, 100],
                "gamma": [0.001, 0.01, 0.1, 1],
                "kernel": ["rbf", "linear", "poly"]
            },
            "logistic_regression": {
                "C": [0.1, 1, 10, 100],
                "penalty": ["l1", "l2"],
                "solver": ["liblinear", "saga"]
            }
        }
        
        return spaces.get(model_type, {})
    
    async def _save_optimization_history(self):
        """Guarda historial de optimizaciones"""
        try:
            # Crear directorio de datos
            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)
            
            # Convertir resultados a formato serializable
            history_data = []
            for result in self.optimization_results.values():
                history_data.append({
                    "optimization_id": result.optimization_id,
                    "optimization_type": result.optimization_type.value,
                    "status": result.status.value,
                    "best_score": result.best_score,
                    "best_parameters": result.best_parameters,
                    "optimization_history": result.optimization_history,
                    "created_at": result.created_at.isoformat(),
                    "completed_at": result.completed_at.isoformat() if result.completed_at else None,
                    "error": result.error
                })
            
            # Guardar archivo
            history_file = data_dir / "optimization_history.json"
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"Error guardando historial de optimizaciones: {e}")
    
    async def get_optimization_status(self, optimization_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene estado de optimización"""
        try:
            if optimization_id not in self.optimization_results:
                return None
            
            result = self.optimization_results[optimization_id]
            
            return {
                "optimization_id": result.optimization_id,
                "optimization_type": result.optimization_type.value,
                "status": result.status.value,
                "best_score": result.best_score,
                "best_parameters": result.best_parameters,
                "created_at": result.created_at.isoformat(),
                "completed_at": result.completed_at.isoformat() if result.completed_at else None,
                "error": result.error,
                "history_length": len(result.optimization_history)
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo estado de optimización: {e}")
            return None
    
    async def get_optimization_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Obtiene historial de optimizaciones"""
        try:
            results = list(self.optimization_results.values())
            results.sort(key=lambda x: x.created_at, reverse=True)
            
            return [
                {
                    "optimization_id": result.optimization_id,
                    "optimization_type": result.optimization_type.value,
                    "status": result.status.value,
                    "best_score": result.best_score,
                    "created_at": result.created_at.isoformat(),
                    "completed_at": result.completed_at.isoformat() if result.completed_at else None,
                    "duration_seconds": (
                        (result.completed_at - result.created_at).total_seconds()
                        if result.completed_at else None
                    )
                }
                for result in results[:limit]
            ]
            
        except Exception as e:
            logger.error(f"Error obteniendo historial de optimizaciones: {e}")
            return []
    
    async def cancel_optimization(self, optimization_id: str) -> bool:
        """Cancela optimización en curso"""
        try:
            if optimization_id not in self.optimization_results:
                return False
            
            result = self.optimization_results[optimization_id]
            result.status = OptimizationStatus.CANCELLED
            result.completed_at = datetime.now()
            
            # Cancelar tarea si está corriendo
            if optimization_id in self.running_optimizations:
                self.running_optimizations[optimization_id].cancel()
                del self.running_optimizations[optimization_id]
            
            logger.info(f"Optimización cancelada: {optimization_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error cancelando optimización: {e}")
            return False
    
    async def get_model_recommendations(self, task_type: str, data_size: int) -> Dict[str, Any]:
        """Obtiene recomendaciones de modelos"""
        try:
            recommendations = {
                "task_type": task_type,
                "data_size": data_size,
                "recommended_models": [],
                "optimization_suggestions": []
            }
            
            # Recomendaciones basadas en tamaño de datos y tipo de tarea
            if task_type == "classification":
                if data_size < 1000:
                    recommendations["recommended_models"] = [
                        {"name": "Logistic Regression", "reason": "Eficiente para datasets pequeños"},
                        {"name": "Naive Bayes", "reason": "Rápido y efectivo para texto"}
                    ]
                elif data_size < 10000:
                    recommendations["recommended_models"] = [
                        {"name": "Random Forest", "reason": "Buen balance entre rendimiento y velocidad"},
                        {"name": "SVM", "reason": "Excelente para clasificación de texto"}
                    ]
                else:
                    recommendations["recommended_models"] = [
                        {"name": "Gradient Boosting", "reason": "Alto rendimiento en datasets grandes"},
                        {"name": "Neural Network", "reason": "Capacidad de aprendizaje complejo"}
                    ]
            
            # Sugerencias de optimización
            recommendations["optimization_suggestions"] = [
                "Usar validación cruzada para evaluación robusta",
                "Aplicar selección de características para reducir dimensionalidad",
                "Considerar ensemble methods para mejorar rendimiento",
                "Optimizar hiperparámetros con búsqueda automática"
            ]
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error obteniendo recomendaciones: {e}")
            return {"error": str(e)}


