"""
AI Model Optimization and Learning System for AI History Comparison
Sistema de optimización y aprendizaje automático para análisis de historial de IA
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import pickle
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import xgboost as xgb
import lightgbm as lgb

# Deep Learning
try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Tipos de modelos ML"""
    LINEAR_REGRESSION = "linear_regression"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    SVM = "svm"
    NEURAL_NETWORK = "neural_network"
    DEEP_LEARNING = "deep_learning"
    ENSEMBLE = "ensemble"

class OptimizationGoal(Enum):
    """Objetivos de optimización"""
    MAXIMIZE_QUALITY = "maximize_quality"
    MINIMIZE_COST = "minimize_cost"
    MAXIMIZE_SPEED = "maximize_speed"
    BALANCE_ALL = "balance_all"
    CUSTOM = "custom"

class LearningMode(Enum):
    """Modos de aprendizaje"""
    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"
    REINFORCEMENT = "reinforcement"
    TRANSFER = "transfer"
    CONTINUOUS = "continuous"

@dataclass
class ModelPerformance:
    """Rendimiento del modelo"""
    model_id: str
    model_type: ModelType
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    mse: float
    r2_score: float
    training_time: float
    prediction_time: float
    memory_usage: float
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class OptimizationResult:
    """Resultado de optimización"""
    id: str
    goal: OptimizationGoal
    best_model: str
    best_parameters: Dict[str, Any]
    performance_improvement: float
    cost_reduction: float
    speed_improvement: float
    quality_improvement: float
    recommendations: List[str]
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class LearningInsight:
    """Insight de aprendizaje"""
    id: str
    insight_type: str
    description: str
    confidence: float
    impact_score: float
    actionable_recommendations: List[str]
    related_patterns: List[str]
    created_at: datetime = field(default_factory=datetime.now)

class AIOptimizer:
    """
    Optimizador de IA y sistema de aprendizaje automático
    """
    
    def __init__(
        self,
        enable_deep_learning: bool = True,
        enable_auto_ml: bool = True,
        enable_continuous_learning: bool = True,
        models_directory: str = "models/ai_optimizer/"
    ):
        self.enable_deep_learning = enable_deep_learning and TENSORFLOW_AVAILABLE
        self.enable_auto_ml = enable_auto_ml
        self.enable_continuous_learning = enable_continuous_learning
        self.models_directory = Path(models_directory)
        self.models_directory.mkdir(parents=True, exist_ok=True)
        
        # Almacenamiento de modelos
        self.trained_models: Dict[str, Any] = {}
        self.model_performances: Dict[str, ModelPerformance] = {}
        self.optimization_results: Dict[str, OptimizationResult] = {}
        self.learning_insights: Dict[str, LearningInsight] = {}
        
        # Datos de entrenamiento
        self.training_data: Optional[pd.DataFrame] = None
        self.feature_columns: List[str] = []
        self.target_columns: List[str] = []
        
        # Configuración
        self.config = {
            "test_size": 0.2,
            "random_state": 42,
            "cv_folds": 5,
            "max_iterations": 1000,
            "early_stopping_rounds": 50,
            "learning_rate": 0.01,
            "batch_size": 32,
            "epochs": 100
        }
        
        # Inicializar modelos base
        self._initialize_base_models()
        
        # Configurar TensorFlow si está disponible
        if self.enable_deep_learning:
            self._configure_tensorflow()
    
    def _initialize_base_models(self):
        """Inicializar modelos base"""
        self.base_models = {
            ModelType.LINEAR_REGRESSION: LinearRegression(),
            ModelType.RIDGE: Ridge(alpha=1.0),
            ModelType.LASSO: Lasso(alpha=1.0),
            ModelType.RANDOM_FOREST: RandomForestRegressor(n_estimators=100, random_state=42),
            ModelType.GRADIENT_BOOSTING: GradientBoostingRegressor(n_estimators=100, random_state=42),
            ModelType.XGBOOST: xgb.XGBRegressor(n_estimators=100, random_state=42),
            ModelType.LIGHTGBM: lgb.LGBMRegressor(n_estimators=100, random_state=42),
            ModelType.SVM: SVR(kernel='rbf'),
            ModelType.NEURAL_NETWORK: MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
        }
    
    def _configure_tensorflow(self):
        """Configurar TensorFlow"""
        if TENSORFLOW_AVAILABLE:
            # Configurar para usar CPU si no hay GPU
            tf.config.set_visible_devices([], 'GPU')
            logger.info("TensorFlow configured for CPU usage")
    
    async def prepare_training_data(
        self,
        data: pd.DataFrame,
        feature_columns: List[str],
        target_columns: List[str],
        preprocess: bool = True
    ) -> bool:
        """
        Preparar datos de entrenamiento
        
        Args:
            data: DataFrame con los datos
            feature_columns: Columnas de características
            target_columns: Columnas objetivo
            preprocess: Si preprocesar los datos
            
        Returns:
            True si se prepararon exitosamente
        """
        try:
            logger.info("Preparing training data")
            
            self.training_data = data.copy()
            self.feature_columns = feature_columns
            self.target_columns = target_columns
            
            if preprocess:
                await self._preprocess_data()
            
            logger.info(f"Training data prepared: {len(self.training_data)} samples, {len(feature_columns)} features")
            return True
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return False
    
    async def _preprocess_data(self):
        """Preprocesar datos"""
        try:
            # Manejar valores faltantes
            self.training_data = self.training_data.fillna(self.training_data.mean())
            
            # Codificar variables categóricas
            for column in self.feature_columns:
                if self.training_data[column].dtype == 'object':
                    le = LabelEncoder()
                    self.training_data[column] = le.fit_transform(self.training_data[column].astype(str))
            
            # Normalizar características
            self.scaler = StandardScaler()
            self.training_data[self.feature_columns] = self.scaler.fit_transform(
                self.training_data[self.feature_columns]
            )
            
            logger.info("Data preprocessing completed")
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
    
    async def train_model(
        self,
        model_type: ModelType,
        model_id: str = None,
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> ModelPerformance:
        """
        Entrenar un modelo
        
        Args:
            model_type: Tipo de modelo a entrenar
            model_id: ID único del modelo
            hyperparameters: Hiperparámetros personalizados
            
        Returns:
            Rendimiento del modelo entrenado
        """
        try:
            if model_id is None:
                model_id = f"{model_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            logger.info(f"Training model: {model_type.value}")
            
            # Preparar datos
            X = self.training_data[self.feature_columns]
            y = self.training_data[self.target_columns[0]]  # Usar primera columna objetivo
            
            # Dividir datos
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config["test_size"], random_state=self.config["random_state"]
            )
            
            # Crear modelo
            if model_type == ModelType.DEEP_LEARNING and self.enable_deep_learning:
                model = await self._create_deep_learning_model(X_train.shape[1])
            else:
                model = self.base_models[model_type]
                if hyperparameters:
                    model.set_params(**hyperparameters)
            
            # Entrenar modelo
            start_time = time.time()
            
            if model_type == ModelType.DEEP_LEARNING and self.enable_deep_learning:
                await self._train_deep_learning_model(model, X_train, y_train, X_test, y_test)
            else:
                model.fit(X_train, y_train)
            
            training_time = time.time() - start_time
            
            # Evaluar modelo
            start_time = time.time()
            y_pred = model.predict(X_test)
            prediction_time = time.time() - start_time
            
            # Calcular métricas
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            # Crear objeto de rendimiento
            performance = ModelPerformance(
                model_id=model_id,
                model_type=model_type,
                accuracy=r2,  # Usar R² como accuracy para regresión
                precision=0.0,  # No aplicable para regresión
                recall=0.0,  # No aplicable para regresión
                f1_score=0.0,  # No aplicable para regresión
                mse=mse,
                r2_score=r2,
                training_time=training_time,
                prediction_time=prediction_time,
                memory_usage=self._estimate_memory_usage(model)
            )
            
            # Almacenar modelo y rendimiento
            self.trained_models[model_id] = model
            self.model_performances[model_id] = performance
            
            # Guardar modelo
            await self._save_model(model, model_id)
            
            logger.info(f"Model {model_id} trained successfully. R²: {r2:.4f}")
            return performance
            
        except Exception as e:
            logger.error(f"Error training model {model_type.value}: {e}")
            raise
    
    async def _create_deep_learning_model(self, input_dim: int) -> Any:
        """Crear modelo de deep learning"""
        if not self.enable_deep_learning:
            raise ValueError("Deep learning not available")
        
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config["learning_rate"]),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    async def _train_deep_learning_model(self, model: Any, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
        """Entrenar modelo de deep learning"""
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config["early_stopping_rounds"],
                restore_best_weights=True
            )
        ]
        
        model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=self.config["epochs"],
            batch_size=self.config["batch_size"],
            callbacks=callbacks,
            verbose=0
        )
    
    def _estimate_memory_usage(self, model: Any) -> float:
        """Estimar uso de memoria del modelo"""
        try:
            import sys
            return sys.getsizeof(model) / 1024 / 1024  # MB
        except:
            return 0.0
    
    async def _save_model(self, model: Any, model_id: str):
        """Guardar modelo"""
        try:
            model_path = self.models_directory / f"{model_id}.joblib"
            
            if hasattr(model, 'save'):  # Modelo de TensorFlow
                model.save(str(model_path))
            else:
                joblib.dump(model, model_path)
            
            logger.info(f"Model {model_id} saved to {model_path}")
            
        except Exception as e:
            logger.error(f"Error saving model {model_id}: {e}")
    
    async def optimize_models(
        self,
        goal: OptimizationGoal,
        models_to_optimize: List[ModelType] = None
    ) -> OptimizationResult:
        """
        Optimizar modelos para un objetivo específico
        
        Args:
            goal: Objetivo de optimización
            models_to_optimize: Modelos a optimizar
            
        Returns:
            Resultado de optimización
        """
        try:
            logger.info(f"Starting model optimization for goal: {goal.value}")
            
            if models_to_optimize is None:
                models_to_optimize = list(ModelType)
            
            best_model_id = None
            best_performance = -float('inf')
            best_parameters = {}
            
            # Optimizar cada modelo
            for model_type in models_to_optimize:
                if model_type in self.base_models:
                    optimized_model_id = await self._optimize_single_model(model_type, goal)
                    
                    if optimized_model_id and optimized_model_id in self.model_performances:
                        performance = self.model_performances[optimized_model_id]
                        
                        # Evaluar según el objetivo
                        score = self._evaluate_optimization_goal(performance, goal)
                        
                        if score > best_performance:
                            best_performance = score
                            best_model_id = optimized_model_id
                            best_parameters = self._get_model_parameters(optimized_model_id)
            
            # Calcular mejoras
            improvements = await self._calculate_improvements(best_model_id, goal)
            
            # Generar recomendaciones
            recommendations = await self._generate_optimization_recommendations(best_model_id, goal)
            
            # Crear resultado de optimización
            result = OptimizationResult(
                id=f"optimization_{goal.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                goal=goal,
                best_model=best_model_id or "none",
                best_parameters=best_parameters,
                performance_improvement=improvements.get("performance", 0.0),
                cost_reduction=improvements.get("cost", 0.0),
                speed_improvement=improvements.get("speed", 0.0),
                quality_improvement=improvements.get("quality", 0.0),
                recommendations=recommendations
            )
            
            self.optimization_results[result.id] = result
            
            logger.info(f"Optimization completed. Best model: {best_model_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error in model optimization: {e}")
            raise
    
    async def _optimize_single_model(self, model_type: ModelType, goal: OptimizationGoal) -> Optional[str]:
        """Optimizar un modelo individual"""
        try:
            # Definir grid de hiperparámetros según el tipo de modelo
            param_grid = self._get_hyperparameter_grid(model_type, goal)
            
            if not param_grid:
                # Entrenar con parámetros por defecto
                return await self.train_model(model_type)
            
            # Usar GridSearchCV para optimización
            X = self.training_data[self.feature_columns]
            y = self.training_data[self.target_columns[0]]
            
            base_model = self.base_models[model_type]
            
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=self.config["cv_folds"],
                scoring='r2',
                n_jobs=-1
            )
            
            grid_search.fit(X, y)
            
            # Entrenar modelo con mejores parámetros
            model_id = f"{model_type.value}_optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            optimized_model = self.base_models[model_type]
            optimized_model.set_params(**grid_search.best_params_)
            optimized_model.fit(X, y)
            
            # Evaluar modelo optimizado
            y_pred = optimized_model.predict(X)
            r2 = r2_score(y, y_pred)
            
            # Crear rendimiento
            performance = ModelPerformance(
                model_id=model_id,
                model_type=model_type,
                accuracy=r2,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                mse=mean_squared_error(y, y_pred),
                r2_score=r2,
                training_time=0.0,
                prediction_time=0.0,
                memory_usage=self._estimate_memory_usage(optimized_model)
            )
            
            self.trained_models[model_id] = optimized_model
            self.model_performances[model_id] = performance
            
            return model_id
            
        except Exception as e:
            logger.error(f"Error optimizing model {model_type.value}: {e}")
            return None
    
    def _get_hyperparameter_grid(self, model_type: ModelType, goal: OptimizationGoal) -> Dict[str, List]:
        """Obtener grid de hiperparámetros"""
        grids = {
            ModelType.RANDOM_FOREST: {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            },
            ModelType.GRADIENT_BOOSTING: {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            ModelType.XGBOOST: {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            ModelType.LIGHTGBM: {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            ModelType.SVM: {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto', 0.1, 1]
            },
            ModelType.RIDGE: {
                'alpha': [0.1, 1, 10, 100]
            },
            ModelType.LASSO: {
                'alpha': [0.1, 1, 10, 100]
            }
        }
        
        return grids.get(model_type, {})
    
    def _evaluate_optimization_goal(self, performance: ModelPerformance, goal: OptimizationGoal) -> float:
        """Evaluar rendimiento según el objetivo"""
        if goal == OptimizationGoal.MAXIMIZE_QUALITY:
            return performance.r2_score
        elif goal == OptimizationGoal.MINIMIZE_COST:
            return 1.0 / (performance.memory_usage + 1)  # Menor uso de memoria = mejor
        elif goal == OptimizationGoal.MAXIMIZE_SPEED:
            return 1.0 / (performance.prediction_time + 0.001)  # Menor tiempo = mejor
        elif goal == OptimizationGoal.BALANCE_ALL:
            # Combinación balanceada de métricas
            return (performance.r2_score * 0.4 + 
                   (1.0 / (performance.memory_usage + 1)) * 0.3 +
                   (1.0 / (performance.prediction_time + 0.001)) * 0.3)
        else:
            return performance.r2_score
    
    def _get_model_parameters(self, model_id: str) -> Dict[str, Any]:
        """Obtener parámetros del modelo"""
        if model_id in self.trained_models:
            model = self.trained_models[model_id]
            if hasattr(model, 'get_params'):
                return model.get_params()
        return {}
    
    async def _calculate_improvements(self, best_model_id: str, goal: OptimizationGoal) -> Dict[str, float]:
        """Calcular mejoras obtenidas"""
        if not best_model_id or best_model_id not in self.model_performances:
            return {"performance": 0.0, "cost": 0.0, "speed": 0.0, "quality": 0.0}
        
        best_performance = self.model_performances[best_model_id]
        
        # Calcular mejoras basadas en el objetivo
        improvements = {
            "performance": best_performance.r2_score,
            "cost": 1.0 / (best_performance.memory_usage + 1),
            "speed": 1.0 / (best_performance.prediction_time + 0.001),
            "quality": best_performance.r2_score
        }
        
        return improvements
    
    async def _generate_optimization_recommendations(self, best_model_id: str, goal: OptimizationGoal) -> List[str]:
        """Generar recomendaciones de optimización"""
        recommendations = []
        
        if not best_model_id or best_model_id not in self.model_performances:
            recommendations.append("No se encontró un modelo óptimo. Revisar datos de entrenamiento.")
            return recommendations
        
        best_performance = self.model_performances[best_model_id]
        
        if goal == OptimizationGoal.MAXIMIZE_QUALITY:
            recommendations.extend([
                f"Modelo {best_performance.model_type.value} alcanzó R² de {best_performance.r2_score:.3f}",
                "Considerar ensemble de modelos para mayor precisión",
                "Recolectar más datos de entrenamiento si es posible"
            ])
        elif goal == OptimizationGoal.MINIMIZE_COST:
            recommendations.extend([
                f"Modelo optimizado usa {best_performance.memory_usage:.2f} MB de memoria",
                "Considerar modelos más simples para reducir costos computacionales",
                "Implementar compresión de modelos si es necesario"
            ])
        elif goal == OptimizationGoal.MAXIMIZE_SPEED:
            recommendations.extend([
                f"Tiempo de predicción: {best_performance.prediction_time:.4f} segundos",
                "Considerar modelos lineales para máxima velocidad",
                "Implementar cache de predicciones para consultas repetidas"
            ])
        else:
            recommendations.extend([
                f"Modelo balanceado con R² de {best_performance.r2_score:.3f}",
                f"Uso de memoria: {best_performance.memory_usage:.2f} MB",
                f"Tiempo de predicción: {best_performance.prediction_time:.4f} segundos"
            ])
        
        return recommendations
    
    async def generate_learning_insights(self) -> List[LearningInsight]:
        """Generar insights de aprendizaje automático"""
        insights = []
        
        try:
            # Insight 1: Mejor modelo por rendimiento
            if self.model_performances:
                best_model = max(self.model_performances.values(), key=lambda x: x.r2_score)
                
                insight = LearningInsight(
                    id=f"best_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    insight_type="model_performance",
                    description=f"El modelo {best_model.model_type.value} muestra el mejor rendimiento con R² de {best_model.r2_score:.3f}",
                    confidence=0.9,
                    impact_score=0.8,
                    actionable_recommendations=[
                        f"Usar {best_model.model_type.value} para predicciones de alta precisión",
                        "Considerar este modelo como baseline para futuras comparaciones"
                    ],
                    related_patterns=[best_model.model_type.value]
                )
                insights.append(insight)
            
            # Insight 2: Análisis de características importantes
            if self.training_data is not None and len(self.trained_models) > 0:
                feature_importance = await self._analyze_feature_importance()
                
                if feature_importance:
                    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                    
                    insight = LearningInsight(
                        id=f"feature_importance_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        insight_type="feature_analysis",
                        description=f"Las características más importantes son: {', '.join([f[0] for f in top_features])}",
                        confidence=0.8,
                        impact_score=0.7,
                        actionable_recommendations=[
                            "Enfocar recolección de datos en características más importantes",
                            "Considerar eliminar características de baja importancia",
                            "Priorizar calidad de datos para características top"
                        ],
                        related_patterns=[f[0] for f in top_features]
                    )
                    insights.append(insight)
            
            # Insight 3: Recomendaciones de optimización
            if self.optimization_results:
                latest_optimization = max(self.optimization_results.values(), key=lambda x: x.created_at)
                
                insight = LearningInsight(
                    id=f"optimization_recommendation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    insight_type="optimization",
                    description=f"Optimización para {latest_optimization.goal.value} resultó en {latest_optimization.performance_improvement:.1%} de mejora",
                    confidence=0.85,
                    impact_score=0.9,
                    actionable_recommendations=latest_optimization.recommendations,
                    related_patterns=[latest_optimization.goal.value]
                )
                insights.append(insight)
            
            # Almacenar insights
            for insight in insights:
                self.learning_insights[insight.id] = insight
            
            logger.info(f"Generated {len(insights)} learning insights")
            return insights
            
        except Exception as e:
            logger.error(f"Error generating learning insights: {e}")
            return []
    
    async def _analyze_feature_importance(self) -> Dict[str, float]:
        """Analizar importancia de características"""
        try:
            if not self.trained_models:
                return {}
            
            # Usar el modelo con mejor rendimiento que tenga feature_importances_
            best_model = None
            best_performance = -1
            
            for model_id, performance in self.model_performances.items():
                if performance.r2_score > best_performance:
                    model = self.trained_models[model_id]
                    if hasattr(model, 'feature_importances_'):
                        best_model = model
                        best_performance = performance.r2_score
            
            if best_model is None:
                return {}
            
            # Obtener importancia de características
            importance_dict = dict(zip(self.feature_columns, best_model.feature_importances_))
            
            return importance_dict
            
        except Exception as e:
            logger.error(f"Error analyzing feature importance: {e}")
            return {}
    
    async def predict(self, model_id: str, features: Dict[str, Any]) -> float:
        """Realizar predicción con un modelo entrenado"""
        try:
            if model_id not in self.trained_models:
                raise ValueError(f"Model {model_id} not found")
            
            model = self.trained_models[model_id]
            
            # Preparar características
            feature_vector = []
            for column in self.feature_columns:
                value = features.get(column, 0)
                feature_vector.append(value)
            
            feature_array = np.array(feature_vector).reshape(1, -1)
            
            # Normalizar si hay scaler
            if hasattr(self, 'scaler'):
                feature_array = self.scaler.transform(feature_array)
            
            # Realizar predicción
            prediction = model.predict(feature_array)[0]
            
            return float(prediction)
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise
    
    async def get_optimization_summary(self) -> Dict[str, Any]:
        """Obtener resumen de optimización"""
        return {
            "total_models_trained": len(self.trained_models),
            "total_optimizations": len(self.optimization_results),
            "total_insights": len(self.learning_insights),
            "best_model": max(self.model_performances.values(), key=lambda x: x.r2_score).model_id if self.model_performances else None,
            "average_performance": np.mean([p.r2_score for p in self.model_performances.values()]) if self.model_performances else 0.0,
            "models_by_type": {
                model_type.value: len([p for p in self.model_performances.values() if p.model_type == model_type])
                for model_type in ModelType
            },
            "last_optimization": max(self.optimization_results.values(), key=lambda x: x.created_at).created_at.isoformat() if self.optimization_results else None
        }
    
    async def export_optimization_data(self, filepath: str = None) -> str:
        """Exportar datos de optimización"""
        try:
            if filepath is None:
                filepath = f"exports/ai_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Crear directorio si no existe
            import os
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Preparar datos para exportación
            export_data = {
                "model_performances": {
                    model_id: {
                        "model_id": perf.model_id,
                        "model_type": perf.model_type.value,
                        "accuracy": perf.accuracy,
                        "r2_score": perf.r2_score,
                        "mse": perf.mse,
                        "training_time": perf.training_time,
                        "prediction_time": perf.prediction_time,
                        "memory_usage": perf.memory_usage,
                        "created_at": perf.created_at.isoformat()
                    }
                    for model_id, perf in self.model_performances.items()
                },
                "optimization_results": {
                    opt_id: {
                        "id": result.id,
                        "goal": result.goal.value,
                        "best_model": result.best_model,
                        "best_parameters": result.best_parameters,
                        "performance_improvement": result.performance_improvement,
                        "cost_reduction": result.cost_reduction,
                        "speed_improvement": result.speed_improvement,
                        "quality_improvement": result.quality_improvement,
                        "recommendations": result.recommendations,
                        "created_at": result.created_at.isoformat()
                    }
                    for opt_id, result in self.optimization_results.items()
                },
                "learning_insights": {
                    insight_id: {
                        "id": insight.id,
                        "insight_type": insight.insight_type,
                        "description": insight.description,
                        "confidence": insight.confidence,
                        "impact_score": insight.impact_score,
                        "actionable_recommendations": insight.actionable_recommendations,
                        "related_patterns": insight.related_patterns,
                        "created_at": insight.created_at.isoformat()
                    }
                    for insight_id, insight in self.learning_insights.items()
                },
                "summary": await self.get_optimization_summary(),
                "exported_at": datetime.now().isoformat()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"AI optimization data exported to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting optimization data: {e}")
            raise

























