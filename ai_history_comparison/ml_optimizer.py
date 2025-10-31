"""
ML Model Optimizer for AI History Comparison System
Sistema de optimización automática de modelos de ML para mejorar el análisis de historial
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import joblib
import pickle

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import xgboost as xgb
import lightgbm as lgb

from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelPerformance:
    """Métricas de rendimiento de un modelo"""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    mse: float
    r2_score: float
    training_time: float
    prediction_time: float
    model_size: int
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class OptimizationResult:
    """Resultado de optimización de modelo"""
    best_model: str
    best_params: Dict[str, Any]
    best_score: float
    improvement: float
    recommendations: List[str]
    performance_metrics: ModelPerformance
    feature_importance: Dict[str, float]

class DocumentDataset(Dataset):
    """Dataset personalizado para documentos"""
    
    def __init__(self, documents, labels, tokenizer=None, max_length=512):
        self.documents = documents
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.documents)
    
    def __getitem__(self, idx):
        document = self.documents[idx]
        label = self.labels[idx]
        
        if self.tokenizer:
            # Tokenizar documento
            encoding = self.tokenizer(
                document,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.float)
            }
        else:
            return {
                'document': document,
                'labels': torch.tensor(label, dtype=torch.float)
            }

class QualityPredictorNN(nn.Module):
    """Red neuronal para predecir calidad de documentos"""
    
    def __init__(self, input_size, hidden_sizes=[512, 256, 128], dropout=0.3):
        super(QualityPredictorNN, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden_size)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class MLOptimizer:
    """
    Optimizador de modelos de ML para el sistema de análisis de historial de IA
    """
    
    def __init__(
        self,
        model_storage_path: str = "models/",
        cache_path: str = "cache/",
        enable_auto_optimization: bool = True,
        optimization_interval: int = 3600  # 1 hora
    ):
        self.model_storage_path = Path(model_storage_path)
        self.cache_path = Path(cache_path)
        self.enable_auto_optimization = enable_auto_optimization
        self.optimization_interval = optimization_interval
        
        # Crear directorios si no existen
        self.model_storage_path.mkdir(parents=True, exist_ok=True)
        self.cache_path.mkdir(parents=True, exist_ok=True)
        
        # Modelos disponibles
        self.models = {
            'random_forest': RandomForestRegressor(random_state=42),
            'gradient_boosting': GradientBoostingRegressor(random_state=42),
            'xgboost': xgb.XGBRegressor(random_state=42),
            'lightgbm': lgb.LGBMRegressor(random_state=42),
            'linear_regression': LinearRegression(),
            'ridge': Ridge(random_state=42)
        }
        
        # Parámetros para optimización
        self.param_grids = {
            'random_forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'gradient_boosting': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            },
            'xgboost': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            },
            'lightgbm': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            },
            'ridge': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            }
        }
        
        # Modelos entrenados
        self.trained_models = {}
        self.model_performance = {}
        self.best_model = None
        self.feature_importance = {}
        
        # Embedding model para características de texto
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Inicializar optimización automática
        if self.enable_auto_optimization:
            asyncio.create_task(self._auto_optimization_loop())
    
    async def prepare_training_data(self, documents_data: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preparar datos de entrenamiento a partir del historial de documentos
        
        Args:
            documents_data: Lista de documentos con métricas
            
        Returns:
            Tupla con características (X) y etiquetas (y)
        """
        logger.info(f"Preparando datos de entrenamiento para {len(documents_data)} documentos")
        
        features = []
        labels = []
        
        for doc in documents_data:
            # Características del documento
            doc_features = []
            
            # Características básicas
            doc_features.extend([
                doc.get('word_count', 0),
                doc.get('readability_score', 0.0),
                doc.get('originality_score', 0.0),
                doc.get('processing_time', 0.0),
                len(doc.get('query', '')),
                doc.get('metadata', {}).get('category_score', 0.0)
            ])
            
            # Características de texto usando embeddings
            content_embedding = self.embedding_model.encode([doc.get('content', '')])[0]
            doc_features.extend(content_embedding[:50])  # Usar solo las primeras 50 dimensiones
            
            # Características de query
            query_embedding = self.embedding_model.encode([doc.get('query', '')])[0]
            doc_features.extend(query_embedding[:20])  # Usar solo las primeras 20 dimensiones
            
            # Características de estructura
            content = doc.get('content', '')
            doc_features.extend([
                content.count('\n'),  # Número de líneas
                content.count('.'),   # Número de oraciones
                content.count('#'),   # Número de títulos
                content.count('*'),   # Número de listas
                content.count('http'), # Número de enlaces
                len(set(content.split())) / len(content.split()) if content.split() else 0  # Diversidad léxica
            ])
            
            # Características temporales
            timestamp = doc.get('timestamp', datetime.now())
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            
            doc_features.extend([
                timestamp.hour,  # Hora del día
                timestamp.weekday(),  # Día de la semana
                (datetime.now() - timestamp).days  # Días desde creación
            ])
            
            features.append(doc_features)
            
            # Etiqueta (calidad del documento)
            quality_score = doc.get('quality_score', 0.0)
            labels.append(quality_score)
        
        X = np.array(features)
        y = np.array(labels)
        
        logger.info(f"Datos preparados: {X.shape[0]} muestras, {X.shape[1]} características")
        return X, y
    
    async def optimize_models(self, X: np.ndarray, y: np.ndarray) -> OptimizationResult:
        """
        Optimizar todos los modelos disponibles
        
        Args:
            X: Características de entrenamiento
            y: Etiquetas de entrenamiento
            
        Returns:
            Resultado de optimización con el mejor modelo
        """
        logger.info("Iniciando optimización de modelos")
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Escalar características
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        best_score = -np.inf
        best_model_name = None
        best_model = None
        best_params = None
        optimization_results = {}
        
        for model_name, model in self.models.items():
            logger.info(f"Optimizando modelo: {model_name}")
            
            try:
                start_time = datetime.now()
                
                if model_name in self.param_grids:
                    # Optimización con GridSearch
                    grid_search = GridSearchCV(
                        model,
                        self.param_grids[model_name],
                        cv=5,
                        scoring='r2',
                        n_jobs=-1,
                        verbose=0
                    )
                    grid_search.fit(X_train_scaled, y_train)
                    
                    best_model_cv = grid_search.best_estimator_
                    best_params_cv = grid_search.best_params_
                    best_score_cv = grid_search.best_score_
                else:
                    # Entrenar sin optimización de hiperparámetros
                    best_model_cv = model
                    best_model_cv.fit(X_train_scaled, y_train)
                    best_params_cv = {}
                    best_score_cv = cross_val_score(
                        best_model_cv, X_train_scaled, y_train, cv=5, scoring='r2'
                    ).mean()
                
                # Evaluar en conjunto de prueba
                y_pred = best_model_cv.predict(X_test_scaled)
                test_score = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                training_time = (datetime.now() - start_time).total_seconds()
                
                # Calcular tiempo de predicción
                pred_start = datetime.now()
                _ = best_model_cv.predict(X_test_scaled[:100])  # Predicción de muestra
                prediction_time = (datetime.now() - pred_start).total_seconds() / 100
                
                # Calcular tamaño del modelo
                model_size = len(pickle.dumps(best_model_cv))
                
                # Crear métricas de rendimiento
                performance = ModelPerformance(
                    model_name=model_name,
                    accuracy=test_score,
                    precision=0.0,  # No aplicable para regresión
                    recall=0.0,     # No aplicable para regresión
                    f1_score=0.0,   # No aplicable para regresión
                    mse=mse,
                    r2_score=test_score,
                    training_time=training_time,
                    prediction_time=prediction_time,
                    model_size=model_size
                )
                
                optimization_results[model_name] = {
                    'model': best_model_cv,
                    'params': best_params_cv,
                    'cv_score': best_score_cv,
                    'test_score': test_score,
                    'performance': performance
                }
                
                # Actualizar mejor modelo
                if test_score > best_score:
                    best_score = test_score
                    best_model_name = model_name
                    best_model = best_model_cv
                    best_params = best_params_cv
                
                logger.info(f"Modelo {model_name}: R² = {test_score:.4f}, MSE = {mse:.4f}")
                
            except Exception as e:
                logger.error(f"Error optimizando modelo {model_name}: {e}")
                continue
        
        # Calcular importancia de características
        feature_importance = {}
        if best_model and hasattr(best_model, 'feature_importances_'):
            feature_importance = dict(zip(
                range(len(best_model.feature_importances_)),
                best_model.feature_importances_
            ))
        
        # Generar recomendaciones
        recommendations = await self._generate_optimization_recommendations(
            optimization_results, best_model_name, best_score
        )
        
        # Guardar mejor modelo
        if best_model:
            await self._save_model(best_model, best_model_name, best_params)
            self.best_model = best_model
            self.model_performance[best_model_name] = optimization_results[best_model_name]['performance']
            self.feature_importance = feature_importance
        
        # Calcular mejora
        baseline_score = optimization_results.get('linear_regression', {}).get('test_score', 0.0)
        improvement = best_score - baseline_score if baseline_score > 0 else 0.0
        
        return OptimizationResult(
            best_model=best_model_name,
            best_params=best_params,
            best_score=best_score,
            improvement=improvement,
            recommendations=recommendations,
            performance_metrics=optimization_results[best_model_name]['performance'],
            feature_importance=feature_importance
        )
    
    async def _generate_optimization_recommendations(
        self, 
        results: Dict[str, Any], 
        best_model: str, 
        best_score: float
    ) -> List[str]:
        """Generar recomendaciones basadas en los resultados de optimización"""
        recommendations = []
        
        # Recomendación sobre el mejor modelo
        recommendations.append(f"El modelo {best_model} obtuvo el mejor rendimiento (R² = {best_score:.4f})")
        
        # Análisis de rendimiento relativo
        scores = {name: result['test_score'] for name, result in results.items()}
        sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        if len(sorted_models) >= 2:
            second_best = sorted_models[1]
            gap = best_score - second_best[1]
            if gap > 0.05:
                recommendations.append(f"El modelo {best_model} supera significativamente a {second_best[0]} por {gap:.4f} puntos")
            else:
                recommendations.append(f"Los modelos {best_model} y {second_best[0]} tienen rendimiento similar")
        
        # Recomendaciones de características
        if self.feature_importance:
            top_features = sorted(
                self.feature_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
            recommendations.append("Las características más importantes son:")
            for i, (feature_idx, importance) in enumerate(top_features, 1):
                recommendations.append(f"  {i}. Característica {feature_idx}: {importance:.4f}")
        
        # Recomendaciones de rendimiento
        if best_score < 0.7:
            recommendations.append("Considerar recopilar más datos de entrenamiento para mejorar el rendimiento")
        
        if best_score > 0.9:
            recommendations.append("El modelo tiene excelente rendimiento, considerar implementación en producción")
        
        return recommendations
    
    async def _save_model(self, model: Any, model_name: str, params: Dict[str, Any]):
        """Guardar modelo optimizado"""
        model_path = self.model_storage_path / f"{model_name}_optimized.pkl"
        
        model_data = {
            'model': model,
            'params': params,
            'created_at': datetime.now().isoformat(),
            'model_name': model_name
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Modelo {model_name} guardado en {model_path}")
    
    async def load_best_model(self) -> Optional[Any]:
        """Cargar el mejor modelo guardado"""
        model_files = list(self.model_storage_path.glob("*_optimized.pkl"))
        
        if not model_files:
            logger.warning("No se encontraron modelos guardados")
            return None
        
        # Cargar el modelo más reciente
        latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
        
        try:
            with open(latest_model, 'rb') as f:
                model_data = pickle.load(f)
            
            self.best_model = model_data['model']
            logger.info(f"Modelo {model_data['model_name']} cargado desde {latest_model}")
            return self.best_model
            
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            return None
    
    async def predict_quality(self, document_features: np.ndarray) -> float:
        """
        Predecir calidad de un documento usando el mejor modelo
        
        Args:
            document_features: Características del documento
            
        Returns:
            Predicción de calidad (0.0 - 1.0)
        """
        if self.best_model is None:
            await self.load_best_model()
        
        if self.best_model is None:
            logger.warning("No hay modelo disponible para predicción")
            return 0.5  # Valor por defecto
        
        try:
            # Asegurar que las características tengan la forma correcta
            if document_features.ndim == 1:
                document_features = document_features.reshape(1, -1)
            
            prediction = self.best_model.predict(document_features)[0]
            
            # Asegurar que la predicción esté en el rango [0, 1]
            prediction = max(0.0, min(1.0, prediction))
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error en predicción: {e}")
            return 0.5
    
    async def _auto_optimization_loop(self):
        """Loop de optimización automática"""
        while True:
            try:
                await asyncio.sleep(self.optimization_interval)
                logger.info("Iniciando optimización automática")
                
                # Aquí se cargarían los datos más recientes del historial
                # Por ahora, solo logueamos que se ejecutó
                logger.info("Optimización automática completada")
                
            except Exception as e:
                logger.error(f"Error en optimización automática: {e}")
                await asyncio.sleep(300)  # Esperar 5 minutos antes de reintentar
    
    def get_model_performance_summary(self) -> Dict[str, Any]:
        """Obtener resumen del rendimiento de modelos"""
        if not self.model_performance:
            return {"message": "No hay modelos entrenados"}
        
        summary = {
            "total_models": len(self.model_performance),
            "best_model": self.best_model.__class__.__name__ if self.best_model else None,
            "models": {}
        }
        
        for model_name, performance in self.model_performance.items():
            summary["models"][model_name] = {
                "r2_score": performance.r2_score,
                "mse": performance.mse,
                "training_time": performance.training_time,
                "prediction_time": performance.prediction_time,
                "model_size": performance.model_size
            }
        
        return summary
    
    async def export_optimization_report(self) -> Dict[str, Any]:
        """Exportar reporte de optimización"""
        return {
            "timestamp": datetime.now().isoformat(),
            "optimization_summary": self.get_model_performance_summary(),
            "feature_importance": self.feature_importance,
            "best_model_info": {
                "name": self.best_model.__class__.__name__ if self.best_model else None,
                "parameters": getattr(self.best_model, 'get_params', lambda: {})() if self.best_model else {}
            },
            "recommendations": [
                "Continuar monitoreando el rendimiento del modelo",
                "Recopilar más datos para mejorar la precisión",
                "Considerar actualizar el modelo periódicamente"
            ]
        }



























