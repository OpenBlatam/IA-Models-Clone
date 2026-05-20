"""
Sistema de Machine Learning mejorado
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import numpy as np
import statistics


class MLModelType(str, Enum):
    """Tipos de modelos ML"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    DEEP_LEARNING = "deep_learning"


@dataclass
class MLPrediction:
    """Predicción ML"""
    model_type: MLModelType
    predictions: Dict[str, Any]
    confidence: float
    model_version: str
    inference_time: float
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "model_type": self.model_type.value,
            "predictions": self.predictions,
            "confidence": self.confidence,
            "model_version": self.model_version,
            "inference_time": self.inference_time,
            "timestamp": self.timestamp
        }


class EnhancedMLSystem:
    """Sistema de ML mejorado"""
    
    def __init__(self):
        """Inicializa el sistema"""
        self.models: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Dict] = {}
        self.prediction_history: List[MLPrediction] = []
    
    def register_model(self, model_id: str, model_type: MLModelType,
                      model: Any, metadata: Dict):
        """
        Registra un modelo ML
        
        Args:
            model_id: ID del modelo
            model_type: Tipo de modelo
            model: Modelo ML
            metadata: Metadatos del modelo
        """
        self.models[model_id] = {
            "model": model,
            "type": model_type,
            "metadata": metadata
        }
        self.model_metadata[model_id] = metadata
    
    def predict(self, model_id: str, features: np.ndarray) -> MLPrediction:
        """
        Realiza predicción
        
        Args:
            model_id: ID del modelo
            features: Features de entrada
            
        Returns:
            Predicción
        """
        import time
        
        if model_id not in self.models:
            raise ValueError(f"Modelo no encontrado: {model_id}")
        
        model_info = self.models[model_id]
        model = model_info["model"]
        model_type = model_info["type"]
        
        start_time = time.time()
        
        try:
            # Realizar predicción (placeholder - implementar según modelo real)
            if model_type == MLModelType.CLASSIFICATION:
                # predictions = model.predict(features)
                predictions = {"class": "normal", "probabilities": {"normal": 0.7, "dry": 0.3}}
            elif model_type == MLModelType.REGRESSION:
                # predictions = model.predict(features)
                predictions = {"value": 75.5, "range": [70, 80]}
            else:
                predictions = {"result": "prediction"}
            
            inference_time = time.time() - start_time
            
            prediction = MLPrediction(
                model_type=model_type,
                predictions=predictions,
                confidence=0.85,  # Placeholder
                model_version=model_info["metadata"].get("version", "1.0"),
                inference_time=inference_time
            )
            
            self.prediction_history.append(prediction)
            
            return prediction
        
        except Exception as e:
            raise RuntimeError(f"Error en predicción: {str(e)}")
    
    def batch_predict(self, model_id: str, features_list: List[np.ndarray]) -> List[MLPrediction]:
        """Realiza predicciones en lote"""
        predictions = []
        
        for features in features_list:
            try:
                prediction = self.predict(model_id, features)
                predictions.append(prediction)
            except Exception as e:
                # Continuar con siguiente si falla
                continue
        
        return predictions
    
    def get_model_info(self, model_id: str) -> Optional[Dict]:
        """Obtiene información de un modelo"""
        if model_id not in self.models:
            return None
        
        model_info = self.models[model_id]
        metadata = self.model_metadata.get(model_id, {})
        
        return {
            "model_id": model_id,
            "type": model_info["type"].value,
            "metadata": metadata,
            "registered": True
        }
    
    def get_prediction_stats(self) -> Dict:
        """Obtiene estadísticas de predicciones"""
        if not self.prediction_history:
            return {
                "total_predictions": 0,
                "average_confidence": 0,
                "average_inference_time": 0
            }
        
        confidences = [p.confidence for p in self.prediction_history]
        inference_times = [p.inference_time for p in self.prediction_history]
        
        return {
            "total_predictions": len(self.prediction_history),
            "average_confidence": statistics.mean(confidences),
            "average_inference_time": statistics.mean(inference_times),
            "models_used": len(set(p.model_version for p in self.prediction_history))
        }

