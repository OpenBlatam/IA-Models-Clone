"""
Predictive Analytics - Sistema de análisis predictivo avanzado
===============================================================
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


class PredictiveAnalytics:
    """Sistema de análisis predictivo avanzado"""
    
    def __init__(self):
        self.models: Dict[str, Dict[str, Any]] = {}
        self.predictions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.training_data: Dict[str, List[Dict[str, Any]]] = {}
    
    def train_model(self, model_id: str, model_type: str,
                   training_data: List[Dict[str, Any]], target_variable: str):
        """Entrena un modelo predictivo"""
        logger.info(f"Entrenando modelo: {model_id} - Tipo: {model_type}")
        
        # En producción, esto entrenaría un modelo real (scikit-learn, tensorflow, etc.)
        model = {
            "id": model_id,
            "type": model_type,
            "target_variable": target_variable,
            "trained_at": datetime.now().isoformat(),
            "training_samples": len(training_data),
            "status": "trained",
            "accuracy": 0.85,  # Simulado
            "features": list(training_data[0].keys()) if training_data else []
        }
        
        self.models[model_id] = model
        self.training_data[model_id] = training_data
        
        logger.info(f"Modelo entrenado: {model_id}")
        return model
    
    def predict(self, model_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Hace una predicción"""
        model = self.models.get(model_id)
        if not model:
            raise ValueError(f"Modelo no encontrado: {model_id}")
        
        # Predicción simplificada (en producción usaría el modelo real)
        prediction_value = sum(input_data.values()) / len(input_data) if input_data else 0
        
        prediction = {
            "model_id": model_id,
            "input": input_data,
            "prediction": prediction_value,
            "confidence": model.get("accuracy", 0.85),
            "timestamp": datetime.now().isoformat()
        }
        
        self.predictions[model_id].append(prediction)
        
        return prediction
    
    def forecast_time_series(self, data: List[Dict[str, Any]], periods: int = 10) -> List[Dict[str, Any]]:
        """Predice serie temporal"""
        if not data:
            return []
        
        # Calcular tendencia (simplificado)
        values = [d.get("value", 0) for d in data if isinstance(d.get("value"), (int, float))]
        
        if len(values) < 2:
            return []
        
        # Calcular promedio y tendencia
        avg = sum(values) / len(values)
        trend = (values[-1] - values[0]) / len(values) if len(values) > 1 else 0
        
        forecast = []
        last_value = values[-1] if values else avg
        
        for i in range(periods):
            predicted_value = last_value + trend * (i + 1)
            forecast.append({
                "period": i + 1,
                "predicted_value": round(predicted_value, 2),
                "confidence": 0.75
            })
        
        return forecast
    
    def predict_churn(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predice probabilidad de churn"""
        # Factores de churn (simplificado)
        churn_score = 0.0
        
        # Menos actividad = mayor churn
        if user_data.get("days_since_last_activity", 0) > 30:
            churn_score += 0.3
        
        # Menos uso = mayor churn
        if user_data.get("usage_frequency", 0) < 1:
            churn_score += 0.2
        
        # Problemas reportados = mayor churn
        if user_data.get("support_tickets", 0) > 3:
            churn_score += 0.2
        
        churn_score = min(1.0, churn_score)
        
        return {
            "user_id": user_data.get("user_id"),
            "churn_probability": round(churn_score, 4),
            "risk_level": "high" if churn_score > 0.5 else "medium" if churn_score > 0.3 else "low",
            "recommendations": self._get_churn_recommendations(churn_score)
        }
    
    def _get_churn_recommendations(self, churn_score: float) -> List[str]:
        """Obtiene recomendaciones para reducir churn"""
        recommendations = []
        
        if churn_score > 0.5:
            recommendations.extend([
                "Contactar usuario inmediatamente",
                "Ofrecer descuento o incentivo",
                "Revisar experiencia del usuario"
            ])
        elif churn_score > 0.3:
            recommendations.extend([
                "Enviar comunicación proactiva",
                "Ofrecer tutoriales o recursos"
            ])
        
        return recommendations
    
    def get_model_performance(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene rendimiento de un modelo"""
        model = self.models.get(model_id)
        if not model:
            return None
        
        predictions = self.predictions.get(model_id, [])
        
        return {
            "model": model,
            "total_predictions": len(predictions),
            "recent_predictions": predictions[-10:] if predictions else [],
            "average_confidence": sum(p.get("confidence", 0) for p in predictions) / len(predictions) if predictions else 0
        }




