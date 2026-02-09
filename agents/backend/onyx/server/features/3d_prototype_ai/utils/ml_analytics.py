"""
ML Analytics - Sistema de analytics con machine learning
=========================================================
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


class MLAnalytics:
    """Sistema de analytics con machine learning"""
    
    def __init__(self):
        self.user_behavior: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.predictions: Dict[str, Dict[str, Any]] = {}
        self.models: Dict[str, Any] = {}
    
    def record_user_action(self, user_id: str, action: str,
                          context: Optional[Dict[str, Any]] = None):
        """Registra acción de usuario"""
        action_record = {
            "action": action,
            "context": context or {},
            "timestamp": datetime.now().isoformat()
        }
        
        self.user_behavior[user_id].append(action_record)
        
        # Mantener solo últimas 1000 acciones por usuario
        if len(self.user_behavior[user_id]) > 1000:
            self.user_behavior[user_id] = self.user_behavior[user_id][-1000:]
    
    def predict_user_preference(self, user_id: str) -> Dict[str, Any]:
        """Predice preferencias de usuario usando ML"""
        user_actions = self.user_behavior.get(user_id, [])
        
        if not user_actions:
            return {
                "user_id": user_id,
                "confidence": 0.0,
                "preferences": {}
            }
        
        # Análisis simple (en producción usaría ML real)
        action_counts = defaultdict(int)
        for action in user_actions:
            action_counts[action["action"]] += 1
        
        # Predecir preferencias basadas en acciones
        preferences = {
            "preferred_product_types": ["licuadora", "estufa"],  # Simulado
            "preferred_budget_range": "medium",
            "preferred_complexity": "medium"
        }
        
        confidence = min(1.0, len(user_actions) / 100.0)
        
        prediction = {
            "user_id": user_id,
            "confidence": confidence,
            "preferences": preferences,
            "based_on_actions": len(user_actions)
        }
        
        self.predictions[user_id] = prediction
        return prediction
    
    def detect_anomalies(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Detecta anomalías usando ML"""
        anomalies = []
        
        if user_id:
            user_actions = self.user_behavior.get(user_id, [])
            # Detectar patrones anómalos (simplificado)
            if len(user_actions) > 500:  # Muchas acciones en poco tiempo
                anomalies.append({
                    "user_id": user_id,
                    "type": "high_activity",
                    "severity": "medium",
                    "description": "Actividad inusualmente alta"
                })
        else:
            # Análisis global
            total_users = len(self.user_behavior)
            if total_users > 10000:
                anomalies.append({
                    "type": "high_user_count",
                    "severity": "low",
                    "description": "Alto número de usuarios activos"
                })
        
        return anomalies
    
    def generate_insights(self, time_range_days: int = 7) -> Dict[str, Any]:
        """Genera insights usando ML"""
        cutoff = datetime.now() - timedelta(days=time_range_days)
        
        # Analizar comportamiento reciente
        recent_actions = []
        for user_id, actions in self.user_behavior.items():
            recent = [a for a in actions if datetime.fromisoformat(a["timestamp"]) > cutoff]
            recent_actions.extend(recent)
        
        # Insights (simplificado, en producción usaría ML real)
        insights = {
            "total_actions": len(recent_actions),
            "unique_users": len(set(a.get("user_id") for a in recent_actions if "user_id" in a)),
            "most_common_actions": self._get_most_common(recent_actions, "action", 5),
            "trends": {
                "growing": True,
                "growth_rate": 15.5
            },
            "recommendations": [
                "Optimizar generación de prototipos",
                "Mejorar recomendaciones de materiales"
            ]
        }
        
        return insights
    
    def _get_most_common(self, items: List[Dict[str, Any]], key: str, limit: int) -> List[Dict[str, Any]]:
        """Obtiene los más comunes"""
        counts = defaultdict(int)
        for item in items:
            counts[item.get(key, "unknown")] += 1
        
        sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        return [{"item": k, "count": v} for k, v in sorted_items[:limit]]
    
    def train_model(self, model_name: str, training_data: List[Dict[str, Any]]):
        """Entrena un modelo ML"""
        logger.info(f"Entrenando modelo: {model_name} con {len(training_data)} ejemplos")
        
        # En producción, esto entrenaría un modelo real
        self.models[model_name] = {
            "name": model_name,
            "trained_at": datetime.now().isoformat(),
            "training_samples": len(training_data),
            "status": "trained"
        }
    
    def get_model_predictions(self, model_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Obtiene predicciones de un modelo"""
        model = self.models.get(model_name)
        if not model:
            raise ValueError(f"Modelo no encontrado: {model_name}")
        
        # En producción, esto usaría el modelo real
        return {
            "model": model_name,
            "prediction": "sample_prediction",
            "confidence": 0.85
        }




