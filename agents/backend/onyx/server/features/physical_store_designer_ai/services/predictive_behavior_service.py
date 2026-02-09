"""
Predictive Behavior Service - Análisis de comportamiento predictivo
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
from ..services.llm_service import LLMService

logger = logging.getLogger(__name__)


class PredictiveBehaviorService:
    """Servicio para análisis de comportamiento predictivo"""
    
    def __init__(self, llm_service: Optional[LLMService] = None):
        self.llm_service = llm_service or LLMService()
        self.behavior_data: Dict[str, List[Dict[str, Any]]] = {}
        self.predictions: Dict[str, Dict[str, Any]] = {}
    
    def record_behavior(
        self,
        store_id: str,
        customer_id: str,
        behavior_type: str,  # "movement", "interaction", "purchase", "browse"
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Registrar comportamiento"""
        
        behavior = {
            "behavior_id": f"beh_{store_id}_{len(self.behavior_data.get(store_id, [])) + 1}",
            "store_id": store_id,
            "customer_id": customer_id,
            "type": behavior_type,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        
        if store_id not in self.behavior_data:
            self.behavior_data[store_id] = []
        
        self.behavior_data[store_id].append(behavior)
        
        return behavior
    
    async def predict_customer_behavior(
        self,
        customer_id: str,
        time_horizon: str = "next_visit"  # "next_visit", "next_purchase", "churn"
    ) -> Dict[str, Any]:
        """Predecir comportamiento del cliente"""
        
        # Obtener historial del cliente
        all_behaviors = []
        for store_behaviors in self.behavior_data.values():
            all_behaviors.extend([b for b in store_behaviors if b["customer_id"] == customer_id])
        
        if not all_behaviors:
            return {
                "customer_id": customer_id,
                "prediction": "insufficient_data",
                "message": "No hay datos suficientes para predecir"
            }
        
        if self.llm_service.client:
            try:
                return await self._predict_with_llm(customer_id, all_behaviors, time_horizon)
            except Exception as e:
                logger.error(f"Error prediciendo comportamiento: {e}")
                return self._predict_basic(customer_id, all_behaviors, time_horizon)
        else:
            return self._predict_basic(customer_id, all_behaviors, time_horizon)
    
    async def _predict_with_llm(
        self,
        customer_id: str,
        behaviors: List[Dict[str, Any]],
        time_horizon: str
    ) -> Dict[str, Any]:
        """Predecir usando LLM"""
        prompt = f"""Basado en este historial de comportamiento del cliente {customer_id}:
        {behaviors[-20:]}
        
        Predice el comportamiento para: {time_horizon}
        
        Proporciona:
        - Probabilidad
        - Factores clave
        - Recomendaciones"""
        
        result = await self.llm_service.generate_structured(
            prompt=prompt,
            system_prompt="Eres un experto en análisis predictivo de comportamiento."
        )
        
        return {
            "customer_id": customer_id,
            "time_horizon": time_horizon,
            "prediction": result if result else {},
            "predicted_at": datetime.now().isoformat()
        }
    
    def _predict_basic(
        self,
        customer_id: str,
        behaviors: List[Dict[str, Any]],
        time_horizon: str
    ) -> Dict[str, Any]:
        """Predicción básica"""
        purchase_count = len([b for b in behaviors if b["type"] == "purchase"])
        visit_count = len([b for b in behaviors if b["type"] == "movement"])
        
        probability = min(0.9, 0.5 + (purchase_count * 0.1))
        
        return {
            "customer_id": customer_id,
            "time_horizon": time_horizon,
            "probability": round(probability, 2),
            "factors": ["Historial de compras", "Frecuencia de visitas"],
            "predicted_at": datetime.now().isoformat()
        }
    
    async def predict_store_traffic(
        self,
        store_id: str,
        date: str
    ) -> Dict[str, Any]:
        """Predecir tráfico de la tienda"""
        
        store_behaviors = self.behavior_data.get(store_id, [])
        
        # Analizar patrones históricos
        target_date = datetime.fromisoformat(date)
        day_of_week = target_date.weekday()
        
        historical = [
            b for b in store_behaviors
            if datetime.fromisoformat(b["timestamp"]).weekday() == day_of_week
        ]
        
        # Calcular promedio
        hourly_counts = defaultdict(int)
        for behavior in historical:
            hour = datetime.fromisoformat(behavior["timestamp"]).hour
            hourly_counts[hour] += 1
        
        avg_traffic = sum(hourly_counts.values()) / len(hourly_counts) if hourly_counts else 0
        
        return {
            "store_id": store_id,
            "date": date,
            "predicted_traffic": round(avg_traffic, 0),
            "hourly_breakdown": dict(hourly_counts),
            "confidence": 0.75,
            "predicted_at": datetime.now().isoformat()
        }




