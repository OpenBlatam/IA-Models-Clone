"""
Customer Behavior Service - Análisis de comportamiento del cliente
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
from ..services.llm_service import LLMService

logger = logging.getLogger(__name__)


class CustomerBehaviorService:
    """Servicio para análisis de comportamiento del cliente"""
    
    def __init__(self, llm_service: Optional[LLMService] = None):
        self.llm_service = llm_service or LLMService()
        self.interactions: Dict[str, List[Dict[str, Any]]] = {}
        self.profiles: Dict[str, Dict[str, Any]] = {}
        self.heatmaps: Dict[str, Dict[str, Any]] = {}
    
    def record_interaction(
        self,
        store_id: str,
        customer_id: str,
        interaction_type: str,  # "entry", "browse", "purchase", "exit"
        location: Optional[str] = None,
        duration: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Registrar interacción del cliente"""
        
        interaction = {
            "interaction_id": f"int_{store_id}_{len(self.interactions.get(store_id, [])) + 1}",
            "store_id": store_id,
            "customer_id": customer_id,
            "type": interaction_type,
            "location": location,
            "duration_seconds": duration,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        }
        
        if store_id not in self.interactions:
            self.interactions[store_id] = []
        
        self.interactions[store_id].append(interaction)
        
        return interaction
    
    def build_customer_profile(
        self,
        customer_id: str
    ) -> Dict[str, Any]:
        """Construir perfil del cliente"""
        
        # Obtener todas las interacciones del cliente
        all_interactions = []
        for store_interactions in self.interactions.values():
            all_interactions.extend([i for i in store_interactions if i["customer_id"] == customer_id])
        
        if not all_interactions:
            return {
                "customer_id": customer_id,
                "message": "No hay interacciones registradas"
            }
        
        # Analizar comportamiento
        total_visits = len([i for i in all_interactions if i["type"] == "entry"])
        total_purchases = len([i for i in all_interactions if i["type"] == "purchase"])
        avg_duration = sum(i.get("duration_seconds", 0) for i in all_interactions) / len(all_interactions)
        
        # Zonas más visitadas
        locations = [i.get("location") for i in all_interactions if i.get("location")]
        location_counts = defaultdict(int)
        for loc in locations:
            location_counts[loc] += 1
        
        profile = {
            "customer_id": customer_id,
            "total_visits": total_visits,
            "total_purchases": total_purchases,
            "conversion_rate": (total_purchases / total_visits * 100) if total_visits > 0 else 0,
            "average_duration_seconds": round(avg_duration, 2),
            "preferred_locations": dict(sorted(location_counts.items(), key=lambda x: x[1], reverse=True)[:5]),
            "first_interaction": all_interactions[0]["timestamp"] if all_interactions else None,
            "last_interaction": all_interactions[-1]["timestamp"] if all_interactions else None,
            "customer_segment": self._determine_segment(total_visits, total_purchases, avg_duration)
        }
        
        self.profiles[customer_id] = profile
        
        return profile
    
    def _determine_segment(
        self,
        visits: int,
        purchases: int,
        avg_duration: float
    ) -> str:
        """Determinar segmento del cliente"""
        if purchases >= 10:
            return "vip"
        elif purchases >= 5:
            return "regular"
        elif visits >= 3:
            return "browser"
        else:
            return "new"
    
    def generate_heatmap(
        self,
        store_id: str,
        hours: int = 24
    ) -> Dict[str, Any]:
        """Generar heatmap de actividad"""
        
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        store_interactions = self.interactions.get(store_id, [])
        recent_interactions = [
            i for i in store_interactions
            if start_time <= datetime.fromisoformat(i["timestamp"]) <= end_time
        ]
        
        # Agrupar por ubicación
        location_activity = defaultdict(int)
        for interaction in recent_interactions:
            location = interaction.get("location", "unknown")
            location_activity[location] += 1
        
        heatmap = {
            "store_id": store_id,
            "period_hours": hours,
            "total_interactions": len(recent_interactions),
            "location_activity": dict(location_activity),
            "hot_spots": dict(sorted(location_activity.items(), key=lambda x: x[1], reverse=True)[:10]),
            "generated_at": datetime.now().isoformat()
        }
        
        self.heatmaps[store_id] = heatmap
        
        return heatmap
    
    async def analyze_customer_journey(
        self,
        customer_id: str
    ) -> Dict[str, Any]:
        """Analizar journey del cliente"""
        
        profile = self.build_customer_profile(customer_id)
        
        if "message" in profile:
            return profile
        
        # Obtener interacciones ordenadas
        all_interactions = []
        for store_interactions in self.interactions.values():
            all_interactions.extend([i for i in store_interactions if i["customer_id"] == customer_id])
        
        all_interactions.sort(key=lambda x: x["timestamp"])
        
        # Analizar journey
        journey_stages = []
        current_stage = None
        
        for interaction in all_interactions:
            if interaction["type"] == "entry":
                current_stage = "entry"
            elif interaction["type"] == "browse":
                if current_stage != "browse":
                    journey_stages.append("browse")
                current_stage = "browse"
            elif interaction["type"] == "purchase":
                journey_stages.append("purchase")
                current_stage = "purchase"
            elif interaction["type"] == "exit":
                journey_stages.append("exit")
                current_stage = None
        
        journey = {
            "customer_id": customer_id,
            "profile": profile,
            "journey_stages": journey_stages,
            "total_stages": len(journey_stages),
            "drop_off_points": self._identify_drop_offs(journey_stages),
            "recommendations": await self._generate_journey_recommendations(profile, journey_stages)
        }
        
        return journey
    
    def _identify_drop_offs(self, stages: List[str]) -> List[str]:
        """Identificar puntos de abandono"""
        drop_offs = []
        
        if "entry" in stages and "browse" not in stages:
            drop_offs.append("After entry")
        elif "browse" in stages and "purchase" not in stages:
            drop_offs.append("After browsing")
        
        return drop_offs
    
    async def _generate_journey_recommendations(
        self,
        profile: Dict[str, Any],
        stages: List[str]
    ) -> List[str]:
        """Generar recomendaciones basadas en journey"""
        recommendations = []
        
        if profile["conversion_rate"] < 20:
            recommendations.append("Mejorar estrategia de conversión")
        
        if "browse" in stages and "purchase" not in stages:
            recommendations.append("Implementar incentivos de compra")
        
        if profile["average_duration_seconds"] < 60:
            recommendations.append("Mejorar engagement para aumentar tiempo en tienda")
        
        return recommendations




