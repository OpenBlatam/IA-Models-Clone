"""
Sistema de recomendaciones basadas en horarios (mañana/noche)
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, time


@dataclass
class TimeBasedRoutine:
    """Rutina basada en horario"""
    time_of_day: str  # "morning", "evening", "night"
    steps: List[Dict]
    products: List[str]
    duration_minutes: int
    priority: int
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "time_of_day": self.time_of_day,
            "steps": self.steps,
            "products": self.products,
            "duration_minutes": self.duration_minutes,
            "priority": self.priority
        }


@dataclass
class TimeBasedRecommendation:
    """Recomendación basada en horario"""
    user_id: str
    morning_routine: TimeBasedRoutine
    evening_routine: TimeBasedRoutine
    night_routine: Optional[TimeBasedRoutine] = None
    tips: List[str] = None
    
    def __post_init__(self):
        if self.tips is None:
            self.tips = []
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "user_id": self.user_id,
            "morning_routine": self.morning_routine.to_dict(),
            "evening_routine": self.evening_routine.to_dict(),
            "night_routine": self.night_routine.to_dict() if self.night_routine else None,
            "tips": self.tips
        }


class TimeBasedRecommendations:
    """Sistema de recomendaciones basadas en horarios"""
    
    def __init__(self):
        """Inicializa el sistema"""
        self.recommendations: Dict[str, TimeBasedRecommendation] = {}  # user_id -> recommendation
    
    def generate_time_based_routine(self, user_id: str, skin_type: str,
                                   concerns: List[str]) -> TimeBasedRecommendation:
        """Genera rutina basada en horarios"""
        
        # Rutina matutina
        morning_steps = [
            {"step": 1, "action": "Limpieza suave", "product_type": "gentle_cleanser"},
            {"step": 2, "action": "Serum antioxidante", "product_type": "vitamin_c_serum"},
            {"step": 3, "action": "Hidratante ligera", "product_type": "lightweight_moisturizer"},
            {"step": 4, "action": "Protector solar", "product_type": "sunscreen", "priority": "high"}
        ]
        
        morning_products = ["Gentle Cleanser", "Vitamin C Serum", "Lightweight Moisturizer", "SPF 30+"]
        
        morning_routine = TimeBasedRoutine(
            time_of_day="morning",
            steps=morning_steps,
            products=morning_products,
            duration_minutes=10,
            priority=1
        )
        
        # Rutina nocturna
        evening_steps = [
            {"step": 1, "action": "Doble limpieza", "product_type": "oil_cleanser"},
            {"step": 2, "action": "Limpieza secundaria", "product_type": "water_cleanser"},
            {"step": 3, "action": "Exfoliante (2-3x/semana)", "product_type": "exfoliant"},
            {"step": 4, "action": "Serum activo", "product_type": "active_serum"},
            {"step": 5, "action": "Hidratante", "product_type": "moisturizer"}
        ]
        
        evening_products = ["Oil Cleanser", "Water Cleanser", "AHA/BHA Exfoliant", "Retinol Serum", "Moisturizer"]
        
        evening_routine = TimeBasedRoutine(
            time_of_day="evening",
            steps=evening_steps,
            products=evening_products,
            duration_minutes=15,
            priority=1
        )
        
        # Rutina nocturna opcional
        night_routine = None
        if "dryness" in concerns or "aging" in concerns:
            night_steps = [
                {"step": 1, "action": "Aplicar crema reparadora", "product_type": "night_cream"},
                {"step": 2, "action": "Aceite facial (opcional)", "product_type": "facial_oil"}
            ]
            
            night_products = ["Night Repair Cream", "Facial Oil"]
            
            night_routine = TimeBasedRoutine(
                time_of_day="night",
                steps=night_steps,
                products=night_products,
                duration_minutes=5,
                priority=2
            )
        
        # Tips personalizados
        tips = [
            "Aplica productos en orden de consistencia: más ligeros primero",
            "Espera 2-3 minutos entre capas para mejor absorción",
            "Nunca olvides el protector solar por la mañana",
            "La rutina nocturna es para reparación y renovación"
        ]
        
        if "acne" in concerns:
            tips.append("Para acné, usa productos con ácido salicílico en la noche")
        
        if "aging" in concerns:
            tips.append("Retinol solo en la noche, nunca con sol")
        
        recommendation = TimeBasedRecommendation(
            user_id=user_id,
            morning_routine=morning_routine,
            evening_routine=evening_routine,
            night_routine=night_routine,
            tips=tips
        )
        
        self.recommendations[user_id] = recommendation
        return recommendation
    
    def get_time_based_routine(self, user_id: str) -> Optional[TimeBasedRecommendation]:
        """Obtiene rutina basada en horarios"""
        return self.recommendations.get(user_id)
    
    def get_routine_for_time(self, user_id: str, current_time: time) -> Optional[TimeBasedRoutine]:
        """Obtiene rutina para el horario actual"""
        recommendation = self.recommendations.get(user_id)
        if not recommendation:
            return None
        
        hour = current_time.hour
        
        if 5 <= hour < 12:
            return recommendation.morning_routine
        elif 12 <= hour < 20:
            return recommendation.evening_routine
        else:
            return recommendation.night_routine or recommendation.evening_routine






