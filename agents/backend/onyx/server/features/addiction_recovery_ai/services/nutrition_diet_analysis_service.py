"""
Servicio de Análisis de Nutrición y Dieta - Sistema completo de nutrición
"""

from typing import Dict, List, Optional
from datetime import datetime
import statistics


class NutritionDietAnalysisService:
    """Servicio de análisis de nutrición y dieta"""
    
    def __init__(self):
        """Inicializa el servicio de nutrición"""
        pass
    
    def record_meal(
        self,
        user_id: str,
        meal_data: Dict
    ) -> Dict:
        """
        Registra una comida
        
        Args:
            user_id: ID del usuario
            meal_data: Datos de la comida
        
        Returns:
            Comida registrada
        """
        meal = {
            "id": f"meal_{datetime.now().timestamp()}",
            "user_id": user_id,
            "meal_data": meal_data,
            "meal_type": meal_data.get("meal_type", "other"),
            "calories": meal_data.get("calories", 0),
            "nutrients": meal_data.get("nutrients", {}),
            "timestamp": meal_data.get("timestamp", datetime.now().isoformat()),
            "recorded_at": datetime.now().isoformat()
        }
        
        return meal
    
    def analyze_nutrition_patterns(
        self,
        user_id: str,
        meals: List[Dict],
        days: int = 30
    ) -> Dict:
        """
        Analiza patrones nutricionales
        
        Args:
            user_id: ID del usuario
            meals: Lista de comidas
            days: Número de días
        
        Returns:
            Análisis de patrones nutricionales
        """
        if not meals:
            return {
                "user_id": user_id,
                "analysis": "no_data"
            }
        
        total_calories = sum(m.get("calories", 0) for m in meals)
        avg_daily_calories = total_calories / days if days > 0 else 0
        
        return {
            "user_id": user_id,
            "period_days": days,
            "total_meals": len(meals),
            "total_calories": total_calories,
            "average_daily_calories": round(avg_daily_calories, 2),
            "nutrition_breakdown": self._analyze_nutrients(meals),
            "meal_timing_patterns": self._analyze_meal_timing(meals),
            "recommendations": self._generate_nutrition_recommendations(meals, avg_daily_calories),
            "generated_at": datetime.now().isoformat()
        }
    
    def detect_nutrition_triggers(
        self,
        user_id: str,
        meals: List[Dict]
    ) -> Dict:
        """
        Detecta triggers nutricionales
        
        Args:
            user_id: ID del usuario
            meals: Lista de comidas
        
        Returns:
            Triggers nutricionales detectados
        """
        triggers = []
        
        # Detectar comidas irregulares
        meal_times = [m.get("timestamp") for m in meals if m.get("timestamp")]
        if len(meal_times) < 2:
            triggers.append({
                "type": "irregular_meals",
                "severity": "medium",
                "description": "Patrón de comidas irregular detectado"
            })
        
        # Detectar déficit calórico extremo
        daily_calories = [sum(m.get("calories", 0) for m in meals if self._is_same_day(m.get("timestamp"), day)) 
                         for day in range(7)]
        if any(cal < 800 for cal in daily_calories if cal > 0):
            triggers.append({
                "type": "low_calorie_intake",
                "severity": "high",
                "description": "Ingesta calórica muy baja detectada"
            })
        
        return {
            "user_id": user_id,
            "triggers_detected": triggers,
            "total_triggers": len(triggers),
            "recommendations": self._generate_trigger_recommendations(triggers),
            "generated_at": datetime.now().isoformat()
        }
    
    def _analyze_nutrients(self, meals: List[Dict]) -> Dict:
        """Analiza nutrientes"""
        total_nutrients = {}
        
        for meal in meals:
            nutrients = meal.get("nutrients", {})
            for nutrient, value in nutrients.items():
                total_nutrients[nutrient] = total_nutrients.get(nutrient, 0) + value
        
        return total_nutrients
    
    def _analyze_meal_timing(self, meals: List[Dict]) -> Dict:
        """Analiza patrones de horarios de comida"""
        meal_times = []
        
        for meal in meals:
            timestamp = meal.get("timestamp")
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp)
                    meal_times.append(dt.hour)
                except:
                    pass
        
        if not meal_times:
            return {}
        
        return {
            "average_meal_time": round(statistics.mean(meal_times), 2),
            "meal_frequency": len(meal_times),
            "consistency": "regular" if len(set(meal_times)) <= 3 else "irregular"
        }
    
    def _is_same_day(self, timestamp: Optional[str], day_offset: int) -> bool:
        """Verifica si timestamp corresponde al día especificado"""
        if not timestamp:
            return False
        try:
            dt = datetime.fromisoformat(timestamp)
            target_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            return (dt.date() - target_date.date()).days == day_offset
        except:
            return False
    
    def _generate_nutrition_recommendations(self, meals: List[Dict], avg_calories: float) -> List[str]:
        """Genera recomendaciones nutricionales"""
        recommendations = []
        
        if avg_calories < 1200:
            recommendations.append("Considera aumentar tu ingesta calórica diaria")
        elif avg_calories > 3000:
            recommendations.append("Considera moderar tu ingesta calórica")
        
        return recommendations
    
    def _generate_trigger_recommendations(self, triggers: List[Dict]) -> List[str]:
        """Genera recomendaciones basadas en triggers"""
        if triggers:
            return [
                "Mantén un horario regular de comidas",
                "Asegúrate de consumir suficientes calorías diarias"
            ]
        return []

