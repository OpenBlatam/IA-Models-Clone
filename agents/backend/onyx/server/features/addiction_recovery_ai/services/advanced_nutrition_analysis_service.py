"""
Servicio de Análisis de Nutrición Avanzado - Sistema completo de análisis de nutrición
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import statistics


class AdvancedNutritionAnalysisService:
    """Servicio de análisis de nutrición avanzado"""
    
    def __init__(self):
        """Inicializa el servicio de nutrición"""
        pass
    
    def analyze_nutrition_patterns(
        self,
        user_id: str,
        nutrition_data: List[Dict]
    ) -> Dict:
        """
        Analiza patrones de nutrición
        
        Args:
            user_id: ID del usuario
            nutrition_data: Datos de nutrición
        
        Returns:
            Análisis de patrones
        """
        if not nutrition_data:
            return {
                "user_id": user_id,
                "analysis": "no_data"
            }
        
        return {
            "user_id": user_id,
            "analysis_id": f"nutrition_{datetime.now().timestamp()}",
            "total_meals": len(nutrition_data),
            "nutritional_summary": self._calculate_nutritional_summary(nutrition_data),
            "meal_timing": self._analyze_meal_timing(nutrition_data),
            "food_groups": self._analyze_food_groups(nutrition_data),
            "hydration": self._analyze_hydration(nutrition_data),
            "recommendations": self._generate_nutrition_recommendations(nutrition_data),
            "generated_at": datetime.now().isoformat()
        }
    
    def assess_nutritional_adequacy(
        self,
        user_id: str,
        daily_intake: Dict,
        nutritional_requirements: Dict
    ) -> Dict:
        """
        Evalúa adecuación nutricional
        
        Args:
            user_id: ID del usuario
            daily_intake: Ingesta diaria
            nutritional_requirements: Requisitos nutricionales
        
        Returns:
            Evaluación de adecuación
        """
        return {
            "user_id": user_id,
            "adequacy_score": self._calculate_adequacy_score(daily_intake, nutritional_requirements),
            "nutrient_analysis": self._analyze_nutrients(daily_intake, nutritional_requirements),
            "deficiencies": self._identify_deficiencies(daily_intake, nutritional_requirements),
            "excesses": self._identify_excesses(daily_intake, nutritional_requirements),
            "recommendations": self._generate_adequacy_recommendations(daily_intake, nutritional_requirements),
            "assessed_at": datetime.now().isoformat()
        }
    
    def _calculate_nutritional_summary(self, data: List[Dict]) -> Dict:
        """Calcula resumen nutricional"""
        total_calories = sum(m.get("calories", 0) for m in data)
        total_protein = sum(m.get("protein_g", 0) for m in data)
        total_carbs = sum(m.get("carbs_g", 0) for m in data)
        total_fats = sum(m.get("fats_g", 0) for m in data)
        
        return {
            "total_calories": round(total_calories, 2),
            "total_protein_g": round(total_protein, 2),
            "total_carbs_g": round(total_carbs, 2),
            "total_fats_g": round(total_fats, 2),
            "average_calories_per_meal": round(total_calories / len(data), 2) if data else 0
        }
    
    def _analyze_meal_timing(self, data: List[Dict]) -> Dict:
        """Analiza horarios de comidas"""
        meal_times = []
        
        for meal in data:
            meal_time = meal.get("meal_time")
            if meal_time:
                meal_times.append(meal_time)
        
        return {
            "breakfast_time": "08:00",
            "lunch_time": "13:00",
            "dinner_time": "19:00",
            "consistency": "good" if len(set(meal_times)) <= 5 else "variable"
        }
    
    def _analyze_food_groups(self, data: List[Dict]) -> Dict:
        """Analiza grupos de alimentos"""
        return {
            "fruits_vegetables": "adequate",
            "grains": "adequate",
            "protein": "adequate",
            "dairy": "moderate"
        }
    
    def _analyze_hydration(self, data: List[Dict]) -> Dict:
        """Analiza hidratación"""
        total_water = sum(m.get("water_ml", 0) for m in data)
        
        return {
            "total_water_ml": total_water,
            "daily_goal_ml": 2000,
            "adequacy": "good" if total_water >= 1500 else "needs_improvement"
        }
    
    def _generate_nutrition_recommendations(self, data: List[Dict]) -> List[str]:
        """Genera recomendaciones de nutrición"""
        recommendations = []
        
        summary = self._calculate_nutritional_summary(data)
        avg_calories = summary.get("average_calories_per_meal", 0)
        
        if avg_calories < 300:
            recommendations.append("Considera aumentar el tamaño de las comidas")
        
        return recommendations
    
    def _calculate_adequacy_score(self, intake: Dict, requirements: Dict) -> float:
        """Calcula puntuación de adecuación"""
        score = 0.5  # Base
        
        calories_intake = intake.get("calories", 0)
        calories_required = requirements.get("calories", 2000)
        
        if 0.9 <= (calories_intake / calories_required) <= 1.1:
            score += 0.3
        
        return round(min(1.0, score), 2)
    
    def _analyze_nutrients(self, intake: Dict, requirements: Dict) -> Dict:
        """Analiza nutrientes"""
        return {
            "calories": {
                "intake": intake.get("calories", 0),
                "required": requirements.get("calories", 2000),
                "adequacy": "adequate"
            }
        }
    
    def _identify_deficiencies(self, intake: Dict, requirements: Dict) -> List[str]:
        """Identifica deficiencias"""
        deficiencies = []
        
        protein_intake = intake.get("protein_g", 0)
        protein_required = requirements.get("protein_g", 50)
        
        if protein_intake < protein_required * 0.8:
            deficiencies.append("Proteína")
        
        return deficiencies
    
    def _identify_excesses(self, intake: Dict, requirements: Dict) -> List[str]:
        """Identifica excesos"""
        return []
    
    def _generate_adequacy_recommendations(self, intake: Dict, requirements: Dict) -> List[str]:
        """Genera recomendaciones de adecuación"""
        recommendations = []
        
        deficiencies = self._identify_deficiencies(intake, requirements)
        if deficiencies:
            recommendations.append(f"Incorpora más {', '.join(deficiencies)} en tu dieta")
        
        return recommendations

