"""
Sistema de seguimiento de hábitos alimenticios
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import uuid
import statistics


@dataclass
class DietRecord:
    """Registro de dieta"""
    id: str
    user_id: str
    record_date: str
    meal_type: str  # "breakfast", "lunch", "dinner", "snack"
    foods: List[str] = None
    water_intake_ml: Optional[int] = None
    alcohol_consumption: Optional[str] = None  # "none", "light", "moderate", "heavy"
    notes: Optional[str] = None
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.foods is None:
            self.foods = []
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "record_date": self.record_date,
            "meal_type": self.meal_type,
            "foods": self.foods,
            "water_intake_ml": self.water_intake_ml,
            "alcohol_consumption": self.alcohol_consumption,
            "notes": self.notes,
            "created_at": self.created_at
        }


@dataclass
class DietAnalysis:
    """Análisis de dieta"""
    user_id: str
    average_water_intake: Optional[float] = None
    skin_beneficial_foods: List[str] = None
    problematic_foods: List[str] = None
    skin_impact: Dict = None
    recommendations: List[str] = None
    days_analyzed: int = 0
    
    def __post_init__(self):
        if self.skin_beneficial_foods is None:
            self.skin_beneficial_foods = []
        if self.problematic_foods is None:
            self.problematic_foods = []
        if self.skin_impact is None:
            self.skin_impact = {}
        if self.recommendations is None:
            self.recommendations = []
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "user_id": self.user_id,
            "average_water_intake": self.average_water_intake,
            "skin_beneficial_foods": self.skin_beneficial_foods,
            "problematic_foods": self.problematic_foods,
            "skin_impact": self.skin_impact,
            "recommendations": self.recommendations,
            "days_analyzed": self.days_analyzed
        }


class DietTracker:
    """Sistema de seguimiento de hábitos alimenticios"""
    
    def __init__(self):
        """Inicializa el tracker"""
        self.records: Dict[str, List[DietRecord]] = {}
    
    def add_diet_record(self, user_id: str, record_date: str, meal_type: str,
                       foods: Optional[List[str]] = None,
                       water_intake_ml: Optional[int] = None,
                       alcohol_consumption: Optional[str] = None,
                       notes: Optional[str] = None) -> DietRecord:
        """Agrega registro de dieta"""
        record = DietRecord(
            id=str(uuid.uuid4()),
            user_id=user_id,
            record_date=record_date,
            meal_type=meal_type,
            foods=foods or [],
            water_intake_ml=water_intake_ml,
            alcohol_consumption=alcohol_consumption,
            notes=notes
        )
        
        if user_id not in self.records:
            self.records[user_id] = []
        
        self.records[user_id].append(record)
        return record
    
    def analyze_diet(self, user_id: str, days: int = 30) -> DietAnalysis:
        """Analiza dieta"""
        user_records = self.records.get(user_id, [])
        
        if not user_records:
            return DietAnalysis(
                user_id=user_id,
                recommendations=["Agrega registros de dieta para análisis"]
            )
        
        cutoff = datetime.now().date() - timedelta(days=days)
        recent_records = [
            r for r in user_records
            if datetime.fromisoformat(r.record_date).date() >= cutoff
        ]
        
        if not recent_records:
            return DietAnalysis(
                user_id=user_id,
                recommendations=["No hay registros recientes"]
            )
        
        # Calcular promedio de agua
        water_values = [r.water_intake_ml for r in recent_records if r.water_intake_ml]
        avg_water = statistics.mean(water_values) if water_values else None
        
        # Alimentos beneficiosos para la piel
        beneficial_foods = ["salmon", "avocado", "berries", "nuts", "green_tea", "dark_chocolate", 
                           "spinach", "sweet_potato", "tomatoes", "walnuts"]
        
        # Alimentos problemáticos
        problematic_foods = ["sugar", "processed_foods", "dairy", "fried_foods", "alcohol"]
        
        # Analizar alimentos consumidos
        all_foods = []
        for record in recent_records:
            all_foods.extend([f.lower() for f in record.foods])
        
        beneficial_found = [f for f in all_foods if any(bf in f for bf in beneficial_foods)]
        problematic_found = [f for f in all_foods if any(pf in f for pf in problematic_foods)]
        
        # Impacto en la piel
        skin_impact = {
            "hydration_status": "good" if avg_water and avg_water >= 2000 else "needs_improvement",
            "antioxidant_intake": "good" if len(beneficial_found) > 5 else "needs_improvement",
            "inflammation_risk": "high" if len(problematic_found) > 3 else "low"
        }
        
        # Recomendaciones
        recommendations = []
        
        if avg_water and avg_water < 2000:
            recommendations.append(f"Consumo de agua bajo ({avg_water:.0f}ml/día). Objetivo: 2000-3000ml")
        
        if len(beneficial_found) < 3:
            recommendations.append("Aumenta consumo de alimentos beneficiosos para la piel (pescado, frutas, verduras)")
        
        if len(problematic_found) > 5:
            recommendations.append("Reduce consumo de alimentos procesados y azúcares")
        
        if any(r.alcohol_consumption in ["moderate", "heavy"] for r in recent_records):
            recommendations.append("El alcohol puede deshidratar la piel. Modera el consumo")
        
        if not recommendations:
            recommendations.append("Tu dieta parece balanceada para la salud de la piel")
        
        return DietAnalysis(
            user_id=user_id,
            average_water_intake=avg_water,
            skin_beneficial_foods=list(set(beneficial_found)),
            problematic_foods=list(set(problematic_found)),
            skin_impact=skin_impact,
            recommendations=recommendations,
            days_analyzed=len(recent_records)
        )


