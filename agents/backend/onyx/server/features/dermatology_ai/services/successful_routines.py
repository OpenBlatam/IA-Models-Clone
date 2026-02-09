"""
Sistema de rutinas exitosas de otros usuarios
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import uuid
import statistics


@dataclass
class SuccessfulRoutine:
    """Rutina exitosa"""
    id: str
    user_id: str
    routine_name: str
    products: List[Dict]
    skin_type: str
    age_range: str
    improvement_percentage: float
    time_to_results_weeks: int
    user_rating: float
    verified: bool = False
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "routine_name": self.routine_name,
            "products": self.products,
            "skin_type": self.skin_type,
            "age_range": self.age_range,
            "improvement_percentage": self.improvement_percentage,
            "time_to_results_weeks": self.time_to_results_weeks,
            "user_rating": self.user_rating,
            "verified": self.verified
        }


@dataclass
class RoutineMatch:
    """Coincidencia de rutina"""
    routine: SuccessfulRoutine
    match_score: float
    similarity_factors: List[str]
    recommendation: str
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "routine": self.routine.to_dict(),
            "match_score": self.match_score,
            "similarity_factors": self.similarity_factors,
            "recommendation": self.recommendation
        }


class SuccessfulRoutinesSystem:
    """Sistema de rutinas exitosas"""
    
    def __init__(self):
        """Inicializa el sistema"""
        self.routines: Dict[str, SuccessfulRoutine] = {}  # routine_id -> routine
    
    def add_successful_routine(self, user_id: str, routine_name: str, products: List[Dict],
                              skin_type: str, age_range: str, improvement_percentage: float,
                              time_to_results_weeks: int, user_rating: float,
                              verified: bool = False) -> SuccessfulRoutine:
        """Agrega una rutina exitosa"""
        routine = SuccessfulRoutine(
            id=str(uuid.uuid4()),
            user_id=user_id,
            routine_name=routine_name,
            products=products,
            skin_type=skin_type,
            age_range=age_range,
            improvement_percentage=improvement_percentage,
            time_to_results_weeks=time_to_results_weeks,
            user_rating=user_rating,
            verified=verified
        )
        
        self.routines[routine.id] = routine
        return routine
    
    def find_matching_routines(self, user_skin_type: str, user_age_range: str,
                              user_concerns: List[str], limit: int = 5) -> List[RoutineMatch]:
        """Encuentra rutinas que coincidan con el usuario"""
        matches = []
        
        for routine in self.routines.values():
            match_score = 0.0
            similarity_factors = []
            
            # Coincidencia de tipo de piel
            if routine.skin_type == user_skin_type:
                match_score += 0.4
                similarity_factors.append("Mismo tipo de piel")
            
            # Coincidencia de rango de edad
            if routine.age_range == user_age_range:
                match_score += 0.3
                similarity_factors.append("Mismo rango de edad")
            
            # Coincidencia de productos (simplificado)
            # En producción sería más sofisticado
            match_score += 0.2  # Base
            
            # Bonus por alta mejora
            if routine.improvement_percentage > 30:
                match_score += 0.1
                similarity_factors.append("Alta tasa de mejora")
            
            if match_score > 0.3:  # Umbral mínimo
                recommendation = f"Rutina con {routine.improvement_percentage:.1f}% de mejora en {routine.time_to_results_weeks} semanas"
                
                match = RoutineMatch(
                    routine=routine,
                    match_score=match_score,
                    similarity_factors=similarity_factors,
                    recommendation=recommendation
                )
                matches.append(match)
        
        # Ordenar por score de coincidencia
        matches.sort(key=lambda x: x.match_score, reverse=True)
        
        return matches[:limit]
    
    def get_top_routines(self, skin_type: Optional[str] = None,
                        age_range: Optional[str] = None, limit: int = 10) -> List[SuccessfulRoutine]:
        """Obtiene rutinas top"""
        routines = list(self.routines.values())
        
        # Filtrar
        if skin_type:
            routines = [r for r in routines if r.skin_type == skin_type]
        if age_range:
            routines = [r for r in routines if r.age_range == age_range]
        
        # Ordenar por mejora y rating
        routines.sort(
            key=lambda x: (x.improvement_percentage * 0.6 + x.user_rating * 10 * 0.4),
            reverse=True
        )
        
        return routines[:limit]
    
    def get_routine_statistics(self, routine_id: str) -> Dict:
        """Obtiene estadísticas de una rutina"""
        routine = self.routines.get(routine_id)
        
        if not routine:
            return {}
        
        return {
            "routine_id": routine_id,
            "routine_name": routine.routine_name,
            "total_users": 1,  # Simulado
            "average_improvement": routine.improvement_percentage,
            "average_rating": routine.user_rating,
            "average_time_to_results": routine.time_to_results_weeks,
            "verification_status": "verified" if routine.verified else "unverified"
        }






