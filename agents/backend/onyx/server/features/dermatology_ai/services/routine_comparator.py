"""
Sistema de comparación de rutinas de skincare
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import uuid


@dataclass
class SkincareRoutine:
    """Rutina de skincare"""
    id: str
    user_id: str
    name: str
    products: List[Dict]  # Lista de productos con frecuencia
    steps: List[str]  # Pasos de la rutina
    morning: bool  # True si es rutina matutina
    evening: bool  # True si es rutina nocturna
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "name": self.name,
            "products": self.products,
            "steps": self.steps,
            "morning": self.morning,
            "evening": self.evening,
            "created_at": self.created_at
        }


@dataclass
class RoutineComparison:
    """Comparación de rutinas"""
    routine1_id: str
    routine2_id: str
    similarity_score: float
    differences: List[str]
    recommendations: List[str]
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "routine1_id": self.routine1_id,
            "routine2_id": self.routine2_id,
            "similarity_score": self.similarity_score,
            "differences": self.differences,
            "recommendations": self.recommendations,
            "created_at": self.created_at
        }


class RoutineComparator:
    """Sistema de comparación de rutinas"""
    
    def __init__(self):
        """Inicializa el comparador"""
        self.routines: Dict[str, SkincareRoutine] = {}  # routine_id -> routine
    
    def create_routine(self, user_id: str, name: str, products: List[Dict],
                      steps: List[str], morning: bool = True,
                      evening: bool = False) -> SkincareRoutine:
        """Crea una nueva rutina"""
        routine = SkincareRoutine(
            id=str(uuid.uuid4()),
            user_id=user_id,
            name=name,
            products=products,
            steps=steps,
            morning=morning,
            evening=evening
        )
        
        self.routines[routine.id] = routine
        return routine
    
    def compare_routines(self, routine1_id: str, routine2_id: str) -> RoutineComparison:
        """Compara dos rutinas"""
        routine1 = self.routines.get(routine1_id)
        routine2 = self.routines.get(routine2_id)
        
        if not routine1 or not routine2:
            raise ValueError("Una o ambas rutinas no existen")
        
        # Calcular similitud
        similarity = self._calculate_similarity(routine1, routine2)
        
        # Encontrar diferencias
        differences = self._find_differences(routine1, routine2)
        
        # Generar recomendaciones
        recommendations = self._generate_recommendations(routine1, routine2, differences)
        
        return RoutineComparison(
            routine1_id=routine1_id,
            routine2_id=routine2_id,
            similarity_score=similarity,
            differences=differences,
            recommendations=recommendations
        )
    
    def _calculate_similarity(self, routine1: SkincareRoutine,
                             routine2: SkincareRoutine) -> float:
        """Calcula similitud entre rutinas"""
        # Comparar productos
        products1 = {p.get("id") for p in routine1.products}
        products2 = {p.get("id") for p in routine2.products}
        
        common_products = len(products1 & products2)
        total_products = len(products1 | products2)
        
        product_similarity = common_products / total_products if total_products > 0 else 0.0
        
        # Comparar pasos
        steps1 = set(routine1.steps)
        steps2 = set(routine2.steps)
        
        common_steps = len(steps1 & steps2)
        total_steps = len(steps1 | steps2)
        
        step_similarity = common_steps / total_steps if total_steps > 0 else 0.0
        
        # Similitud combinada
        similarity = (product_similarity * 0.6 + step_similarity * 0.4)
        
        return float(similarity)
    
    def _find_differences(self, routine1: SkincareRoutine,
                         routine2: SkincareRoutine) -> List[str]:
        """Encuentra diferencias entre rutinas"""
        differences = []
        
        # Diferencias en productos
        products1 = {p.get("id") for p in routine1.products}
        products2 = {p.get("id") for p in routine2.products}
        
        only_in_1 = products1 - products2
        only_in_2 = products2 - products1
        
        if only_in_1:
            differences.append(f"Rutina 1 tiene {len(only_in_1)} productos únicos")
        if only_in_2:
            differences.append(f"Rutina 2 tiene {len(only_in_2)} productos únicos")
        
        # Diferencias en pasos
        steps1 = set(routine1.steps)
        steps2 = set(routine2.steps)
        
        if steps1 != steps2:
            differences.append("Los pasos de las rutinas son diferentes")
        
        # Diferencias en horario
        if routine1.morning != routine2.morning:
            differences.append("Diferencia en rutina matutina")
        if routine1.evening != routine2.evening:
            differences.append("Diferencia en rutina nocturna")
        
        return differences
    
    def _generate_recommendations(self, routine1: SkincareRoutine,
                                  routine2: SkincareRoutine,
                                  differences: List[str]) -> List[str]:
        """Genera recomendaciones basadas en comparación"""
        recommendations = []
        
        # Recomendación sobre productos
        products1 = {p.get("id") for p in routine1.products}
        products2 = {p.get("id") for p in routine2.products}
        
        if len(products1) < len(products2):
            recommendations.append("Rutina 2 tiene más productos. Considera agregar más productos a rutina 1.")
        elif len(products1) > len(products2):
            recommendations.append("Rutina 1 tiene más productos. Considera simplificar rutina 2.")
        
        # Recomendación sobre pasos
        if len(routine1.steps) != len(routine2.steps):
            recommendations.append("Las rutinas tienen diferente número de pasos. Considera estandarizar.")
        
        return recommendations
    
    def get_user_routines(self, user_id: str) -> List[SkincareRoutine]:
        """Obtiene rutinas de un usuario"""
        return [r for r in self.routines.values() if r.user_id == user_id]
    
    def get_routine(self, routine_id: str) -> Optional[SkincareRoutine]:
        """Obtiene una rutina por ID"""
        return self.routines.get(routine_id)






