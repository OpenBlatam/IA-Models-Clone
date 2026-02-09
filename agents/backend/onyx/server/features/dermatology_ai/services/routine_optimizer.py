"""
Sistema de optimización de rutinas de skincare
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class RoutineStep:
    """Paso de rutina"""
    step_number: int
    product_name: str
    product_type: str
    application_time: int  # minutos
    wait_time: Optional[int] = None  # minutos antes del siguiente paso
    notes: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "step_number": self.step_number,
            "product_name": self.product_name,
            "product_type": self.product_type,
            "application_time": self.application_time,
            "wait_time": self.wait_time,
            "notes": self.notes
        }


@dataclass
class OptimizedRoutine:
    """Rutina optimizada"""
    user_id: str
    routine_name: str
    time_of_day: str
    total_duration: int  # minutos
    steps: List[RoutineStep]
    optimization_notes: List[str]
    efficiency_score: float  # 0-1
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "user_id": self.user_id,
            "routine_name": self.routine_name,
            "time_of_day": self.time_of_day,
            "total_duration": self.total_duration,
            "steps": [s.to_dict() for s in self.steps],
            "optimization_notes": self.optimization_notes,
            "efficiency_score": self.efficiency_score
        }


class RoutineOptimizer:
    """Sistema de optimización de rutinas"""
    
    def __init__(self):
        """Inicializa el optimizador"""
        self.routines: Dict[str, List[OptimizedRoutine]] = {}
    
    def optimize_routine(self, user_id: str, products: List[Dict],
                        time_of_day: str, max_duration: Optional[int] = None) -> OptimizedRoutine:
        """Optimiza rutina de skincare"""
        
        # Ordenar productos por tipo (orden de aplicación)
        product_order = {
            "oil_cleanser": 1,
            "water_cleanser": 2,
            "toner": 3,
            "exfoliant": 4,
            "serum": 5,
            "moisturizer": 6,
            "sunscreen": 7,
            "eye_cream": 8
        }
        
        sorted_products = sorted(
            products,
            key=lambda p: product_order.get(p.get("type", ""), 99)
        )
        
        # Crear pasos
        steps = []
        total_duration = 0
        optimization_notes = []
        
        for i, product in enumerate(sorted_products):
            product_name = product.get("name", "Unknown")
            product_type = product.get("type", "unknown")
            
            # Tiempo de aplicación según tipo
            application_time = self._get_application_time(product_type)
            
            # Tiempo de espera (solo para algunos productos)
            wait_time = None
            if product_type in ["serum", "exfoliant", "retinol"]:
                wait_time = 2  # Esperar 2 minutos para absorción
            
            step = RoutineStep(
                step_number=i + 1,
                product_name=product_name,
                product_type=product_type,
                application_time=application_time,
                wait_time=wait_time,
                notes=self._get_step_notes(product_type)
            )
            
            steps.append(step)
            total_duration += application_time
            if wait_time:
                total_duration += wait_time
        
        # Verificar duración máxima
        if max_duration and total_duration > max_duration:
            optimization_notes.append(f"Rutina excede duración máxima ({max_duration} min)")
            optimization_notes.append("Considera reducir pasos o productos")
        
        # Optimizaciones
        if len(steps) > 6:
            optimization_notes.append("Rutina con muchos pasos. Considera simplificar")
        
        # Score de eficiencia
        efficiency_score = min(1.0, max(0.0, 1.0 - (total_duration / 30)))
        
        routine = OptimizedRoutine(
            user_id=user_id,
            routine_name=f"Rutina {time_of_day} optimizada",
            time_of_day=time_of_day,
            total_duration=total_duration,
            steps=steps,
            optimization_notes=optimization_notes,
            efficiency_score=efficiency_score
        )
        
        if user_id not in self.routines:
            self.routines[user_id] = []
        
        self.routines[user_id].append(routine)
        return routine
    
    def _get_application_time(self, product_type: str) -> int:
        """Obtiene tiempo de aplicación según tipo"""
        times = {
            "oil_cleanser": 2,
            "water_cleanser": 1,
            "toner": 1,
            "exfoliant": 2,
            "serum": 1,
            "moisturizer": 2,
            "sunscreen": 1,
            "eye_cream": 1
        }
        return times.get(product_type, 1)
    
    def _get_step_notes(self, product_type: str) -> Optional[str]:
        """Obtiene notas para el paso"""
        notes = {
            "oil_cleanser": "Masajea suavemente por 1-2 minutos",
            "serum": "Aplica sobre piel limpia y húmeda",
            "sunscreen": "Reaplica cada 2 horas si estás al sol"
        }
        return notes.get(product_type)






