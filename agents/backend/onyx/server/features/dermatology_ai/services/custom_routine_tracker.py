"""
Sistema de seguimiento de rutinas personalizadas
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import uuid
import statistics


@dataclass
class RoutineStep:
    """Paso de rutina"""
    step_id: str
    step_number: int
    product_name: str
    product_category: str
    application_time: str  # "morning", "evening", "both"
    frequency: str  # "daily", "weekly", "as_needed"
    notes: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "step_id": self.step_id,
            "step_number": self.step_number,
            "product_name": self.product_name,
            "product_category": self.product_category,
            "application_time": self.application_time,
            "frequency": self.frequency,
            "notes": self.notes
        }


@dataclass
class CustomRoutine:
    """Rutina personalizada"""
    routine_id: str
    user_id: str
    routine_name: str
    steps: List[RoutineStep]
    created_at: str = None
    last_modified: str = None
    is_active: bool = True
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.last_modified is None:
            self.last_modified = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "routine_id": self.routine_id,
            "user_id": self.user_id,
            "routine_name": self.routine_name,
            "steps": [s.to_dict() for s in self.steps],
            "created_at": self.created_at,
            "last_modified": self.last_modified,
            "is_active": self.is_active
        }


@dataclass
class RoutineUsage:
    """Uso de rutina"""
    usage_id: str
    routine_id: str
    user_id: str
    usage_date: str
    steps_completed: List[str]  # step_ids
    completion_percentage: float
    notes: Optional[str] = None
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "usage_id": self.usage_id,
            "routine_id": self.routine_id,
            "user_id": self.user_id,
            "usage_date": self.usage_date,
            "steps_completed": self.steps_completed,
            "completion_percentage": self.completion_percentage,
            "notes": self.notes,
            "created_at": self.created_at
        }


@dataclass
class RoutineAnalysis:
    """Análisis de rutina"""
    routine_id: str
    user_id: str
    total_uses: int
    average_completion: float
    consistency_score: float
    most_skipped_steps: List[str]
    recommendations: List[str]
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "routine_id": self.routine_id,
            "user_id": self.user_id,
            "total_uses": self.total_uses,
            "average_completion": self.average_completion,
            "consistency_score": self.consistency_score,
            "most_skipped_steps": self.most_skipped_steps,
            "recommendations": self.recommendations
        }


class CustomRoutineTracker:
    """Sistema de seguimiento de rutinas personalizadas"""
    
    def __init__(self):
        """Inicializa el tracker"""
        self.routines: Dict[str, CustomRoutine] = {}
        self.usages: Dict[str, List[RoutineUsage]] = {}  # routine_id -> [usages]
        self.user_routines: Dict[str, List[str]] = {}  # user_id -> [routine_ids]
    
    def create_routine(self, user_id: str, routine_name: str, steps: List[Dict]) -> CustomRoutine:
        """Crea rutina personalizada"""
        routine_id = str(uuid.uuid4())
        
        routine_steps = []
        for idx, step_data in enumerate(steps, 1):
            step = RoutineStep(
                step_id=str(uuid.uuid4()),
                step_number=idx,
                product_name=step_data.get("product_name", ""),
                product_category=step_data.get("product_category", ""),
                application_time=step_data.get("application_time", "both"),
                frequency=step_data.get("frequency", "daily"),
                notes=step_data.get("notes")
            )
            routine_steps.append(step)
        
        routine = CustomRoutine(
            routine_id=routine_id,
            user_id=user_id,
            routine_name=routine_name,
            steps=routine_steps
        )
        
        self.routines[routine_id] = routine
        
        if user_id not in self.user_routines:
            self.user_routines[user_id] = []
        self.user_routines[user_id].append(routine_id)
        
        return routine
    
    def record_usage(self, routine_id: str, user_id: str, usage_date: str,
                     steps_completed: List[str], notes: Optional[str] = None) -> RoutineUsage:
        """Registra uso de rutina"""
        routine = self.routines.get(routine_id)
        if not routine or routine.user_id != user_id:
            raise ValueError("Rutina no encontrada")
        
        total_steps = len(routine.steps)
        completed_count = len(steps_completed)
        completion_percentage = (completed_count / total_steps * 100) if total_steps > 0 else 0.0
        
        usage = RoutineUsage(
            usage_id=str(uuid.uuid4()),
            routine_id=routine_id,
            user_id=user_id,
            usage_date=usage_date,
            steps_completed=steps_completed,
            completion_percentage=completion_percentage,
            notes=notes
        )
        
        if routine_id not in self.usages:
            self.usages[routine_id] = []
        self.usages[routine_id].append(usage)
        
        return usage
    
    def analyze_routine(self, routine_id: str, user_id: str, days: int = 30) -> RoutineAnalysis:
        """Analiza uso de rutina"""
        routine = self.routines.get(routine_id)
        if not routine or routine.user_id != user_id:
            raise ValueError("Rutina no encontrada")
        
        usages = self.usages.get(routine_id, [])
        
        if not usages:
            return RoutineAnalysis(
                routine_id=routine_id,
                user_id=user_id,
                total_uses=0,
                average_completion=0.0,
                consistency_score=0.0,
                most_skipped_steps=[],
                recommendations=["No hay datos de uso"]
            )
        
        cutoff = datetime.now().date() - timedelta(days=days)
        recent_usages = [
            u for u in usages
            if datetime.fromisoformat(u.usage_date).date() >= cutoff
        ]
        
        if not recent_usages:
            return RoutineAnalysis(
                routine_id=routine_id,
                user_id=user_id,
                total_uses=0,
                average_completion=0.0,
                consistency_score=0.0,
                most_skipped_steps=[],
                recommendations=["No hay datos recientes"]
            )
        
        # Calcular promedio de completitud
        completions = [u.completion_percentage for u in recent_usages]
        avg_completion = statistics.mean(completions) if completions else 0.0
        
        # Calcular consistencia (frecuencia de uso)
        usage_dates = [datetime.fromisoformat(u.usage_date).date() for u in recent_usages]
        unique_days = len(set(usage_dates))
        consistency_score = (unique_days / days) * 100 if days > 0 else 0.0
        
        # Pasos más omitidos
        all_step_ids = {step.step_id for step in routine.steps}
        step_skip_counts = {step_id: 0 for step_id in all_step_ids}
        
        for usage in recent_usages:
            completed = set(usage.steps_completed)
            for step_id in all_step_ids:
                if step_id not in completed:
                    step_skip_counts[step_id] += 1
        
        most_skipped = sorted(step_skip_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        most_skipped_steps = [step_id for step_id, count in most_skipped if count > 0]
        
        # Recomendaciones
        recommendations = []
        
        if avg_completion < 70:
            recommendations.append("Completitud baja. Considera simplificar la rutina")
        
        if consistency_score < 50:
            recommendations.append("Consistencia baja. Intenta usar la rutina más regularmente")
        
        if most_skipped_steps:
            skipped_names = [s.product_name for s in routine.steps if s.step_id in most_skipped_steps]
            recommendations.append(f"Pasos frecuentemente omitidos: {', '.join(skipped_names)}")
        
        if not recommendations:
            recommendations.append("Excelente adherencia a la rutina")
        
        return RoutineAnalysis(
            routine_id=routine_id,
            user_id=user_id,
            total_uses=len(recent_usages),
            average_completion=avg_completion,
            consistency_score=consistency_score,
            most_skipped_steps=most_skipped_steps,
            recommendations=recommendations
        )
    
    def get_user_routines(self, user_id: str) -> List[CustomRoutine]:
        """Obtiene rutinas del usuario"""
        routine_ids = self.user_routines.get(user_id, [])
        return [self.routines[rid] for rid in routine_ids if rid in self.routines]


