"""
Routine Manager
===============

Gestión de rutinas diarias para artistas.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, time, timedelta
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class RoutineType(Enum):
    """Tipos de rutinas."""
    MORNING = "morning"
    AFTERNOON = "afternoon"
    EVENING = "evening"
    NIGHT = "night"
    DAILY = "daily"
    WEEKLY = "weekly"
    CUSTOM = "custom"


class RoutineStatus(Enum):
    """Estado de la rutina."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    OVERDUE = "overdue"


@dataclass
class RoutineTask:
    """Tarea de rutina."""
    id: str
    title: str
    description: str
    routine_type: RoutineType
    scheduled_time: time
    duration_minutes: int
    priority: int = 5  # 1-10, 10 es más importante
    days_of_week: List[int] = None  # 0=Lunes, 6=Domingo
    is_required: bool = True
    reminders: List[time] = None
    notes: Optional[str] = None
    
    def __post_init__(self):
        if self.days_of_week is None:
            self.days_of_week = list(range(7))  # Todos los días
        if self.reminders is None:
            self.reminders = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        data = asdict(self)
        data['routine_type'] = self.routine_type.value
        data['scheduled_time'] = self.scheduled_time.strftime("%H:%M:%S")
        if self.reminders:
            data['reminders'] = [r.strftime("%H:%M:%S") for r in self.reminders]
        return data


@dataclass
class RoutineCompletion:
    """Registro de completación de rutina."""
    task_id: str
    completed_at: datetime
    status: RoutineStatus
    notes: Optional[str] = None
    actual_duration_minutes: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        data = asdict(self)
        data['status'] = self.status.value
        data['completed_at'] = self.completed_at.isoformat()
        return data


class RoutineManager:
    """Gestor de rutinas para artistas."""
    
    def __init__(self, artist_id: str):
        """
        Inicializar gestor de rutinas.
        
        Args:
            artist_id: ID del artista
        """
        self.artist_id = artist_id
        self.routines: Dict[str, RoutineTask] = {}
        self.completions: List[RoutineCompletion] = []
        self._logger = logger
    
    def add_routine(self, routine: RoutineTask) -> RoutineTask:
        """
        Agregar rutina.
        
        Args:
            routine: Rutina a agregar
        
        Returns:
            Rutina agregada
        """
        if routine.id in self.routines:
            raise ValueError(f"Routine with id {routine.id} already exists")
        
        self.routines[routine.id] = routine
        self._logger.info(f"Added routine {routine.id} for artist {self.artist_id}")
        return routine
    
    def get_routine(self, routine_id: str) -> Optional[RoutineTask]:
        """
        Obtener rutina por ID.
        
        Args:
            routine_id: ID de la rutina
        
        Returns:
            Rutina o None si no existe
        """
        return self.routines.get(routine_id)
    
    def update_routine(self, routine_id: str, **updates) -> RoutineTask:
        """
        Actualizar rutina.
        
        Args:
            routine_id: ID de la rutina
            **updates: Campos a actualizar
        
        Returns:
            Rutina actualizada
        """
        if routine_id not in self.routines:
            raise ValueError(f"Routine {routine_id} not found")
        
        routine = self.routines[routine_id]
        for key, value in updates.items():
            if hasattr(routine, key):
                setattr(routine, key, value)
        
        self._logger.info(f"Updated routine {routine_id} for artist {self.artist_id}")
        return routine
    
    def delete_routine(self, routine_id: str) -> bool:
        """
        Eliminar rutina.
        
        Args:
            routine_id: ID de la rutina
        
        Returns:
            True si se eliminó, False si no existía
        """
        if routine_id in self.routines:
            del self.routines[routine_id]
            self._logger.info(f"Deleted routine {routine_id} for artist {self.artist_id}")
            return True
        return False
    
    def get_routines_for_today(self) -> List[RoutineTask]:
        """
        Obtener rutinas para hoy.
        
        Returns:
            Lista de rutinas programadas para hoy
        """
        today = datetime.now()
        day_of_week = today.weekday()
        
        return [
            routine for routine in self.routines.values()
            if day_of_week in routine.days_of_week
        ]
    
    def get_routines_by_type(self, routine_type: RoutineType) -> List[RoutineTask]:
        """
        Obtener rutinas por tipo.
        
        Args:
            routine_type: Tipo de rutina
        
        Returns:
            Lista de rutinas del tipo especificado
        """
        return [
            routine for routine in self.routines.values()
            if routine.routine_type == routine_type
        ]
    
    def mark_completed(
        self,
        task_id: str,
        status: RoutineStatus = RoutineStatus.COMPLETED,
        notes: Optional[str] = None,
        actual_duration_minutes: Optional[int] = None
    ) -> RoutineCompletion:
        """
        Marcar rutina como completada.
        
        Args:
            task_id: ID de la tarea
            status: Estado de la rutina
            notes: Notas adicionales
            actual_duration_minutes: Duración real en minutos
        
        Returns:
            Registro de completación
        """
        if task_id not in self.routines:
            raise ValueError(f"Routine {task_id} not found")
        
        completion = RoutineCompletion(
            task_id=task_id,
            completed_at=datetime.now(),
            status=status,
            notes=notes,
            actual_duration_minutes=actual_duration_minutes
        )
        
        self.completions.append(completion)
        self._logger.info(f"Marked routine {task_id} as {status.value} for artist {self.artist_id}")
        return completion
    
    def get_completion_history(self, task_id: Optional[str] = None, days: int = 30) -> List[RoutineCompletion]:
        """
        Obtener historial de completaciones.
        
        Args:
            task_id: ID de tarea específica (opcional)
            days: Número de días hacia atrás
        
        Returns:
            Lista de completaciones
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        completions = [
            comp for comp in self.completions
            if comp.completed_at >= cutoff_date
        ]
        
        if task_id:
            completions = [comp for comp in completions if comp.task_id == task_id]
        
        return sorted(completions, key=lambda c: c.completed_at, reverse=True)
    
    def get_completion_rate(self, task_id: str, days: int = 30) -> float:
        """
        Obtener tasa de completación de una rutina.
        
        Args:
            task_id: ID de la tarea
            days: Número de días a analizar
        
        Returns:
            Tasa de completación (0.0 a 1.0)
        """
        if task_id not in self.routines:
            return 0.0
        
        routine = self.routines[task_id]
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Contar días esperados
        today = datetime.now()
        expected_count = 0
        for i in range(days):
            check_date = today - timedelta(days=i)
            if check_date.weekday() in routine.days_of_week:
                expected_count += 1
        
        if expected_count == 0:
            return 0.0
        
        # Contar completaciones
        completed_count = len([
            comp for comp in self.completions
            if comp.task_id == task_id
            and comp.status == RoutineStatus.COMPLETED
            and comp.completed_at >= cutoff_date
        ])
        
        return completed_count / expected_count if expected_count > 0 else 0.0
    
    def get_pending_routines(self) -> List[RoutineTask]:
        """
        Obtener rutinas pendientes para hoy.
        
        Returns:
            Lista de rutinas pendientes
        """
        today_routines = self.get_routines_for_today()
        now = datetime.now()
        current_time = now.time()
        
        pending = []
        for routine in today_routines:
            # Verificar si ya fue completada hoy
            today_completions = [
                comp for comp in self.completions
                if comp.task_id == routine.id
                and comp.completed_at.date() == now.date()
                and comp.status == RoutineStatus.COMPLETED
            ]
            
            if not today_completions and current_time >= routine.scheduled_time:
                pending.append(routine)
        
        return sorted(pending, key=lambda r: r.scheduled_time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "artist_id": self.artist_id,
            "routines": [routine.to_dict() for routine in self.routines.values()],
            "total_routines": len(self.routines),
            "total_completions": len(self.completions)
        }

