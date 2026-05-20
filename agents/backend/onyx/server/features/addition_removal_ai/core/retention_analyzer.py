"""
Retention Analyzer - Sistema de análisis de retención
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class UserSession:
    """Sesión de usuario"""
    user_id: str
    content_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    completed: bool = False


class RetentionAnalyzer:
    """Analizador de retención"""

    def __init__(self):
        """Inicializar analizador"""
        self.sessions: List[UserSession] = []
        self.user_first_visit: Dict[str, datetime] = {}
        self.user_last_visit: Dict[str, datetime] = {}
        self.user_visit_count: Dict[str, int] = defaultdict(int)

    def record_visit(
        self,
        user_id: str,
        content_id: str,
        start_time: Optional[datetime] = None
    ) -> str:
        """
        Registrar visita de usuario.

        Args:
            user_id: ID del usuario
            content_id: ID del contenido
            start_time: Tiempo de inicio (opcional)

        Returns:
            ID de sesión
        """
        import uuid
        
        session_id = str(uuid.uuid4())
        start = start_time or datetime.utcnow()
        
        # Actualizar primera y última visita
        if user_id not in self.user_first_visit:
            self.user_first_visit[user_id] = start
        self.user_last_visit[user_id] = start
        self.user_visit_count[user_id] += 1
        
        session = UserSession(
            user_id=user_id,
            content_id=content_id,
            start_time=start
        )
        
        self.sessions.append(session)
        logger.debug(f"Visita registrada: {user_id} - {content_id}")
        
        return session_id

    def complete_session(
        self,
        user_id: str,
        content_id: str,
        completed: bool = True,
        duration: Optional[float] = None
    ):
        """
        Completar sesión de usuario.

        Args:
            user_id: ID del usuario
            content_id: ID del contenido
            completed: Si se completó el contenido
            duration: Duración en segundos
        """
        # Buscar sesión más reciente sin completar
        for session in reversed(self.sessions):
            if (session.user_id == user_id and
                session.content_id == content_id and
                not session.completed):
                session.end_time = datetime.utcnow()
                session.duration = duration
                session.completed = completed
                logger.debug(f"Sesión completada: {user_id} - {content_id}")
                return
        
        logger.warning(f"No se encontró sesión para completar: {user_id} - {content_id}")

    def calculate_retention_rate(
        self,
        period_days: int = 30,
        cohort: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calcular tasa de retención.

        Args:
            period_days: Período en días
            cohort: Cohorte específica (opcional)

        Returns:
            Análisis de retención
        """
        cutoff_date = datetime.utcnow() - timedelta(days=period_days)
        
        # Filtrar usuarios que visitaron en el período
        active_users = set()
        for user_id, first_visit in self.user_first_visit.items():
            if first_visit >= cutoff_date:
                active_users.add(user_id)
        
        if not active_users:
            return {"error": "No hay usuarios activos en el período"}
        
        # Calcular retención por día
        retention_by_day = {}
        for day in range(period_days):
            day_date = cutoff_date + timedelta(days=day)
            next_day = day_date + timedelta(days=1)
            
            # Usuarios que visitaron en el día inicial
            initial_users = {
                user_id for user_id, first_visit in self.user_first_visit.items()
                if cutoff_date <= first_visit < cutoff_date + timedelta(days=1)
            }
            
            # Usuarios que regresaron en este día
            returning_users = {
                user_id for user_id, last_visit in self.user_last_visit.items()
                if day_date <= last_visit < next_day and user_id in initial_users
            }
            
            retention_rate = (
                len(returning_users) / len(initial_users)
                if initial_users else 0.0
            )
            
            retention_by_day[day] = {
                "day": day,
                "retention_rate": retention_rate,
                "returning_users": len(returning_users),
                "initial_users": len(initial_users)
            }
        
        # Tasa de retención general
        total_users = len(active_users)
        returning_users = {
            user_id for user_id, last_visit in self.user_last_visit.items()
            if last_visit >= cutoff_date
        }
        overall_retention = len(returning_users) / total_users if total_users > 0 else 0
        
        return {
            "period_days": period_days,
            "total_active_users": total_users,
            "returning_users": len(returning_users),
            "overall_retention_rate": overall_retention,
            "retention_by_day": retention_by_day
        }

    def analyze_user_retention(
        self,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Analizar retención de un usuario específico.

        Args:
            user_id: ID del usuario

        Returns:
            Análisis de retención del usuario
        """
        if user_id not in self.user_first_visit:
            return {"error": "Usuario no encontrado"}
        
        first_visit = self.user_first_visit[user_id]
        last_visit = self.user_last_visit[user_id]
        visit_count = self.user_visit_count[user_id]
        
        # Calcular días desde primera visita
        days_since_first = (datetime.utcnow() - first_visit).days + 1
        
        # Calcular días desde última visita
        days_since_last = (datetime.utcnow() - last_visit).days
        
        # Calcular frecuencia de visitas
        visit_frequency = visit_count / days_since_first if days_since_first > 0 else 0
        
        # Sesiones del usuario
        user_sessions = [s for s in self.sessions if s.user_id == user_id]
        completed_sessions = [s for s in user_sessions if s.completed]
        completion_rate = (
            len(completed_sessions) / len(user_sessions)
            if user_sessions else 0.0
        )
        
        # Tiempo promedio de sesión
        durations = [s.duration for s in user_sessions if s.duration]
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        return {
            "user_id": user_id,
            "first_visit": first_visit.isoformat(),
            "last_visit": last_visit.isoformat(),
            "total_visits": visit_count,
            "days_since_first": days_since_first,
            "days_since_last": days_since_last,
            "visit_frequency": visit_frequency,
            "total_sessions": len(user_sessions),
            "completed_sessions": len(completed_sessions),
            "completion_rate": completion_rate,
            "average_session_duration": avg_duration
        }

    def get_retention_cohorts(
        self,
        cohort_size_days: int = 7
    ) -> Dict[str, Any]:
        """
        Obtener análisis de cohortes de retención.

        Args:
            cohort_size_days: Tamaño de cohorte en días

        Returns:
            Análisis de cohortes
        """
        cohorts = defaultdict(list)
        
        # Agrupar usuarios por cohorte
        for user_id, first_visit in self.user_first_visit.items():
            # Redondear a la semana
            cohort_start = first_visit.replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            cohort_key = cohort_start.strftime("%Y-%m-%d")
            cohorts[cohort_key].append(user_id)
        
        cohort_analysis = []
        for cohort_key, users in cohorts.items():
            cohort_retention = []
            
            for user_id in users:
                user_analysis = self.analyze_user_retention(user_id)
                if "error" not in user_analysis:
                    cohort_retention.append(user_analysis)
            
            if cohort_retention:
                avg_retention = sum(
                    u.get("completion_rate", 0) for u in cohort_retention
                ) / len(cohort_retention)
                
                cohort_analysis.append({
                    "cohort": cohort_key,
                    "user_count": len(users),
                    "average_completion_rate": avg_retention,
                    "users": cohort_retention[:10]  # Primeros 10 usuarios
                })
        
        return {
            "cohort_size_days": cohort_size_days,
            "total_cohorts": len(cohort_analysis),
            "cohorts": cohort_analysis
        }






