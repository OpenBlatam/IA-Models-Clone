"""
Behavior Analyzer - Sistema de análisis de comportamiento del usuario
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


@dataclass
class UserBehavior:
    """Comportamiento de usuario"""
    user_id: str
    content_id: str
    action_type: str
    duration: Optional[float] = None
    timestamp: datetime = None
    metadata: Dict[str, Any] = None


class BehaviorAnalyzer:
    """Analizador de comportamiento"""

    def __init__(self):
        """Inicializar analizador"""
        self.behaviors: List[UserBehavior] = []
        self.user_sessions: Dict[str, List[UserBehavior]] = defaultdict(list)

    def record_behavior(
        self,
        user_id: str,
        content_id: str,
        action_type: str,
        duration: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Registrar comportamiento de usuario.

        Args:
            user_id: ID del usuario
            content_id: ID del contenido
            action_type: Tipo de acción (view, read, share, like, etc.)
            duration: Duración en segundos (opcional)
            metadata: Metadatos adicionales
        """
        behavior = UserBehavior(
            user_id=user_id,
            content_id=content_id,
            action_type=action_type,
            duration=duration,
            timestamp=datetime.utcnow(),
            metadata=metadata or {}
        )
        
        self.behaviors.append(behavior)
        self.user_sessions[user_id].append(behavior)
        
        # Limitar tamaño de sesiones
        if len(self.user_sessions[user_id]) > 1000:
            self.user_sessions[user_id] = self.user_sessions[user_id][-1000:]
        
        logger.debug(f"Comportamiento registrado: {user_id} - {action_type} - {content_id}")

    def analyze_user_behavior(
        self,
        user_id: str,
        period_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Analizar comportamiento de un usuario.

        Args:
            user_id: ID del usuario
            period_days: Período en días (opcional)

        Returns:
            Análisis de comportamiento
        """
        user_behaviors = self.user_sessions.get(user_id, [])
        
        if period_days:
            cutoff_date = datetime.utcnow() - timedelta(days=period_days)
            user_behaviors = [
                b for b in user_behaviors
                if b.timestamp and b.timestamp >= cutoff_date
            ]
        
        if not user_behaviors:
            return {"error": "No hay comportamiento registrado para este usuario"}
        
        # Estadísticas de acciones
        action_counts = Counter(b.action_type for b in user_behaviors)
        
        # Contenidos más visitados
        content_counts = Counter(b.content_id for b in user_behaviors)
        top_content = content_counts.most_common(10)
        
        # Tiempo total de lectura
        read_durations = [
            b.duration for b in user_behaviors
            if b.action_type == "read" and b.duration
        ]
        total_read_time = sum(read_durations) if read_durations else 0
        
        # Patrones de tiempo
        hour_distribution = Counter(
            b.timestamp.hour for b in user_behaviors if b.timestamp
        )
        
        # Frecuencia de uso
        if len(user_behaviors) >= 2:
            first_action = min(b.timestamp for b in user_behaviors if b.timestamp)
            last_action = max(b.timestamp for b in user_behaviors if b.timestamp)
            days_active = (last_action - first_action).days + 1
            avg_actions_per_day = len(user_behaviors) / days_active if days_active > 0 else 0
        else:
            days_active = 1
            avg_actions_per_day = len(user_behaviors)
        
        return {
            "user_id": user_id,
            "total_actions": len(user_behaviors),
            "action_distribution": dict(action_counts),
            "top_content": [
                {"content_id": content_id, "interactions": count}
                for content_id, count in top_content
            ],
            "total_read_time_seconds": total_read_time,
            "average_read_time": total_read_time / len(read_durations) if read_durations else 0,
            "hour_distribution": dict(hour_distribution),
            "days_active": days_active,
            "average_actions_per_day": avg_actions_per_day,
            "period_days": period_days
        }

    def analyze_content_behavior(
        self,
        content_id: str,
        period_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Analizar comportamiento de usuarios con un contenido.

        Args:
            content_id: ID del contenido
            period_days: Período en días (opcional)

        Returns:
            Análisis de comportamiento del contenido
        """
        content_behaviors = [
            b for b in self.behaviors
            if b.content_id == content_id
        ]
        
        if period_days:
            cutoff_date = datetime.utcnow() - timedelta(days=period_days)
            content_behaviors = [
                b for b in content_behaviors
                if b.timestamp and b.timestamp >= cutoff_date
            ]
        
        if not content_behaviors:
            return {"error": "No hay comportamiento registrado para este contenido"}
        
        # Estadísticas de acciones
        action_counts = Counter(b.action_type for b in content_behaviors)
        
        # Usuarios únicos
        unique_users = len(set(b.user_id for b in content_behaviors))
        
        # Tiempo promedio de lectura
        read_durations = [
            b.duration for b in content_behaviors
            if b.action_type == "read" and b.duration
        ]
        avg_read_time = sum(read_durations) / len(read_durations) if read_durations else 0
        
        # Tasa de conversión (si hay acciones de conversión)
        conversion_actions = ["purchase", "subscribe", "signup", "download"]
        conversions = sum(
            1 for b in content_behaviors
            if b.action_type in conversion_actions
        )
        views = action_counts.get("view", 0)
        conversion_rate = conversions / views if views > 0 else 0
        
        return {
            "content_id": content_id,
            "total_interactions": len(content_behaviors),
            "unique_users": unique_users,
            "action_distribution": dict(action_counts),
            "average_read_time_seconds": avg_read_time,
            "conversion_rate": conversion_rate,
            "conversions": conversions,
            "views": views,
            "period_days": period_days
        }

    def get_behavior_patterns(
        self,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Obtener patrones de comportamiento.

        Args:
            user_id: ID del usuario (opcional, si no se proporciona analiza todos)

        Returns:
            Patrones de comportamiento
        """
        behaviors = self.behaviors
        
        if user_id:
            behaviors = [b for b in behaviors if b.user_id == user_id]
        
        if not behaviors:
            return {"error": "No hay comportamiento disponible"}
        
        # Patrón de días de la semana
        day_distribution = Counter(
            b.timestamp.weekday() for b in behaviors if b.timestamp
        )
        day_names = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
        day_pattern = {
            day_names[day]: count
            for day, count in day_distribution.items()
        }
        
        # Patrón de horas
        hour_distribution = Counter(
            b.timestamp.hour for b in behaviors if b.timestamp
        )
        
        # Secuencia de acciones más común
        action_sequences = []
        for i in range(len(behaviors) - 1):
            if behaviors[i].timestamp and behaviors[i+1].timestamp:
                seq = f"{behaviors[i].action_type} -> {behaviors[i+1].action_type}"
                action_sequences.append(seq)
        
        common_sequences = Counter(action_sequences).most_common(5)
        
        return {
            "day_pattern": day_pattern,
            "hour_pattern": dict(hour_distribution),
            "common_action_sequences": [
                {"sequence": seq, "frequency": count}
                for seq, count in common_sequences
            ],
            "total_behaviors": len(behaviors)
        }






