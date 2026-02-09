"""
Statistics - Sistema de estadísticas avanzadas
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)


class StatisticsCollector:
    """Recolector de estadísticas avanzadas"""

    def __init__(self):
        """Inicializar recolector"""
        self.operation_stats: Dict[str, List[float]] = defaultdict(list)
        self.daily_stats: Dict[str, Dict[str, Any]] = {}
        self.user_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "operations": 0,
            "total_changes": 0,
            "avg_operation_time": 0.0
        })

    def record_operation(
        self,
        operation_type: str,
        duration: float,
        user_id: Optional[str] = None,
        content_length: int = 0
    ):
        """
        Registrar una operación.

        Args:
            operation_type: Tipo de operación
            duration: Duración
            user_id: ID del usuario
            content_length: Longitud del contenido
        """
        today = datetime.utcnow().date().isoformat()
        
        # Estadísticas por tipo
        self.operation_stats[operation_type].append(duration)
        
        # Estadísticas diarias
        if today not in self.daily_stats:
            self.daily_stats[today] = {
                "operations": 0,
                "total_duration": 0.0,
                "total_content_length": 0
            }
        
        self.daily_stats[today]["operations"] += 1
        self.daily_stats[today]["total_duration"] += duration
        self.daily_stats[today]["total_content_length"] += content_length
        
        # Estadísticas por usuario
        if user_id:
            self.user_stats[user_id]["operations"] += 1
            self.user_stats[user_id]["total_changes"] += content_length
            
            # Actualizar promedio
            ops = self.user_stats[user_id]["operations"]
            current_avg = self.user_stats[user_id]["avg_operation_time"]
            self.user_stats[user_id]["avg_operation_time"] = (
                (current_avg * (ops - 1) + duration) / ops
            )

    def get_operation_statistics(self, operation_type: str) -> Dict[str, Any]:
        """
        Obtener estadísticas de un tipo de operación.

        Args:
            operation_type: Tipo de operación

        Returns:
            Estadísticas
        """
        durations = self.operation_stats.get(operation_type, [])
        
        if not durations:
            return {
                "operation_type": operation_type,
                "count": 0,
                "avg_duration": 0.0,
                "min_duration": 0.0,
                "max_duration": 0.0,
                "median_duration": 0.0
            }
        
        return {
            "operation_type": operation_type,
            "count": len(durations),
            "avg_duration": statistics.mean(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "median_duration": statistics.median(durations),
            "std_deviation": statistics.stdev(durations) if len(durations) > 1 else 0.0
        }

    def get_daily_statistics(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Obtener estadísticas diarias.

        Args:
            days: Número de días

        Returns:
            Lista de estadísticas diarias
        """
        end_date = datetime.utcnow().date()
        start_date = end_date - timedelta(days=days - 1)
        
        stats = []
        current_date = start_date
        
        while current_date <= end_date:
            date_str = current_date.isoformat()
            daily = self.daily_stats.get(date_str, {
                "operations": 0,
                "total_duration": 0.0,
                "total_content_length": 0
            })
            
            stats.append({
                "date": date_str,
                "operations": daily["operations"],
                "avg_duration": (
                    daily["total_duration"] / daily["operations"]
                    if daily["operations"] > 0 else 0.0
                ),
                "total_content_length": daily["total_content_length"]
            })
            
            current_date += timedelta(days=1)
        
        return stats

    def get_user_statistics(self, user_id: str) -> Dict[str, Any]:
        """
        Obtener estadísticas de un usuario.

        Args:
            user_id: ID del usuario

        Returns:
            Estadísticas del usuario
        """
        return self.user_stats.get(user_id, {
            "operations": 0,
            "total_changes": 0,
            "avg_operation_time": 0.0
        })

    def get_top_users(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Obtener usuarios más activos.

        Args:
            limit: Número de usuarios

        Returns:
            Lista de usuarios
        """
        sorted_users = sorted(
            self.user_stats.items(),
            key=lambda x: x[1]["operations"],
            reverse=True
        )
        
        return [
            {
                "user_id": user_id,
                **stats
            }
            for user_id, stats in sorted_users[:limit]
        ]

    def get_summary(self) -> Dict[str, Any]:
        """
        Obtener resumen general.

        Returns:
            Resumen de estadísticas
        """
        total_operations = sum(len(durations) for durations in self.operation_stats.values())
        total_duration = sum(
            sum(durations) for durations in self.operation_stats.values()
        )
        
        return {
            "total_operations": total_operations,
            "total_duration": total_duration,
            "avg_duration": total_duration / total_operations if total_operations > 0 else 0.0,
            "operation_types": len(self.operation_stats),
            "active_users": len(self.user_stats),
            "days_tracked": len(self.daily_stats)
        }






