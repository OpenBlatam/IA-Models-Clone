"""
Analytics Service - Dashboard y métricas
==========================================

Sistema de analytics y métricas para el dashboard.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class UserMetrics:
    """Métricas del usuario"""
    user_id: str
    total_points: int
    current_level: int
    current_streak: int
    jobs_applied: int
    jobs_saved: int
    jobs_liked: int
    steps_completed: int
    skills_learned: int
    challenges_completed: int
    badges_earned: int
    last_activity: Optional[datetime] = None


@dataclass
class ActivityStats:
    """Estadísticas de actividad"""
    date: str
    actions_count: int
    points_earned: int
    steps_completed: int
    jobs_viewed: int


class AnalyticsService:
    """Servicio de analytics"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.user_metrics: Dict[str, UserMetrics] = {}
        self.activity_logs: Dict[str, List[ActivityStats]] = {}
        logger.info("AnalyticsService initialized")
    
    def get_dashboard_data(self, user_id: str) -> Dict[str, Any]:
        """Obtener datos completos del dashboard"""
        metrics = self.get_user_metrics(user_id)
        activity_stats = self.get_activity_stats(user_id, days=30)
        progress_trend = self.get_progress_trend(user_id, days=7)
        
        return {
            "metrics": {
                "total_points": metrics.total_points,
                "current_level": metrics.current_level,
                "current_streak": metrics.current_streak,
                "jobs_applied": metrics.jobs_applied,
                "jobs_saved": metrics.jobs_saved,
                "steps_completed": metrics.steps_completed,
                "skills_learned": metrics.skills_learned,
                "challenges_completed": metrics.challenges_completed,
                "badges_earned": metrics.badges_earned,
            },
            "activity_stats": activity_stats,
            "progress_trend": progress_trend,
            "last_activity": metrics.last_activity.isoformat() if metrics.last_activity else None,
        }
    
    def get_user_metrics(self, user_id: str) -> UserMetrics:
        """Obtener métricas del usuario"""
        if user_id not in self.user_metrics:
            self.user_metrics[user_id] = UserMetrics(
                user_id=user_id,
                total_points=0,
                current_level=1,
                current_streak=0,
                jobs_applied=0,
                jobs_saved=0,
                jobs_liked=0,
                steps_completed=0,
                skills_learned=0,
                challenges_completed=0,
                badges_earned=0,
            )
        return self.user_metrics[user_id]
    
    def update_metric(
        self,
        user_id: str,
        metric_name: str,
        value: Any
    ):
        """Actualizar una métrica"""
        metrics = self.get_user_metrics(user_id)
        
        if hasattr(metrics, metric_name):
            setattr(metrics, metric_name, value)
            metrics.last_activity = datetime.now()
    
    def increment_metric(self, user_id: str, metric_name: str, amount: int = 1):
        """Incrementar una métrica"""
        metrics = self.get_user_metrics(user_id)
        
        if hasattr(metrics, metric_name):
            current_value = getattr(metrics, metric_name)
            if isinstance(current_value, int):
                setattr(metrics, metric_name, current_value + amount)
                metrics.last_activity = datetime.now()
    
    def get_activity_stats(
        self,
        user_id: str,
        days: int = 30
    ) -> List[ActivityStats]:
        """Obtener estadísticas de actividad"""
        if user_id not in self.activity_logs:
            return []
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        stats = [
            stat for stat in self.activity_logs[user_id]
            if datetime.fromisoformat(stat.date) >= start_date
        ]
        
        return stats
    
    def get_progress_trend(
        self,
        user_id: str,
        days: int = 7
    ) -> Dict[str, Any]:
        """Obtener tendencia de progreso"""
        activity_stats = self.get_activity_stats(user_id, days)
        
        if not activity_stats:
            return {
                "trend": "stable",
                "growth_rate": 0.0,
                "data_points": [],
            }
        
        # Calcular tendencia
        total_actions = sum(stat.actions_count for stat in activity_stats)
        avg_daily = total_actions / len(activity_stats) if activity_stats else 0
        
        # Comparar primera mitad vs segunda mitad
        mid_point = len(activity_stats) // 2
        first_half = sum(stat.actions_count for stat in activity_stats[:mid_point])
        second_half = sum(stat.actions_count for stat in activity_stats[mid_point:])
        
        if first_half == 0:
            growth_rate = 1.0 if second_half > 0 else 0.0
        else:
            growth_rate = (second_half - first_half) / first_half
        
        if growth_rate > 0.1:
            trend = "increasing"
        elif growth_rate < -0.1:
            trend = "decreasing"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "growth_rate": round(growth_rate, 2),
            "average_daily_actions": round(avg_daily, 2),
            "data_points": [
                {
                    "date": stat.date,
                    "actions": stat.actions_count,
                    "points": stat.points_earned,
                }
                for stat in activity_stats
            ],
        }
    
    def log_activity(
        self,
        user_id: str,
        action: str,
        points_earned: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Registrar una actividad"""
        today = datetime.now().date().isoformat()
        
        if user_id not in self.activity_logs:
            self.activity_logs[user_id] = []
        
        # Buscar stats del día
        today_stats = next(
            (stat for stat in self.activity_logs[user_id] if stat.date == today),
            None
        )
        
        if today_stats:
            today_stats.actions_count += 1
            today_stats.points_earned += points_earned
        else:
            new_stats = ActivityStats(
                date=today,
                actions_count=1,
                points_earned=points_earned,
                steps_completed=0,
                jobs_viewed=0,
            )
            self.activity_logs[user_id].append(new_stats)
        
        # Actualizar métricas según acción
        if action == "complete_step":
            self.increment_metric(user_id, "steps_completed")
        elif action == "apply_job":
            self.increment_metric(user_id, "jobs_applied")
        elif action == "save_job":
            self.increment_metric(user_id, "jobs_saved")
        elif action == "like_job":
            self.increment_metric(user_id, "jobs_liked")
        elif action == "learn_skill":
            self.increment_metric(user_id, "skills_learned")
        elif action == "complete_challenge":
            self.increment_metric(user_id, "challenges_completed")
        elif action == "earn_badge":
            self.increment_metric(user_id, "badges_earned")
    
    def get_leaderboard_stats(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Obtener estadísticas para leaderboard"""
        sorted_users = sorted(
            self.user_metrics.values(),
            key=lambda x: x.total_points,
            reverse=True
        )
        
        return [
            {
                "user_id": metrics.user_id,
                "total_points": metrics.total_points,
                "level": metrics.current_level,
                "streak": metrics.current_streak,
                "jobs_applied": metrics.jobs_applied,
            }
            for metrics in sorted_users[:limit]
        ]




