"""
Scheduler Helper - Ayudante del Programador
============================================

Utilidades para ayudar con la programación de posts.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


class SchedulerHelper:
    """Ayudante para programación inteligente"""
    
    @staticmethod
    def suggest_optimal_times(
        platform: str,
        posts_per_day: int = 3,
        start_hour: int = 8,
        end_hour: int = 20
    ) -> List[datetime]:
        """
        Sugerir horarios óptimos para publicar
        
        Args:
            platform: Plataforma
            posts_per_day: Posts por día
            start_hour: Hora de inicio
            end_hour: Hora de fin
            
        Returns:
            Lista de datetimes sugeridos
        """
        from ..utils.content_optimizer import ContentOptimizer
        
        # Obtener mejores horarios de la plataforma
        best_hours = ContentOptimizer.suggest_posting_time(platform)
        
        # Si no hay suficientes, distribuir uniformemente
        if len(best_hours) < posts_per_day:
            hours = []
            interval = (end_hour - start_hour) / posts_per_day
            for i in range(posts_per_day):
                hour = int(start_hour + (interval * i))
                hours.append(hour)
        else:
            hours = best_hours[:posts_per_day]
        
        # Generar datetimes para hoy
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        suggested_times = []
        
        for hour in hours:
            suggested_time = today.replace(hour=hour, minute=0)
            if suggested_time > datetime.now():
                suggested_times.append(suggested_time)
        
        return suggested_times
    
    @staticmethod
    def distribute_posts_over_time(
        posts: List[Dict[str, Any]],
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """
        Distribuir posts uniformemente en un rango de tiempo
        
        Args:
            posts: Lista de posts a distribuir
            start_date: Fecha de inicio
            end_date: Fecha de fin
            
        Returns:
            Lista de posts con fechas asignadas
        """
        if not posts:
            return []
        
        total_days = (end_date - start_date).days + 1
        posts_per_day = len(posts) / total_days if total_days > 0 else 1
        
        distributed = []
        current_date = start_date
        post_index = 0
        
        for day in range(total_days):
            posts_today = int(posts_per_day * (day + 1)) - int(posts_per_day * day)
            
            for _ in range(posts_today):
                if post_index < len(posts):
                    post = posts[post_index].copy()
                    # Distribuir horas del día
                    hour = 9 + (post_index % 8)  # Entre 9 AM y 5 PM
                    post["scheduled_time"] = current_date.replace(hour=hour, minute=0)
                    distributed.append(post)
                    post_index += 1
            
            current_date += timedelta(days=1)
        
        return distributed
    
    @staticmethod
    def check_schedule_conflicts(
        scheduled_posts: List[Dict[str, Any]],
        new_post: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Verificar conflictos en el calendario
        
        Args:
            scheduled_posts: Posts ya programados
            new_post: Nuevo post a verificar
            
        Returns:
            Lista de posts en conflicto
        """
        new_time = new_post.get("scheduled_time")
        new_platforms = set(new_post.get("platforms", []))
        
        if not new_time:
            return []
        
        conflicts = []
        
        for post in scheduled_posts:
            post_time = post.get("scheduled_time")
            post_platforms = set(post.get("platforms", []))
            
            if not post_time:
                continue
            
            # Verificar si hay solapamiento de tiempo (misma hora)
            time_diff = abs((new_time - post_time).total_seconds())
            
            if time_diff < 3600:  # Menos de 1 hora de diferencia
                # Verificar si comparten plataformas
                shared_platforms = new_platforms & post_platforms
                
                if shared_platforms:
                    conflicts.append({
                        "post": post,
                        "shared_platforms": list(shared_platforms),
                        "time_diff_minutes": time_diff / 60
                    })
        
        return conflicts
    
    @staticmethod
    def optimize_schedule(
        posts: List[Dict[str, Any]],
        platforms: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """
        Optimizar calendario de publicaciones
        
        Args:
            posts: Lista de posts
            platforms: Plataformas objetivo
            start_date: Fecha de inicio
            end_date: Fecha de fin
            
        Returns:
            Lista de posts optimizados
        """
        # Agrupar posts por plataforma
        posts_by_platform = defaultdict(list)
        
        for post in posts:
            for platform in post.get("platforms", []):
                posts_by_platform[platform].append(post)
        
        optimized = []
        
        # Optimizar cada plataforma
        for platform, platform_posts in posts_by_platform.items():
            # Obtener horarios sugeridos
            suggested_times = SchedulerHelper.suggest_optimal_times(
                platform,
                posts_per_day=len(platform_posts) // ((end_date - start_date).days + 1)
            )
            
            # Asignar horarios
            for i, post in enumerate(platform_posts):
                if i < len(suggested_times):
                    post_copy = post.copy()
                    post_copy["scheduled_time"] = suggested_times[i]
                    optimized.append(post_copy)
                else:
                    optimized.append(post)
        
        return optimized




