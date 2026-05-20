"""
Schedule Optimizer Script - Optimizador de Calendario
=====================================================

Script para optimizar el calendario de publicaciones.
"""

import logging
import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from community_manager_ai import CommunityManager
from community_manager_ai.utils.scheduler_helper import SchedulerHelper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def optimize_schedule(
    days_ahead: int = 7,
    posts_per_day: int = 3
):
    """
    Optimizar calendario de publicaciones
    
    Args:
        days_ahead: Días hacia adelante
        posts_per_day: Posts por día
    """
    manager = CommunityManager()
    
    # Obtener posts programados
    all_posts = manager.scheduler.get_all_posts(status="scheduled")
    
    if not all_posts:
        logger.info("No hay posts programados para optimizar")
        return
    
    logger.info(f"Optimizando {len(all_posts)} posts")
    
    # Definir rango de fechas
    start_date = datetime.now()
    end_date = start_date + timedelta(days=days_ahead)
    
    # Obtener todas las plataformas únicas
    all_platforms = set()
    for post in all_posts:
        all_platforms.update(post.get("platforms", []))
    
    # Optimizar para cada plataforma
    optimized_posts = []
    
    for platform in all_platforms:
        platform_posts = [
            p for p in all_posts
            if platform in p.get("platforms", [])
        ]
        
        if not platform_posts:
            continue
        
        # Obtener horarios sugeridos
        suggested_times = SchedulerHelper.suggest_optimal_times(
            platform,
            posts_per_day=posts_per_day
        )
        
        # Redistribuir posts
        for i, post in enumerate(platform_posts):
            if i < len(suggested_times):
                # Actualizar fecha programada
                post_id = post.get("id")
                new_time = suggested_times[i % len(suggested_times)]
                
                # Actualizar en el scheduler
                # TODO: Implementar método update_post en scheduler
                logger.info(f"Post {post_id} optimizado para {platform} a las {new_time}")
    
    logger.info("Optimización completada")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimizador de calendario")
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Días hacia adelante (default: 7)"
    )
    parser.add_argument(
        "--posts-per-day",
        type=int,
        default=3,
        help="Posts por día (default: 3)"
    )
    
    args = parser.parse_args()
    
    optimize_schedule(
        days_ahead=args.days,
        posts_per_day=args.posts_per_day
    )




