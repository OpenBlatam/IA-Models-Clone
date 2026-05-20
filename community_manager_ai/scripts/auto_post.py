"""
Auto Post Script - Script de Publicación Automática
====================================================

Script para publicar automáticamente posts programados.
"""

import logging
import time
from datetime import datetime
from typing import Optional
import sys
import os

# Agregar el directorio raíz al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from community_manager_ai import CommunityManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_auto_poster(
    check_interval: int = 60,
    max_posts_per_run: int = 10
):
    """
    Ejecutar el auto-poster
    
    Args:
        check_interval: Intervalo en segundos para verificar posts pendientes
        max_posts_per_run: Máximo de posts a publicar por ejecución
    """
    manager = CommunityManager()
    
    logger.info("Auto-poster iniciado")
    logger.info(f"Verificando cada {check_interval} segundos")
    
    try:
        while True:
            # Obtener posts pendientes
            pending_posts = manager.scheduler.get_pending_posts(
                limit=max_posts_per_run
            )
            
            if pending_posts:
                logger.info(f"Encontrados {len(pending_posts)} posts pendientes")
                
                for post in pending_posts:
                    try:
                        post_id = post.get("id")
                        content = post.get("content")
                        platforms = post.get("platforms", [])
                        media_paths = post.get("media_paths", [])
                        
                        logger.info(f"Publicando post {post_id} en {platforms}")
                        
                        # Publicar en todas las plataformas
                        results = manager.publish_now(
                            content=content,
                            platforms=platforms,
                            media_paths=media_paths
                        )
                        
                        # Marcar como publicado
                        manager.scheduler.mark_as_published(post_id, results)
                        
                        logger.info(f"Post {post_id} publicado exitosamente")
                        
                    except Exception as e:
                        logger.error(f"Error publicando post {post_id}: {e}")
            else:
                logger.debug("No hay posts pendientes")
            
            # Esperar antes de la siguiente verificación
            time.sleep(check_interval)
            
    except KeyboardInterrupt:
        logger.info("Auto-poster detenido por el usuario")
    except Exception as e:
        logger.error(f"Error en auto-poster: {e}")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Auto-poster para Community Manager AI")
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Intervalo de verificación en segundos (default: 60)"
    )
    parser.add_argument(
        "--max-posts",
        type=int,
        default=10,
        help="Máximo de posts por ejecución (default: 10)"
    )
    
    args = parser.parse_args()
    
    run_auto_poster(
        check_interval=args.interval,
        max_posts_per_run=args.max_posts
    )




