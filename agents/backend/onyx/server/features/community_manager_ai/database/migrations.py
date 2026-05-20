"""
Database Migrations - Migraciones de Base de Datos
===================================================

Utilidades para migraciones de base de datos.
"""

import logging
from sqlalchemy import text
from .database import engine, Base
from .models import (
    Post, Meme, Template, PlatformConnection,
    AnalyticsMetric, Notification
)

logger = logging.getLogger(__name__)


def create_tables():
    """Crear todas las tablas"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Tablas creadas exitosamente")
    except Exception as e:
        logger.error(f"Error creando tablas: {e}")
        raise


def drop_tables():
    """Eliminar todas las tablas"""
    try:
        Base.metadata.drop_all(bind=engine)
        logger.info("Tablas eliminadas exitosamente")
    except Exception as e:
        logger.error(f"Error eliminando tablas: {e}")
        raise


def add_indexes():
    """Agregar índices adicionales para performance"""
    try:
        with engine.connect() as conn:
            # Índices para Posts
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_posts_status_scheduled_time 
                ON posts(status, scheduled_time)
            """))
            
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_posts_platforms 
                ON posts USING GIN(platforms)
            """))
            
            # Índices para Analytics
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_analytics_platform_recorded 
                ON analytics_metrics(platform, recorded_at)
            """))
            
            # Índices para Notifications
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_notifications_type_timestamp 
                ON notifications(type, timestamp)
            """))
            
            conn.commit()
            logger.info("Índices agregados exitosamente")
    except Exception as e:
        logger.warning(f"Error agregando índices (puede que ya existan): {e}")


def migrate_to_v2():
    """Migración a versión 2 (ejemplo)"""
    try:
        with engine.connect() as conn:
            # Agregar nuevas columnas si no existen
            try:
                conn.execute(text("""
                    ALTER TABLE posts 
                    ADD COLUMN IF NOT EXISTS engagement_score FLOAT DEFAULT 0.0
                """))
                conn.commit()
                logger.info("Migración a v2 completada")
            except Exception as e:
                logger.warning(f"Columna puede que ya exista: {e}")
    except Exception as e:
        logger.error(f"Error en migración: {e}")




