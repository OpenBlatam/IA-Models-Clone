"""
Servicio de métricas y analytics para el sistema
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import sqlite3
from collections import defaultdict

from config.settings import settings

logger = logging.getLogger(__name__)


class MetricsService:
    """Servicio para métricas y analytics"""
    
    def __init__(self):
        self.db_path = settings.database_url.replace("sqlite:///", "")
        self._init_metrics_tables()
    
    def _init_metrics_tables(self):
        """Inicializa tablas de métricas"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Tabla de métricas de generación
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS generation_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    song_id TEXT,
                    user_id TEXT,
                    prompt TEXT,
                    duration INTEGER,
                    generation_time REAL,
                    model_used TEXT,
                    status TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Tabla de uso por usuario
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    date DATE NOT NULL,
                    songs_generated INTEGER DEFAULT 0,
                    total_duration REAL DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, date)
                )
            """)
            
            conn.commit()
            conn.close()
            logger.info("Metrics tables initialized")
        except Exception as e:
            logger.error(f"Error initializing metrics tables: {e}")
    
    def record_generation(
        self,
        song_id: str,
        user_id: Optional[str],
        prompt: str,
        duration: int,
        generation_time: float,
        model_used: str,
        status: str
    ):
        """Registra una generación"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO generation_metrics 
                (song_id, user_id, prompt, duration, generation_time, model_used, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (song_id, user_id, prompt, duration, generation_time, model_used, status))
            
            # Actualizar uso diario
            if user_id:
                today = datetime.now().date()
                cursor.execute("""
                    INSERT OR REPLACE INTO user_usage 
                    (user_id, date, songs_generated, total_duration)
                    VALUES (?, ?, 
                        COALESCE((SELECT songs_generated FROM user_usage 
                                 WHERE user_id = ? AND date = ?), 0) + 1,
                        COALESCE((SELECT total_duration FROM user_usage 
                                 WHERE user_id = ? AND date = ?), 0) + ?
                    )
                """, (user_id, today, user_id, today, user_id, today, duration))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error recording generation: {e}")
    
    def get_stats(self, days: int = 7) -> Dict:
        """Obtiene estadísticas generales"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            since_date = datetime.now() - timedelta(days=days)
            
            # Total de canciones generadas
            cursor.execute("""
                SELECT COUNT(*) as total, 
                       SUM(duration) as total_duration,
                       AVG(generation_time) as avg_generation_time
                FROM generation_metrics
                WHERE created_at >= ? AND status = 'completed'
            """, (since_date,))
            
            row = cursor.fetchone()
            total_songs = row['total'] if row else 0
            total_duration = row['total_duration'] if row else 0
            avg_time = row['avg_generation_time'] if row else 0
            
            # Canciones por día
            cursor.execute("""
                SELECT DATE(created_at) as date, COUNT(*) as count
                FROM generation_metrics
                WHERE created_at >= ? AND status = 'completed'
                GROUP BY DATE(created_at)
                ORDER BY date DESC
            """, (since_date,))
            
            daily_stats = [dict(row) for row in cursor.fetchall()]
            
            # Géneros más populares (extraer del prompt)
            cursor.execute("""
                SELECT prompt, COUNT(*) as count
                FROM generation_metrics
                WHERE created_at >= ? AND status = 'completed'
                GROUP BY prompt
                ORDER BY count DESC
                LIMIT 10
            """, (since_date,))
            
            popular_prompts = [dict(row) for row in cursor.fetchall()]
            
            conn.close()
            
            return {
                "total_songs": total_songs,
                "total_duration_hours": total_duration / 3600 if total_duration else 0,
                "avg_generation_time_seconds": avg_time,
                "daily_stats": daily_stats,
                "popular_prompts": popular_prompts,
                "period_days": days
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}
    
    def get_user_stats(self, user_id: str, days: int = 30) -> Dict:
        """Obtiene estadísticas de un usuario"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            since_date = datetime.now() - timedelta(days=days)
            
            cursor.execute("""
                SELECT COUNT(*) as total_songs,
                       SUM(duration) as total_duration,
                       AVG(generation_time) as avg_generation_time
                FROM generation_metrics
                WHERE user_id = ? AND created_at >= ? AND status = 'completed'
            """, (user_id, since_date))
            
            row = cursor.fetchone()
            
            conn.close()
            
            return {
                "user_id": user_id,
                "total_songs": row['total_songs'] if row else 0,
                "total_duration_minutes": (row['total_duration'] / 60) if row and row['total_duration'] else 0,
                "avg_generation_time_seconds": row['avg_generation_time'] if row else 0,
                "period_days": days
            }
        except Exception as e:
            logger.error(f"Error getting user stats: {e}")
            return {}


# Instancia global
_metrics_service: Optional[MetricsService] = None


def get_metrics_service() -> MetricsService:
    """Obtiene la instancia global del servicio de métricas"""
    global _metrics_service
    if _metrics_service is None:
        _metrics_service = MetricsService()
    return _metrics_service

