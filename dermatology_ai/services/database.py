"""
Sistema de base de datos para historial persistente
"""

import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
from contextlib import contextmanager
import threading


class DatabaseManager:
    """Gestor de base de datos SQLite para historial"""
    
    def __init__(self, db_path: str = "dermatology_history.db"):
        """
        Inicializa el gestor de base de datos
        
        Args:
            db_path: Path a la base de datos SQLite
        """
        self.db_path = Path(db_path)
        self._local = threading.local()
        self._init_database()
    
    def _get_connection(self):
        """Obtiene conexión a la base de datos (thread-safe)"""
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False
            )
            self._local.connection.row_factory = sqlite3.Row
        return self._local.connection
    
    @contextmanager
    def get_cursor(self):
        """Context manager para cursor de base de datos"""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()
    
    def _init_database(self):
        """Inicializa las tablas de la base de datos"""
        with self.get_cursor() as cursor:
            # Tabla de análisis
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analyses (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    timestamp TEXT NOT NULL,
                    analysis_type TEXT NOT NULL,
                    quality_scores TEXT NOT NULL,
                    conditions TEXT NOT NULL,
                    skin_type TEXT NOT NULL,
                    recommendations_priority TEXT NOT NULL,
                    image_hash TEXT,
                    metadata TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            
            # Tabla de recomendaciones
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS recommendations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_id TEXT NOT NULL,
                    routine TEXT NOT NULL,
                    specific_recommendations TEXT NOT NULL,
                    tips TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (analysis_id) REFERENCES analyses(id)
                )
            """)
            
            # Tabla de comparaciones
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS comparisons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    record_id1 TEXT NOT NULL,
                    record_id2 TEXT NOT NULL,
                    comparison_data TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (record_id1) REFERENCES analyses(id),
                    FOREIGN KEY (record_id2) REFERENCES analyses(id)
                )
            """)
            
            # Índices para mejor rendimiento
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_id ON analyses(user_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON analyses(timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_analysis_id ON recommendations(analysis_id)
            """)
    
    def save_analysis(self, analysis_id: str, user_id: Optional[str],
                     analysis_result: Dict, image_hash: Optional[str] = None,
                     metadata: Optional[Dict] = None) -> bool:
        """
        Guarda un análisis en la base de datos
        
        Args:
            analysis_id: ID único del análisis
            user_id: ID del usuario
            analysis_result: Resultado del análisis
            image_hash: Hash de la imagen
            metadata: Metadatos adicionales
            
        Returns:
            True si se guardó correctamente
        """
        try:
            timestamp = datetime.now().isoformat()
            
            with self.get_cursor() as cursor:
                cursor.execute("""
                    INSERT OR REPLACE INTO analyses 
                    (id, user_id, timestamp, analysis_type, quality_scores, 
                     conditions, skin_type, recommendations_priority, 
                     image_hash, metadata, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    analysis_id,
                    user_id,
                    timestamp,
                    analysis_result.get("analysis_type", "image"),
                    json.dumps(analysis_result.get("quality_scores", {})),
                    json.dumps(analysis_result.get("conditions", [])),
                    analysis_result.get("skin_type", "unknown"),
                    json.dumps(analysis_result.get("recommendations_priority", [])),
                    image_hash,
                    json.dumps(metadata or {}),
                    timestamp
                ))
            
            return True
        except Exception as e:
            print(f"Error guardando análisis: {e}")
            return False
    
    def get_analysis(self, analysis_id: str) -> Optional[Dict]:
        """
        Obtiene un análisis por ID
        
        Args:
            analysis_id: ID del análisis
            
        Returns:
            Diccionario con el análisis o None
        """
        try:
            with self.get_cursor() as cursor:
                cursor.execute("""
                    SELECT * FROM analyses WHERE id = ?
                """, (analysis_id,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                return {
                    "id": row["id"],
                    "user_id": row["user_id"],
                    "timestamp": row["timestamp"],
                    "analysis_type": row["analysis_type"],
                    "quality_scores": json.loads(row["quality_scores"]),
                    "conditions": json.loads(row["conditions"]),
                    "skin_type": row["skin_type"],
                    "recommendations_priority": json.loads(row["recommendations_priority"]),
                    "image_hash": row["image_hash"],
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {}
                }
        except Exception as e:
            print(f"Error obteniendo análisis: {e}")
            return None
    
    def get_user_history(self, user_id: str, limit: int = 50, 
                        offset: int = 0) -> List[Dict]:
        """
        Obtiene historial de un usuario
        
        Args:
            user_id: ID del usuario
            limit: Límite de registros
            offset: Offset para paginación
            
        Returns:
            Lista de análisis
        """
        try:
            with self.get_cursor() as cursor:
                cursor.execute("""
                    SELECT * FROM analyses 
                    WHERE user_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ? OFFSET ?
                """, (user_id, limit, offset))
                
                rows = cursor.fetchall()
                return [self._row_to_dict(row) for row in rows]
        except Exception as e:
            print(f"Error obteniendo historial: {e}")
            return []
    
    def get_statistics(self, user_id: Optional[str] = None) -> Dict:
        """
        Obtiene estadísticas de análisis
        
        Args:
            user_id: ID del usuario (None para todas las estadísticas)
            
        Returns:
            Diccionario con estadísticas
        """
        try:
            with self.get_cursor() as cursor:
                if user_id:
                    cursor.execute("""
                        SELECT 
                            COUNT(*) as total,
                            AVG(CAST(json_extract(quality_scores, '$.overall_score') AS REAL)) as avg_score,
                            MIN(CAST(json_extract(quality_scores, '$.overall_score') AS REAL)) as min_score,
                            MAX(CAST(json_extract(quality_scores, '$.overall_score') AS REAL)) as max_score
                        FROM analyses
                        WHERE user_id = ?
                    """, (user_id,))
                else:
                    cursor.execute("""
                        SELECT 
                            COUNT(*) as total,
                            AVG(CAST(json_extract(quality_scores, '$.overall_score') AS REAL)) as avg_score,
                            MIN(CAST(json_extract(quality_scores, '$.overall_score') AS REAL)) as min_score,
                            MAX(CAST(json_extract(quality_scores, '$.overall_score') AS REAL)) as max_score
                        FROM analyses
                    """)
                
                row = cursor.fetchone()
                return {
                    "total_analyses": row["total"] if row else 0,
                    "average_score": round(row["avg_score"] or 0, 2) if row else 0,
                    "min_score": round(row["min_score"] or 0, 2) if row else 0,
                    "max_score": round(row["max_score"] or 0, 2) if row else 0
                }
        except Exception as e:
            print(f"Error obteniendo estadísticas: {e}")
            return {}
    
    def save_recommendation(self, analysis_id: str, recommendations: Dict) -> int:
        """
        Guarda recomendaciones asociadas a un análisis
        
        Args:
            analysis_id: ID del análisis
            recommendations: Diccionario con recomendaciones
            
        Returns:
            ID del registro creado
        """
        try:
            timestamp = datetime.now().isoformat()
            
            with self.get_cursor() as cursor:
                cursor.execute("""
                    INSERT INTO recommendations 
                    (analysis_id, routine, specific_recommendations, tips, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    analysis_id,
                    json.dumps(recommendations.get("routine", {})),
                    json.dumps(recommendations.get("specific_recommendations", [])),
                    json.dumps(recommendations.get("tips", [])),
                    timestamp
                ))
                
                return cursor.lastrowid
        except Exception as e:
            print(f"Error guardando recomendaciones: {e}")
            return -1
    
    def _row_to_dict(self, row) -> Dict:
        """Convierte una fila de SQLite a diccionario"""
        return {
            "id": row["id"],
            "user_id": row["user_id"],
            "timestamp": row["timestamp"],
            "analysis_type": row["analysis_type"],
            "quality_scores": json.loads(row["quality_scores"]),
            "conditions": json.loads(row["conditions"]),
            "skin_type": row["skin_type"],
            "recommendations_priority": json.loads(row["recommendations_priority"]),
            "image_hash": row["image_hash"],
            "metadata": json.loads(row["metadata"]) if row["metadata"] else {}
        }
    
    def close(self):
        """Cierra la conexión a la base de datos"""
        if hasattr(self._local, 'connection'):
            self._local.connection.close()
            delattr(self._local, 'connection')






