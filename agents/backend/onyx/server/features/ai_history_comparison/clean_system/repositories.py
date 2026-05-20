"""
Repositorios simples para acceso a datos
=======================================

Solo lo esencial para persistir y recuperar datos.
"""

import json
import sqlite3
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path

from .models import HistoryEntry, ComparisonResult, AnalysisJob


class HistoryRepository:
    """Repositorio simple para entradas de historial."""
    
    def __init__(self, db_path: str = "ai_history.db"):
        """Inicializar repositorio."""
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Inicializar base de datos."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS history_entries (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    quality_score REAL NOT NULL,
                    word_count INTEGER NOT NULL,
                    readability_score REAL NOT NULL,
                    sentiment_score REAL NOT NULL,
                    metadata TEXT
                )
            """)
            conn.commit()
    
    def save(self, entry: HistoryEntry) -> HistoryEntry:
        """Guardar entrada de historial."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO history_entries 
                (id, content, model_version, timestamp, quality_score, 
                 word_count, readability_score, sentiment_score, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.id,
                entry.content,
                entry.model_version,
                entry.timestamp.isoformat(),
                entry.quality_score,
                entry.word_count,
                entry.readability_score,
                entry.sentiment_score,
                json.dumps(entry.metadata)
            ))
            conn.commit()
        
        return entry
    
    def find_by_id(self, entry_id: str) -> Optional[HistoryEntry]:
        """Buscar entrada por ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM history_entries WHERE id = ?", (entry_id,)
            )
            row = cursor.fetchone()
            
            if row:
                return self._row_to_entry(row)
        
        return None
    
    def find_by_model_version(self, model_version: str, limit: int = 100) -> List[HistoryEntry]:
        """Buscar entradas por versión de modelo."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM history_entries WHERE model_version = ? ORDER BY timestamp DESC LIMIT ?",
                (model_version, limit)
            )
            rows = cursor.fetchall()
            
            return [self._row_to_entry(row) for row in rows]
    
    def find_recent(self, days: int = 7, limit: int = 100) -> List[HistoryEntry]:
        """Buscar entradas recientes."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM history_entries WHERE timestamp >= ? ORDER BY timestamp DESC LIMIT ?",
                (cutoff.isoformat(), limit)
            )
            rows = cursor.fetchall()
            
            return [self._row_to_entry(row) for row in rows]
    
    def search(self, query: str, limit: int = 50) -> List[HistoryEntry]:
        """Buscar entradas por contenido."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM history_entries WHERE content LIKE ? ORDER BY timestamp DESC LIMIT ?",
                (f"%{query}%", limit)
            )
            rows = cursor.fetchall()
            
            return [self._row_to_entry(row) for row in rows]
    
    def count_by_model(self) -> Dict[str, int]:
        """Contar entradas por modelo."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT model_version, COUNT(*) FROM history_entries GROUP BY model_version"
            )
            rows = cursor.fetchall()
            
            return {row[0]: row[1] for row in rows}
    
    def delete(self, entry_id: str) -> bool:
        """Eliminar entrada."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM history_entries WHERE id = ?", (entry_id,)
            )
            conn.commit()
            
            return cursor.rowcount > 0
    
    def _row_to_entry(self, row) -> HistoryEntry:
        """Convertir fila de BD a HistoryEntry."""
        return HistoryEntry(
            id=row[0],
            content=row[1],
            model_version=row[2],
            timestamp=datetime.fromisoformat(row[3]),
            quality_score=row[4],
            word_count=row[5],
            readability_score=row[6],
            sentiment_score=row[7],
            metadata=json.loads(row[8]) if row[8] else {}
        )


class ComparisonRepository:
    """Repositorio simple para resultados de comparación."""
    
    def __init__(self, db_path: str = "ai_history.db"):
        """Inicializar repositorio."""
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Inicializar base de datos."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS comparison_results (
                    id TEXT PRIMARY KEY,
                    model_a TEXT NOT NULL,
                    model_b TEXT NOT NULL,
                    similarity_score REAL NOT NULL,
                    quality_difference REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    details TEXT
                )
            """)
            conn.commit()
    
    def save(self, result: ComparisonResult) -> ComparisonResult:
        """Guardar resultado de comparación."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO comparison_results 
                (id, model_a, model_b, similarity_score, quality_difference, timestamp, details)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                result.id,
                result.model_a,
                result.model_b,
                result.similarity_score,
                result.quality_difference,
                result.timestamp.isoformat(),
                json.dumps(result.details)
            ))
            conn.commit()
        
        return result
    
    def find_by_id(self, result_id: str) -> Optional[ComparisonResult]:
        """Buscar resultado por ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM comparison_results WHERE id = ?", (result_id,)
            )
            row = cursor.fetchone()
            
            if row:
                return self._row_to_result(row)
        
        return None
    
    def find_by_models(self, model_a: str, model_b: str, limit: int = 10) -> List[ComparisonResult]:
        """Buscar comparaciones entre modelos específicos."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """SELECT * FROM comparison_results 
                   WHERE (model_a = ? AND model_b = ?) OR (model_a = ? AND model_b = ?)
                   ORDER BY timestamp DESC LIMIT ?""",
                (model_a, model_b, model_b, model_a, limit)
            )
            rows = cursor.fetchall()
            
            return [self._row_to_result(row) for row in rows]
    
    def find_recent(self, days: int = 7, limit: int = 50) -> List[ComparisonResult]:
        """Buscar comparaciones recientes."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM comparison_results WHERE timestamp >= ? ORDER BY timestamp DESC LIMIT ?",
                (cutoff.isoformat(), limit)
            )
            rows = cursor.fetchall()
            
            return [self._row_to_result(row) for row in rows]
    
    def get_model_stats(self) -> Dict[str, Dict[str, float]]:
        """Obtener estadísticas por modelo."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT model_a, model_b, AVG(similarity_score), AVG(quality_difference)
                FROM comparison_results 
                GROUP BY model_a, model_b
            """)
            rows = cursor.fetchall()
            
            stats = {}
            for row in rows:
                model_a, model_b, avg_similarity, avg_quality_diff = row
                key = f"{model_a}_vs_{model_b}"
                stats[key] = {
                    'avg_similarity': avg_similarity,
                    'avg_quality_difference': avg_quality_diff
                }
            
            return stats
    
    def _row_to_result(self, row) -> ComparisonResult:
        """Convertir fila de BD a ComparisonResult."""
        return ComparisonResult(
            id=row[0],
            model_a=row[1],
            model_b=row[2],
            similarity_score=row[3],
            quality_difference=row[4],
            timestamp=datetime.fromisoformat(row[5]),
            details=json.loads(row[6]) if row[6] else {}
        )




