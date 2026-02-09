"""
Database Optimizer
==================

Optimizador de base de datos.
"""

import logging
import sqlite3
from typing import Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)


class DatabaseOptimizer:
    """Optimizador de base de datos."""
    
    def __init__(self, db_path: str):
        """
        Inicializar optimizador.
        
        Args:
            db_path: Ruta a la base de datos
        """
        self.db_path = db_path
        self._logger = logger
    
    def analyze(self) -> Dict[str, Any]:
        """
        Analizar base de datos.
        
        Returns:
            Análisis de la base de datos
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Obtener información de tablas
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        analysis = {
            "tables": {},
            "total_size": Path(self.db_path).stat().st_size if Path(self.db_path).exists() else 0,
            "recommendations": []
        }
        
        for table in tables:
            # Contar filas
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            row_count = cursor.fetchone()[0]
            
            # Obtener tamaño aproximado
            cursor.execute(f"SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='{table}'")
            
            analysis["tables"][table] = {
                "row_count": row_count
            }
        
        conn.close()
        
        # Recomendaciones
        if analysis["total_size"] > 10 * 1024 * 1024:  # > 10MB
            analysis["recommendations"].append("Considerar VACUUM para optimizar tamaño")
        
        return analysis
    
    def vacuum(self) -> bool:
        """
        Ejecutar VACUUM para optimizar.
        
        Returns:
            True si se ejecutó correctamente
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("VACUUM")
            conn.close()
            
            self._logger.info("Database VACUUM completed")
            return True
        except Exception as e:
            self._logger.error(f"Error running VACUUM: {str(e)}")
            return False
    
    def reindex(self) -> bool:
        """
        Reindexar base de datos.
        
        Returns:
            True si se ejecutó correctamente
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("REINDEX")
            conn.close()
            
            self._logger.info("Database REINDEX completed")
            return True
        except Exception as e:
            self._logger.error(f"Error running REINDEX: {str(e)}")
            return False
    
    def optimize(self) -> Dict[str, Any]:
        """
        Optimizar base de datos completamente.
        
        Returns:
            Resultado de optimización
        """
        result = {
            "vacuum": self.vacuum(),
            "reindex": self.reindex(),
            "analysis": self.analyze()
        }
        
        return result




