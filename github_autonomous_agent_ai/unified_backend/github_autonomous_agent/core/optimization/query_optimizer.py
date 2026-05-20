"""
Optimizador de Queries y Sistema de Índices.
"""

from typing import Dict, Any, List, Optional, Set
from datetime import datetime
from collections import defaultdict

from config.logging_config import get_logger
from core.database.connection_pool import get_pool

logger = get_logger(__name__)


class QueryOptimizer:
    """Optimizador de queries con análisis y sugerencias."""
    
    def __init__(self):
        """Inicializar optimizador."""
        self.query_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "count": 0,
            "total_time": 0.0,
            "avg_time": 0.0,
            "max_time": 0.0,
            "min_time": float('inf'),
            "last_executed": None
        })
        self.slow_queries: List[Dict[str, Any]] = []
        self.max_slow_queries = 100
    
    def record_query(
        self,
        query: str,
        duration: float,
        params: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Registrar ejecución de query.
        
        Args:
            query: SQL query
            duration: Duración en segundos
            params: Parámetros de la query (opcional)
        """
        stats = self.query_stats[query]
        stats["count"] += 1
        stats["total_time"] += duration
        stats["avg_time"] = stats["total_time"] / stats["count"]
        stats["max_time"] = max(stats["max_time"], duration)
        stats["min_time"] = min(stats["min_time"], duration)
        stats["last_executed"] = datetime.now().isoformat()
        
        # Registrar queries lentas
        if duration > 1.0:  # Más de 1 segundo
            self.slow_queries.append({
                "query": query[:200],  # Limitar longitud
                "duration": duration,
                "params": params,
                "timestamp": datetime.now().isoformat()
            })
            if len(self.slow_queries) > self.max_slow_queries:
                self.slow_queries.pop(0)
    
    async def analyze_table(self, table_name: str) -> Dict[str, Any]:
        """
        Analizar tabla y sugerir índices.
        
        Args:
            table_name: Nombre de la tabla
            
        Returns:
            Análisis y sugerencias
        """
        pool = get_pool()
        async with pool.acquire() as conn:
            # Obtener información de la tabla
            async with conn.execute(f"PRAGMA table_info({table_name})") as cursor:
                columns = await cursor.fetchall()
            
            # Obtener índices existentes
            async with conn.execute(f"SELECT name, sql FROM sqlite_master WHERE type='index' AND tbl_name='{table_name}'") as cursor:
                indexes = await cursor.fetchall()
            
            # Analizar queries frecuentes para esta tabla
            relevant_queries = [
                q for q, stats in self.query_stats.items()
                if table_name.lower() in q.lower()
            ]
            
            suggestions = []
            
            # Sugerir índices basados en queries frecuentes
            for query, stats in relevant_queries:
                if stats["count"] > 10 and stats["avg_time"] > 0.1:
                    # Analizar WHERE clauses
                    if "WHERE" in query.upper():
                        # Extraer columnas en WHERE (simplificado)
                        where_part = query.upper().split("WHERE")[1].split()[0:3]
                        if where_part:
                            col = where_part[0].strip()
                            if col and col not in [idx[0] for idx in indexes]:
                                suggestions.append({
                                    "column": col,
                                    "reason": f"Frecuentemente usado en WHERE (usado {stats['count']} veces, avg {stats['avg_time']:.3f}s)",
                                    "query_example": query[:100]
                                })
            
            return {
                "table": table_name,
                "columns": [dict(col) for col in columns],
                "existing_indexes": [{"name": idx[0], "sql": idx[1]} for idx in indexes],
                "suggestions": suggestions,
                "query_count": len(relevant_queries)
            }
    
    async def create_index(self, table_name: str, columns: List[str], index_name: Optional[str] = None) -> bool:
        """
        Crear índice.
        
        Args:
            table_name: Nombre de la tabla
            columns: Columnas para el índice
            index_name: Nombre del índice (opcional)
            
        Returns:
            True si se creó exitosamente
        """
        if index_name is None:
            index_name = f"idx_{table_name}_{'_'.join(columns)}"
        
        columns_str = ", ".join(columns)
        sql = f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name}({columns_str})"
        
        try:
            pool = get_pool()
            async with pool.acquire() as conn:
                await conn.execute(sql)
                await conn.commit()
            logger.info(f"Índice creado: {index_name} en {table_name}({columns_str})")
            return True
        except Exception as e:
            logger.error(f"Error creando índice {index_name}: {e}", exc_info=True)
            return False
    
    def get_slow_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Obtener queries más lentas.
        
        Args:
            limit: Número máximo de queries
            
        Returns:
            Lista de queries lentas
        """
        sorted_queries = sorted(
            self.slow_queries,
            key=lambda x: x["duration"],
            reverse=True
        )
        return sorted_queries[:limit]
    
    def get_query_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de queries."""
        return {
            "total_queries": len(self.query_stats),
            "slow_queries_count": len(self.slow_queries),
            "queries": {
                query: {
                    "count": stats["count"],
                    "avg_time": stats["avg_time"],
                    "max_time": stats["max_time"],
                    "min_time": stats["min_time"] if stats["min_time"] != float('inf') else 0
                }
                for query, stats in self.query_stats.items()
            }
        }
    
    def suggest_optimizations(self) -> List[Dict[str, Any]]:
        """
        Sugerir optimizaciones basadas en estadísticas.
        
        Returns:
            Lista de sugerencias
        """
        suggestions = []
        
        # Queries lentas frecuentes
        for query, stats in self.query_stats.items():
            if stats["count"] > 5 and stats["avg_time"] > 0.5:
                suggestions.append({
                    "type": "slow_query",
                    "query": query[:200],
                    "count": stats["count"],
                    "avg_time": stats["avg_time"],
                    "suggestion": "Considerar agregar índices o optimizar query"
                })
        
        return suggestions


# Instancia global
_query_optimizer: Optional[QueryOptimizer] = None


def get_query_optimizer() -> QueryOptimizer:
    """Obtener optimizador de queries."""
    global _query_optimizer
    if _query_optimizer is None:
        _query_optimizer = QueryOptimizer()
    return _query_optimizer



