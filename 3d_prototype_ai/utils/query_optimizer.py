"""
Query Optimizer - Sistema de optimización de consultas
=======================================================
"""

import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


class QueryOptimizer:
    """Sistema de optimización de consultas"""
    
    def __init__(self):
        self.query_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.slow_queries: List[Dict[str, Any]] = []
        self.query_plans: Dict[str, Dict[str, Any]] = {}
        self.indexes: Dict[str, List[str]] = {}
    
    def record_query(self, query_id: str, query_type: str, duration: float,
                    query_string: Optional[str] = None, parameters: Optional[Dict[str, Any]] = None):
        """Registra una consulta"""
        query_record = {
            "query_id": query_id,
            "query_type": query_type,
            "duration": duration,
            "query_string": query_string,
            "parameters": parameters,
            "timestamp": datetime.now().isoformat()
        }
        
        self.query_history[query_type].append(query_record)
        
        # Mantener solo últimas 1000 por tipo
        if len(self.query_history[query_type]) > 1000:
            self.query_history[query_type] = self.query_history[query_type][-1000:]
        
        # Registrar consultas lentas
        if duration > 1.0:  # Más de 1 segundo
            self.slow_queries.append(query_record)
            if len(self.slow_queries) > 100:
                self.slow_queries = self.slow_queries[-100:]
        
        logger.debug(f"Consulta registrada: {query_type} - {duration:.3f}s")
    
    def analyze_query(self, query_string: str) -> Dict[str, Any]:
        """Analiza una consulta y sugiere optimizaciones"""
        suggestions = []
        
        # Detectar consultas sin índices
        if "WHERE" in query_string.upper() and "JOIN" in query_string.upper():
            suggestions.append({
                "type": "index",
                "message": "Considera agregar índices en las columnas de JOIN",
                "priority": "high"
            })
        
        # Detectar SELECT *
        if "SELECT *" in query_string.upper():
            suggestions.append({
                "type": "select",
                "message": "Evita SELECT *, especifica solo las columnas necesarias",
                "priority": "medium"
            })
        
        # Detectar consultas sin LIMIT
        if "SELECT" in query_string.upper() and "LIMIT" not in query_string.upper():
            suggestions.append({
                "type": "limit",
                "message": "Considera agregar LIMIT para evitar cargar demasiados datos",
                "priority": "medium"
            })
        
        return {
            "query": query_string,
            "suggestions": suggestions,
            "optimization_score": 100 - (len(suggestions) * 20)
        }
    
    def suggest_indexes(self, query_type: str) -> List[Dict[str, Any]]:
        """Sugiere índices basados en consultas históricas"""
        queries = self.query_history.get(query_type, [])
        
        if not queries:
            return []
        
        # Analizar patrones de WHERE
        where_patterns = defaultdict(int)
        for query in queries:
            if query.get("query_string"):
                # Extraer columnas de WHERE (simplificado)
                query_str = query["query_string"].upper()
                if "WHERE" in query_str:
                    where_patterns["common_columns"] += 1
        
        suggestions = []
        if where_patterns:
            suggestions.append({
                "table": "prototypes",
                "columns": ["product_type", "created_at"],
                "type": "composite_index",
                "reason": "Frecuentemente usado en WHERE clauses"
            })
        
        return suggestions
    
    def get_slow_queries(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Obtiene consultas lentas"""
        return sorted(self.slow_queries, key=lambda x: x["duration"], reverse=True)[:limit]
    
    def get_query_stats(self, query_type: Optional[str] = None) -> Dict[str, Any]:
        """Obtiene estadísticas de consultas"""
        if query_type:
            queries = self.query_history.get(query_type, [])
        else:
            queries = [q for queries_list in self.query_history.values() for q in queries_list]
        
        if not queries:
            return {"count": 0}
        
        durations = [q["duration"] for q in queries]
        
        return {
            "count": len(queries),
            "avg_duration": sum(durations) / len(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "p95_duration": sorted(durations)[int(len(durations) * 0.95)] if durations else 0,
            "p99_duration": sorted(durations)[int(len(durations) * 0.99)] if durations else 0
        }
    
    def optimize_query(self, query_string: str) -> str:
        """Optimiza una consulta automáticamente"""
        optimized = query_string
        
        # Remover espacios extra
        optimized = " ".join(optimized.split())
        
        # Agregar LIMIT si no existe y es SELECT
        if "SELECT" in optimized.upper() and "LIMIT" not in optimized.upper():
            if ";" in optimized:
                optimized = optimized.replace(";", " LIMIT 100;")
            else:
                optimized += " LIMIT 100"
        
        return optimized




