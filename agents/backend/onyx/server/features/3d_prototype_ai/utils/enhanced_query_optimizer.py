"""
Enhanced Query Optimizer - Sistema de optimización de consultas mejorado
=========================================================================
"""

import logging
import re
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


class EnhancedQueryOptimizer:
    """Sistema de optimización de consultas mejorado"""
    
    def __init__(self):
        self.query_cache: Dict[str, Dict[str, Any]] = {}
        self.query_patterns: Dict[str, List[str]] = defaultdict(list)
        self.optimization_suggestions: Dict[str, List[Dict[str, Any]]] = {}
    
    def analyze_and_optimize(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analiza y optimiza una consulta"""
        query_hash = self._hash_query(query)
        
        # Verificar cache
        if query_hash in self.query_cache:
            cached = self.query_cache[query_hash]
            if datetime.fromisoformat(cached["cached_at"]) > datetime.now() - timedelta(hours=1):
                return {
                    "original_query": query,
                    "optimized_query": cached["optimized_query"],
                    "optimizations": cached["optimizations"],
                    "from_cache": True
                }
        
        optimizations = []
        optimized_query = query
        
        # Optimización 1: Remover espacios extra
        if "  " in query:
            optimized_query = re.sub(r'\s+', ' ', optimized_query).strip()
            optimizations.append({
                "type": "whitespace",
                "description": "Espacios extra removidos",
                "impact": "low"
            })
        
        # Optimización 2: Agregar índices sugeridos
        if "WHERE" in query.upper():
            where_match = re.search(r'WHERE\s+(\w+)\s*=', query, re.IGNORECASE)
            if where_match:
                column = where_match.group(1)
                optimizations.append({
                    "type": "index",
                    "description": f"Considera agregar índice en columna: {column}",
                    "impact": "high",
                    "suggestion": f"CREATE INDEX idx_{column} ON table({column})"
                })
        
        # Optimización 3: Agregar LIMIT si falta
        if "SELECT" in query.upper() and "LIMIT" not in query.upper():
            if not query.rstrip().endswith(";"):
                optimized_query += " LIMIT 100"
            else:
                optimized_query = optimized_query.rstrip(";") + " LIMIT 100;"
            optimizations.append({
                "type": "limit",
                "description": "LIMIT agregado para limitar resultados",
                "impact": "medium"
            })
        
        # Optimización 4: Optimizar JOINs
        if "JOIN" in query.upper():
            optimizations.append({
                "type": "join",
                "description": "Verifica que las columnas de JOIN tengan índices",
                "impact": "high"
            })
        
        # Guardar en cache
        self.query_cache[query_hash] = {
            "original_query": query,
            "optimized_query": optimized_query,
            "optimizations": optimizations,
            "cached_at": datetime.now().isoformat()
        }
        
        return {
            "original_query": query,
            "optimized_query": optimized_query,
            "optimizations": optimizations,
            "from_cache": False,
            "estimated_improvement": self._estimate_improvement(optimizations)
        }
    
    def _hash_query(self, query: str) -> str:
        """Genera hash de consulta"""
        import hashlib
        normalized = re.sub(r'\s+', ' ', query.strip().upper())
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _estimate_improvement(self, optimizations: List[Dict[str, Any]]) -> float:
        """Estima mejora de rendimiento"""
        improvement = 0.0
        
        for opt in optimizations:
            if opt["impact"] == "high":
                improvement += 0.3
            elif opt["impact"] == "medium":
                improvement += 0.15
            else:
                improvement += 0.05
        
        return min(1.0, improvement)
    
    def suggest_indexes(self, query_pattern: str) -> List[Dict[str, Any]]:
        """Sugiere índices basados en patrón de consulta"""
        suggestions = []
        
        # Extraer columnas de WHERE
        where_matches = re.findall(r'WHERE\s+(\w+)\s*[=<>]', query_pattern, re.IGNORECASE)
        for column in set(where_matches):
            suggestions.append({
                "column": column,
                "type": "btree",
                "reason": "Usado frecuentemente en WHERE clauses"
            })
        
        # Extraer columnas de JOIN
        join_matches = re.findall(r'JOIN\s+\w+\s+ON\s+\w+\.(\w+)\s*=', query_pattern, re.IGNORECASE)
        for column in set(join_matches):
            suggestions.append({
                "column": column,
                "type": "btree",
                "reason": "Usado en JOIN operations"
            })
        
        return suggestions
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de optimización"""
        total_queries = len(self.query_cache)
        queries_with_optimizations = sum(
            1 for cached in self.query_cache.values()
            if len(cached.get("optimizations", [])) > 0
        )
        
        return {
            "total_queries_cached": total_queries,
            "queries_optimized": queries_with_optimizations,
            "optimization_rate": (queries_with_optimizations / total_queries * 100) if total_queries > 0 else 0,
            "cache_hit_rate": 0.75  # Simulado
        }




