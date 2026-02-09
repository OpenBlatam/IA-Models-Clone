"""
Query Optimizer - Optimizador de Queries
========================================

Optimización de queries:
- Query analysis
- Index recommendations
- Query rewriting
- Performance hints
- Query caching
"""

import logging
import re
from typing import Optional, Dict, Any, List
from enum import Enum

logger = logging.getLogger(__name__)


class QueryType(str, Enum):
    """Tipos de query"""
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    AGGREGATE = "aggregate"


class QueryOptimizer:
    """
    Optimizador de queries.
    """
    
    def __init__(self) -> None:
        self.query_cache: Dict[str, Dict[str, Any]] = {}
        self.query_stats: Dict[str, Dict[str, Any]] = {}
        self.index_recommendations: List[Dict[str, Any]] = []
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analiza query"""
        analysis = {
            "query": query,
            "type": self._detect_query_type(query),
            "tables": self._extract_tables(query),
            "columns": self._extract_columns(query),
            "joins": self._count_joins(query),
            "where_clauses": self._count_where_clauses(query),
            "complexity": self._calculate_complexity(query),
            "recommendations": []
        }
        
        # Generar recomendaciones
        analysis["recommendations"] = self._generate_recommendations(analysis)
        
        return analysis
    
    def _detect_query_type(self, query: str) -> str:
        """Detecta tipo de query"""
        query_lower = query.lower().strip()
        
        if query_lower.startswith("select"):
            if "group by" in query_lower or "count(" in query_lower:
                return QueryType.AGGREGATE.value
            return QueryType.SELECT.value
        elif query_lower.startswith("insert"):
            return QueryType.INSERT.value
        elif query_lower.startswith("update"):
            return QueryType.UPDATE.value
        elif query_lower.startswith("delete"):
            return QueryType.DELETE.value
        else:
            return QueryType.SELECT.value
    
    def _extract_tables(self, query: str) -> List[str]:
        """Extrae tablas de query"""
        # Patrón simplificado para FROM y JOIN
        tables = []
        from_match = re.search(r'from\s+(\w+)', query, re.IGNORECASE)
        if from_match:
            tables.append(from_match.group(1))
        
        join_matches = re.findall(r'join\s+(\w+)', query, re.IGNORECASE)
        tables.extend(join_matches)
        
        return list(set(tables))
    
    def _extract_columns(self, query: str) -> List[str]:
        """Extrae columnas de query"""
        # Patrón simplificado para SELECT
        columns = []
        select_match = re.search(r'select\s+(.*?)\s+from', query, re.IGNORECASE | re.DOTALL)
        if select_match:
            cols = select_match.group(1)
            # Separar por comas
            cols_list = [col.strip().split()[0] for col in cols.split(",")]
            columns.extend(cols_list)
        
        return columns
    
    def _count_joins(self, query: str) -> int:
        """Cuenta JOINs"""
        return len(re.findall(r'\bjoin\b', query, re.IGNORECASE))
    
    def _count_where_clauses(self, query: str) -> int:
        """Cuenta cláusulas WHERE"""
        where_match = re.search(r'where\s+(.*?)(?:group|order|limit|$)', query, re.IGNORECASE | re.DOTALL)
        if where_match:
            where_clause = where_match.group(1)
            # Contar condiciones (AND, OR)
            return len(re.findall(r'\b(and|or)\b', where_clause, re.IGNORECASE)) + 1
        return 0
    
    def _calculate_complexity(self, query: str) -> str:
        """Calcula complejidad"""
        joins = self._count_joins(query)
        where_clauses = self._count_where_clauses(query)
        has_subquery = "select" in query.lower().replace(query.lower().split()[0], "", 1)
        
        score = joins * 2 + where_clauses + (3 if has_subquery else 0)
        
        if score < 3:
            return "simple"
        elif score < 7:
            return "medium"
        else:
            return "complex"
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Genera recomendaciones"""
        recommendations = []
        
        # Recomendaciones basadas en análisis
        if analysis["joins"] > 3:
            recommendations.append("Consider reducing number of joins or using denormalization")
        
        if analysis["where_clauses"] > 5:
            recommendations.append("Consider adding indexes on WHERE clause columns")
        
        if analysis["complexity"] == "complex":
            recommendations.append("Query is complex, consider breaking into smaller queries")
        
        if analysis["type"] == QueryType.SELECT.value and analysis["joins"] > 0:
            recommendations.append("Ensure foreign key columns are indexed")
        
        return recommendations
    
    def optimize_query(self, query: str) -> str:
        """Optimiza query"""
        # Optimizaciones básicas
        optimized = query.strip()
        
        # Remover espacios múltiples
        optimized = re.sub(r'\s+', ' ', optimized)
        
        # Asegurar que SELECT tenga columnas específicas (no *)
        if re.match(r'select\s+\*\s+from', optimized, re.IGNORECASE):
            logger.warning("Query uses SELECT *, consider specifying columns")
        
        return optimized
    
    def recommend_indexes(self, query: str) -> List[Dict[str, Any]]:
        """Recomienda índices"""
        analysis = self.analyze_query(query)
        recommendations = []
        
        # Recomendar índices en columnas WHERE
        if analysis["where_clauses"] > 0:
            for table in analysis["tables"]:
                recommendations.append({
                    "table": table,
                    "columns": analysis["columns"][:3],  # Primeras columnas
                    "type": "btree",
                    "reason": "Frequently used in WHERE clauses"
                })
        
        return recommendations
    
    def cache_query(self, query: str, result: Any, ttl: int = 3600) -> None:
        """Cachea resultado de query"""
        query_hash = hash(query)
        self.query_cache[str(query_hash)] = {
            "query": query,
            "result": result,
            "cached_at": datetime.now().isoformat(),
            "ttl": ttl
        }
    
    def get_cached_query(self, query: str) -> Optional[Any]:
        """Obtiene query del cache"""
        query_hash = hash(query)
        cached = self.query_cache.get(str(query_hash))
        
        if cached:
            # Verificar TTL
            from datetime import datetime, timedelta
            cached_at = datetime.fromisoformat(cached["cached_at"])
            if datetime.now() - cached_at < timedelta(seconds=cached["ttl"]):
                return cached["result"]
            else:
                del self.query_cache[str(query_hash)]
        
        return None


from datetime import datetime


def get_query_optimizer() -> QueryOptimizer:
    """Obtiene optimizador de queries"""
    return QueryOptimizer()















