"""
Advanced Search - Búsqueda avanzada con múltiples filtros
===========================================================
"""

import logging
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import re

logger = logging.getLogger(__name__)


class SearchOperator(Enum):
    """Operadores de búsqueda"""
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    GREATER_EQUAL = "greater_equal"
    LESS_EQUAL = "less_equal"
    IN = "in"
    NOT_IN = "not_in"
    BETWEEN = "between"
    REGEX = "regex"
    EXISTS = "exists"
    NOT_EXISTS = "not_exists"


@dataclass
class SearchFilter:
    """Filtro de búsqueda"""
    field: str
    operator: SearchOperator
    value: Any
    case_sensitive: bool = False


@dataclass
class SearchQuery:
    """Query de búsqueda"""
    query: Optional[str] = None
    filters: List[SearchFilter] = field(default_factory=list)
    sort_by: Optional[str] = None
    sort_order: str = "asc"  # "asc" or "desc"
    page: int = 1
    page_size: int = 20
    fields: Optional[List[str]] = None  # Campos a retornar
    aggregations: Optional[Dict[str, Any]] = None


class AdvancedSearch:
    """Sistema de búsqueda avanzada"""
    
    def __init__(self):
        self.indexes: Dict[str, List[Dict[str, Any]]] = {}
        self.field_types: Dict[str, Dict[str, str]] = {}  # index -> field -> type
    
    def create_index(self, index_name: str, field_types: Dict[str, str]):
        """Crea un nuevo índice"""
        self.indexes[index_name] = []
        self.field_types[index_name] = field_types
        logger.info(f"Índice {index_name} creado")
    
    def index_document(self, index_name: str, document: Dict[str, Any]):
        """Indexa un documento"""
        if index_name not in self.indexes:
            raise ValueError(f"Índice {index_name} no existe")
        
        # Agregar metadata
        document["_indexed_at"] = datetime.now().isoformat()
        self.indexes[index_name].append(document)
    
    def search(
        self,
        index_name: str,
        search_query: SearchQuery
    ) -> Dict[str, Any]:
        """Realiza una búsqueda"""
        if index_name not in self.indexes:
            raise ValueError(f"Índice {index_name} no existe")
        
        documents = self.indexes[index_name]
        
        # Aplicar query de texto
        if search_query.query:
            documents = self._apply_text_query(documents, search_query.query)
        
        # Aplicar filtros
        for filter_obj in search_query.filters:
            documents = self._apply_filter(documents, filter_obj)
        
        # Ordenar
        if search_query.sort_by:
            documents = self._sort_documents(
                documents,
                search_query.sort_by,
                search_query.sort_order
            )
        
        # Paginación
        total = len(documents)
        start = (search_query.page - 1) * search_query.page_size
        end = start + search_query.page_size
        paginated_docs = documents[start:end]
        
        # Seleccionar campos
        if search_query.fields:
            paginated_docs = [
                {k: v for k, v in doc.items() if k in search_query.fields or k.startswith("_")}
                for doc in paginated_docs
            ]
        
        # Agregaciones
        aggregations = None
        if search_query.aggregations:
            aggregations = self._compute_aggregations(
                documents,
                search_query.aggregations
            )
        
        return {
            "results": paginated_docs,
            "total": total,
            "page": search_query.page,
            "page_size": search_query.page_size,
            "total_pages": (total + search_query.page_size - 1) // search_query.page_size,
            "aggregations": aggregations
        }
    
    def _apply_text_query(
        self,
        documents: List[Dict[str, Any]],
        query: str
    ) -> List[Dict[str, Any]]:
        """Aplica búsqueda de texto"""
        query_lower = query.lower()
        results = []
        
        for doc in documents:
            # Buscar en todos los campos de texto
            for value in doc.values():
                if isinstance(value, str) and query_lower in value.lower():
                    results.append(doc)
                    break
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, str) and query_lower in item.lower():
                            results.append(doc)
                            break
                    if doc in results:
                        break
        
        return results
    
    def _apply_filter(
        self,
        documents: List[Dict[str, Any]],
        filter_obj: SearchFilter
    ) -> List[Dict[str, Any]]:
        """Aplica un filtro"""
        results = []
        
        for doc in documents:
            field_value = doc.get(filter_obj.field)
            
            if filter_obj.operator == SearchOperator.EXISTS:
                if field_value is not None:
                    results.append(doc)
                continue
            
            if filter_obj.operator == SearchOperator.NOT_EXISTS:
                if field_value is None:
                    results.append(doc)
                continue
            
            if field_value is None:
                continue
            
            matches = False
            
            if filter_obj.operator == SearchOperator.EQUALS:
                matches = self._compare_values(field_value, filter_obj.value, "==", filter_obj.case_sensitive)
            elif filter_obj.operator == SearchOperator.NOT_EQUALS:
                matches = not self._compare_values(field_value, filter_obj.value, "==", filter_obj.case_sensitive)
            elif filter_obj.operator == SearchOperator.CONTAINS:
                matches = self._compare_values(field_value, filter_obj.value, "contains", filter_obj.case_sensitive)
            elif filter_obj.operator == SearchOperator.NOT_CONTAINS:
                matches = not self._compare_values(field_value, filter_obj.value, "contains", filter_obj.case_sensitive)
            elif filter_obj.operator == SearchOperator.STARTS_WITH:
                matches = self._compare_values(field_value, filter_obj.value, "starts_with", filter_obj.case_sensitive)
            elif filter_obj.operator == SearchOperator.ENDS_WITH:
                matches = self._compare_values(field_value, filter_obj.value, "ends_with", filter_obj.case_sensitive)
            elif filter_obj.operator == SearchOperator.GREATER_THAN:
                matches = self._compare_values(field_value, filter_obj.value, ">", filter_obj.case_sensitive)
            elif filter_obj.operator == SearchOperator.LESS_THAN:
                matches = self._compare_values(field_value, filter_obj.value, "<", filter_obj.case_sensitive)
            elif filter_obj.operator == SearchOperator.GREATER_EQUAL:
                matches = self._compare_values(field_value, filter_obj.value, ">=", filter_obj.case_sensitive)
            elif filter_obj.operator == SearchOperator.LESS_EQUAL:
                matches = self._compare_values(field_value, filter_obj.value, "<=", filter_obj.case_sensitive)
            elif filter_obj.operator == SearchOperator.IN:
                matches = field_value in filter_obj.value if isinstance(filter_obj.value, list) else False
            elif filter_obj.operator == SearchOperator.NOT_IN:
                matches = field_value not in filter_obj.value if isinstance(filter_obj.value, list) else True
            elif filter_obj.operator == SearchOperator.BETWEEN:
                if isinstance(filter_obj.value, list) and len(filter_obj.value) == 2:
                    matches = filter_obj.value[0] <= field_value <= filter_obj.value[1]
            elif filter_obj.operator == SearchOperator.REGEX:
                if isinstance(field_value, str):
                    flags = 0 if filter_obj.case_sensitive else re.IGNORECASE
                    matches = bool(re.search(filter_obj.value, field_value, flags))
            
            if matches:
                results.append(doc)
        
        return results
    
    def _compare_values(
        self,
        value1: Any,
        value2: Any,
        operator: str,
        case_sensitive: bool
    ) -> bool:
        """Compara dos valores"""
        if operator == "==":
            if isinstance(value1, str) and isinstance(value2, str):
                if not case_sensitive:
                    return value1.lower() == value2.lower()
            return value1 == value2
        elif operator == "contains":
            if isinstance(value1, str) and isinstance(value2, str):
                if not case_sensitive:
                    return value2.lower() in value1.lower()
                return value2 in value1
            return False
        elif operator == "starts_with":
            if isinstance(value1, str) and isinstance(value2, str):
                if not case_sensitive:
                    return value1.lower().startswith(value2.lower())
                return value1.startswith(value2)
            return False
        elif operator == "ends_with":
            if isinstance(value1, str) and isinstance(value2, str):
                if not case_sensitive:
                    return value1.lower().endswith(value2.lower())
                return value1.endswith(value2)
            return False
        elif operator in [">", "<", ">=", "<="]:
            try:
                if operator == ">":
                    return value1 > value2
                elif operator == "<":
                    return value1 < value2
                elif operator == ">=":
                    return value1 >= value2
                elif operator == "<=":
                    return value1 <= value2
            except TypeError:
                return False
        
        return False
    
    def _sort_documents(
        self,
        documents: List[Dict[str, Any]],
        sort_by: str,
        sort_order: str
    ) -> List[Dict[str, Any]]:
        """Ordena documentos"""
        reverse = sort_order == "desc"
        
        def get_sort_value(doc):
            value = doc.get(sort_by)
            if value is None:
                return "" if reverse else "zzz"
            return value
        
        return sorted(documents, key=get_sort_value, reverse=reverse)
    
    def _compute_aggregations(
        self,
        documents: List[Dict[str, Any]],
        aggregations: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calcula agregaciones"""
        results = {}
        
        for agg_name, agg_config in aggregations.items():
            agg_type = agg_config.get("type")
            field = agg_config.get("field")
            
            if agg_type == "count":
                results[agg_name] = len(documents)
            elif agg_type == "sum":
                values = [doc.get(field, 0) for doc in documents if isinstance(doc.get(field), (int, float))]
                results[agg_name] = sum(values)
            elif agg_type == "avg":
                values = [doc.get(field, 0) for doc in documents if isinstance(doc.get(field), (int, float))]
                results[agg_name] = sum(values) / len(values) if values else 0
            elif agg_type == "min":
                values = [doc.get(field) for doc in documents if doc.get(field) is not None]
                results[agg_name] = min(values) if values else None
            elif agg_type == "max":
                values = [doc.get(field) for doc in documents if doc.get(field) is not None]
                results[agg_name] = max(values) if values else None
            elif agg_type == "terms":
                # Agrupar por términos
                terms = {}
                for doc in documents:
                    value = doc.get(field)
                    if value is not None:
                        terms[value] = terms.get(value, 0) + 1
                results[agg_name] = terms
        
        return results




