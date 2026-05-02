"""
Servicio de Búsqueda y Filtrado Avanzado.
"""

from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from enum import Enum

from config.logging_config import get_logger

logger = get_logger(__name__)


class SearchOperator(str, Enum):
    """Operadores de búsqueda."""
    EQUALS = "equals"
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    GREATER_THAN = "gt"
    LESS_THAN = "lt"
    GREATER_EQUAL = "gte"
    LESS_EQUAL = "lte"
    IN = "in"
    NOT_IN = "not_in"
    REGEX = "regex"


class SearchFilter:
    """
    Filtro de búsqueda con validaciones.
    
    Attributes:
        field: Campo a filtrar
        operator: Operador de búsqueda
        value: Valor a buscar
    """
    
    def __init__(
        self,
        field: str,
        operator: SearchOperator,
        value: Any
    ):
        """
        Inicializar filtro con validaciones.
        
        Args:
            field: Campo a filtrar (debe ser string no vacío)
            operator: Operador de búsqueda (debe ser SearchOperator)
            value: Valor a buscar
            
        Raises:
            ValueError: Si los parámetros son inválidos
        """
        # Validaciones
        if not field or not isinstance(field, str) or not field.strip():
            raise ValueError(f"field debe ser un string no vacío, recibido: {field}")
        
        if not isinstance(operator, SearchOperator):
            raise ValueError(f"operator debe ser un SearchOperator, recibido: {type(operator)}")
        
        self.field = field.strip()
        self.operator = operator
        self.value = value
        
        logger.debug(f"SearchFilter creado: field={self.field}, operator={operator.value}")
    
    def matches(self, item: Dict[str, Any]) -> bool:
        """
        Verificar si el item coincide con el filtro con validaciones.
        
        Args:
            item: Item a verificar (debe ser diccionario)
            
        Returns:
            True si coincide, False si no
            
        Raises:
            ValueError: Si item no es un diccionario
        """
        # Validación
        if not isinstance(item, dict):
            raise ValueError(f"item debe ser un diccionario, recibido: {type(item)}")
        
        field_value = item.get(self.field)
        
        if field_value is None:
            return False
        
        try:
            if self.operator == SearchOperator.EQUALS:
                return field_value == self.value
            
            elif self.operator == SearchOperator.CONTAINS:
                return str(self.value).lower() in str(field_value).lower()
            
            elif self.operator == SearchOperator.STARTS_WITH:
                return str(field_value).startswith(str(self.value))
            
            elif self.operator == SearchOperator.ENDS_WITH:
                return str(field_value).endswith(str(self.value))
            
            elif self.operator == SearchOperator.GREATER_THAN:
                return field_value > self.value
            
            elif self.operator == SearchOperator.LESS_THAN:
                return field_value < self.value
            
            elif self.operator == SearchOperator.GREATER_EQUAL:
                return field_value >= self.value
            
            elif self.operator == SearchOperator.LESS_EQUAL:
                return field_value <= self.value
            
            elif self.operator == SearchOperator.IN:
                if not isinstance(self.value, list):
                    logger.warning(f"Operador IN requiere lista, recibido: {type(self.value)}")
                    return False
                return field_value in self.value
            
            elif self.operator == SearchOperator.NOT_IN:
                if not isinstance(self.value, list):
                    logger.warning(f"Operador NOT_IN requiere lista, recibido: {type(self.value)}")
                    return True
                return field_value not in self.value
            
            elif self.operator == SearchOperator.REGEX:
                import re
                try:
                    return bool(re.search(str(self.value), str(field_value)))
                except re.error as e:
                    logger.warning(f"Error en regex pattern '{self.value}': {e}")
                    return False
            
            logger.warning(f"Operador desconocido: {self.operator}")
            return False
        except Exception as e:
            logger.warning(
                f"Error al aplicar filtro {self.field} {self.operator.value}: {e}",
                exc_info=True
            )
            return False


class SearchService:
    """
    Servicio de búsqueda y filtrado con mejoras.
    
    Attributes:
        search_history: Historial de búsquedas
        max_history: Número máximo de búsquedas en historial
    """
    
    def __init__(self, max_history: int = 100):
        """
        Inicializar servicio de búsqueda con validaciones.
        
        Args:
            max_history: Número máximo de búsquedas en historial (debe ser entero positivo)
            
        Raises:
            ValueError: Si max_history es inválido
        """
        # Validación
        if not isinstance(max_history, int) or max_history < 1:
            raise ValueError(f"max_history debe ser un entero positivo, recibido: {max_history}")
        
        self.search_history: List[Dict[str, Any]] = []
        self.max_history = max_history
        
        logger.info(f"✅ SearchService inicializado: max_history={max_history}")
    
    def search(
        self,
        items: List[Dict[str, Any]],
        query: Optional[str] = None,
        filters: Optional[List[SearchFilter]] = None,
        sort_by: Optional[str] = None,
        sort_order: str = "asc",
        limit: Optional[int] = None,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Buscar y filtrar items con validaciones.
        
        Args:
            items: Lista de items a buscar (debe ser lista)
            query: Query de texto (opcional, debe ser string si se proporciona)
            filters: Filtros adicionales (opcional, debe ser lista de SearchFilter)
            sort_by: Campo para ordenar (opcional, debe ser string si se proporciona)
            sort_order: Orden (asc/desc, debe ser string)
            limit: Límite de resultados (opcional, debe ser entero positivo)
            offset: Offset para paginación (debe ser entero no negativo)
            
        Returns:
            Resultados de búsqueda con metadatos
            
        Raises:
            ValueError: Si los parámetros son inválidos
        """
        # Validaciones
        if not isinstance(items, list):
            raise ValueError(f"items debe ser una lista, recibido: {type(items)}")
        
        if query is not None:
            if not isinstance(query, str):
                raise ValueError(f"query debe ser un string si se proporciona, recibido: {type(query)}")
            query = query.strip() if query else None
        
        if filters is not None:
            if not isinstance(filters, list):
                raise ValueError(f"filters debe ser una lista si se proporciona, recibido: {type(filters)}")
            for f in filters:
                if not isinstance(f, SearchFilter):
                    raise ValueError(f"Todos los filtros deben ser SearchFilter, recibido: {type(f)}")
        
        if sort_by is not None:
            if not isinstance(sort_by, str) or not sort_by.strip():
                raise ValueError(f"sort_by debe ser un string no vacío si se proporciona, recibido: {sort_by}")
            sort_by = sort_by.strip()
        
        if not isinstance(sort_order, str) or sort_order.lower() not in ["asc", "desc"]:
            raise ValueError(f"sort_order debe ser 'asc' o 'desc', recibido: {sort_order}")
        
        if limit is not None:
            if not isinstance(limit, int) or limit < 1:
                raise ValueError(f"limit debe ser un entero positivo si se proporciona, recibido: {limit}")
        
        if not isinstance(offset, int) or offset < 0:
            raise ValueError(f"offset debe ser un entero no negativo, recibido: {offset}")
        
        logger.debug(
            f"🔍 Búsqueda iniciada: items={len(items)}, query={query or 'None'}, "
            f"filters={len(filters) if filters else 0}, sort_by={sort_by or 'None'}, "
            f"sort_order={sort_order}, limit={limit}, offset={offset}"
        )
        
        results = items.copy()
        initial_count = len(results)
        
        # Aplicar query de texto
        if query:
            query_lower = query.lower()
            before_query = len(results)
            results = [
                item for item in results
                if any(
                    query_lower in str(value).lower()
                    for value in item.values()
                    if value is not None
                )
            ]
            logger.debug(f"Query aplicada: {before_query} -> {len(results)} resultados")
        
        # Aplicar filtros
        if filters:
            for i, filter_obj in enumerate(filters):
                before_filter = len(results)
                results = [item for item in results if filter_obj.matches(item)]
                logger.debug(
                    f"Filtro {i+1}/{len(filters)} aplicado ({filter_obj.field} {filter_obj.operator.value}): "
                    f"{before_filter} -> {len(results)} resultados"
                )
        
        # Ordenar
        if sort_by:
            reverse = sort_order.lower() == "desc"
            try:
                results.sort(
                    key=lambda x: x.get(sort_by, ""),
                    reverse=reverse
                )
                logger.debug(f"Ordenamiento aplicado: {sort_by} ({sort_order})")
            except Exception as e:
                logger.warning(f"Error ordenando por {sort_by}: {e}", exc_info=True)
        
        # Paginación
        total = len(results)
        if offset > 0:
            results = results[offset:]
        if limit:
            results = results[:limit]
        
        # Guardar en historial
        self._add_to_history(query, filters, len(results))
        
        result = {
            "results": results,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": (offset + len(results)) < total
        }
        
        logger.info(
            f"✅ Búsqueda completada: {len(results)}/{total} resultados "
            f"(inicial: {initial_count}, query: {query or 'None'}, "
            f"filters: {len(filters) if filters else 0})"
        )
        
        return result
    
    def _add_to_history(
        self,
        query: Optional[str],
        filters: Optional[List[SearchFilter]],
        result_count: int
    ) -> None:
        """Agregar búsqueda al historial."""
        self.search_history.append({
            "query": query,
            "filters_count": len(filters) if filters else 0,
            "result_count": result_count,
            "timestamp": datetime.now().isoformat()
        })
        
        if len(self.search_history) > self.max_history:
            self.search_history.pop(0)
    
    def get_search_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Obtener historial de búsquedas con validaciones.
        
        Args:
            limit: Número máximo de resultados (debe ser entero positivo)
            
        Returns:
            Historial de búsquedas (más recientes primero)
            
        Raises:
            ValueError: Si limit es inválido
        """
        # Validación
        if not isinstance(limit, int) or limit < 1:
            raise ValueError(f"limit debe ser un entero positivo, recibido: {limit}")
        
        # Limitar a máximo razonable
        if limit > 1000:
            logger.warning(f"limit muy alto ({limit}), limitando a 1000")
            limit = 1000
        
        result = self.search_history[-limit:]
        logger.debug(f"Obtenido historial de búsquedas: {len(result)} resultados (limit: {limit})")
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        return {
            "total_searches": len(self.search_history),
            "recent_searches": len([s for s in self.search_history if 
                (datetime.now() - datetime.fromisoformat(s["timestamp"])).days < 7])
        }

