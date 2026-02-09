"""
GraphQL API System
==================

Sistema de API GraphQL para consultas flexibles.
"""

import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class GraphQLQuery:
    """Query GraphQL."""
    query_id: str
    query: str
    variables: Dict[str, Any] = field(default_factory=dict)
    operation_name: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class GraphQLResolver:
    """Resolver GraphQL."""
    field_name: str
    resolver_func: callable
    description: str = ""


class GraphQLAPI:
    """
    API GraphQL.
    
    Gestiona queries GraphQL y resolvers.
    """
    
    def __init__(self):
        """Inicializar API GraphQL."""
        self.resolvers: Dict[str, GraphQLResolver] = {}
        self.query_history: List[GraphQLQuery] = []
        self.max_history = 10000
    
    def register_resolver(
        self,
        field_name: str,
        resolver_func: callable,
        description: str = ""
    ) -> GraphQLResolver:
        """
        Registrar resolver.
        
        Args:
            field_name: Nombre del campo
            resolver_func: Función resolver
            description: Descripción
            
        Returns:
            Resolver registrado
        """
        resolver = GraphQLResolver(
            field_name=field_name,
            resolver_func=resolver_func,
            description=description
        )
        
        self.resolvers[field_name] = resolver
        logger.info(f"Registered GraphQL resolver: {field_name}")
        
        return resolver
    
    async def execute_query(
        self,
        query: str,
        variables: Optional[Dict[str, Any]] = None,
        operation_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Ejecutar query GraphQL.
        
        Args:
            query: Query GraphQL
            variables: Variables
            operation_name: Nombre de operación
            
        Returns:
            Resultado de la query
        """
        query_id = f"query_{len(self.query_history)}"
        
        # Registrar query
        graphql_query = GraphQLQuery(
            query_id=query_id,
            query=query,
            variables=variables or {},
            operation_name=operation_name
        )
        self.query_history.append(graphql_query)
        if len(self.query_history) > self.max_history:
            self.query_history = self.query_history[-self.max_history:]
        
        # Ejecutar query (simplificado - en producción usar librería GraphQL)
        try:
            result = await self._execute_simple_query(query, variables or {})
            return {
                "data": result,
                "errors": None
            }
        except Exception as e:
            logger.error(f"Error executing GraphQL query: {e}", exc_info=True)
            return {
                "data": None,
                "errors": [{"message": str(e)}]
            }
    
    async def _execute_simple_query(
        self,
        query: str,
        variables: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Ejecutar query simple (parser básico).
        
        Nota: En producción usar librería GraphQL completa como strawberry o graphene.
        """
        result = {}
        
        # Parser básico (simplificado)
        if "trajectory" in query.lower():
            from .trajectory_optimizer import get_trajectory_optimizer
            optimizer = get_trajectory_optimizer()
            result["trajectory"] = {
                "cache_size": len(optimizer._trajectory_cache),
                "statistics": optimizer.get_statistics()
            }
        
        if "metrics" in query.lower():
            from .metrics import get_metrics_collector
            metrics = get_metrics_collector()
            result["metrics"] = metrics.get_all_metrics()
        
        if "performance" in query.lower():
            from .performance import get_performance_monitor
            monitor = get_performance_monitor()
            result["performance"] = monitor.get_performance_metrics()
        
        return result
    
    def get_resolver(self, field_name: str) -> Optional[GraphQLResolver]:
        """Obtener resolver por nombre de campo."""
        return self.resolvers.get(field_name)
    
    def list_resolvers(self) -> List[GraphQLResolver]:
        """Listar todos los resolvers."""
        return list(self.resolvers.values())
    
    def get_query_history(
        self,
        limit: int = 100
    ) -> List[GraphQLQuery]:
        """
        Obtener historial de queries.
        
        Args:
            limit: Límite de resultados
            
        Returns:
            Lista de queries
        """
        return self.query_history[-limit:]


# Instancia global
_graphql_api: Optional[GraphQLAPI] = None


def get_graphql_api() -> GraphQLAPI:
    """Obtener instancia global de la API GraphQL."""
    global _graphql_api
    if _graphql_api is None:
        _graphql_api = GraphQLAPI()
    return _graphql_api

