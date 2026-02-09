"""
GraphQL API System
==================
Sistema de API GraphQL con resolvers y schema management
"""

import json
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum


class GraphQLOperationType(Enum):
    """Tipos de operación GraphQL"""
    QUERY = "query"
    MUTATION = "mutation"
    SUBSCRIPTION = "subscription"


@dataclass
class GraphQLField:
    """Campo GraphQL"""
    name: str
    type: str
    description: Optional[str] = None
    args: Dict[str, str] = None  # arg_name -> arg_type
    resolver: Optional[Callable] = None
    
    def __post_init__(self):
        if self.args is None:
            self.args = {}


@dataclass
class GraphQLType:
    """Tipo GraphQL"""
    name: str
    fields: List[GraphQLField]
    description: Optional[str] = None


@dataclass
class GraphQLQuery:
    """Query GraphQL"""
    id: str
    query: str
    variables: Dict[str, Any]
    operation_name: Optional[str] = None
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class GraphQLAPI:
    """
    Sistema de API GraphQL
    """
    
    def __init__(self):
        self.types: Dict[str, GraphQLType] = {}
        self.queries: Dict[str, GraphQLField] = {}
        self.mutations: Dict[str, GraphQLField] = {}
        self.subscriptions: Dict[str, GraphQLField] = {}
        self.query_history: List[GraphQLQuery] = []
        self.schema_cache: Optional[str] = None
    
    def register_type(self, type_def: GraphQLType):
        """Registrar tipo GraphQL"""
        self.types[type_def.name] = type_def
        self.schema_cache = None  # Invalidate cache
    
    def register_query(
        self,
        name: str,
        return_type: str,
        resolver: Callable,
        args: Optional[Dict[str, str]] = None,
        description: Optional[str] = None
    ):
        """Registrar query"""
        field = GraphQLField(
            name=name,
            type=return_type,
            description=description,
            args=args or {},
            resolver=resolver
        )
        self.queries[name] = field
        self.schema_cache = None
    
    def register_mutation(
        self,
        name: str,
        return_type: str,
        resolver: Callable,
        args: Optional[Dict[str, str]] = None,
        description: Optional[str] = None
    ):
        """Registrar mutation"""
        field = GraphQLField(
            name=name,
            type=return_type,
            description=description,
            args=args or {},
            resolver=resolver
        )
        self.mutations[name] = field
        self.schema_cache = None
    
    def register_subscription(
        self,
        name: str,
        return_type: str,
        resolver: Callable,
        args: Optional[Dict[str, str]] = None,
        description: Optional[str] = None
    ):
        """Registrar subscription"""
        field = GraphQLField(
            name=name,
            type=return_type,
            description=description,
            args=args or {},
            resolver=resolver
        )
        self.subscriptions[name] = field
        self.schema_cache = None
    
    def generate_schema(self) -> str:
        """Generar schema GraphQL"""
        if self.schema_cache:
            return self.schema_cache
        
        schema_parts = []
        
        # Types
        for type_name, type_def in self.types.items():
            type_schema = f"type {type_name} {{\n"
            for field in type_def.fields:
                args_str = ""
                if field.args:
                    args_str = "(" + ", ".join(
                        f"{arg_name}: {arg_type}"
                        for arg_name, arg_type in field.args.items()
                    ) + ")"
                type_schema += f"  {field.name}{args_str}: {field.type}\n"
            type_schema += "}\n"
            schema_parts.append(type_schema)
        
        # Query type
        if self.queries:
            query_schema = "type Query {\n"
            for name, field in self.queries.items():
                args_str = ""
                if field.args:
                    args_str = "(" + ", ".join(
                        f"{arg_name}: {arg_type}"
                        for arg_name, arg_type in field.args.items()
                    ) + ")"
                query_schema += f"  {name}{args_str}: {field.type}\n"
            query_schema += "}\n"
            schema_parts.append(query_schema)
        
        # Mutation type
        if self.mutations:
            mutation_schema = "type Mutation {\n"
            for name, field in self.mutations.items():
                args_str = ""
                if field.args:
                    args_str = "(" + ", ".join(
                        f"{arg_name}: {arg_type}"
                        for arg_name, arg_type in field.args.items()
                    ) + ")"
                mutation_schema += f"  {name}{args_str}: {field.type}\n"
            mutation_schema += "}\n"
            schema_parts.append(mutation_schema)
        
        # Subscription type
        if self.subscriptions:
            subscription_schema = "type Subscription {\n"
            for name, field in self.subscriptions.items():
                args_str = ""
                if field.args:
                    args_str = "(" + ", ".join(
                        f"{arg_name}: {arg_type}"
                        for arg_name, arg_type in field.args.items()
                    ) + ")"
                subscription_schema += f"  {name}{args_str}: {field.type}\n"
            subscription_schema += "}\n"
            schema_parts.append(subscription_schema)
        
        schema = "\n".join(schema_parts)
        self.schema_cache = schema
        return schema
    
    def execute_query(
        self,
        query: str,
        variables: Optional[Dict[str, Any]] = None,
        operation_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Ejecutar query GraphQL
        
        Args:
            query: Query GraphQL
            variables: Variables
            operation_name: Nombre de operación
        """
        query_obj = GraphQLQuery(
            id=f"query_{int(time.time() * 1000)}",
            query=query,
            variables=variables or {},
            operation_name=operation_name
        )
        
        self.query_history.append(query_obj)
        
        # Parse query (simplificado)
        # En implementación real, usar librería GraphQL
        operation_type = self._detect_operation_type(query)
        
        try:
            if operation_type == GraphQLOperationType.QUERY:
                result = self._execute_query_operation(query_obj)
            elif operation_type == GraphQLOperationType.MUTATION:
                result = self._execute_mutation_operation(query_obj)
            else:
                result = {"error": "Subscription not implemented"}
            
            return {
                "data": result,
                "errors": []
            }
        except Exception as e:
            return {
                "data": None,
                "errors": [{"message": str(e)}]
            }
    
    def _detect_operation_type(self, query: str) -> GraphQLOperationType:
        """Detectar tipo de operación"""
        query_lower = query.lower().strip()
        if query_lower.startswith("mutation"):
            return GraphQLOperationType.MUTATION
        elif query_lower.startswith("subscription"):
            return GraphQLOperationType.SUBSCRIPTION
        else:
            return GraphQLOperationType.QUERY
    
    def _execute_query_operation(self, query_obj: GraphQLQuery) -> Any:
        """Ejecutar operación query"""
        # Parse query name (simplificado)
        query_name = self._extract_operation_name(query_obj.query)
        
        if query_name not in self.queries:
            raise ValueError(f"Query '{query_name}' not found")
        
        field = self.queries[query_name]
        if not field.resolver:
            raise ValueError(f"No resolver for query '{query_name}'")
        
        # Ejecutar resolver
        return field.resolver(**query_obj.variables)
    
    def _execute_mutation_operation(self, query_obj: GraphQLQuery) -> Any:
        """Ejecutar operación mutation"""
        mutation_name = self._extract_operation_name(query_obj.query)
        
        if mutation_name not in self.mutations:
            raise ValueError(f"Mutation '{mutation_name}' not found")
        
        field = self.mutations[mutation_name]
        if not field.resolver:
            raise ValueError(f"No resolver for mutation '{mutation_name}'")
        
        return field.resolver(**query_obj.variables)
    
    def _extract_operation_name(self, query: str) -> str:
        """Extraer nombre de operación (simplificado)"""
        # En implementación real, usar parser GraphQL
        lines = query.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('{') and not line.startswith('#'):
                # Primera palabra que no sea query/mutation/subscription
                words = line.split()
                if len(words) > 1:
                    return words[1].split('(')[0]
        return "unknown"
    
    def get_query_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas de queries"""
        return {
            'total_queries': len(self.query_history),
            'total_types': len(self.types),
            'total_queries_registered': len(self.queries),
            'total_mutations_registered': len(self.mutations),
            'total_subscriptions_registered': len(self.subscriptions),
            'recent_queries': [
                {
                    'id': q.id,
                    'operation': q.operation_name or 'unknown',
                    'timestamp': q.timestamp
                }
                for q in self.query_history[-10:]
            ]
        }


# Instancia global
graphql_api = GraphQLAPI()

