"""
Schema Manager
==============

GraphQL schema management.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GraphQLType:
    """GraphQL type definition."""
    name: str
    type: str  # Object, Scalar, Enum, etc.
    fields: Dict[str, str] = None  # field_name -> field_type
    description: Optional[str] = None
    
    def __post_init__(self):
        if self.fields is None:
            self.fields = {}


class SchemaManager:
    """GraphQL schema manager."""
    
    def __init__(self):
        self._types: Dict[str, GraphQLType] = {}
        self._queries: Dict[str, Any] = {}
        self._mutations: Dict[str, Any] = {}
        self._subscriptions: Dict[str, Any] = {}
        self._schema_string: Optional[str] = None
    
    def register_type(self, graphql_type: GraphQLType):
        """Register GraphQL type."""
        self._types[graphql_type.name] = graphql_type
        logger.info(f"Registered GraphQL type: {graphql_type.name}")
    
    def register_query(self, name: str, resolver: Any, return_type: str):
        """Register GraphQL query."""
        self._queries[name] = {
            "resolver": resolver,
            "return_type": return_type
        }
        logger.info(f"Registered GraphQL query: {name}")
    
    def register_mutation(self, name: str, resolver: Any, return_type: str):
        """Register GraphQL mutation."""
        self._mutations[name] = {
            "resolver": resolver,
            "return_type": return_type
        }
        logger.info(f"Registered GraphQL mutation: {name}")
    
    def register_subscription(self, name: str, resolver: Any, return_type: str):
        """Register GraphQL subscription."""
        self._subscriptions[name] = {
            "resolver": resolver,
            "return_type": return_type
        }
        logger.info(f"Registered GraphQL subscription: {name}")
    
    def build_schema(self) -> str:
        """Build GraphQL schema string."""
        schema_parts = []
        
        # Add type definitions
        for type_name, graphql_type in self._types.items():
            type_def = f"type {graphql_type.name} {{\n"
            for field_name, field_type in graphql_type.fields.items():
                type_def += f"  {field_name}: {field_type}\n"
            type_def += "}\n"
            schema_parts.append(type_def)
        
        # Add Query type
        if self._queries:
            schema_parts.append("type Query {\n")
            for query_name, query_info in self._queries.items():
                schema_parts.append(f"  {query_name}: {query_info['return_type']}\n")
            schema_parts.append("}\n")
        
        # Add Mutation type
        if self._mutations:
            schema_parts.append("type Mutation {\n")
            for mutation_name, mutation_info in self._mutations.items():
                schema_parts.append(f"  {mutation_name}: {mutation_info['return_type']}\n")
            schema_parts.append("}\n")
        
        # Add Subscription type
        if self._subscriptions:
            schema_parts.append("type Subscription {\n")
            for sub_name, sub_info in self._subscriptions.items():
                schema_parts.append(f"  {sub_name}: {sub_info['return_type']}\n")
            schema_parts.append("}\n")
        
        self._schema_string = "\n".join(schema_parts)
        return self._schema_string
    
    def get_schema(self) -> Optional[str]:
        """Get GraphQL schema."""
        if not self._schema_string:
            return self.build_schema()
        return self._schema_string
    
    def validate_query(self, query: str) -> tuple:
        """Validate GraphQL query."""
        # In production, use a proper GraphQL parser
        # This is a simplified version
        if not query.strip():
            return False, "Empty query"
        
        # Basic validation
        if "query" not in query.lower() and "mutation" not in query.lower():
            return False, "Query must start with 'query' or 'mutation'"
        
        return True, None
    
    def get_schema_stats(self) -> Dict[str, Any]:
        """Get schema statistics."""
        return {
            "total_types": len(self._types),
            "total_queries": len(self._queries),
            "total_mutations": len(self._mutations),
            "total_subscriptions": len(self._subscriptions),
            "types": list(self._types.keys())
        }

