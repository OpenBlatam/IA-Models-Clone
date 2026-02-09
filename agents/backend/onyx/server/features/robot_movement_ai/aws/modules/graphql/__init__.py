"""
GraphQL Support
==============

GraphQL integration modules.
"""

from aws.modules.graphql.schema_manager import SchemaManager, GraphQLType
from aws.modules.graphql.resolver_manager import ResolverManager
from aws.modules.graphql.query_optimizer import QueryOptimizer, QueryAnalysis

__all__ = [
    "SchemaManager",
    "GraphQLType",
    "ResolverManager",
    "QueryOptimizer",
    "QueryAnalysis",
]

