"""
GraphQL Routes
==============

Rutas GraphQL.
"""

from fastapi import APIRouter
from strawberry.fastapi import GraphQLRouter
from ..graphql.schema import schema

router = APIRouter()

# Crear router GraphQL
graphql_app = GraphQLRouter(schema=schema)

# Incluir en el router principal
router.include_router(graphql_app, prefix="/graphql")




