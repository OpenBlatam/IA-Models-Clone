"""
GraphQL Schema Definition
"""

import logging

logger = logging.getLogger(__name__)

try:
    from strawberry.fastapi import GraphQLRouter
    from strawberry import Schema
    GRAPHQL_AVAILABLE = True
except ImportError:
    GRAPHQL_AVAILABLE = False
    logger.warning("Strawberry GraphQL not available. Install with: pip install strawberry-graphql[fastapi]")


if GRAPHQL_AVAILABLE:
    from strawberry import type, field
    from typing import List, Optional
    from datetime import datetime
    
    @type
    class SkinMetricsType:
        """GraphQL type for skin metrics"""
        overall_score: float
        texture_score: float
        hydration_score: float
        elasticity_score: float
        pigmentation_score: float
        pore_size_score: float
        wrinkles_score: float
        redness_score: float
        dark_spots_score: float
    
    @type
    class ConditionType:
        """GraphQL type for skin condition"""
        name: str
        confidence: float
        severity: str
        description: Optional[str] = None
    
    @type
    class AnalysisType:
        """GraphQL type for analysis"""
        id: str
        user_id: str
        metrics: Optional[SkinMetricsType] = None
        conditions: List[ConditionType] = field(default_factory=list)
        status: str
        created_at: datetime
        completed_at: Optional[datetime] = None
    
    @type
    class Query:
        """GraphQL Query type"""
        
        @field
        async def get_analysis(self, analysis_id: str) -> Optional[AnalysisType]:
            """Get analysis by ID"""
            # This would use a query handler
            return None
        
        @field
        async def get_analysis_history(
            self,
            user_id: str,
            limit: int = 10
        ) -> List[AnalysisType]:
            """Get analysis history"""
            return []
    
    # Create schema
    schema = Schema(query=Query)
    
    def create_graphql_router():
        """Create GraphQL router"""
        return GraphQLRouter(schema)
else:
    schema = None
    
    def create_graphql_router():
        """Placeholder when GraphQL not available"""
        return None

