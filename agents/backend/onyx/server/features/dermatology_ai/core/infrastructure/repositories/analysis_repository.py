from typing import List, Optional
import logging

from ...domain.entities import Analysis
from ...domain.interfaces import IAnalysisRepository
from ..adapters import IDatabaseAdapter
from ..mappers import AnalysisMapper
from ..query_optimizer import QueryOptimizer
from ....utils.retry import retry, RetryConfig

logger = logging.getLogger(__name__)

# Retry config for critical database operations
DB_RETRY_CONFIG = RetryConfig(
    max_attempts=3,
    initial_delay=0.5,
    max_delay=5.0,
    exponential_base=2.0,
    jitter=True
)


class AnalysisRepository(IAnalysisRepository):
    
    def __init__(self, database: IDatabaseAdapter):
        self.database = database
        self.table_name = "analyses"
    
    @retry(config=DB_RETRY_CONFIG)
    async def create(self, analysis: Analysis) -> Analysis:
        """Create analysis with retry logic for resilience"""
        data = AnalysisMapper.to_dict(analysis)
        await self.database.insert(self.table_name, data)
        logger.debug(f"Created analysis {analysis.id}")
        return analysis
    
    @retry(config=DB_RETRY_CONFIG)
    async def get_by_id(self, analysis_id: str) -> Optional[Analysis]:
        """Get analysis by ID with retry logic"""
        data = await self.database.get(self.table_name, {"id": analysis_id})
        if not data:
            return None
        
        return AnalysisMapper.to_entity(data)
    
    async def get_by_user(self, user_id: str, limit: int = 10) -> List[Analysis]:
        """Get analyses by user (read operation, less critical for retry)"""
        # Optimize query with indexed field
        optimized_limit, _ = QueryOptimizer.optimize_limit_offset(limit, 0)
        optimized_filters = QueryOptimizer.optimize_filter_conditions(
            {"user_id": user_id},
            indexed_fields=["user_id", "created_at"]
        )
        
        results = await self.database.query(
            self.table_name,
            filter_conditions=optimized_filters,
            limit=optimized_limit
        )
        
        return [AnalysisMapper.to_entity(data) for data in results] if results else []
    
    @retry(config=DB_RETRY_CONFIG)
    async def update(self, analysis: Analysis) -> Analysis:
        """Update analysis with retry logic for resilience"""
        data = AnalysisMapper.to_update_dict(analysis)
        await self.database.update(self.table_name, {"id": analysis.id}, data)
        logger.debug(f"Updated analysis {analysis.id}")
        return analysis
    
    async def delete(self, analysis_id: str) -> bool:
        return await self.database.delete(self.table_name, {"id": analysis_id})

