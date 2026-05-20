from typing import List
import logging

from ..domain.entities import Analysis
from ..domain.interfaces import IAnalysisRepository
from .base import UseCase
from .exceptions import ValidationError, ProcessingError
from .validators import UserIdValidator, PaginationValidator
from ...infrastructure.logging_utils import StructuredLogger

logger = logging.getLogger(__name__)
structured_logger = StructuredLogger(__name__)


class GetAnalysisHistoryUseCase(UseCase):
    
    def __init__(self, analysis_repository: IAnalysisRepository):
        self.analysis_repository = analysis_repository
    
    async def execute(
        self,
        user_id: str,
        limit: int = 10,
        offset: int = 0
    ) -> List[Analysis]:
        # Validate inputs using centralized validators
        UserIdValidator.validate_user_id(user_id)
        PaginationValidator.validate_pagination(limit, offset)
        
        structured_logger.set_context(user_id=user_id, limit=limit, offset=offset)
        
        with structured_logger.operation("get_history", user_id=user_id):
            try:
                analyses = await self.analysis_repository.get_by_user(user_id, limit + offset)
                return analyses[offset:offset + limit]
            except Exception as e:
                raise ProcessingError(f"Failed to get analysis history: {e}") from e

