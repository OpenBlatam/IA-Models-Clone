"""
Base service class with common functionality.
"""

from typing import Dict, Any, Optional, Tuple, List, TypeVar, Type, Callable
from sqlalchemy.orm import Session
from sqlalchemy.ext.declarative import DeclarativeMeta
import logging
import uuid
from datetime import datetime

from ..utils.decorators import log_execution_time, handle_errors

logger = logging.getLogger(__name__)

T = TypeVar('T')


class BaseService:
    """Base service class with common functionality."""
    
    def __init__(self, db: Session):
        """Initialize base service."""
        self.db = db
    
    def serialize_model(self, model: Any) -> Dict[str, Any]:
        """
        Serialize a single model instance.
        
        Args:
            model: SQLAlchemy model instance
            
        Returns:
            Serialized dictionary
        """
        from ..utils.serializers import serialize_model
        return serialize_model(model)
    
    def serialize_list(self, models: List[Any]) -> List[Dict[str, Any]]:
        """
        Serialize a list of model instances.
        
        Args:
            models: List of SQLAlchemy model instances
            
        Returns:
            List of serialized dictionaries
        """
        from ..utils.serializers import serialize_list
        return serialize_list(models)
    
    def get_or_raise_not_found(
        self,
        repository,
        entity_id: str,
        entity_name: str = "Entity"
    ) -> Any:
        """
        Get entity by ID or raise NotFoundError.
        
        Args:
            repository: Repository instance with get_by_id method
            entity_id: Entity ID
            entity_name: Entity name for error message
            
        Returns:
            Entity instance
            
        Raises:
            NotFoundError: If entity doesn't exist
        """
        from ..exceptions import NotFoundError
        
        entity = repository.get_by_id(entity_id)
        if not entity:
            raise NotFoundError(entity_name, entity_id)
        return entity
    
    def execute_in_transaction(
        self,
        operation: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute an operation within a database transaction.
        
        Args:
            operation: Callable to execute
            *args: Positional arguments for operation
            **kwargs: Keyword arguments for operation
            
        Returns:
            Result of operation
            
        Raises:
            Exception: If operation fails, transaction is rolled back
        """
        try:
            result = operation(*args, **kwargs)
            self.db.commit()
            return result
        except Exception as e:
            self.db.rollback()
            logger.error(f"Transaction failed: {e}", exc_info=True)
            raise
    
    def batch_get_by_ids(
        self,
        repository,
        ids: List[str],
        eager_load: Optional[List[str]] = None
    ) -> List[Any]:
        """
        Get multiple entities by IDs with optional eager loading.
        
        Args:
            repository: Repository instance with get_by_ids method
            ids: List of entity IDs
            eager_load: Optional list of relationships to eager load
            
        Returns:
            List of entities
        """
        if hasattr(repository, 'get_by_ids'):
            return repository.get_by_ids(ids, eager_load=eager_load)
        else:
            # Fallback to individual queries if get_by_ids not available
            return [repository.get_by_id(id) for id in ids if repository.get_by_id(id)]
    
    @log_execution_time
    @handle_errors
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the service.
        
        Returns:
            Dictionary with health status
        """
        try:
            # Simple query to check database connection
            from sqlalchemy import text
            self.db.execute(text("SELECT 1"))
            return {
                "status": "healthy",
                "service": self.__class__.__name__,
                "database": "connected"
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "service": self.__class__.__name__,
                "database": "disconnected",
                "error": str(e)
            }
    
    def validate_pagination_params(
        self,
        page: int,
        page_size: int,
        max_page_size: int = 100
    ) -> Tuple[int, int]:
        """
        Validate and normalize pagination parameters.
        
        Args:
            page: Page number
            page_size: Items per page
            max_page_size: Maximum allowed page size
            
        Returns:
            Tuple of (validated page, validated page_size)
            
        Raises:
            ValidationError: If parameters are invalid
        """
        from ..exceptions import ValidationError
        from ..constants import MIN_PAGE_SIZE, MAX_PAGE_SIZE
        
        if page < 1:
            raise ValidationError("Page must be greater than 0")
        
        if page_size < MIN_PAGE_SIZE:
            raise ValidationError(f"Page size must be at least {MIN_PAGE_SIZE}")
        
        if page_size > max_page_size:
            raise ValidationError(f"Page size cannot exceed {max_page_size}")
        
        return page, page_size
    
    def validate_limit(
        self,
        limit: int,
        max_limit: int = 100,
        min_limit: int = 1
    ) -> int:
        """
        Validate and normalize limit parameter.
        
        Args:
            limit: Limit value
            max_limit: Maximum allowed limit
            min_limit: Minimum allowed limit
            
        Returns:
            Validated limit
            
        Raises:
            ValidationError: If limit is invalid
        """
        from ..exceptions import ValidationError
        
        if limit < min_limit:
            raise ValidationError(f"Limit must be at least {min_limit}")
        
        if limit > max_limit:
            raise ValidationError(f"Limit cannot exceed {max_limit}")
        
        return limit
    
    def validate_with_conversion(
        self,
        validator_func: Callable,
        value: Any,
        *args,
        **kwargs
    ) -> Any:
        """
        Validate a value using a validator function, converting ValueError to ValidationError.
        
        Args:
            validator_func: Validator function to call
            value: Value to validate
            *args: Additional positional arguments for validator
            **kwargs: Additional keyword arguments for validator
            
        Returns:
            Validated value
            
        Raises:
            ValidationError: If validation fails
        """
        from ..exceptions import ValidationError
        
        try:
            return validator_func(value, *args, **kwargs)
        except ValueError as e:
            raise ValidationError(str(e))
    
    def generate_id(self) -> str:
        """
        Generate a new UUID as string.
        
        Returns:
            UUID string
        """
        return str(uuid.uuid4())
    
    def get_current_timestamp(self) -> datetime:
        """
        Get current datetime.
        
        Returns:
            Current datetime
        """
        return datetime.now()
    
    def create_entity_data(
        self,
        **fields
    ) -> Dict[str, Any]:
        """
        Create entity data dictionary with common fields.
        
        Args:
            **fields: Additional fields to include in entity data
            
        Returns:
            Dictionary with entity data including id and created_at if not provided
        """
        entity_data = fields.copy()
        
        # Add ID if not provided
        if "id" not in entity_data:
            entity_data["id"] = self.generate_id()
        
        # Add created_at if not provided
        if "created_at" not in entity_data:
            entity_data["created_at"] = self.get_current_timestamp()
        
        return entity_data






