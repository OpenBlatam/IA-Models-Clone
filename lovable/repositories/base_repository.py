"""
Base repository class with common functionality.
"""

from typing import Dict, Any, List, Optional, Type, TypeVar, Tuple
from sqlalchemy.orm import Session, selectinload, joinedload
from sqlalchemy import desc, asc
import logging
import time

from ..utils.query_helpers import apply_pagination, apply_sorting, safe_query_execute

logger = logging.getLogger(__name__)

T = TypeVar('T')


class BaseRepository:
    """Base repository class with common functionality."""
    
    def __init__(self, db: Session, model_class: Type[T]):
        """
        Initialize base repository.
        
        Args:
            db: Database session
            model_class: SQLAlchemy model class
        """
        self.db = db
        self.model_class = model_class
    
    def get_by_id(self, id: str) -> Optional[T]:
        """
        Get entity by ID.
        
        Args:
            id: Entity ID
            
        Returns:
            Entity or None if not found
        """
        return self.db.query(self.model_class).filter(
            self.model_class.id == id
        ).first()
    
    def get_all(
        self,
        page: int = 1,
        page_size: int = 20,
        sort_by: Optional[str] = None,
        order: str = "desc",
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> tuple[List[T], int]:
        """
        Get all entities with pagination and filtering.
        
        Args:
            page: Page number (1-indexed)
            page_size: Items per page
            sort_by: Field to sort by
            order: Sort order ('asc' or 'desc')
            filters: Dictionary of filter conditions
            **kwargs: Additional filter parameters (category, user_id, featured, etc.)
            
        Returns:
            Tuple of (entities, total count)
        """
        query = self.db.query(self.model_class)
        
        # Merge kwargs into filters if filters dict provided, otherwise create new dict
        if filters is None:
            filters = {}
        
        # Add kwargs to filters (common filters like category, user_id, featured)
        for key, value in kwargs.items():
            if value is not None:
                filters[key] = value
        
        # Apply filters
        if filters:
            from ..utils.service_helpers import apply_common_filters
            query = apply_common_filters(query, filters, self.model_class)
        
        # Apply sorting
        if sort_by:
            sort_mapping = self._get_sort_mapping()
            query = apply_sorting(query, sort_by, order, sort_mapping)
        
        # Apply pagination
        paginated_query, total = apply_pagination(query, page, page_size)
        
        # Execute query with performance tracking
        start_time = time.time()
        entities = safe_query_execute(paginated_query, "Failed to fetch entities")
        duration = time.time() - start_time
        
        # Record query metrics
        try:
            from ..utils.performance_metrics import get_metrics
            get_metrics().record_query(
                query_type=f"{self.model_class.__name__}.get_all",
                duration=duration,
                success=True
            )
        except Exception:
            pass
        
        return entities, total
    
    def create(self, data: Dict[str, Any]) -> T:
        """
        Create a new entity.
        
        Args:
            data: Entity data
            
        Returns:
            Created entity
        """
        try:
            entity = self.model_class(**data)
            self.db.add(entity)
            self.db.commit()
            self.db.refresh(entity)
            return entity
        except Exception as e:
            self.db.rollback()
            raise
    
    def update(self, id: str, data: Dict[str, Any]) -> Optional[T]:
        """
        Update an entity.
        
        Args:
            id: Entity ID
            data: Update data
            
        Returns:
            Updated entity or None if not found
        """
        entity = self.get_by_id(id)
        if not entity:
            return None
        
        try:
            for key, value in data.items():
                if hasattr(entity, key):
                    setattr(entity, key, value)
            
            self.db.commit()
            self.db.refresh(entity)
            return entity
        except Exception as e:
            self.db.rollback()
            raise
    
    def delete(self, id: str) -> bool:
        """
        Delete an entity.
        
        Args:
            id: Entity ID
            
        Returns:
            True if deleted, False if not found
        """
        entity = self.get_by_id(id)
        if not entity:
            return False
        
        try:
            self.db.delete(entity)
            self.db.commit()
            return True
        except Exception as e:
            self.db.rollback()
            raise
    
    def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """
        Count entities with optional filters.
        
        Args:
            filters: Dictionary of filter conditions
            
        Returns:
            Count of entities
        """
        query = self.db.query(self.model_class)
        
        if filters:
            from ..utils.service_helpers import apply_common_filters
            query = apply_common_filters(query, filters, self.model_class)
        
        return query.count()
    
    def exists(self, id: str) -> bool:
        """
        Check if entity exists.
        
        Args:
            id: Entity ID
            
        Returns:
            True if exists, False otherwise
        """
        return self.get_by_id(id) is not None
    
    def _get_sort_mapping(self) -> Dict[str, Any]:
        """
        Get sort field mapping for the model.
        
        Override in subclasses to provide custom mappings.
        
        Returns:
            Dictionary mapping sort_by values to model attributes
        """
        # Default: try to get attribute directly
        return {}
    
    def get_by_ids(
        self,
        ids: List[str],
        eager_load: Optional[List[str]] = None
    ) -> List[T]:
        """
        Get multiple entities by IDs with optional eager loading.
        
        Args:
            ids: List of entity IDs
            eager_load: Optional list of relationships to eager load
            
        Returns:
            List of entities
        """
        if not ids:
            return []
        
        query = self.db.query(self.model_class).filter(
            self.model_class.id.in_(ids)
        )
        
        # Apply eager loading if specified
        if eager_load:
            for relation in eager_load:
                try:
                    query = query.options(selectinload(getattr(self.model_class, relation)))
                except AttributeError:
                    logger.warning(f"Relation {relation} not found in {self.model_class.__name__}")
        
        return query.all()
    
    def bulk_update(
        self,
        updates: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> int:
        """
        Bulk update multiple entities.
        
        Args:
            updates: List of dictionaries with 'id' and update fields
            batch_size: Number of updates per batch
            
        Returns:
            Number of updated entities
        """
        if not updates:
            return 0
        
        updated_count = 0
        
        try:
            # Process in batches
            for i in range(0, len(updates), batch_size):
                batch = updates[i:i + batch_size]
                
                for update_data in batch:
                    entity_id = update_data.pop('id', None)
                    if not entity_id:
                        continue
                    
                    entity = self.get_by_id(entity_id)
                    if entity:
                        for key, value in update_data.items():
                            if hasattr(entity, key):
                                setattr(entity, key, value)
                        updated_count += 1
                
                self.db.commit()
            
            return updated_count
        except Exception as e:
            self.db.rollback()
            logger.error(f"Bulk update failed: {e}")
            raise
    
    def bulk_create(
        self,
        items: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> List[T]:
        """
        Bulk create multiple entities.
        
        Args:
            items: List of dictionaries with entity data
            batch_size: Number of items per batch
            
        Returns:
            List of created entities
        """
        if not items:
            return []
        
        created_entities = []
        
        try:
            # Process in batches
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                
                for item_data in batch:
                    entity = self.model_class(**item_data)
                    self.db.add(entity)
                    created_entities.append(entity)
                
                self.db.commit()
            
            return created_entities
        except Exception as e:
            self.db.rollback()
            logger.error(f"Bulk create failed: {e}")
            raise






