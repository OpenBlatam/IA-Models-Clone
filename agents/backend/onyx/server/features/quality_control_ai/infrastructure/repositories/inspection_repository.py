"""
Inspection Repository

Repository for persisting and retrieving inspections.
"""

import logging
from typing import List, Optional
from datetime import datetime
import json

from ...domain import Inspection
from ...domain.exceptions import InspectionException

logger = logging.getLogger(__name__)


class InspectionRepository:
    """
    Repository for inspection persistence.
    
    This is an abstract implementation that can be extended
    to support different storage backends (database, file system, etc.).
    """
    
    def __init__(self, storage_backend=None):
        """
        Initialize repository.
        
        Args:
            storage_backend: Storage backend adapter (database, file system, etc.)
        """
        self.storage_backend = storage_backend
        self._in_memory_store = {}  # Fallback in-memory storage
    
    def save(self, inspection: Inspection) -> str:
        """
        Save an inspection.
        
        Args:
            inspection: Inspection entity to save
        
        Returns:
            Inspection ID
        
        Raises:
            InspectionException: If save fails
        """
        try:
            if self.storage_backend:
                return self.storage_backend.save_inspection(inspection)
            else:
                # Fallback: in-memory storage
                self._in_memory_store[inspection.id] = inspection.to_dict()
                logger.info(f"Inspection {inspection.id} saved to in-memory store")
                return inspection.id
        except Exception as e:
            logger.error(f"Failed to save inspection: {str(e)}", exc_info=True)
            raise InspectionException(f"Failed to save inspection: {str(e)}")
    
    def find_by_id(self, inspection_id: str) -> Optional[Inspection]:
        """
        Find inspection by ID.
        
        Args:
            inspection_id: Inspection ID
        
        Returns:
            Inspection entity or None if not found
        """
        try:
            if self.storage_backend:
                data = self.storage_backend.get_inspection(inspection_id)
                if data:
                    return self._from_dict(data)
                return None
            else:
                # Fallback: in-memory storage
                data = self._in_memory_store.get(inspection_id)
                if data:
                    return self._from_dict(data)
                return None
        except Exception as e:
            logger.error(f"Failed to find inspection: {str(e)}", exc_info=True)
            return None
    
    def find_all(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Inspection]:
        """
        Find all inspections with optional filtering.
        
        Args:
            limit: Maximum number of results
            offset: Offset for pagination
            start_date: Filter by start date
            end_date: Filter by end date
        
        Returns:
            List of inspection entities
        """
        try:
            if self.storage_backend:
                data_list = self.storage_backend.list_inspections(
                    limit=limit,
                    offset=offset,
                    start_date=start_date,
                    end_date=end_date,
                )
                return [self._from_dict(data) for data in data_list]
            else:
                # Fallback: in-memory storage
                inspections = list(self._in_memory_store.values())
                
                # Apply date filtering
                if start_date or end_date:
                    filtered = []
                    for data in inspections:
                        created_at = datetime.fromisoformat(data.get('created_at', ''))
                        if start_date and created_at < start_date:
                            continue
                        if end_date and created_at > end_date:
                            continue
                        filtered.append(data)
                    inspections = filtered
                
                # Apply pagination
                if offset:
                    inspections = inspections[offset:]
                if limit:
                    inspections = inspections[:limit]
                
                return [self._from_dict(data) for data in inspections]
        except Exception as e:
            logger.error(f"Failed to find inspections: {str(e)}", exc_info=True)
            return []
    
    def delete(self, inspection_id: str) -> bool:
        """
        Delete an inspection.
        
        Args:
            inspection_id: Inspection ID
        
        Returns:
            True if deleted, False if not found
        """
        try:
            if self.storage_backend:
                return self.storage_backend.delete_inspection(inspection_id)
            else:
                # Fallback: in-memory storage
                if inspection_id in self._in_memory_store:
                    del self._in_memory_store[inspection_id]
                    logger.info(f"Inspection {inspection_id} deleted from in-memory store")
                    return True
                return False
        except Exception as e:
            logger.error(f"Failed to delete inspection: {str(e)}", exc_info=True)
            return False
    
    def count(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> int:
        """
        Count inspections with optional date filtering.
        
        Args:
            start_date: Filter by start date
            end_date: Filter by end date
        
        Returns:
            Count of inspections
        """
        try:
            if self.storage_backend:
                return self.storage_backend.count_inspections(
                    start_date=start_date,
                    end_date=end_date,
                )
            else:
                # Fallback: in-memory storage
                count = len(self._in_memory_store)
                if start_date or end_date:
                    filtered = self.find_all(start_date=start_date, end_date=end_date)
                    count = len(filtered)
                return count
        except Exception as e:
            logger.error(f"Failed to count inspections: {str(e)}", exc_info=True)
            return 0
    
    def _from_dict(self, data: dict) -> Inspection:
        """
        Convert dictionary to Inspection entity.
        
        Args:
            data: Dictionary representation
        
        Returns:
            Inspection entity
        """
        # This is a simplified conversion
        # In a real implementation, this would properly reconstruct
        # all domain entities and value objects
        from ...domain import (
            Inspection,
            ImageMetadata,
            QualityScore,
        )
        
        # Reconstruct image metadata
        img_meta = ImageMetadata(
            width=data['image_metadata']['width'],
            height=data['image_metadata']['height'],
            channels=data['image_metadata'].get('channels', 3),
            format=data['image_metadata'].get('format'),
            source=data['image_metadata'].get('source'),
        )
        
        # Reconstruct quality score
        quality_score = QualityScore(
            score=data['quality_score']['score'],
            defects_count=data['quality_score']['defects_count'],
            anomalies_count=data['quality_score']['anomalies_count'],
        )
        
        # Create inspection (defects and anomalies would need to be reconstructed too)
        inspection = Inspection(
            id=data['id'],
            image_metadata=img_meta,
            quality_score=quality_score,
        )
        
        return inspection



