"""
History Repository
=================

Single responsibility: Manage persistence of history entries.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging

from ...domain.entities.history_entry import HistoryEntry
from .base_repository import BaseRepository
from .database_manager import DatabaseManager

logger = logging.getLogger(__name__)


class HistoryRepository(BaseRepository):
    """
    Repository for managing history entries.
    
    Single Responsibility: Handle all database operations for history entries.
    """
    
    def __init__(self, database_manager: DatabaseManager):
        """
        Initialize the repository.
        
        Args:
            database_manager: Database manager instance
        """
        super().__init__(database_manager)
        self._table_name = "history_entries"
    
    async def save(self, entry: HistoryEntry) -> HistoryEntry:
        """
        Save history entry to database.
        
        Args:
            entry: History entry to save
            
        Returns:
            Saved history entry
        """
        try:
            query = f"""
                INSERT OR REPLACE INTO {self._table_name} 
                (id, content, content_hash, model_version, timestamp, metrics, metadata, user_feedback)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            params = (
                entry.id,
                entry.content,
                entry.content_hash,
                entry.model_version,
                entry.timestamp.isoformat(),
                self._serialize_metrics(entry.metrics),
                self._serialize_metadata(entry.metadata),
                self._serialize_feedback(entry.user_feedback)
            )
            
            await self._execute_query(query, params)
            logger.info(f"Saved history entry: {entry.id}")
            
            return entry
            
        except Exception as e:
            logger.error(f"Error saving history entry {entry.id}: {e}")
            raise
    
    async def find_by_id(self, entry_id: str) -> Optional[HistoryEntry]:
        """
        Find history entry by ID.
        
        Args:
            entry_id: Entry ID to find
            
        Returns:
            History entry or None if not found
        """
        try:
            query = f"SELECT * FROM {self._table_name} WHERE id = ?"
            result = await self._fetch_one(query, (entry_id,))
            
            if result:
                return self._deserialize_entry(result)
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding history entry {entry_id}: {e}")
            raise
    
    async def find_by_model_version(
        self,
        model_version: str,
        limit: Optional[int] = None
    ) -> List[HistoryEntry]:
        """
        Find entries by model version.
        
        Args:
            model_version: Model version to filter by
            limit: Maximum number of entries to return
            
        Returns:
            List of history entries
        """
        try:
            query = f"""
                SELECT * FROM {self._table_name} 
                WHERE model_version = ? 
                ORDER BY timestamp DESC
            """
            
            if limit:
                query += f" LIMIT {limit}"
            
            results = await self._fetch_all(query, (model_version,))
            return [self._deserialize_entry(row) for row in results]
            
        except Exception as e:
            logger.error(f"Error finding entries by model version {model_version}: {e}")
            raise
    
    async def find_by_time_range(
        self,
        start_time: datetime,
        end_time: datetime,
        model_version: Optional[str] = None
    ) -> List[HistoryEntry]:
        """
        Find entries within time range.
        
        Args:
            start_time: Start time
            end_time: End time
            model_version: Optional model version filter
            
        Returns:
            List of history entries
        """
        try:
            query = f"""
                SELECT * FROM {self._table_name} 
                WHERE timestamp >= ? AND timestamp <= ?
            """
            params = [start_time.isoformat(), end_time.isoformat()]
            
            if model_version:
                query += " AND model_version = ?"
                params.append(model_version)
            
            query += " ORDER BY timestamp ASC"
            
            results = await self._fetch_all(query, params)
            return [self._deserialize_entry(row) for row in results]
            
        except Exception as e:
            logger.error(f"Error finding entries in time range: {e}")
            raise
    
    async def find_recent_entries(
        self,
        days: int = 7,
        model_version: Optional[str] = None
    ) -> List[HistoryEntry]:
        """
        Find recent entries.
        
        Args:
            days: Number of days to look back
            model_version: Optional model version filter
            
        Returns:
            List of recent history entries
        """
        cutoff_time = datetime.utcnow() - timedelta(days=days)
        return await self.find_by_time_range(cutoff_time, datetime.utcnow(), model_version)
    
    async def find_by_content_hash(self, content_hash: str) -> List[HistoryEntry]:
        """
        Find entries by content hash.
        
        Args:
            content_hash: Content hash to search for
            
        Returns:
            List of history entries with matching hash
        """
        try:
            query = f"""
                SELECT * FROM {self._table_name} 
                WHERE content_hash = ? 
                ORDER BY timestamp DESC
            """
            
            results = await self._fetch_all(query, (content_hash,))
            return [self._deserialize_entry(row) for row in results]
            
        except Exception as e:
            logger.error(f"Error finding entries by content hash: {e}")
            raise
    
    async def count_entries(self, model_version: Optional[str] = None) -> int:
        """
        Count total entries.
        
        Args:
            model_version: Optional model version filter
            
        Returns:
            Total count of entries
        """
        try:
            query = f"SELECT COUNT(*) FROM {self._table_name}"
            params = []
            
            if model_version:
                query += " WHERE model_version = ?"
                params.append(model_version)
            
            result = await self._fetch_one(query, params)
            return result[0] if result else 0
            
        except Exception as e:
            logger.error(f"Error counting entries: {e}")
            raise
    
    async def get_model_versions(self) -> List[str]:
        """
        Get all unique model versions.
        
        Returns:
            List of unique model versions
        """
        try:
            query = f"SELECT DISTINCT model_version FROM {self._table_name}"
            results = await self._fetch_all(query)
            return [row[0] for row in results if row[0]]
            
        except Exception as e:
            logger.error(f"Error getting model versions: {e}")
            raise
    
    async def delete(self, entry_id: str) -> bool:
        """
        Delete history entry.
        
        Args:
            entry_id: Entry ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        try:
            query = f"DELETE FROM {self._table_name} WHERE id = ?"
            result = await self._execute_query(query, (entry_id,))
            
            if result.rowcount > 0:
                logger.info(f"Deleted history entry: {entry_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error deleting history entry {entry_id}: {e}")
            raise
    
    def _serialize_metrics(self, metrics) -> str:
        """Serialize metrics to JSON string."""
        import json
        return json.dumps(metrics.to_dict())
    
    def _serialize_metadata(self, metadata: Dict[str, Any]) -> str:
        """Serialize metadata to JSON string."""
        import json
        return json.dumps(metadata)
    
    def _serialize_feedback(self, feedback: Optional[Dict[str, Any]]) -> Optional[str]:
        """Serialize feedback to JSON string."""
        if feedback is None:
            return None
        import json
        return json.dumps(feedback)
    
    def _deserialize_entry(self, row: tuple) -> HistoryEntry:
        """Deserialize database row to HistoryEntry."""
        import json
        from ...domain.value_objects.content_metrics import ContentMetrics
        
        return HistoryEntry(
            id=row[0],
            content=row[1],
            content_hash=row[2],
            model_version=row[3],
            timestamp=datetime.fromisoformat(row[4]),
            metrics=ContentMetrics.from_dict(json.loads(row[5])),
            metadata=json.loads(row[6]) if row[6] else {},
            user_feedback=json.loads(row[7]) if row[7] else None
        )




