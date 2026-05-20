"""
Record Storage - Advanced Features Extension

This module extends RecordStorage with advanced features like:
- Batch operations
- Query/filter capabilities
- Backup and restore
- Data validation schemas
- Transaction support
"""

import json
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime

from .record_storage import RecordStorage

logger = logging.getLogger(__name__)


class AdvancedRecordStorage(RecordStorage):
    """
    Extended RecordStorage with advanced features.
    
    Inherits all base functionality and adds:
    - Batch operations
    - Query/filter methods
    - Backup/restore
    - Data validation
    """
    
    def batch_add(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Add multiple records in a single operation.
        
        Args:
            records: List of record dictionaries to add
            
        Returns:
            Dictionary with success count and any errors
        """
        if not isinstance(records, list):
            raise ValueError("records must be a list")
        
        existing_records = self.read()
        existing_ids = {r.get('id') for r in existing_records if isinstance(r, dict)}
        
        added = 0
        skipped = 0
        errors = []
        
        for i, record in enumerate(records):
            if not isinstance(record, dict):
                errors.append(f"Index {i}: not a dictionary")
                continue
            
            if 'id' not in record:
                errors.append(f"Index {i}: missing 'id' field")
                continue
            
            if record['id'] in existing_ids:
                skipped += 1
                continue
            
            existing_records.append(record)
            existing_ids.add(record['id'])
            added += 1
        
        if added > 0:
            self.write(existing_records)
        
        return {
            "added": added,
            "skipped": skipped,
            "errors": errors,
            "total": len(records)
        }
    
    def batch_update(self, updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Update multiple records in a single operation.
        
        Args:
            updates: List of dicts with 'id' and fields to update
            
        Returns:
            Dictionary with success count and any errors
        """
        if not isinstance(updates, list):
            raise ValueError("updates must be a list")
        
        records = self.read()
        record_map = {r.get('id'): i for i, r in enumerate(records) if isinstance(r, dict)}
        
        updated = 0
        not_found = 0
        errors = []
        
        for i, update in enumerate(updates):
            if not isinstance(update, dict):
                errors.append(f"Index {i}: not a dictionary")
                continue
            
            record_id = update.get('id')
            if not record_id:
                errors.append(f"Index {i}: missing 'id' field")
                continue
            
            if record_id not in record_map:
                not_found += 1
                continue
            
            update_data = {k: v for k, v in update.items() if k != 'id'}
            if not update_data:
                continue
            
            idx = record_map[record_id]
            original_id = records[idx].get('id')
            records[idx].update(update_data)
            if 'id' not in records[idx] or records[idx].get('id') != original_id:
                records[idx]['id'] = original_id
            updated += 1
        
        if updated > 0:
            self.write(records)
        
        return {
            "updated": updated,
            "not_found": not_found,
            "errors": errors,
            "total": len(updates)
        }
    
    def batch_delete(self, record_ids: List[str]) -> Dict[str, Any]:
        """
        Delete multiple records in a single operation.
        
        Args:
            record_ids: List of record IDs to delete
            
        Returns:
            Dictionary with deletion count
        """
        if not isinstance(record_ids, list):
            raise ValueError("record_ids must be a list")
        
        records = self.read()
        ids_to_delete = set(record_ids)
        
        original_count = len(records)
        records = [
            r for r in records
            if isinstance(r, dict) and r.get('id') not in ids_to_delete
        ]
        
        deleted = original_count - len(records)
        
        if deleted > 0:
            self.write(records)
        
        return {
            "deleted": deleted,
            "not_found": len(ids_to_delete) - deleted,
            "total": len(record_ids)
        }
    
    def filter(self, predicate: Callable[[Dict[str, Any]], bool]) -> List[Dict[str, Any]]:
        """
        Filter records using a predicate function.
        
        Args:
            predicate: Function that takes a record and returns True/False
            
        Returns:
            List of records that match the predicate
        """
        if not callable(predicate):
            raise ValueError("predicate must be callable")
        
        records = self.read()
        return [r for r in records if isinstance(r, dict) and predicate(r)]
    
    def find_all(self, **kwargs) -> List[Dict[str, Any]]:
        """
        Find all records matching the given field values.
        
        Args:
            **kwargs: Field name and value pairs to match
            
        Returns:
            List of matching records
        """
        records = self.read()
        matches = []
        
        for record in records:
            if not isinstance(record, dict):
                continue
            
            match = True
            for key, value in kwargs.items():
                if record.get(key) != value:
                    match = False
                    break
            
            if match:
                matches.append(record)
        
        return matches
    
    def find_one(self, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Find the first record matching the given field values.
        
        Args:
            **kwargs: Field name and value pairs to match
            
        Returns:
            First matching record or None
        """
        matches = self.find_all(**kwargs)
        return matches[0] if matches else None
    
    def count(self, **kwargs) -> int:
        """
        Count records matching the given field values.
        
        Args:
            **kwargs: Field name and value pairs to match
            
        Returns:
            Number of matching records
        """
        return len(self.find_all(**kwargs))
    
    def backup(self, backup_path: Optional[str] = None) -> str:
        """
        Create a backup of the storage file.
        
        Args:
            backup_path: Optional path for backup file
            
        Returns:
            Path to the backup file
        """
        if backup_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{self.file_path}.backup_{timestamp}"
        
        backup_path = Path(backup_path)
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            shutil.copy2(self.file_path, backup_path)
            logger.info(f"Backup created: {backup_path}")
            return str(backup_path)
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            raise RuntimeError(f"Cannot create backup: {e}") from e
    
    def restore(self, backup_path: str) -> bool:
        """
        Restore from a backup file.
        
        Args:
            backup_path: Path to the backup file
            
        Returns:
            True if restore was successful
        """
        backup_path = Path(backup_path)
        
        if not backup_path.exists():
            raise ValueError(f"Backup file does not exist: {backup_path}")
        
        try:
            shutil.copy2(backup_path, self.file_path)
            logger.info(f"Restored from backup: {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to restore backup: {e}")
            raise RuntimeError(f"Cannot restore backup: {e}") from e
    
    def validate_schema(self, record: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
        """
        Validate a record against a schema.
        
        Args:
            record: Record to validate
            schema: Schema definition with field requirements
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        required_fields = schema.get('required', [])
        for field in required_fields:
            if field not in record:
                errors.append(f"Missing required field: {field}")
        
        field_types = schema.get('types', {})
        for field, expected_type in field_types.items():
            if field in record:
                if not isinstance(record[field], expected_type):
                    errors.append(
                        f"Field '{field}' must be {expected_type.__name__}, "
                        f"got {type(record[field]).__name__}"
                    )
        
        return errors
    
    def add_with_validation(self, record: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        """
        Add a record with schema validation.
        
        Args:
            record: Record to add
            schema: Schema definition for validation
            
        Returns:
            True if added successfully
            
        Raises:
            ValueError: If validation fails
        """
        errors = self.validate_schema(record, schema)
        if errors:
            raise ValueError(f"Validation failed: {', '.join(errors)}")
        
        return self.add(record)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the storage.
        
        Returns:
            Dictionary with storage statistics
        """
        records = self.read()
        
        if not records:
            return {
                "total_records": 0,
                "file_size": self.file_path.stat().st_size if self.file_path.exists() else 0,
                "file_path": str(self.file_path)
            }
        
        field_counts = {}
        for record in records:
            if isinstance(record, dict):
                for key in record.keys():
                    field_counts[key] = field_counts.get(key, 0) + 1
        
        return {
            "total_records": len(records),
            "file_size": self.file_path.stat().st_size if self.file_path.exists() else 0,
            "file_path": str(self.file_path),
            "fields": field_counts,
            "most_common_fields": sorted(
                field_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }


