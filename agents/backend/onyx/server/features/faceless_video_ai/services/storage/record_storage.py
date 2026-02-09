"""
Record Storage Service
Manages record storage with proper file handling
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class RecordStorage:
    """
    Manages record storage with proper file operations
    
    Features:
    - Uses context managers for safe file operations
    - Proper error handling and validation
    - Correct record update handling
    """
    
    def __init__(self, file_path: str):
        """
        Initialize record storage
        
        Args:
            file_path: Path to the storage file
            
        Raises:
            ValueError: If file_path is invalid
        """
        if not file_path or not isinstance(file_path, str):
            raise ValueError("file_path must be a non-empty string")
        
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.file_path.exists():
            self._initialize_file()
    
    def _initialize_file(self) -> None:
        """Initialize file with empty structure"""
        try:
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump({"records": []}, f, indent=2, ensure_ascii=False)
            logger.debug(f"Initialized storage file: {self.file_path}")
        except (IOError, OSError) as e:
            logger.error(f"Cannot initialize storage file: {e}")
            raise RuntimeError(f"Cannot initialize storage file: {e}") from e
    
    def read(self) -> List[Dict[str, Any]]:
        """
        Read all records from the file
        
        Returns:
            List of records. Returns empty list if file doesn't exist or is invalid
            
        Raises:
            RuntimeError: If file cannot be read or contains invalid JSON
        """
        if not self.file_path.exists():
            logger.warning(f"Storage file does not exist: {self.file_path}")
            return []
        
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, dict) or 'records' not in data:
                logger.error("Invalid file format: missing 'records' key")
                return []
            
            records = data.get('records', [])
            if not isinstance(records, list):
                logger.error("Invalid file format: 'records' is not a list")
                return []
            
            logger.debug(f"Read {len(records)} records from {self.file_path}")
            return records
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in storage file: {e}")
            raise RuntimeError(f"Cannot parse storage file: {e}") from e
        except (IOError, OSError) as e:
            logger.error(f"Failed to read storage file: {e}")
            raise RuntimeError(f"Cannot read storage file: {e}") from e
    
    def write(self, records: List[Dict[str, Any]]) -> bool:
        """
        Write records to the file
        
        Args:
            records: List of record dictionaries to write
            
        Returns:
            True if write was successful
            
        Raises:
            ValueError: If records is not a list or contains invalid items
            RuntimeError: If file cannot be written
        """
        if not isinstance(records, list):
            raise ValueError("records must be a list")
        
        for i, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(f"Element at index {i} is not a dictionary")
        
        try:
            data = {"records": records}
            
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Wrote {len(records)} records to {self.file_path}")
            return True
            
        except (IOError, OSError) as e:
            logger.error(f"Failed to write storage file: {e}")
            raise RuntimeError(f"Cannot write storage file: {e}") from e
        except (TypeError, ValueError) as e:
            logger.error(f"Failed to serialize data: {e}")
            raise RuntimeError(f"Cannot serialize data: {e}") from e
    
    def update(self, record_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update a specific record
        
        Args:
            record_id: ID of the record to update
            updates: Dictionary of fields to update
            
        Returns:
            True if update was successful, False if record not found
            
        Raises:
            ValueError: If record_id or updates are invalid
            RuntimeError: If file operations fail
        """
        if not isinstance(record_id, str) or not record_id:
            raise ValueError("record_id must be a non-empty string")
        
        if not isinstance(updates, dict):
            raise ValueError("updates must be a dictionary")
        
        if not updates:
            logger.warning("No updates provided")
            return False
        
        try:
            records = self.read()
            
            record_found = False
            for i, record in enumerate(records):
                if not isinstance(record, dict):
                    continue
                
                if record.get('id') == record_id:
                    original_id = record.get('id')
                    records[i].update(updates)
                    if 'id' not in records[i] or records[i].get('id') != original_id:
                        records[i]['id'] = original_id
                    record_found = True
                    break
            
            if not record_found:
                logger.warning(f"Record with id '{record_id}' not found")
                return False
            
            self.write(records)
            logger.debug(f"Updated record with id '{record_id}'")
            return True
            
        except (IOError, OSError) as e:
            logger.error(f"Failed to update record: {e}")
            raise RuntimeError(f"Cannot update record: {e}") from e
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid input for update: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during update: {e}")
            raise RuntimeError(f"Unexpected error during update: {e}") from e


_record_storage: Optional[RecordStorage] = None


def get_record_storage(file_path: Optional[str] = None) -> RecordStorage:
    """
    Get record storage instance (singleton)
    
    Args:
        file_path: Path to storage file (optional, uses default if not provided)
        
    Returns:
        RecordStorage instance
    """
    global _record_storage
    if _record_storage is None:
        if file_path is None:
            file_path = "/tmp/faceless_video/records.json"
        _record_storage = RecordStorage(file_path=file_path)
    return _record_storage


