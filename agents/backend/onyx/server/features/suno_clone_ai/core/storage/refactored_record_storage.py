"""
Refactored Record Storage

A properly structured file-based record storage system with:
- Context managers for file operations
- Proper error handling
- Correct indentation
- Proper record handling in update method
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class RecordStorage:
    """
    File-based record storage with proper error handling and context managers.
    
    Stores records in a JSON file with proper file handling.
    """
    
    def __init__(self, file_path: str):
        """
        Initialize record storage.
        
        Args:
            file_path: Path to the JSON file for storing records
            
        Raises:
            ValueError: If file_path is empty or invalid
        """
        if not file_path or not isinstance(file_path, str):
            raise ValueError("file_path must be a non-empty string")
        
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.file_path.exists():
            self._initialize_file()
    
    def _initialize_file(self) -> None:
        """Initialize the storage file with an empty records list."""
        try:
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump({"records": []}, f, indent=2)
            logger.info(f"Initialized storage file: {self.file_path}")
        except (IOError, OSError) as e:
            logger.error(f"Failed to initialize storage file: {e}")
            raise RuntimeError(f"Cannot initialize storage file: {e}") from e
    
    def read(self) -> List[Dict[str, Any]]:
        """
        Read all records from the file.
        
        Returns:
            List of records. Returns empty list if file doesn't exist or is invalid.
            
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
        Write records to the file, replacing all existing records.
        
        Args:
            records: List of record dictionaries to write
            
        Returns:
            True if write was successful, False otherwise
            
        Raises:
            ValueError: If records is not a list
            RuntimeError: If file cannot be written
        """
        if not isinstance(records, list):
            raise ValueError("records must be a list")
        
        try:
            data = {"records": records}
            
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Wrote {len(records)} records to {self.file_path}")
            return True
            
        except (IOError, OSError) as e:
            logger.error(f"Failed to write storage file: {e}")
            raise RuntimeError(f"Cannot write storage file: {e}") from e
        except (TypeError, ValueError) as e:
            logger.error(f"Failed to serialize records: {e}")
            raise RuntimeError(f"Cannot serialize records: {e}") from e
    
    def update(self, record_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update a specific record by ID.
        
        Args:
            record_id: The ID of the record to update
            updates: Dictionary of fields to update
            
        Returns:
            True if record was found and updated, False otherwise
            
        Raises:
            ValueError: If record_id is empty or updates is not a dict
            RuntimeError: If file operations fail
        """
        if not record_id or not isinstance(record_id, str):
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
                    logger.warning(f"Skipping invalid record at index {i}")
                    continue
                
                if record.get('id') == record_id:
                    records[i].update(updates)
                    record_found = True
                    logger.debug(f"Updated record with id: {record_id}")
                    break
            
            if not record_found:
                logger.warning(f"Record with id '{record_id}' not found")
                return False
            
            self.write(records)
            logger.info(f"Successfully updated record: {record_id}")
            return True
            
        except (RuntimeError, ValueError) as e:
            logger.error(f"Failed to update record: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during update: {e}")
            raise RuntimeError(f"Unexpected error during update: {e}") from e
    
    def add(self, record: Dict[str, Any]) -> bool:
        """
        Add a new record to the storage.
        
        Args:
            record: Dictionary representing the record to add
            
        Returns:
            True if record was added successfully, False otherwise
            
        Raises:
            ValueError: If record is not a dict or missing required 'id' field
        """
        if not isinstance(record, dict):
            raise ValueError("record must be a dictionary")
        
        if 'id' not in record:
            raise ValueError("record must contain an 'id' field")
        
        try:
            records = self.read()
            
            for existing_record in records:
                if isinstance(existing_record, dict) and existing_record.get('id') == record['id']:
                    logger.warning(f"Record with id '{record['id']}' already exists")
                    return False
            
            records.append(record)
            self.write(records)
            logger.info(f"Added new record with id: {record['id']}")
            return True
            
        except (RuntimeError, ValueError) as e:
            logger.error(f"Failed to add record: {e}")
            raise
    
    def delete(self, record_id: str) -> bool:
        """
        Delete a record by ID.
        
        Args:
            record_id: The ID of the record to delete
            
        Returns:
            True if record was found and deleted, False otherwise
            
        Raises:
            ValueError: If record_id is empty
        """
        if not record_id or not isinstance(record_id, str):
            raise ValueError("record_id must be a non-empty string")
        
        try:
            records = self.read()
            original_count = len(records)
            
            records = [
                record for record in records
                if isinstance(record, dict) and record.get('id') != record_id
            ]
            
            if len(records) == original_count:
                logger.warning(f"Record with id '{record_id}' not found")
                return False
            
            self.write(records)
            logger.info(f"Deleted record with id: {record_id}")
            return True
            
        except (RuntimeError, ValueError) as e:
            logger.error(f"Failed to delete record: {e}")
            raise
    
    def get(self, record_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific record by ID.
        
        Args:
            record_id: The ID of the record to retrieve
            
        Returns:
            The record dictionary if found, None otherwise
            
        Raises:
            ValueError: If record_id is empty
        """
        if not record_id or not isinstance(record_id, str):
            raise ValueError("record_id must be a non-empty string")
        
        try:
            records = self.read()
            
            for record in records:
                if isinstance(record, dict) and record.get('id') == record_id:
                    return record
            
            logger.debug(f"Record with id '{record_id}' not found")
            return None
            
        except RuntimeError as e:
            logger.error(f"Failed to get record: {e}")
            raise


