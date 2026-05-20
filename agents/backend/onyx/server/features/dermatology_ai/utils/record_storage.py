"""
Record Storage - Refactored Version
===================================

This module provides a robust record storage system with proper file handling,
error handling, and input validation.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional


class RecordStorage:
    """
    Storage class for managing records in a JSON file.
    
    Features:
    - Uses context managers for safe file operations
    - Proper error handling and input validation
    - Correct record update handling (merges instead of replaces)
    - Type hints and comprehensive documentation
    """
    
    def __init__(self, file_path: str):
        """
        Initialize the RecordStorage instance.
        
        Args:
            file_path: Path to the JSON file for storing records
            
        Raises:
            ValueError: If file_path is invalid
            RuntimeError: If file cannot be initialized
        """
        if not file_path or not isinstance(file_path, str):
            raise ValueError("file_path must be a non-empty string")
        
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.file_path.exists():
            self._initialize_file()
    
    def _initialize_file(self) -> None:
        """Initialize file with empty structure."""
        try:
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump({"records": []}, f, indent=2, ensure_ascii=False)
        except (IOError, OSError) as e:
            raise RuntimeError(f"Cannot initialize file: {e}") from e
    
    def read(self) -> List[Dict[str, Any]]:
        """
        Read all records from the file.
        
        Returns:
            List of record dictionaries
            
        Raises:
            RuntimeError: If file cannot be read or contains invalid JSON
        """
        if not self.file_path.exists():
            return []
        
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, dict) or 'records' not in data:
                return []
            
            records = data.get('records', [])
            if not isinstance(records, list):
                return []
            
            return records
            
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON in file: {e}") from e
        except (IOError, OSError) as e:
            raise RuntimeError(f"Error reading file: {e}") from e
    
    def write(self, records: List[Dict[str, Any]]) -> bool:
        """
        Write records to the file.
        
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
                raise ValueError(f"Item at index {i} is not a dictionary")
        
        try:
            data = {"records": records}
            
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            return True
            
        except (IOError, OSError) as e:
            raise RuntimeError(f"Error writing file: {e}") from e
        except (TypeError, ValueError) as e:
            raise RuntimeError(f"Error serializing data: {e}") from e
    
    def update(self, record_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update a specific record by ID.
        
        Args:
            record_id: ID of the record to update
            updates: Dictionary of fields to update (merges with existing record)
            
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
                return False
            
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump({"records": records}, f, indent=2, ensure_ascii=False)
            
            return True
            
        except (IOError, OSError) as e:
            raise RuntimeError(f"Error updating record: {e}") from e
        except (ValueError, TypeError) as e:
            raise
        except Exception as e:
            raise RuntimeError(f"Unexpected error: {e}") from e


