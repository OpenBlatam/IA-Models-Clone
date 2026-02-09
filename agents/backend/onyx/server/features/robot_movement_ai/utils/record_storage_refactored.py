"""
Record Storage - Refactored Version
====================================

A properly refactored record storage class that demonstrates best practices:
- Using context managers (with statement) for file operations
- Proper error handling for user inputs
- Correct indentation
- Proper record handling and storage in update function
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional


class RecordStorage:
    """
    Record storage class with proper file handling and error management.
    
    This class provides methods to read, write, and update records in a JSON file
    with proper error handling and context managers for safe file operations.
    """
    
    def __init__(self, file_path: str):
        """
        Initialize the RecordStorage with a file path.
        
        Args:
            file_path: Path to the JSON file for storing records
            
        Raises:
            ValueError: If file_path is empty or invalid
            RuntimeError: If the file cannot be initialized
        """
        if not file_path or not isinstance(file_path, str):
            raise ValueError("file_path must be a non-empty string")
        
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.file_path.exists():
            self._initialize_file()
    
    def _initialize_file(self) -> None:
        """
        Initialize the storage file with an empty records structure.
        
        Raises:
            RuntimeError: If the file cannot be created
        """
        try:
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump({"records": []}, f, indent=2, ensure_ascii=False)
        except (IOError, OSError) as e:
            raise RuntimeError(f"Cannot initialize storage file: {e}") from e
    
    def read(self) -> List[Dict[str, Any]]:
        """
        Read all records from the file.
        
        Returns:
            List of records. Returns empty list if file doesn't exist or is invalid.
            
        Raises:
            RuntimeError: If the file cannot be read or contains invalid JSON
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
        Write records to the file, replacing all existing records.
        
        Args:
            records: List of record dictionaries to write
            
        Returns:
            True if write was successful, False otherwise
            
        Raises:
            TypeError: If records is not a list
            ValueError: If records contains invalid elements
            RuntimeError: If the file cannot be written
        """
        if not isinstance(records, list):
            raise TypeError("records must be a list")
        
        for i, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(f"Element at index {i} is not a valid dictionary")
        
        try:
            data = {"records": records}
            
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            return True
            
        except (IOError, OSError) as e:
            raise RuntimeError(f"Error writing file: {e}") from e
        except (TypeError, ValueError) as e:
            raise RuntimeError(f"Error serializing records: {e}") from e
    
    def update(self, record_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update a specific record by ID.
        
        Args:
            record_id: The ID of the record to update
            updates: Dictionary of fields to update
            
        Returns:
            True if the record was found and updated, False otherwise
            
        Raises:
            ValueError: If record_id is empty or updates is not a dictionary
            TypeError: If record_id is not a string
            RuntimeError: If file operations fail
        """
        if not isinstance(record_id, str):
            raise TypeError("record_id must be a string")
        
        if not record_id or not record_id.strip():
            raise ValueError("record_id must be a non-empty string")
        
        if not isinstance(updates, dict):
            raise TypeError("updates must be a dictionary")
        
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
            raise RuntimeError(f"Unexpected error during update: {e}") from e


