"""
Complete Refactored Code - FileStorage Class
============================================

This is the complete, production-ready refactored version of the FileStorage class
that addresses all the requirements:

1. ✅ Uses context managers (`with` statement) for all file operations
2. ✅ Correct indentation in all methods
3. ✅ Proper record handling in update method (merges updates instead of replacing)
4. ✅ Comprehensive error handling for user inputs
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional


class FileStorage:
    """
    File-based storage system with proper error handling and context managers.
    
    This class provides read, write, and update operations for JSON-based
    record storage with robust error handling and input validation.
    """
    
    def __init__(self, file_path: str) -> None:
        """
        Initialize the file storage.
        
        Args:
            file_path: Path to the JSON file for storage
            
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
        """
        Initialize the storage file with an empty structure.
        
        Raises:
            RuntimeError: If file cannot be created
        """
        try:
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump({"records": []}, f, indent=2, ensure_ascii=False)
        except (IOError, OSError) as e:
            raise RuntimeError(f"Cannot initialize storage file: {e}") from e
    
    def read(self) -> List[Dict[str, Any]]:
        """
        Read all records from the storage file.
        
        Returns:
            List of records (dictionaries)
            
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
            raise RuntimeError(f"Invalid JSON in storage file: {e}") from e
        except (IOError, OSError) as e:
            raise RuntimeError(f"Error reading storage file: {e}") from e
    
    def write(self, records: List[Dict[str, Any]]) -> bool:
        """
        Write records to the storage file.
        
        Args:
            records: List of dictionaries to write
            
        Returns:
            True if successful
            
        Raises:
            ValueError: If records is not a valid list of dictionaries
            RuntimeError: If file cannot be written
        """
        if not isinstance(records, list):
            raise ValueError("records must be a list")
        
        for i, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(f"Element at index {i} must be a dictionary")
        
        try:
            data = {"records": records}
            
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            return True
            
        except (IOError, OSError) as e:
            raise RuntimeError(f"Error writing to storage file: {e}") from e
        except (TypeError, ValueError) as e:
            raise RuntimeError(f"Error serializing data: {e}") from e
    
    def update(self, record_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update a specific record by ID.
        
        Args:
            record_id: ID of the record to update
            updates: Dictionary with fields to update (merges with existing record)
            
        Returns:
            True if record was found and updated, False otherwise
            
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
            raise RuntimeError(f"Unexpected error during update: {e}") from e


