"""
Data Storage Utility - Refactored Version
==========================================

A refactored data storage class that demonstrates best practices for:
- Using context managers for file operations
- Proper error handling
- Correct indentation
- Proper record handling and storage
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any


class DataStorage:
    """
    A data storage class that handles file operations with proper error handling
    and context managers.
    """
    
    def __init__(self, file_path: str):
        """
        Initialize the DataStorage with a file path.
        
        Args:
            file_path: Path to the JSON file for storing data
        """
        if not file_path:
            raise ValueError("File path cannot be empty")
        
        self.file_path = Path(file_path)
        self._ensure_directory_exists()
    
    def _ensure_directory_exists(self) -> None:
        """Ensure the directory for the file exists."""
        try:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as e:
            raise RuntimeError(f"Failed to create directory: {e}") from e
    
    def write(self, record: Dict[str, Any]) -> bool:
        """
        Write a new record to the file.
        
        Args:
            record: Dictionary containing the record data
            
        Returns:
            True if write was successful, False otherwise
            
        Raises:
            ValueError: If record is invalid
            IOError: If file operation fails
        """
        if not isinstance(record, dict):
            raise ValueError("Record must be a dictionary")
        
        if not record:
            raise ValueError("Record cannot be empty")
        
        try:
            records = self.read_all()
            if records is None:
                records = []
            
            record_id = record.get('id')
            if record_id and any(r.get('id') == record_id for r in records):
                raise ValueError(f"Record with id '{record_id}' already exists")
            
            records.append(record)
            
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(records, f, indent=2, ensure_ascii=False)
            
            return True
            
        except (IOError, OSError, json.JSONDecodeError) as e:
            raise IOError(f"Failed to write record: {e}") from e
        except ValueError:
            raise
    
    def read(self, record_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Read a record or all records from the file.
        
        Args:
            record_id: Optional ID of the record to retrieve. If None, returns all records.
            
        Returns:
            A single record dictionary if record_id is provided, 
            or a list of all records if record_id is None.
            Returns None if record not found or file doesn't exist.
            
        Raises:
            IOError: If file operation fails
        """
        if not isinstance(record_id, (str, type(None))):
            raise ValueError("record_id must be a string or None")
        
        try:
            if not self.file_path.exists():
                return None if record_id else []
            
            with open(self.file_path, 'r', encoding='utf-8') as f:
                records = json.load(f)
            
            if not isinstance(records, list):
                raise ValueError("File does not contain a valid list of records")
            
            if record_id is None:
                return records
            
            for record in records:
                if isinstance(record, dict) and record.get('id') == record_id:
                    return record
            
            return None
            
        except (IOError, OSError) as e:
            raise IOError(f"Failed to read file: {e}") from e
        except json.JSONDecodeError as e:
            raise IOError(f"Invalid JSON in file: {e}") from e
    
    def read_all(self) -> Optional[List[Dict[str, Any]]]:
        """
        Read all records from the file.
        
        Returns:
            List of all records, or None if file doesn't exist or is empty.
            
        Raises:
            IOError: If file operation fails
        """
        result = self.read()
        if result is None:
            return []
        if isinstance(result, list):
            return result
        return [result]
    
    def update(self, record_id: str, updated_data: Dict[str, Any]) -> bool:
        """
        Update an existing record in the file.
        
        Args:
            record_id: ID of the record to update
            updated_data: Dictionary containing updated fields
            
        Returns:
            True if update was successful, False if record not found
            
        Raises:
            ValueError: If record_id or updated_data is invalid
            IOError: If file operation fails
        """
        if not isinstance(record_id, str) or not record_id:
            raise ValueError("record_id must be a non-empty string")
        
        if not isinstance(updated_data, dict):
            raise ValueError("updated_data must be a dictionary")
        
        if not updated_data:
            raise ValueError("updated_data cannot be empty")
        
        try:
            records = self.read_all()
            if records is None:
                return False
            
            record_found = False
            for i, record in enumerate(records):
                if isinstance(record, dict) and record.get('id') == record_id:
                    records[i] = {**record, **updated_data, 'id': record_id}
                    record_found = True
                    break
            
            if not record_found:
                return False
            
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(records, f, indent=2, ensure_ascii=False)
            
            return True
            
        except (IOError, OSError) as e:
            raise IOError(f"Failed to update record: {e}") from e
        except json.JSONDecodeError as e:
            raise IOError(f"Invalid JSON in file: {e}") from e
    
    def delete(self, record_id: str) -> bool:
        """
        Delete a record from the file.
        
        Args:
            record_id: ID of the record to delete
            
        Returns:
            True if deletion was successful, False if record not found
            
        Raises:
            ValueError: If record_id is invalid
            IOError: If file operation fails
        """
        if not isinstance(record_id, str) or not record_id:
            raise ValueError("record_id must be a non-empty string")
        
        try:
            records = self.read_all()
            if records is None:
                return False
            
            initial_length = len(records)
            records = [r for r in records if not (isinstance(r, dict) and r.get('id') == record_id)]
            
            if len(records) == initial_length:
                return False
            
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(records, f, indent=2, ensure_ascii=False)
            
            return True
            
        except (IOError, OSError) as e:
            raise IOError(f"Failed to delete record: {e}") from e
        except json.JSONDecodeError as e:
            raise IOError(f"Invalid JSON in file: {e}") from e


