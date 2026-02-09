"""
REFACTORED CODE - Complete and Corrected Version
================================================

This is the complete refactored version of the FileStorage class that addresses:
1. Context managers (with statement) for file operations
2. Corrected indentation in read() and update() functions
3. Fixed update() function to properly store updated records back to file
4. Appropriate error handling for write, read, and update methods

All requirements have been fully implemented.
"""

import json
import os
from typing import Dict, List, Any, Optional
from pathlib import Path


class FileStorage:
    """
    File-based storage class with write, read, and update methods.
    Uses context managers for safe file operations.
    """
    
    def __init__(self, file_path: str):
        """
        Initialize FileStorage with a file path.
        
        Args:
            file_path: Path to the JSON file for storage
            
        Raises:
            ValueError: If file_path is empty or invalid
        """
        if not file_path or not isinstance(file_path, str):
            raise ValueError("file_path must be a non-empty string")
        
        self.file_path = file_path
        self._ensure_directory_exists()
    
    def _ensure_directory_exists(self) -> None:
        """Ensure the directory for the file exists."""
        directory = os.path.dirname(self.file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
    
    def write(self, data: List[Dict[str, Any]]) -> None:
        """
        Write data to the file.
        
        REQUIREMENT 1: Uses context manager (with statement)
        REQUIREMENT 4: Appropriate error handling for user inputs
        
        Args:
            data: List of dictionaries to write to file
            
        Raises:
            TypeError: If data is not a list
            ValueError: If data contains invalid entries
            IOError: If file cannot be written
        """
        if not isinstance(data, list):
            raise TypeError("data must be a list")
        
        if not all(isinstance(item, dict) for item in data):
            raise ValueError("All items in data must be dictionaries")
        
        try:
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except IOError as e:
            raise IOError(f"Failed to write to file {self.file_path}: {str(e)}")
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid data format: {str(e)}")
    
    def read(self) -> List[Dict[str, Any]]:
        """
        Read data from the file.
        
        REQUIREMENT 1: Uses context manager (with statement)
        REQUIREMENT 2: Corrected indentation issues
        REQUIREMENT 4: Appropriate error handling for user inputs
        
        Returns:
            List of dictionaries read from file
            
        Raises:
            FileNotFoundError: If file does not exist
            json.JSONDecodeError: If file contains invalid JSON
            IOError: If file cannot be read
        """
        if not os.path.exists(self.file_path):
            return []
        
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if not isinstance(data, list):
                raise ValueError("File does not contain a valid list")
            
            return data
        except FileNotFoundError:
            return []
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Invalid JSON in file {self.file_path}: {str(e)}",
                e.doc,
                e.pos
            )
        except IOError as e:
            raise IOError(f"Failed to read from file {self.file_path}: {str(e)}")
    
    def update(self, record_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update a record in the file by ID.
        
        REQUIREMENT 2: Corrected indentation issues
        REQUIREMENT 3: Rectified errors in record handling and storage
        REQUIREMENT 4: Appropriate error handling for user inputs
        
        Args:
            record_id: ID of the record to update
            updates: Dictionary of fields to update
            
        Returns:
            True if record was updated, False if not found
            
        Raises:
            TypeError: If record_id is not a string or updates is not a dict
            ValueError: If record_id is empty or updates is empty
            IOError: If file operations fail
        """
        if not isinstance(record_id, str):
            raise TypeError("record_id must be a string")
        
        if not record_id:
            raise ValueError("record_id cannot be empty")
        
        if not isinstance(updates, dict):
            raise TypeError("updates must be a dictionary")
        
        if not updates:
            raise ValueError("updates cannot be empty")
        
        try:
            records = self.read()
            
            found = False
            for i, record in enumerate(records):
                if not isinstance(record, dict):
                    continue
                
                if record.get('id') == record_id:
                    records[i].update(updates)
                    found = True
                    break
            
            if found:
                self.write(records)
                return True
            
            return False
        except (IOError, json.JSONDecodeError, ValueError) as e:
            raise IOError(f"Failed to update record: {str(e)}")
    
    def add(self, record: Dict[str, Any]) -> None:
        """
        Add a new record to the file.
        
        Args:
            record: Dictionary representing the record to add
            
        Raises:
            TypeError: If record is not a dictionary
            ValueError: If record is empty
            IOError: If file operations fail
        """
        if not isinstance(record, dict):
            raise TypeError("record must be a dictionary")
        
        if not record:
            raise ValueError("record cannot be empty")
        
        try:
            records = self.read()
            records.append(record)
            self.write(records)
        except (IOError, json.JSONDecodeError, ValueError) as e:
            raise IOError(f"Failed to add record: {str(e)}")
    
    def delete(self, record_id: str) -> bool:
        """
        Delete a record from the file by ID.
        
        Args:
            record_id: ID of the record to delete
            
        Returns:
            True if record was deleted, False if not found
            
        Raises:
            TypeError: If record_id is not a string
            ValueError: If record_id is empty
            IOError: If file operations fail
        """
        if not isinstance(record_id, str):
            raise TypeError("record_id must be a string")
        
        if not record_id:
            raise ValueError("record_id cannot be empty")
        
        try:
            records = self.read()
            
            original_length = len(records)
            records = [r for r in records if isinstance(r, dict) and r.get('id') != record_id]
            
            if len(records) < original_length:
                self.write(records)
                return True
            
            return False
        except (IOError, json.JSONDecodeError, ValueError) as e:
            raise IOError(f"Failed to delete record: {str(e)}")
    
    def get(self, record_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a record by ID.
        
        Args:
            record_id: ID of the record to retrieve
            
        Returns:
            Dictionary representing the record, or None if not found
            
        Raises:
            TypeError: If record_id is not a string
            ValueError: If record_id is empty
            IOError: If file operations fail
        """
        if not isinstance(record_id, str):
            raise TypeError("record_id must be a string")
        
        if not record_id:
            raise ValueError("record_id cannot be empty")
        
        try:
            records = self.read()
            
            for record in records:
                if isinstance(record, dict) and record.get('id') == record_id:
                    return record
            
            return None
        except (IOError, json.JSONDecodeError, ValueError) as e:
            raise IOError(f"Failed to get record: {str(e)}")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Example usage demonstrating all requirements
    
    storage = FileStorage("example_data.json")
    
    # Write initial data
    initial_data = [
        {"id": "1", "name": "Alice", "age": 30},
        {"id": "2", "name": "Bob", "age": 25}
    ]
    storage.write(initial_data)
    print("✓ Data written successfully")
    
    # Read data
    records = storage.read()
    print(f"✓ Read {len(records)} records")
    
    # Update record (REQUIREMENT 3: Now properly writes back to file)
    success = storage.update("1", {"age": 31, "status": "active"})
    if success:
        print("✓ Record updated and saved to file")
        updated = storage.get("1")
        print(f"  Updated record: {updated}")
    
    # Cleanup
    if os.path.exists("example_data.json"):
        os.remove("example_data.json")
        print("✓ Cleanup completed")


