import json
from pathlib import Path
from typing import Dict, Any, List


class RecordStorage:
    def __init__(self, file_path: str):
        if not file_path or not isinstance(file_path, str):
            raise ValueError("file_path must be a non-empty string")
        
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.file_path.exists():
            self._initialize_file()
    
    def _initialize_file(self) -> None:
        try:
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump({"records": []}, f, indent=2, ensure_ascii=False)
        except (IOError, OSError) as e:
            raise RuntimeError(f"Cannot initialize storage file: {e}") from e
    
    def read(self) -> List[Dict[str, Any]]:
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


