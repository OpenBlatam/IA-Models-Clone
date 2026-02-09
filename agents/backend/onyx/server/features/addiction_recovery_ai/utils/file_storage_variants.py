"""
Variants and Extensions of FileStorage
Different implementations for various use cases
"""

import json
import os
import gzip
import pickle
import threading
from typing import Dict, List, Any, Optional
from pathlib import Path
from utils.file_storage import FileStorage


class ThreadSafeFileStorage(FileStorage):
    """
    Thread-safe version of FileStorage
    Uses locks to prevent race conditions
    """
    
    def __init__(self, file_path: str):
        super().__init__(file_path)
        self._lock = threading.RLock()
    
    def read(self) -> List[Dict[str, Any]]:
        with self._lock:
            return super().read()
    
    def write(self, data: List[Dict[str, Any]]) -> None:
        with self._lock:
            super().write(data)
    
    def update(self, record_id: str, updates: Dict[str, Any]) -> bool:
        with self._lock:
            return super().update(record_id, updates)
    
    def add(self, record: Dict[str, Any]) -> None:
        with self._lock:
            super().add(record)
    
    def delete(self, record_id: str) -> bool:
        with self._lock:
            return super().delete(record_id)
    
    def get(self, record_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return super().get(record_id)


class CompressedFileStorage(FileStorage):
    """
    FileStorage with gzip compression
    Useful for large files
    """
    
    def read(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.file_path):
            return []
        
        try:
            with gzip.open(self.file_path, 'rt', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                raise ValueError("File does not contain a valid list")
            
            return data
        except FileNotFoundError:
            return []
        except (json.JSONDecodeError, OSError) as e:
            raise IOError(f"Failed to read compressed file: {str(e)}")
    
    def write(self, data: List[Dict[str, Any]]) -> None:
        if not isinstance(data, list):
            raise TypeError("data must be a list")
        
        if not all(isinstance(item, dict) for item in data):
            raise ValueError("All items in data must be dictionaries")
        
        try:
            with gzip.open(self.file_path, 'wt', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except OSError as e:
            raise IOError(f"Failed to write compressed file: {str(e)}")


class CachedFileStorage(FileStorage):
    """
    FileStorage with in-memory cache
    Reduces file I/O for frequent reads
    """
    
    def __init__(self, file_path: str, cache_ttl: Optional[float] = None):
        super().__init__(file_path)
        self._cache: Optional[List[Dict[str, Any]]] = None
        self._cache_time: Optional[float] = None
        self._cache_ttl = cache_ttl
        self._dirty = False
    
    def read(self, use_cache: bool = True) -> List[Dict[str, Any]]:
        import time
        
        if use_cache and self._cache is not None:
            if self._cache_ttl is None:
                return self._cache
            
            if self._cache_time and (time.time() - self._cache_time) < self._cache_ttl:
                return self._cache
        
        data = super().read()
        
        if use_cache:
            self._cache = data
            self._cache_time = time.time()
        
        return data
    
    def write(self, data: List[Dict[str, Any]]) -> None:
        super().write(data)
        self._cache = data
        import time
        self._cache_time = time.time()
        self._dirty = False
    
    def invalidate_cache(self):
        """Invalidate the cache"""
        self._cache = None
        self._cache_time = None
    
    def update(self, record_id: str, updates: Dict[str, Any]) -> bool:
        result = super().update(record_id, updates)
        if result:
            self.invalidate_cache()
        return result
    
    def add(self, record: Dict[str, Any]) -> None:
        super().add(record)
        self.invalidate_cache()
    
    def delete(self, record_id: str) -> bool:
        result = super().delete(record_id)
        if result:
            self.invalidate_cache()
        return result


class IndexedFileStorage(FileStorage):
    """
    FileStorage with index for fast lookups
    Maintains an index of record IDs to positions
    """
    
    def __init__(self, file_path: str):
        super().__init__(file_path)
        self._index: Dict[str, int] = {}
        self._rebuild_index()
    
    def _rebuild_index(self):
        """Rebuild the index from current data"""
        records = super().read()
        self._index = {
            r.get('id'): i 
            for i, r in enumerate(records) 
            if isinstance(r, dict) and 'id' in r
        }
    
    def read(self) -> List[Dict[str, Any]]:
        records = super().read()
        self._rebuild_index()
        return records
    
    def write(self, data: List[Dict[str, Any]]) -> None:
        super().write(data)
        self._rebuild_index()
    
    def get(self, record_id: str) -> Optional[Dict[str, Any]]:
        if record_id in self._index:
            records = super().read()
            idx = self._index[record_id]
            if 0 <= idx < len(records):
                return records[idx]
        return None
    
    def update(self, record_id: str, updates: Dict[str, Any]) -> bool:
        if record_id not in self._index:
            return False
        
        result = super().update(record_id, updates)
        if result:
            self._rebuild_index()
        return result
    
    def delete(self, record_id: str) -> bool:
        if record_id not in self._index:
            return False
        
        result = super().delete(record_id)
        if result:
            self._rebuild_index()
        return result


class LoggedFileStorage(FileStorage):
    """
    FileStorage with operation logging
    Useful for debugging and auditing
    """
    
    def __init__(self, file_path: str, logger=None):
        super().__init__(file_path)
        if logger is None:
            import logging
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger
    
    def write(self, data: List[Dict[str, Any]]) -> None:
        self.logger.info(f"Writing {len(data)} records to {self.file_path}")
        try:
            super().write(data)
            self.logger.info("Write successful")
        except Exception as e:
            self.logger.error(f"Write failed: {e}")
            raise
    
    def read(self) -> List[Dict[str, Any]]:
        self.logger.debug(f"Reading from {self.file_path}")
        try:
            data = super().read()
            self.logger.debug(f"Read {len(data)} records")
            return data
        except Exception as e:
            self.logger.error(f"Read failed: {e}")
            raise
    
    def update(self, record_id: str, updates: Dict[str, Any]) -> bool:
        self.logger.debug(f"Updating record {record_id}")
        try:
            result = super().update(record_id, updates)
            if result:
                self.logger.info(f"Record {record_id} updated successfully")
            else:
                self.logger.warning(f"Record {record_id} not found")
            return result
        except Exception as e:
            self.logger.error(f"Update failed: {e}")
            raise


class ValidatedFileStorage(FileStorage):
    """
    FileStorage with schema validation
    Validates records against a schema before writing
    """
    
    def __init__(self, file_path: str, schema: Optional[Dict[str, Any]] = None):
        super().__init__(file_path)
        self.schema = schema
    
    def _validate_record(self, record: Dict[str, Any]) -> bool:
        """Validate a record against the schema"""
        if self.schema is None:
            return True
        
        required_fields = self.schema.get('required', [])
        for field in required_fields:
            if field not in record:
                return False
        
        field_types = self.schema.get('properties', {})
        for field, expected_type in field_types.items():
            if field in record:
                if expected_type == 'string' and not isinstance(record[field], str):
                    return False
                elif expected_type == 'integer' and not isinstance(record[field], int):
                    return False
                elif expected_type == 'number' and not isinstance(record[field], (int, float)):
                    return False
        
        return True
    
    def write(self, data: List[Dict[str, Any]]) -> None:
        if not isinstance(data, list):
            raise TypeError("data must be a list")
        
        for record in data:
            if not isinstance(record, dict):
                raise ValueError("All items must be dictionaries")
            
            if not self._validate_record(record):
                raise ValueError(f"Record does not match schema: {record}")
        
        super().write(data)
    
    def add(self, record: Dict[str, Any]) -> None:
        if not self._validate_record(record):
            raise ValueError(f"Record does not match schema: {record}")
        super().add(record)


class BackupFileStorage(FileStorage):
    """
    FileStorage with automatic backup
    Creates backups before write operations
    """
    
    def __init__(self, file_path: str, backup_dir: str = "backups", max_backups: int = 5):
        super().__init__(file_path)
        self.backup_dir = backup_dir
        self.max_backups = max_backups
        os.makedirs(backup_dir, exist_ok=True)
    
    def _create_backup(self):
        """Create a backup of current file"""
        import shutil
        from datetime import datetime
        
        if not os.path.exists(self.file_path):
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(
            self.backup_dir,
            f"{Path(self.file_path).stem}_{timestamp}.json"
        )
        
        shutil.copy2(self.file_path, backup_path)
        
        self._cleanup_old_backups()
    
    def _cleanup_old_backups(self):
        """Remove old backups if exceeding max_backups"""
        backups = sorted(
            [
                os.path.join(self.backup_dir, f)
                for f in os.listdir(self.backup_dir)
                if f.startswith(Path(self.file_path).stem)
            ],
            key=os.path.getmtime,
            reverse=True
        )
        
        for backup in backups[self.max_backups:]:
            try:
                os.remove(backup)
            except Exception:
                pass
    
    def write(self, data: List[Dict[str, Any]]) -> None:
        self._create_backup()
        super().write(data)
    
    def update(self, record_id: str, updates: Dict[str, Any]) -> bool:
        self._create_backup()
        return super().update(record_id, updates)
    
    def delete(self, record_id: str) -> bool:
        self._create_backup()
        return super().delete(record_id)


