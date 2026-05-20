"""
Storage Base
============

Base storage pattern for all storage types.
"""

import logging
from typing import Dict, Any, Optional, List, TypeVar, Generic
from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

T = TypeVar('T')


class BaseStorage(ABC, Generic[T]):
    """Base storage interface."""
    
    def __init__(self, name: str = "Storage"):
        """
        Initialize storage.
        
        Args:
            name: Storage name
        """
        self.name = name
    
    @abstractmethod
    async def save(self, key: str, value: T) -> bool:
        """
        Save value.
        
        Args:
            key: Storage key
            value: Value to save
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    async def load(self, key: str) -> Optional[T]:
        """
        Load value.
        
        Args:
            key: Storage key
            
        Returns:
            Value or None
        """
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """
        Delete value.
        
        Args:
            key: Storage key
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """
        Check if key exists.
        
        Args:
            key: Storage key
            
        Returns:
            True if exists
        """
        pass
    
    @abstractmethod
    async def list_keys(self, prefix: Optional[str] = None) -> List[str]:
        """
        List all keys.
        
        Args:
            prefix: Optional key prefix
            
        Returns:
            List of keys
        """
        pass
    
    @abstractmethod
    async def clear(self, prefix: Optional[str] = None) -> int:
        """
        Clear storage.
        
        Args:
            prefix: Optional key prefix
            
        Returns:
            Number of items cleared
        """
        pass
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        keys = await self.list_keys()
        return {
            "name": self.name,
            "total_items": len(keys)
        }


class FileStorage(BaseStorage[Any]):
    """File-based storage implementation."""
    
    def __init__(self, base_path: Path, name: str = "FileStorage"):
        """
        Initialize file storage.
        
        Args:
            base_path: Base directory path
            name: Storage name
        """
        super().__init__(name)
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def _get_path(self, key: str) -> Path:
        """Get file path for key."""
        # Sanitize key for filesystem
        safe_key = key.replace("/", "_").replace("\\", "_")
        return self.base_path / safe_key
    
    async def save(self, key: str, value: Any) -> bool:
        """Save value to file."""
        import json
        
        try:
            path = self._get_path(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w', encoding='utf-8') as f:
                json.dump({
                    "key": key,
                    "value": value,
                    "timestamp": datetime.now().isoformat()
                }, f, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"Error saving {key}: {e}")
            return False
    
    async def load(self, key: str) -> Optional[Any]:
        """Load value from file."""
        import json
        
        try:
            path = self._get_path(key)
            
            if not path.exists():
                return None
            
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("value")
        except Exception as e:
            logger.error(f"Error loading {key}: {e}")
            return None
    
    async def delete(self, key: str) -> bool:
        """Delete value from file."""
        try:
            path = self._get_path(key)
            
            if path.exists():
                path.unlink()
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        path = self._get_path(key)
        return path.exists()
    
    async def list_keys(self, prefix: Optional[str] = None) -> List[str]:
        """List all keys."""
        keys = []
        
        for path in self.base_path.iterdir():
            if path.is_file():
                key = path.stem
                if prefix is None or key.startswith(prefix):
                    keys.append(key)
        
        return keys
    
    async def clear(self, prefix: Optional[str] = None) -> int:
        """Clear storage."""
        count = 0
        
        for path in self.base_path.iterdir():
            if path.is_file():
                key = path.stem
                if prefix is None or key.startswith(prefix):
                    try:
                        path.unlink()
                        count += 1
                    except Exception as e:
                        logger.error(f"Error deleting {path}: {e}")
        
        return count




