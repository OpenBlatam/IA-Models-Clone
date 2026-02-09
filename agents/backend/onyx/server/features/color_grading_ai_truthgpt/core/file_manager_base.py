"""
File Manager Base for Color Grading AI
=======================================

Base class for services that manage files (JSON, templates, presets, etc.).
"""

import logging
import json
from typing import Dict, Any, List, Optional, TypeVar, Generic, Callable
from pathlib import Path
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

T = TypeVar('T')


class FileManagerBase(Generic[T], ABC):
    """
    Base class for file-based managers.
    
    Provides:
    - JSON file loading/saving
    - Search and filtering
    - CRUD operations
    - File management
    """
    
    def __init__(self, storage_dir: str, file_extension: str = ".json"):
        """
        Initialize file manager.
        
        Args:
            storage_dir: Storage directory
            file_extension: File extension (default: .json)
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.file_extension = file_extension
        self._items: Dict[str, T] = {}
        self._loaded = False
    
    def load_all(self) -> Dict[str, T]:
        """
        Load all items from storage.
        
        Returns:
            Dictionary of items
        """
        if self._loaded:
            return self._items
        
        self._items = {}
        
        for file_path in self.storage_dir.glob(f"*{self.file_extension}"):
            try:
                item = self._load_item(file_path)
                if item:
                    item_id = self._get_item_id(item)
                    self._items[item_id] = item
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        self._loaded = True
        logger.info(f"Loaded {len(self._items)} items from {self.storage_dir}")
        return self._items
    
    def _load_item(self, file_path: Path) -> Optional[T]:
        """Load single item from file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return self._deserialize_item(data)
        except Exception as e:
            logger.error(f"Error loading item from {file_path}: {e}")
            return None
    
    def save_item(self, item: T) -> bool:
        """
        Save item to storage.
        
        Args:
            item: Item to save
            
        Returns:
            True if successful
        """
        try:
            item_id = self._get_item_id(item)
            file_path = self.storage_dir / f"{item_id}{self.file_extension}"
            
            data = self._serialize_item(item)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            self._items[item_id] = item
            logger.info(f"Saved item {item_id} to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving item: {e}")
            return False
    
    def delete_item(self, item_id: str) -> bool:
        """
        Delete item from storage.
        
        Args:
            item_id: Item ID
            
        Returns:
            True if successful
        """
        try:
            file_path = self.storage_dir / f"{item_id}{self.file_extension}"
            if file_path.exists():
                file_path.unlink()
            
            if item_id in self._items:
                del self._items[item_id]
            
            logger.info(f"Deleted item {item_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting item {item_id}: {e}")
            return False
    
    def get_item(self, item_id: str) -> Optional[T]:
        """
        Get item by ID.
        
        Args:
            item_id: Item ID
            
        Returns:
            Item or None
        """
        if not self._loaded:
            self.load_all()
        
        return self._items.get(item_id)
    
    def list_items(self, filter_func: Optional[Callable[[T], bool]] = None) -> List[T]:
        """
        List all items, optionally filtered.
        
        Args:
            filter_func: Optional filter function
            
        Returns:
            List of items
        """
        if not self._loaded:
            self.load_all()
        
        items = list(self._items.values())
        
        if filter_func:
            items = [item for item in items if filter_func(item)]
        
        return items
    
    def search_items(self, query: str, search_fields: List[str]) -> List[T]:
        """
        Search items by query.
        
        Args:
            query: Search query
            search_fields: Fields to search in
            
        Returns:
            List of matching items
        """
        if not self._loaded:
            self.load_all()
        
        query_lower = query.lower()
        results = []
        
        for item in self._items.values():
            item_dict = self._item_to_dict(item)
            for field in search_fields:
                value = item_dict.get(field, "")
                if isinstance(value, str) and query_lower in value.lower():
                    results.append(item)
                    break
                elif isinstance(value, list):
                    if any(query_lower in str(v).lower() for v in value):
                        results.append(item)
                        break
        
        return results
    
    @abstractmethod
    def _get_item_id(self, item: T) -> str:
        """Get item ID."""
        pass
    
    @abstractmethod
    def _serialize_item(self, item: T) -> Dict[str, Any]:
        """Serialize item to dictionary."""
        pass
    
    @abstractmethod
    def _deserialize_item(self, data: Dict[str, Any]) -> T:
        """Deserialize item from dictionary."""
        pass
    
    def _item_to_dict(self, item: T) -> Dict[str, Any]:
        """Convert item to dictionary for searching."""
        return self._serialize_item(item)

