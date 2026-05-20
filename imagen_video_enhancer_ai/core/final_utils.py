"""
Final Utilities
===============

Final utility functions and helpers.
"""

import asyncio
import logging
import hashlib
import uuid
from typing import Dict, Any, Optional, List, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class UtilityHelper:
    """Utility helper class."""
    
    @staticmethod
    def generate_id(prefix: str = "") -> str:
        """
        Generate unique ID.
        
        Args:
            prefix: Optional prefix
            
        Returns:
            Unique ID string
        """
        unique_id = str(uuid.uuid4())
        return f"{prefix}_{unique_id}" if prefix else unique_id
    
    @staticmethod
    def hash_data(data: str, algorithm: str = "sha256") -> str:
        """
        Hash data.
        
        Args:
            data: Data to hash
            algorithm: Hash algorithm
            
        Returns:
            Hash string
        """
        hash_func = getattr(hashlib, algorithm, hashlib.sha256)
        return hash_func(data.encode('utf-8')).hexdigest()
    
    @staticmethod
    def format_duration(seconds: float) -> str:
        """
        Format duration in human-readable format.
        
        Args:
            seconds: Duration in seconds
            
        Returns:
            Formatted duration string
        """
        if seconds < 60:
            return f"{seconds:.2f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.2f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.2f}h"
    
    @staticmethod
    def format_size(bytes_size: int) -> str:
        """
        Format size in human-readable format.
        
        Args:
            bytes_size: Size in bytes
            
        Returns:
            Formatted size string
        """
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_size < 1024.0:
                return f"{bytes_size:.2f} {unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.2f} PB"
    
    @staticmethod
    def safe_get(data: Dict[str, Any], key: str, default: Any = None) -> Any:
        """
        Safely get value from dictionary with dot notation.
        
        Args:
            data: Dictionary
            key: Key (supports dot notation)
            default: Default value
            
        Returns:
            Value or default
        """
        keys = key.split('.')
        value = data
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value if value is not None else default
    
    @staticmethod
    def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
        """
        Chunk list into smaller lists.
        
        Args:
            items: List of items
            chunk_size: Chunk size
            
        Returns:
            List of chunks
        """
        return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
    
    @staticmethod
    def flatten_dict(data: Dict[str, Any], separator: str = ".") -> Dict[str, Any]:
        """
        Flatten nested dictionary.
        
        Args:
            data: Nested dictionary
            separator: Key separator
            
        Returns:
            Flattened dictionary
        """
        def _flatten(obj: Any, parent_key: str = "") -> Dict[str, Any]:
            items = []
            
            if isinstance(obj, dict):
                for k, v in obj.items():
                    new_key = f"{parent_key}{separator}{k}" if parent_key else k
                    items.extend(_flatten(v, new_key).items())
            else:
                items.append((parent_key, obj))
            
            return dict(items)
        
        return _flatten(data)
    
    @staticmethod
    def unflatten_dict(data: Dict[str, Any], separator: str = ".") -> Dict[str, Any]:
        """
        Unflatten dictionary.
        
        Args:
            data: Flattened dictionary
            separator: Key separator
            
        Returns:
            Nested dictionary
        """
        result = {}
        
        for key, value in data.items():
            keys = key.split(separator)
            current = result
            
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            
            current[keys[-1]] = value
        
        return result


class AsyncHelper:
    """Async utility helper."""
    
    @staticmethod
    async def delay(seconds: float):
        """
        Delay execution.
        
        Args:
            seconds: Delay in seconds
        """
        await asyncio.sleep(seconds)
    
    @staticmethod
    async def timeout(
        coro: Awaitable[Any],
        timeout: float,
        default: Any = None
    ) -> Any:
        """
        Execute coroutine with timeout.
        
        Args:
            coro: Coroutine to execute
            timeout: Timeout in seconds
            default: Default value on timeout
            
        Returns:
            Coroutine result or default
        """
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            return default
    
    @staticmethod
    async def retry(
        func: Callable[[], Awaitable[Any]],
        max_attempts: int = 3,
        delay: float = 1.0,
        exponential: bool = True
    ) -> Any:
        """
        Retry async function.
        
        Args:
            func: Async function to retry
            max_attempts: Maximum attempts
            delay: Initial delay
            exponential: Whether to use exponential backoff
            
        Returns:
            Function result
        """
        last_error = None
        
        for attempt in range(max_attempts):
            try:
                return await func()
            except Exception as e:
                last_error = e
                if attempt < max_attempts - 1:
                    wait_time = delay * (2 ** attempt) if exponential else delay
                    await asyncio.sleep(wait_time)
        
        if last_error:
            raise last_error
        raise Exception("All retry attempts failed")


class FileHelper:
    """File utility helper."""
    
    @staticmethod
    def ensure_directory(path: Path):
        """
        Ensure directory exists.
        
        Args:
            path: Directory path
        """
        path.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def read_json(file_path: Path) -> Dict[str, Any]:
        """
        Read JSON file.
        
        Args:
            file_path: File path
            
        Returns:
            JSON data
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def write_json(data: Dict[str, Any], file_path: Path):
        """
        Write JSON file.
        
        Args:
            data: Data to write
            file_path: File path
        """
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
    
    @staticmethod
    def get_file_size(file_path: Path) -> int:
        """
        Get file size.
        
        Args:
            file_path: File path
            
        Returns:
            File size in bytes
        """
        return file_path.stat().st_size if file_path.exists() else 0




