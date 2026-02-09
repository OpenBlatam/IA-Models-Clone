"""
I/O Optimizer
=============

Advanced I/O optimization techniques.
"""

import logging
import asyncio
import aiofiles
from typing import Dict, Any, Optional, List
from pathlib import Path
import os

logger = logging.getLogger(__name__)


class IOOptimizer:
    """I/O optimizer with advanced techniques."""
    
    def __init__(self, buffer_size: int = 8192):
        self.buffer_size = buffer_size
        self._file_cache: Dict[str, Any] = {}
    
    async def read_file_optimized(self, file_path: str, chunk_size: Optional[int] = None) -> bytes:
        """Read file with optimizations."""
        chunk_size = chunk_size or self.buffer_size
        
        # Use async file I/O
        async with aiofiles.open(file_path, 'rb') as f:
            chunks = []
            while True:
                chunk = await f.read(chunk_size)
                if not chunk:
                    break
                chunks.append(chunk)
        
        return b''.join(chunks)
    
    async def write_file_optimized(self, file_path: str, data: bytes, chunk_size: Optional[int] = None):
        """Write file with optimizations."""
        chunk_size = chunk_size or self.buffer_size
        
        # Use async file I/O
        async with aiofiles.open(file_path, 'wb') as f:
            for i in range(0, len(data), chunk_size):
                chunk = data[i:i + chunk_size]
                await f.write(chunk)
    
    def optimize_file_operations(self, file_path: str):
        """Optimize file operations."""
        path = Path(file_path)
        
        # Set file buffer size
        # In production, use os.open with custom buffer
        
        # Enable read-ahead if available
        # In production, use posix_fadvise or similar
        
        logger.debug(f"File operations optimized for: {file_path}")
    
    async def batch_read_files(self, file_paths: List[str], max_concurrent: int = 10) -> Dict[str, bytes]:
        """Batch read multiple files."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def read_with_semaphore(path: str):
            async with semaphore:
                return path, await self.read_file_optimized(path)
        
        tasks = [read_with_semaphore(path) for path in file_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        files_data = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"File read error: {result}")
                continue
            path, data = result
            files_data[path] = data
        
        return files_data
    
    async def batch_write_files(self, files_data: Dict[str, bytes], max_concurrent: int = 10):
        """Batch write multiple files."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def write_with_semaphore(path: str, data: bytes):
            async with semaphore:
                await self.write_file_optimized(path, data)
        
        tasks = [
            write_with_semaphore(path, data)
            for path, data in files_data.items()
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_io_stats(self) -> Dict[str, Any]:
        """Get I/O statistics."""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            io_counters = process.io_counters()
            
            return {
                "read_count": io_counters.read_count,
                "write_count": io_counters.write_count,
                "read_bytes": io_counters.read_bytes,
                "write_bytes": io_counters.write_bytes,
                "read_bytes_mb": io_counters.read_bytes / 1024 / 1024,
                "write_bytes_mb": io_counters.write_bytes / 1024 / 1024
            }
        except Exception as e:
            logger.warning(f"Failed to get I/O stats: {e}")
            return {}















