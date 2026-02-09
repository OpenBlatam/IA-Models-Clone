"""
Storage Optimizations

Optimizations for:
- File storage
- Compression
- Deduplication
- Caching
- Streaming
"""

import logging
import os
import gzip
import hashlib
from typing import Optional, Dict, Any, BinaryIO
from pathlib import Path
import shutil

logger = logging.getLogger(__name__)


class StorageOptimizer:
    """File storage optimizations."""
    
    @staticmethod
    def compress_file(
        input_path: str,
        output_path: Optional[str] = None,
        compression_level: int = 6
    ) -> str:
        """
        Compress file efficiently.
        
        Args:
            input_path: Input file path
            output_path: Output file path (optional)
            compression_level: Compression level (1-9)
            
        Returns:
            Path to compressed file
        """
        if output_path is None:
            output_path = f"{input_path}.gz"
        
        with open(input_path, 'rb') as f_in:
            with gzip.open(output_path, 'wb', compresslevel=compression_level) as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        logger.info(f"Compressed {input_path} to {output_path}")
        return output_path
    
    @staticmethod
    def decompress_file(compressed_path: str, output_path: Optional[str] = None) -> str:
        """
        Decompress file.
        
        Args:
            compressed_path: Compressed file path
            output_path: Output file path (optional)
            
        Returns:
            Path to decompressed file
        """
        if output_path is None:
            output_path = compressed_path[:-3] if compressed_path.endswith('.gz') else f"{compressed_path}.decompressed"
        
        with gzip.open(compressed_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        return output_path
    
    @staticmethod
    def calculate_file_hash(file_path: str) -> str:
        """
        Calculate file hash for deduplication.
        
        Args:
            file_path: File path
            
        Returns:
            File hash (MD5)
        """
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    @staticmethod
    def deduplicate_files(
        file_paths: List[str],
        storage_dir: str
    ) -> Dict[str, str]:
        """
        Deduplicate files by hash.
        
        Args:
            file_paths: List of file paths
            storage_dir: Storage directory
            
        Returns:
            Dictionary mapping original paths to deduplicated paths
        """
        storage_path = Path(storage_dir)
        storage_path.mkdir(parents=True, exist_ok=True)
        
        hash_map: Dict[str, str] = {}
        result_map: Dict[str, str] = {}
        
        for file_path in file_paths:
            file_hash = StorageOptimizer.calculate_file_hash(file_path)
            
            if file_hash in hash_map:
                # File already exists, use existing
                result_map[file_path] = hash_map[file_hash]
            else:
                # New file, store it
                stored_path = storage_path / f"{file_hash}"
                shutil.copy2(file_path, stored_path)
                hash_map[file_hash] = str(stored_path)
                result_map[file_path] = str(stored_path)
        
        return result_map


class StreamingStorage:
    """Streaming file operations."""
    
    @staticmethod
    def stream_file(
        file_path: str,
        chunk_size: int = 8192
    ):
        """
        Stream file in chunks.
        
        Args:
            file_path: File path
            chunk_size: Chunk size in bytes
            
        Yields:
            File chunks
        """
        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk
    
    @staticmethod
    def stream_compress(
        input_path: str,
        output_path: str,
        chunk_size: int = 8192
    ) -> None:
        """
        Stream compress file.
        
        Args:
            input_path: Input file path
            output_path: Output file path
            chunk_size: Chunk size
        """
        with open(input_path, 'rb') as f_in:
            with gzip.open(output_path, 'wb') as f_out:
                while True:
                    chunk = f_in.read(chunk_size)
                    if not chunk:
                        break
                    f_out.write(chunk)


class StorageCache:
    """Storage metadata cache."""
    
    def __init__(self, cache_file: str = ".storage_cache.json"):
        """
        Initialize storage cache.
        
        Args:
            cache_file: Cache file path
        """
        self.cache_file = cache_file
        self.cache: Dict[str, Dict[str, Any]] = self._load_cache()
    
    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        """Load cache from file."""
        if os.path.exists(self.cache_file):
            try:
                import json
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def _save_cache(self) -> None:
        """Save cache to file."""
        try:
            import json
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def get(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get file metadata from cache."""
        return self.cache.get(file_path)
    
    def set(self, file_path: str, metadata: Dict[str, Any]) -> None:
        """Set file metadata in cache."""
        self.cache[file_path] = metadata
        self._save_cache()
    
    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()
        self._save_cache()








