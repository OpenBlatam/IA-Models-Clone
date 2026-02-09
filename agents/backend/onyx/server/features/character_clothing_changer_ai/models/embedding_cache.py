"""
Embedding Cache
===============

Caching system for character and clothing embeddings to improve performance.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union
import torch
from safetensors.torch import save_file, load_file

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Cache for embeddings to avoid recomputation."""
    
    def __init__(self, cache_dir: Union[str, Path] = "./embedding_cache", max_size: int = 1000):
        """
        Initialize embedding cache.
        
        Args:
            cache_dir: Directory to store cache
            max_size: Maximum number of cached items
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size
        self.cache_index_file = self.cache_dir / "index.json"
        self.cache_index = self._load_index()
    
    def _load_index(self) -> Dict[str, Any]:
        """Load cache index."""
        if self.cache_index_file.exists():
            try:
                with open(self.cache_index_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading cache index: {e}")
        return {}
    
    def _save_index(self) -> None:
        """Save cache index."""
        try:
            with open(self.cache_index_file, "w") as f:
                json.dump(self.cache_index, f, indent=2)
        except Exception as e:
            logger.warning(f"Error saving cache index: {e}")
    
    def _get_key(self, content: Union[str, bytes, torch.Tensor]) -> str:
        """Generate cache key from content."""
        if isinstance(content, str):
            return hashlib.md5(content.encode()).hexdigest()
        elif isinstance(content, bytes):
            return hashlib.md5(content).hexdigest()
        elif isinstance(content, torch.Tensor):
            # Use tensor hash (shape + dtype + first few values)
            tensor_str = f"{content.shape}_{content.dtype}_{content.flatten()[:10].tolist()}"
            return hashlib.md5(tensor_str.encode()).hexdigest()
        else:
            return hashlib.md5(str(content).encode()).hexdigest()
    
    def get_character_embedding(self, image_path: Union[str, Path]) -> Optional[torch.Tensor]:
        """
        Get cached character embedding.
        
        Args:
            image_path: Path to image
            
        Returns:
            Cached embedding or None
        """
        # Use file hash as key
        try:
            with open(image_path, "rb") as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
        except Exception:
            return None
        
        cache_key = f"character_{file_hash}"
        
        if cache_key in self.cache_index:
            cache_file = self.cache_dir / f"{cache_key}.safetensors"
            if cache_file.exists():
                try:
                    data = load_file(str(cache_file))
                    logger.debug(f"Cache hit for character embedding: {cache_key}")
                    return data["embedding"]
                except Exception as e:
                    logger.warning(f"Error loading cached embedding: {e}")
        
        return None
    
    def save_character_embedding(
        self,
        image_path: Union[str, Path],
        embedding: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save character embedding to cache.
        
        Args:
            image_path: Path to image
            embedding: Embedding tensor
            metadata: Optional metadata
        """
        try:
            with open(image_path, "rb") as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
        except Exception:
            return
        
        cache_key = f"character_{file_hash}"
        cache_file = self.cache_dir / f"{cache_key}.safetensors"
        
        try:
            save_file({"embedding": embedding.cpu()}, str(cache_file))
            
            self.cache_index[cache_key] = {
                "type": "character",
                "file": str(cache_file),
                "metadata": metadata or {},
            }
            
            self._save_index()
            logger.debug(f"Cached character embedding: {cache_key}")
        except Exception as e:
            logger.warning(f"Error saving cached embedding: {e}")
    
    def get_clothing_embedding(self, description: str) -> Optional[torch.Tensor]:
        """
        Get cached clothing embedding.
        
        Args:
            description: Clothing description
            
        Returns:
            Cached embedding or None
        """
        cache_key = f"clothing_{self._get_key(description)}"
        
        if cache_key in self.cache_index:
            cache_file = self.cache_dir / f"{cache_key}.safetensors"
            if cache_file.exists():
                try:
                    data = load_file(str(cache_file))
                    logger.debug(f"Cache hit for clothing embedding: {cache_key}")
                    return data["embedding"]
                except Exception as e:
                    logger.warning(f"Error loading cached embedding: {e}")
        
        return None
    
    def save_clothing_embedding(
        self,
        description: str,
        embedding: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save clothing embedding to cache.
        
        Args:
            description: Clothing description
            embedding: Embedding tensor
            metadata: Optional metadata
        """
        cache_key = f"clothing_{self._get_key(description)}"
        cache_file = self.cache_dir / f"{cache_key}.safetensors"
        
        try:
            save_file({"embedding": embedding.cpu()}, str(cache_file))
            
            self.cache_index[cache_key] = {
                "type": "clothing",
                "file": str(cache_file),
                "description": description,
                "metadata": metadata or {},
            }
            
            self._save_index()
            logger.debug(f"Cached clothing embedding: {cache_key}")
        except Exception as e:
            logger.warning(f"Error saving cached embedding: {e}")
    
    def clear_cache(self) -> None:
        """Clear all cached embeddings."""
        try:
            for cache_file in self.cache_dir.glob("*.safetensors"):
                cache_file.unlink()
            
            self.cache_index = {}
            self._save_index()
            logger.info("Cache cleared")
        except Exception as e:
            logger.warning(f"Error clearing cache: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        character_count = sum(1 for k in self.cache_index.keys() if k.startswith("character_"))
        clothing_count = sum(1 for k in self.cache_index.keys() if k.startswith("clothing_"))
        
        total_size = sum(
            f.stat().st_size
            for f in self.cache_dir.glob("*.safetensors")
        )
        
        return {
            "total_items": len(self.cache_index),
            "character_embeddings": character_count,
            "clothing_embeddings": clothing_count,
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "cache_dir": str(self.cache_dir),
        }


