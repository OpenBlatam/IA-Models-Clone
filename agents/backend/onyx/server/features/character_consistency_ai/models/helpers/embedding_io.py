"""
Embedding I/O Utilities
========================

Utilities for saving and loading character embeddings.
"""

import torch
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import json
import logging
from safetensors.torch import save_file, load_file

logger = logging.getLogger(__name__)


class EmbeddingIO:
    """Handles saving and loading of character embeddings."""
    
    @staticmethod
    def save_embedding(
        embedding: torch.Tensor,
        output_path: Path,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save character embedding as safe tensor.
        
        Args:
            embedding: Character embedding tensor
            output_path: Path to save safe tensor
            metadata: Optional metadata to include
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save embedding as safe tensor
        save_file({"character_embedding": embedding.cpu()}, str(output_path))
        
        # Save metadata if provided
        if metadata:
            metadata_path = output_path.with_suffix(".json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            logger.debug(f"Metadata saved to {metadata_path}")
        
        logger.info(f"Character embedding saved to {output_path}")
    
    @staticmethod
    def load_embedding(
        embedding_path: Path,
        device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Load character embedding from safe tensor.
        
        Args:
            embedding_path: Path to safe tensor file
            device: Device to load tensor on
            
        Returns:
            Tuple of (embedding tensor, metadata dict)
        """
        embedding_path = Path(embedding_path)
        
        if not embedding_path.exists():
            raise FileNotFoundError(f"Embedding file not found: {embedding_path}")
        
        # Load embedding
        data = load_file(str(embedding_path))
        embedding = data["character_embedding"]
        
        # Move to device if specified
        if device:
            embedding = embedding.to(device)
        
        # Load metadata if available
        metadata = None
        metadata_path = embedding_path.with_suffix(".json")
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        
        logger.info(f"Character embedding loaded from {embedding_path}")
        return embedding, metadata
    
    @staticmethod
    def validate_embedding(embedding: torch.Tensor, expected_dim: Optional[int] = None) -> bool:
        """
        Validate embedding tensor.
        
        Args:
            embedding: Embedding tensor to validate
            expected_dim: Expected embedding dimension (optional)
            
        Returns:
            True if valid, raises ValueError if invalid
        """
        if not isinstance(embedding, torch.Tensor):
            raise ValueError(f"Embedding must be torch.Tensor, got {type(embedding)}")
        
        if embedding.dim() != 1:
            raise ValueError(f"Embedding must be 1D tensor, got shape {embedding.shape}")
        
        if expected_dim is not None and embedding.shape[0] != expected_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {expected_dim}, got {embedding.shape[0]}"
            )
        
        if torch.isnan(embedding).any():
            raise ValueError("Embedding contains NaN values")
        
        if torch.isinf(embedding).any():
            raise ValueError("Embedding contains Inf values")
        
        return True


