"""
Sequence Handling Utilities

Functional utilities for sequence processing (padding, truncation, etc.).
"""

import torch
import numpy as np
from typing import List, Union, Optional
import logging

logger = logging.getLogger(__name__)


class SequenceHandler:
    """Handler for sequence operations."""
    
    @staticmethod
    def pad_sequences(
        sequences: List[torch.Tensor],
        max_length: Optional[int] = None,
        padding_value: float = 0.0,
        padding_side: str = "right"
    ) -> torch.Tensor:
        """
        Pad sequences to same length.
        
        Args:
            sequences: List of sequences to pad
            max_length: Maximum length (uses max if None)
            padding_value: Value to pad with
            padding_side: 'left' or 'right'
            
        Returns:
            Padded tensor
        """
        if not sequences:
            raise ValueError("Empty sequences list")
        
        # Get max length
        if max_length is None:
            max_length = max(len(seq) for seq in sequences)
        
        # Pad each sequence
        padded = []
        for seq in sequences:
            current_len = len(seq)
            if current_len < max_length:
                pad_length = max_length - current_len
                if padding_side == "right":
                    pad = torch.full((pad_length,), padding_value, dtype=seq.dtype)
                    padded_seq = torch.cat([seq, pad])
                else:
                    pad = torch.full((pad_length,), padding_value, dtype=seq.dtype)
                    padded_seq = torch.cat([pad, seq])
            else:
                padded_seq = seq[:max_length]
            
            padded.append(padded_seq)
        
        return torch.stack(padded)
    
    @staticmethod
    def truncate_sequences(
        sequences: List[torch.Tensor],
        max_length: int,
        truncation_side: str = "right"
    ) -> List[torch.Tensor]:
        """
        Truncate sequences to max length.
        
        Args:
            sequences: List of sequences
            max_length: Maximum length
            truncation_side: 'left' or 'right'
            
        Returns:
            List of truncated sequences
        """
        truncated = []
        for seq in sequences:
            if len(seq) > max_length:
                if truncation_side == "right":
                    truncated.append(seq[:max_length])
                else:
                    truncated.append(seq[-max_length:])
            else:
                truncated.append(seq)
        
        return truncated


def pad_sequences(
    sequences: List[torch.Tensor],
    max_length: Optional[int] = None,
    padding_value: float = 0.0
) -> torch.Tensor:
    """Convenience function for padding sequences."""
    return SequenceHandler.pad_sequences(sequences, max_length, padding_value)


def truncate_sequences(
    sequences: List[torch.Tensor],
    max_length: int
) -> List[torch.Tensor]:
    """Convenience function for truncating sequences."""
    return SequenceHandler.truncate_sequences(sequences, max_length)



