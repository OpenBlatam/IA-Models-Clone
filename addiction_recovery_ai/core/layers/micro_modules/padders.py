"""
Padders - Ultra-Specific Padding Components
Each padding strategy in its own focused implementation
"""

from typing import Optional, Callable
import torch
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class PadderBase(ABC):
    """Base class for all padders"""
    
    def __init__(self, name: str = "Padder"):
        self.name = name
    
    @abstractmethod
    def pad(self, sequence: torch.Tensor, target_length: int) -> torch.Tensor:
        """Pad sequence to target length"""
        pass
    
    def pad_batch(self, sequences: List[torch.Tensor], target_length: int) -> torch.Tensor:
        """Pad batch of sequences"""
        padded = [self.pad(seq, target_length) for seq in sequences]
        return torch.stack(padded)


class ZeroPadder(PadderBase):
    """Zero padding"""
    
    def __init__(self, pad_value: float = 0.0):
        super().__init__("ZeroPadder")
        self.pad_value = pad_value
    
    def pad(self, sequence: torch.Tensor, target_length: int) -> torch.Tensor:
        """Pad sequence with zeros"""
        current_length = sequence.size(0)
        
        if current_length >= target_length:
            return sequence[:target_length]
        
        padding_size = target_length - current_length
        
        if sequence.dim() == 1:
            padding = torch.full(
                (padding_size,),
                self.pad_value,
                dtype=sequence.dtype,
                device=sequence.device
            )
            return torch.cat([sequence, padding])
        else:
            # Multi-dimensional padding
            pad_shape = list(sequence.shape)
            pad_shape[0] = padding_size
            padding = torch.full(
                pad_shape,
                self.pad_value,
                dtype=sequence.dtype,
                device=sequence.device
            )
            return torch.cat([sequence, padding], dim=0)


class RepeatPadder(PadderBase):
    """Repeat last value padding"""
    
    def __init__(self):
        super().__init__("RepeatPadder")
    
    def pad(self, sequence: torch.Tensor, target_length: int) -> torch.Tensor:
        """Pad by repeating last value"""
        current_length = sequence.size(0)
        
        if current_length >= target_length:
            return sequence[:target_length]
        
        if current_length == 0:
            raise ValueError("Cannot pad empty sequence with repeat padding")
        
        padding_size = target_length - current_length
        last_value = sequence[-1:]
        
        if sequence.dim() == 1:
            padding = last_value.repeat(padding_size)
        else:
            padding = last_value.repeat(padding_size, *[1] * (sequence.dim() - 1))
        
        return torch.cat([sequence, padding])


class ReflectPadder(PadderBase):
    """Reflect padding (mirror edge values)"""
    
    def __init__(self):
        super().__init__("ReflectPadder")
    
    def pad(self, sequence: torch.Tensor, target_length: int) -> torch.Tensor:
        """Pad by reflecting sequence"""
        current_length = sequence.size(0)
        
        if current_length >= target_length:
            return sequence[:target_length]
        
        padding_size = target_length - current_length
        
        # Reflect the sequence
        if current_length > 0:
            reflected = torch.flip(sequence, dims=[0])
            # Take needed amount
            if padding_size <= current_length:
                padding = reflected[:padding_size]
            else:
                # Repeat reflection if needed
                repeats = (padding_size // current_length) + 1
                padding = reflected.repeat(repeats)[:padding_size]
        else:
            raise ValueError("Cannot pad empty sequence with reflect padding")
        
        return torch.cat([sequence, padding])


class CircularPadder(PadderBase):
    """Circular padding (wrap around)"""
    
    def __init__(self):
        super().__init__("CircularPadder")
    
    def pad(self, sequence: torch.Tensor, target_length: int) -> torch.Tensor:
        """Pad by circular repetition"""
        current_length = sequence.size(0)
        
        if current_length >= target_length:
            return sequence[:target_length]
        
        if current_length == 0:
            raise ValueError("Cannot pad empty sequence with circular padding")
        
        padding_size = target_length - current_length
        repeats = (padding_size // current_length) + 1
        padding = sequence.repeat(repeats)[:padding_size]
        
        return torch.cat([sequence, padding])


class CustomPadder(PadderBase):
    """Custom padding with user-defined function"""
    
    def __init__(self, pad_fn: Callable[[torch.Tensor, int], torch.Tensor]):
        super().__init__("CustomPadder")
        self.pad_fn = pad_fn
    
    def pad(self, sequence: torch.Tensor, target_length: int) -> torch.Tensor:
        """Pad using custom function"""
        return self.pad_fn(sequence, target_length)


# Factory for padders
class PadderFactory:
    """Factory for creating padders"""
    
    _registry = {
        'zero': ZeroPadder,
        'repeat': RepeatPadder,
        'reflect': ReflectPadder,
        'circular': CircularPadder,
    }
    
    @classmethod
    def create(cls, padder_type: str, **kwargs) -> PadderBase:
        """Create padder"""
        padder_type = padder_type.lower()
        if padder_type not in cls._registry:
            raise ValueError(f"Unknown padder type: {padder_type}")
        return cls._registry[padder_type](**kwargs)
    
    @classmethod
    def register(cls, name: str, padder_class: type):
        """Register custom padder"""
        cls._registry[name.lower()] = padder_class


__all__ = [
    "PadderBase",
    "ZeroPadder",
    "RepeatPadder",
    "ReflectPadder",
    "CircularPadder",
    "CustomPadder",
    "PadderFactory",
]



