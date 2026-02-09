"""
Initializers - Ultra-Specific Weight Initialization Components
Each initialization strategy in its own focused implementation
"""

import torch
import torch.nn as nn
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class InitializerBase(ABC):
    """Base class for all initializers"""
    
    def __init__(self, name: str = "Initializer"):
        self.name = name
    
    @abstractmethod
    def initialize(self, module: nn.Module):
        """Initialize module weights"""
        pass


class XavierInitializer(InitializerBase):
    """Xavier/Glorot uniform initialization"""
    
    def __init__(self, gain: float = 1.0):
        super().__init__("XavierInitializer")
        self.gain = gain
    
    def initialize(self, module: nn.Module):
        """Apply Xavier initialization"""
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=self.gain)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class KaimingInitializer(InitializerBase):
    """Kaiming/He initialization"""
    
    def __init__(self, mode: str = 'fan_in', nonlinearity: str = 'relu'):
        super().__init__("KaimingInitializer")
        self.mode = mode
        self.nonlinearity = nonlinearity
    
    def initialize(self, module: nn.Module):
        """Apply Kaiming initialization"""
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode=self.mode,
                    nonlinearity=self.nonlinearity
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class OrthogonalInitializer(InitializerBase):
    """Orthogonal initialization"""
    
    def __init__(self, gain: float = 1.0):
        super().__init__("OrthogonalInitializer")
        self.gain = gain
    
    def initialize(self, module: nn.Module):
        """Apply orthogonal initialization"""
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=self.gain)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class UniformInitializer(InitializerBase):
    """Uniform initialization"""
    
    def __init__(self, a: float = 0.0, b: float = 1.0):
        super().__init__("UniformInitializer")
        self.a = a
        self.b = b
    
    def initialize(self, module: nn.Module):
        """Apply uniform initialization"""
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, a=self.a, b=self.b)
                if m.bias is not None:
                    nn.init.uniform_(m.bias, a=self.a, b=self.b)


class NormalInitializer(InitializerBase):
    """Normal/Gaussian initialization"""
    
    def __init__(self, mean: float = 0.0, std: float = 0.02):
        super().__init__("NormalInitializer")
        self.mean = mean
        self.std = std
    
    def initialize(self, module: nn.Module):
        """Apply normal initialization"""
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=self.mean, std=self.std)
                if m.bias is not None:
                    nn.init.normal_(m.bias, mean=self.mean, std=self.std)


class ZeroInitializer(InitializerBase):
    """Zero initialization"""
    
    def __init__(self):
        super().__init__("ZeroInitializer")
    
    def initialize(self, module: nn.Module):
        """Apply zero initialization"""
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class OnesInitializer(InitializerBase):
    """Ones initialization"""
    
    def __init__(self):
        super().__init__("OnesInitializer")
    
    def initialize(self, module: nn.Module):
        """Apply ones initialization"""
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.ones_(m.bias)


# Factory for initializers
class InitializerFactory:
    """Factory for creating initializers"""
    
    _registry = {
        'xavier': XavierInitializer,
        'kaiming': KaimingInitializer,
        'orthogonal': OrthogonalInitializer,
        'uniform': UniformInitializer,
        'normal': NormalInitializer,
        'zero': ZeroInitializer,
        'ones': OnesInitializer,
    }
    
    @classmethod
    def create(cls, initializer_type: str, **kwargs) -> InitializerBase:
        """Create initializer"""
        initializer_type = initializer_type.lower()
        if initializer_type not in cls._registry:
            raise ValueError(f"Unknown initializer type: {initializer_type}")
        return cls._registry[initializer_type](**kwargs)
    
    @classmethod
    def register(cls, name: str, initializer_class: type):
        """Register custom initializer"""
        cls._registry[name.lower()] = initializer_class


__all__ = [
    "InitializerBase",
    "XavierInitializer",
    "KaimingInitializer",
    "OrthogonalInitializer",
    "UniformInitializer",
    "NormalInitializer",
    "ZeroInitializer",
    "OnesInitializer",
    "InitializerFactory",
]



