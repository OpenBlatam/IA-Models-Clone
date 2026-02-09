"""
Base Tracer - Clase base para tracers
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager


class BaseTracer(ABC):
    """Clase base abstracta para tracers"""
    
    @abstractmethod
    @asynccontextmanager
    async def trace(self, operation_name: str, **kwargs):
        """Crea un span de tracing"""
        pass
    
    @abstractmethod
    def log(self, level: str, message: str, **kwargs):
        """Registra un log"""
        pass
    
    @abstractmethod
    def metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Registra una métrica"""
        pass

