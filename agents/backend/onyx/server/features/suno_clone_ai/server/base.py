"""
Base Server - Clase base para servidor
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseServer(ABC):
    """Clase base abstracta para servidores"""

    @abstractmethod
    def start(self) -> None:
        """Inicia el servidor"""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Detiene el servidor"""
        pass

