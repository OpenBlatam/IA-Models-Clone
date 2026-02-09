"""
Base Context Search - Clase base para búsqueda contextual
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseContextSearch(ABC):
    """Clase base abstracta para búsqueda contextual"""

    @abstractmethod
    async def search(self, query: str, context: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
        """Busca en el contexto"""
        pass

