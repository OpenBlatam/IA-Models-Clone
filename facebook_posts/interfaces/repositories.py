from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from datetime import datetime
from ..models.facebook_models import (
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
🎯 Repository Interfaces - Clean Architecture
============================================

Interfaces para acceso a datos siguiendo Repository Pattern.
"""


    FacebookPostEntity, FacebookPostAnalysis, ContentIdentifier
)


class FacebookPostRepository(ABC):
    """Interface para repositorio de Facebook posts."""
    
    @abstractmethod
    async def save(self, post: FacebookPostEntity) -> bool:
        """Guardar post."""
        pass
    
    @abstractmethod
    async def find_by_id(self, post_id: str) -> Optional[FacebookPostEntity]:
        """Buscar post por ID."""
        pass
    
    @abstractmethod
    async def find_by_workspace(self, workspace_id: str) -> List[FacebookPostEntity]:
        """Buscar posts por workspace."""
        pass
    
    @abstractmethod
    async def find_by_user(self, user_id: str) -> List[FacebookPostEntity]:
        """Buscar posts por usuario."""
        pass
    
    @abstractmethod
    async def update(self, post: FacebookPostEntity) -> bool:
        """Actualizar post."""
        pass
    
    @abstractmethod
    async def delete(self, post_id: str) -> bool:
        """Eliminar post."""
        pass
    
    @abstractmethod
    async def search(self, query: str, filters: Dict[str, Any]) -> List[FacebookPostEntity]:
        """Buscar posts con filtros."""
        pass


class AnalysisRepository(ABC):
    """Interface para repositorio de análisis."""
    
    @abstractmethod
    async def save_analysis(self, post_id: str, analysis: FacebookPostAnalysis) -> bool:
        """Guardar análisis."""
        pass
    
    @abstractmethod
    async def get_analysis(self, post_id: str) -> Optional[FacebookPostAnalysis]:
        """Obtener análisis."""
        pass
    
    @abstractmethod
    async def get_analytics(self, workspace_id: str, date_from: datetime, date_to: datetime) -> Dict[str, Any]:
        """Obtener analytics agregadas."""
        pass


class CacheRepository(ABC):
    """Interface para repositorio de cache."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Obtener del cache."""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> bool:
        """Guardar en cache."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Eliminar del cache."""
        pass
    
    @abstractmethod
    async def clear_pattern(self, pattern: str) -> int:
        """Limpiar por patrón."""
        pass 