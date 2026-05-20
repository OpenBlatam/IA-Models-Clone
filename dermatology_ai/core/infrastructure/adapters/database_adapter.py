from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, List


class IDatabaseAdapter(ABC):
    
    @abstractmethod
    async def connect(self):
        pass
    
    @abstractmethod
    async def insert(self, table: str, data: Dict[str, Any]) -> str:
        pass
    
    @abstractmethod
    async def get(self, table: str, key: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        pass
    
    @abstractmethod
    async def query(
        self,
        table: str,
        filter_conditions: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    async def update(
        self,
        table: str,
        key: Dict[str, Any],
        updates: Dict[str, Any]
    ) -> bool:
        pass
    
    @abstractmethod
    async def delete(self, table: str, key: Dict[str, Any]) -> bool:
        pass















