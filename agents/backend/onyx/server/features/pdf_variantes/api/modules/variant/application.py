"""
Variant Module - Application Layer
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass

from .domain import VariantEntity


@dataclass
class GenerateVariantsCommand:
    """Command to generate variants"""
    document_id: str
    user_id: str
    variant_count: int = 10
    variant_type: str = "standard"


@dataclass
class GetVariantQuery:
    """Query to get variant"""
    variant_id: str
    user_id: str


@dataclass
class ListVariantsQuery:
    """Query to list variants"""
    document_id: str
    user_id: str
    limit: int = 20
    offset: int = 0


class GenerateVariantsUseCase(ABC):
    """Generate variants use case"""
    
    @abstractmethod
    async def execute(self, command: GenerateVariantsCommand) -> List[VariantEntity]:
        """Execute generation"""
        pass


class GetVariantUseCase(ABC):
    """Get variant use case"""
    
    @abstractmethod
    async def execute(self, query: GetVariantQuery) -> Optional[VariantEntity]:
        """Execute get"""
        pass


class ListVariantsUseCase(ABC):
    """List variants use case"""
    
    @abstractmethod
    async def execute(self, query: ListVariantsQuery) -> List[VariantEntity]:
        """Execute list"""
        pass






