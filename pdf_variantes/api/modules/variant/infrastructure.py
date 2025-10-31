"""
Variant Module - Infrastructure Layer
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from .domain import VariantEntity


class VariantRepository(ABC):
    """Variant repository interface"""
    
    @abstractmethod
    async def get_by_id(self, variant_id: str) -> Optional[VariantEntity]:
        """Get variant by ID"""
        pass
    
    @abstractmethod
    async def save(self, variant: VariantEntity) -> VariantEntity:
        """Save variant"""
        pass
    
    @abstractmethod
    async def save_batch(self, variants: List[VariantEntity]) -> List[VariantEntity]:
        """Save multiple variants"""
        pass
    
    @abstractmethod
    async def delete(self, variant_id: str) -> bool:
        """Delete variant"""
        pass
    
    @abstractmethod
    async def find_by_document(
        self,
        document_id: str,
        limit: int = 20,
        offset: int = 0
    ) -> List[VariantEntity]:
        """Find variants by document"""
        pass






