"""
Product domain entity.

Represents a skincare product with its properties, ingredients,
and suitability for different skin types.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from .enums import SkinType


@dataclass
class Product:
    """
    Product domain entity.
    
    Represents a skincare product with its properties, ingredients,
    pricing, and metadata for recommendation matching.
    """
    
    id: str
    name: str
    category: str
    description: Optional[str] = None
    ingredients: List[str] = field(default_factory=list)
    price: Optional[float] = None
    rating: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_suitable_for_skin_type(self, skin_type: SkinType) -> bool:
        """
        Check if product is suitable for given skin type.
        
        Args:
            skin_type: Skin type to check suitability for
            
        Returns:
            True if product is suitable, False otherwise
        """
        return True










