from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from pydantic import BaseModel, Field
from decimal import Decimal
from typing import Dict, Any, List, Optional

from typing import Any, List, Dict, Optional
import logging
import asyncio
class Money(BaseModel):
    amount: Decimal = Field(..., gt=0, description="Monto positivo")
    currency: str = Field("USD", min_length=3, max_length=3, description="Código de moneda ISO 4217")

    @classmethod
    def validate_currency(cls, v) -> bool:
        if len(v) != 3:
            raise ValueError("Código de moneda debe tener 3 caracteres")
        return v

    def to_dict(self) -> Dict[str, Any]:
        return {"amount": float(self.amount), "currency": self.currency}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Money":
        return cls(amount=Decimal(str(data["amount"])), currency=data["currency"])

class Dimensions(BaseModel):
    length: float = Field(..., gt=0)
    width: float = Field(..., gt=0)
    height: float = Field(..., gt=0)
    weight: float = Field(..., gt=0)
    unit: str = Field("cm")
    weight_unit: str = Field("kg")
    
    def volume(self) -> float:
        return self.length * self.width * self.height
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "length": self.length,
            "width": self.width,
            "height": self.height,
            "weight": self.weight,
            "unit": self.unit,
            "weight_unit": self.weight_unit,
            "volume": self.volume()
        }

class SEOData(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    meta_title: Optional[str] = None
    meta_description: Optional[str] = None
    slug: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return self.dict() 