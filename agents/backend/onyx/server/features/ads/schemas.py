from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from pydantic import BaseModel, Field
from typing import Optional, Dict
from uuid import UUID

from typing import Any, List, Dict, Optional
import logging
import asyncio
class AdCreate(BaseModel):
    """Schema for creating an Ad (input)."""
    title: str = Field(..., min_length=2, max_length=128)
    content: str = Field(..., min_length=1)
    metadata: Optional[Dict] = Field(default_factory=dict)

class AdRead(BaseModel):
    """Schema for reading an Ad (output)."""
    id: UUID
    title: str
    content: str
    metadata: Optional[Dict] 