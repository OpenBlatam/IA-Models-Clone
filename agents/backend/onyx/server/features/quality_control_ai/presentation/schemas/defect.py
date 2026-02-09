"""Defect Schema"""
from pydantic import BaseModel, Field
from typing import Optional, Dict
from datetime import datetime

class DefectSchema(BaseModel):
    id: str
    type: str
    severity: str
    location: Dict[str, int]
    confidence: float = Field(..., ge=0.0, le=1.0)
    description: Optional[str] = None
    penalty_score: float = 0.0
    detected_at: Optional[datetime] = None



