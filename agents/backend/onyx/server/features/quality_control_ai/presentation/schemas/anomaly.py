"""Anomaly Schema"""
from pydantic import BaseModel, Field
from typing import Optional, Dict
from datetime import datetime

class AnomalySchema(BaseModel):
    id: str
    type: str
    severity: str
    location: Dict[str, int]
    score: float = Field(..., ge=0.0, le=1.0)
    penalty_score: float = 0.0
    detected_at: Optional[datetime] = None



