"""Quality Metrics Schema"""
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class QualityMetricsSchema(BaseModel):
    total_inspections: int = Field(..., ge=0)
    average_quality_score: float = Field(..., ge=0.0, le=100.0)
    defects_count: int = 0
    anomalies_count: int = 0
    rejection_rate: float = Field(0.0, ge=0.0, le=100.0)
    acceptance_rate: float = Field(100.0, ge=0.0, le=100.0)
    defects_per_inspection: float = 0.0
    anomalies_per_inspection: float = 0.0
    calculated_at: Optional[datetime] = None
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None



