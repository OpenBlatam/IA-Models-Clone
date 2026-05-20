"""
Pydantic schemas for recovery plan endpoints
"""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from datetime import datetime


class CreateRecoveryPlanRequest(BaseModel):
    """Request schema for creating recovery plan"""
    user_id: str = Field(..., description="User ID")
    addiction_type: str = Field(..., description="Type of addiction")
    assessment_data: Dict = Field(..., description="Assessment data")
    approach: Optional[str] = Field(default=None, description="Recovery approach")


class RecoveryPlanResponse(BaseModel):
    """Response schema for recovery plan"""
    user_id: str = Field(..., description="User ID")
    plan_id: str = Field(..., description="Plan ID")
    addiction_type: str = Field(..., description="Type of addiction")
    approach: str = Field(..., description="Recovery approach")
    phases: List[Dict] = Field(default_factory=list, description="Recovery phases")
    strategies: List[str] = Field(default_factory=list, description="Recovery strategies")
    milestones: List[Dict] = Field(default_factory=list, description="Recovery milestones")
    created_at: datetime = Field(default_factory=datetime.now, description="Plan creation date")
    updated_at: Optional[datetime] = Field(default=None, description="Last update date")


class UpdateRecoveryPlanRequest(BaseModel):
    """Request schema for updating recovery plan"""
    phases: Optional[List[Dict]] = Field(default=None, description="Updated phases")
    strategies: Optional[List[str]] = Field(default=None, description="Updated strategies")
    milestones: Optional[List[Dict]] = Field(default=None, description="Updated milestones")
    notes: Optional[str] = Field(default=None, description="Additional notes")

