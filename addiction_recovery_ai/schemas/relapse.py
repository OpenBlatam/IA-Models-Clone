"""
Pydantic schemas for relapse prevention endpoints
"""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from datetime import datetime


class RelapseRiskCheckRequest(BaseModel):
    """Request schema for relapse risk check"""
    user_id: str = Field(..., description="User ID")
    days_sober: int = Field(..., ge=0, description="Days sober")
    stress_level: int = Field(..., ge=0, le=10, description="Stress level (0-10)")
    support_level: int = Field(..., ge=0, le=10, description="Support level (0-10)")
    triggers: List[str] = Field(default_factory=list, description="Current triggers")
    previous_relapses: int = Field(default=0, ge=0, description="Number of previous relapses")
    isolation: bool = Field(default=False, description="Feeling isolated")
    negative_thinking: bool = Field(default=False, description="Negative thinking patterns")
    romanticizing: bool = Field(default=False, description="Romanticizing substance use")
    skipping_support: bool = Field(default=False, description="Skipping support meetings")


class RelapseRiskResponse(BaseModel):
    """Response schema for relapse risk assessment"""
    user_id: str = Field(..., description="User ID")
    risk_level: str = Field(..., description="Risk level (low, moderate, high, critical)")
    risk_score: float = Field(..., ge=0, le=100, description="Risk score (0-100)")
    risk_factors: List[str] = Field(default_factory=list, description="Identified risk factors")
    protective_factors: List[str] = Field(default_factory=list, description="Protective factors")
    recommendations: List[str] = Field(default_factory=list, description="Risk mitigation recommendations")
    assessed_at: datetime = Field(default_factory=datetime.now, description="Assessment timestamp")


class CopingStrategiesRequest(BaseModel):
    """Request schema for coping strategies"""
    situation: str = Field(..., description="Current situation")
    trigger_type: Optional[str] = Field(default=None, description="Type of trigger")


class CopingStrategiesResponse(BaseModel):
    """Response schema for coping strategies"""
    situation: str = Field(..., description="Current situation")
    trigger_type: Optional[str] = Field(default=None, description="Type of trigger")
    strategies: List[Dict] = Field(default_factory=list, description="Coping strategies")
    immediate_actions: List[str] = Field(default_factory=list, description="Immediate actions to take")


class EmergencyPlanRequest(BaseModel):
    """Request schema for emergency plan generation"""
    user_id: str = Field(..., description="User ID")
    current_situation: Dict = Field(..., description="Current situation details")


class EmergencyPlanResponse(BaseModel):
    """Response schema for emergency plan"""
    user_id: str = Field(..., description="User ID")
    plan_id: str = Field(..., description="Emergency plan ID")
    immediate_steps: List[str] = Field(default_factory=list, description="Immediate steps to take")
    contacts: List[Dict] = Field(default_factory=list, description="Emergency contacts")
    resources: List[Dict] = Field(default_factory=list, description="Available resources")
    created_at: datetime = Field(default_factory=datetime.now, description="Plan creation timestamp")

