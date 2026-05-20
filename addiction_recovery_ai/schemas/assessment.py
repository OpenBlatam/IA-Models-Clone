"""
Pydantic schemas for assessment endpoints
"""

from typing import List, Optional
from pydantic import BaseModel, Field, validator
from datetime import datetime


class AssessmentRequest(BaseModel):
    """Request schema for addiction assessment"""
    addiction_type: str = Field(..., description="Type of addiction")
    severity: str = Field(..., description="Severity level")
    frequency: str = Field(..., description="Frequency of use")
    duration_years: Optional[float] = Field(default=None, ge=0, description="Duration in years")
    daily_cost: Optional[float] = Field(default=None, ge=0, description="Daily cost")
    triggers: List[str] = Field(default_factory=list, description="List of triggers")
    motivations: List[str] = Field(default_factory=list, description="List of motivations")
    previous_attempts: int = Field(default=0, ge=0, description="Number of previous attempts")
    support_system: bool = Field(default=False, description="Has support system")
    medical_conditions: List[str] = Field(default_factory=list, description="Medical conditions")
    additional_info: Optional[str] = Field(default=None, description="Additional information")

    @validator('addiction_type')
    def validate_addiction_type(cls, v):
        """Validate addiction type"""
        valid_types = ['smoking', 'alcohol', 'drugs', 'gambling', 'internet', 'other']
        if v.lower() not in valid_types:
            raise ValueError(f"Invalid addiction type. Must be one of: {valid_types}")
        return v.lower()

    @validator('severity')
    def validate_severity(cls, v):
        """Validate severity level"""
        valid_levels = ['low', 'moderate', 'high', 'severe']
        if v.lower() not in valid_levels:
            raise ValueError(f"Invalid severity. Must be one of: {valid_levels}")
        return v.lower()

    @validator('frequency')
    def validate_frequency(cls, v):
        """Validate frequency"""
        valid_frequencies = ['daily', 'weekly', 'monthly', 'occasional']
        if v.lower() not in valid_frequencies:
            raise ValueError(f"Invalid frequency. Must be one of: {valid_frequencies}")
        return v.lower()


class AssessmentResponse(BaseModel):
    """Response schema for addiction assessment"""
    user_id: Optional[str] = Field(default=None, description="User ID")
    assessment_id: str = Field(..., description="Assessment ID")
    addiction_type: str = Field(..., description="Type of addiction")
    severity_score: float = Field(..., ge=0, le=10, description="Severity score")
    risk_level: str = Field(..., description="Risk level")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")
    next_steps: List[str] = Field(default_factory=list, description="Next steps")
    assessed_at: datetime = Field(default_factory=datetime.now, description="Assessment timestamp")


class ProfileResponse(BaseModel):
    """Response schema for user profile"""
    user_id: str = Field(..., description="User ID")
    email: Optional[str] = Field(default=None, description="User email")
    name: Optional[str] = Field(default=None, description="User name")
    addiction_type: Optional[str] = Field(default=None, description="Primary addiction type")
    days_sober: Optional[int] = Field(default=None, ge=0, description="Days sober")
    created_at: Optional[datetime] = Field(default=None, description="Profile creation date")
    updated_at: Optional[datetime] = Field(default=None, description="Last update date")


class UpdateProfileRequest(BaseModel):
    """Request schema for updating user profile"""
    email: Optional[str] = Field(default=None, description="User email")
    name: Optional[str] = Field(default=None, description="User name")
    addiction_type: Optional[str] = Field(default=None, description="Primary addiction type")
    additional_info: Optional[dict] = Field(default=None, description="Additional profile information")

