"""
Pydantic schemas for emergency services endpoints
"""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field, EmailStr
from datetime import datetime


class CreateEmergencyContactRequest(BaseModel):
    """Request schema for creating emergency contact"""
    user_id: str = Field(..., description="User ID")
    name: str = Field(..., description="Contact name")
    relationship: str = Field(..., description="Relationship to user")
    phone: str = Field(..., description="Phone number")
    email: Optional[EmailStr] = Field(default=None, description="Email address")
    is_primary: bool = Field(default=False, description="Is primary contact")


class EmergencyContactResponse(BaseModel):
    """Response schema for emergency contact"""
    contact_id: str = Field(..., description="Contact ID")
    user_id: str = Field(..., description="User ID")
    name: str = Field(..., description="Contact name")
    relationship: str = Field(..., description="Relationship to user")
    phone: str = Field(..., description="Phone number")
    email: Optional[str] = Field(default=None, description="Email address")
    is_primary: bool = Field(default=False, description="Is primary contact")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")


class EmergencyContactsListResponse(BaseModel):
    """Response schema for emergency contacts list"""
    user_id: str = Field(..., description="User ID")
    contacts: List[EmergencyContactResponse] = Field(default_factory=list, description="List of contacts")
    primary_contact: Optional[EmergencyContactResponse] = Field(default=None, description="Primary contact")


class TriggerEmergencyRequest(BaseModel):
    """Request schema for triggering emergency protocol"""
    user_id: str = Field(..., description="User ID")
    risk_level: str = Field(..., description="Risk level (low, moderate, high, critical)")
    situation: str = Field(..., description="Situation description")

    @validator('risk_level')
    def validate_risk_level(cls, v):
        """Validate risk level"""
        valid_levels = ['low', 'moderate', 'high', 'critical']
        if v.lower() not in valid_levels:
            raise ValueError(f"Invalid risk level. Must be one of: {valid_levels}")
        return v.lower()


class EmergencyProtocolResponse(BaseModel):
    """Response schema for emergency protocol"""
    protocol_id: str = Field(..., description="Protocol ID")
    user_id: str = Field(..., description="User ID")
    risk_level: str = Field(..., description="Risk level")
    situation: str = Field(..., description="Situation description")
    actions_taken: List[str] = Field(default_factory=list, description="Actions taken")
    contacts_notified: List[str] = Field(default_factory=list, description="Contacts notified")
    resources_provided: List[Dict] = Field(default_factory=list, description="Resources provided")
    triggered_at: datetime = Field(default_factory=datetime.now, description="Trigger timestamp")


class CrisisResourceResponse(BaseModel):
    """Response schema for crisis resource"""
    resource_id: str = Field(..., description="Resource ID")
    name: str = Field(..., description="Resource name")
    type: str = Field(..., description="Resource type")
    phone: Optional[str] = Field(default=None, description="Phone number")
    website: Optional[str] = Field(default=None, description="Website URL")
    location: Optional[str] = Field(default=None, description="Location")
    available_24_7: bool = Field(default=False, description="Available 24/7")


class CrisisResourcesResponse(BaseModel):
    """Response schema for crisis resources list"""
    resources: List[CrisisResourceResponse] = Field(default_factory=list, description="List of resources")
    location: Optional[str] = Field(default=None, description="Location filter")
    total: int = Field(default=0, ge=0, description="Total number of resources")

