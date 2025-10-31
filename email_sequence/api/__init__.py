"""
Email Sequence API Module

This module provides FastAPI routes and schemas for the email sequence system.
"""

from .routes import email_sequence_router
from .schemas import (
    SequenceCreateRequest,
    SequenceUpdateRequest,
    SequenceResponse,
    CampaignCreateRequest,
    CampaignResponse,
    TemplateCreateRequest,
    TemplateResponse,
    SubscriberCreateRequest,
    SubscriberResponse,
    AnalyticsResponse,
    ErrorResponse
)

__all__ = [
    "email_sequence_router",
    "SequenceCreateRequest",
    "SequenceUpdateRequest", 
    "SequenceResponse",
    "CampaignCreateRequest",
    "CampaignResponse",
    "TemplateCreateRequest",
    "TemplateResponse",
    "SubscriberCreateRequest",
    "SubscriberResponse",
    "AnalyticsResponse",
    "ErrorResponse"
]






