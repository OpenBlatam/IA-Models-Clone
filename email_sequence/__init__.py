"""
Email Sequence Module - LangChain Integration

This module provides a comprehensive email sequence system that integrates with LangChain
for intelligent email automation, personalization, and sequence management.

Features:
- LangChain-powered email content generation
- Intelligent sequence timing and triggers
- A/B testing capabilities
- Performance analytics and optimization
- Multi-channel email delivery
- Advanced personalization using AI
"""

from typing import Any, List, Dict, Optional, Union, Tuple
from typing_extensions import Literal, TypedDict

# Core Engine
from .core.email_sequence_engine import EmailSequenceEngine
from .core.sequence_manager import SequenceManager
from .core.personalization_engine import PersonalizationEngine
from .core.analytics_engine import AnalyticsEngine

# Models
from .models.sequence import EmailSequence, SequenceStep, SequenceTrigger
from .models.template import EmailTemplate, TemplateVariable
from .models.campaign import EmailCampaign, CampaignMetrics
from .models.subscriber import Subscriber, SubscriberSegment

# Services
from .services.langchain_service import LangChainEmailService
from .services.delivery_service import EmailDeliveryService
from .services.analytics_service import EmailAnalyticsService

# API
from .api.routes import email_sequence_router
from .api.schemas import (
    SequenceCreateRequest,
    SequenceUpdateRequest,
    CampaignCreateRequest,
    TemplateCreateRequest,
    SubscriberCreateRequest
)

__version__ = "2.0.0"
__author__ = "Blatam Academy Team"

__all__ = [
    # Core Engine
    "EmailSequenceEngine",
    "SequenceManager", 
    "PersonalizationEngine",
    "AnalyticsEngine",
    
    # Models
    "EmailSequence",
    "SequenceStep", 
    "SequenceTrigger",
    "EmailTemplate",
    "TemplateVariable",
    "EmailCampaign",
    "CampaignMetrics",
    "Subscriber",
    "SubscriberSegment",
    
    # Services
    "LangChainEmailService",
    "EmailDeliveryService",
    "EmailAnalyticsService",
    
    # API
    "email_sequence_router",
    "SequenceCreateRequest",
    "SequenceUpdateRequest", 
    "CampaignCreateRequest",
    "TemplateCreateRequest",
    "SubscriberCreateRequest"
] 