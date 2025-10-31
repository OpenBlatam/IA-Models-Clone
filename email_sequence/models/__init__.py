"""
Models module for Email Sequence System

This module contains the data models for sequences, templates, subscribers,
and campaigns using Pydantic v2.
"""

from .sequence import (
    EmailSequence,
    SequenceStep,
    SequenceTrigger,
    TriggerType,
    StepType,
    SequenceStatus
)
from .template import (
    EmailTemplate,
    TemplateVariable,
    TemplateStatus,
    VariableType
)
from .subscriber import (
    Subscriber,
    SubscriberSegment,
    SubscriberStatus
)
from .campaign import (
    EmailCampaign,
    CampaignMetrics,
    CampaignStatus
)

__all__ = [
    # Sequence models
    "EmailSequence",
    "SequenceStep",
    "SequenceTrigger",
    "TriggerType",
    "StepType",
    "SequenceStatus",
    
    # Template models
    "EmailTemplate",
    "TemplateVariable",
    "TemplateStatus",
    "VariableType",
    
    # Subscriber models
    "Subscriber",
    "SubscriberSegment",
    "SubscriberStatus",
    
    # Campaign models
    "EmailCampaign",
    "CampaignMetrics",
    "CampaignStatus"
]