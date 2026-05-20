"""
Data Models for Multimodal Interactive Agent

Enums and dataclasses for multimodal interactions.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


class ModalityType(Enum):
    """Input/output modality types."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    MULTIMODAL = "multimodal"


class InteractionType(Enum):
    """Types of interactions."""
    QUESTION = "question"
    COMMAND = "command"
    CONVERSATION = "conversation"
    TASK = "task"
    CLARIFICATION = "clarification"


@dataclass
class MultimodalInput:
    """Multimodal input data."""
    input_id: str
    modality: ModalityType
    content: Any  # Can be text, image path, audio path, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MultimodalOutput:
    """Multimodal output data."""
    output_id: str
    modality: ModalityType
    content: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Interaction:
    """An interaction with the agent."""
    interaction_id: str
    interaction_type: InteractionType
    input: MultimodalInput
    output: Optional[MultimodalOutput] = None
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)



