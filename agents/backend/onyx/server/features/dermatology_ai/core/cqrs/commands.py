"""
Commands - Write operations in CQRS pattern
Commands represent intent to change state
"""

from abc import ABC
from dataclasses import dataclass
from typing import Any, Optional
from datetime import datetime
import uuid


class Command(ABC):
    """Base class for commands"""
    pass


@dataclass
class AnalyzeImageCommand(Command):
    """Command to analyze an image"""
    user_id: str
    image_data: bytes
    metadata: Optional[dict] = None
    command_id: str = None
    
    def __post_init__(self):
        if self.command_id is None:
            self.command_id = str(uuid.uuid4())


@dataclass
class CreateUserCommand(Command):
    """Command to create a user"""
    email: str
    name: Optional[str] = None
    skin_type: Optional[str] = None
    command_id: str = None
    
    def __post_init__(self):
        if self.command_id is None:
            self.command_id = str(uuid.uuid4())


@dataclass
class UpdateUserPreferencesCommand(Command):
    """Command to update user preferences"""
    user_id: str
    preferences: dict
    command_id: str = None
    
    def __post_init__(self):
        if self.command_id is None:
            self.command_id = str(uuid.uuid4())


@dataclass
class GenerateRecommendationsCommand(Command):
    """Command to generate recommendations"""
    analysis_id: str
    user_id: Optional[str] = None
    command_id: str = None
    
    def __post_init__(self):
        if self.command_id is None:
            self.command_id = str(uuid.uuid4())















