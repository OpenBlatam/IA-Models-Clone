"""
Slack Bot Base Classes and Interfaces
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass


@dataclass
class SlackEvent:
    """Slack event"""
    event_type: str
    data: Dict[str, Any]
    timestamp: datetime
    user_id: Optional[str] = None
    channel_id: Optional[str] = None


@dataclass
class SlackMessage:
    """Slack message"""
    text: str
    channel: str
    user: str
    timestamp: datetime
    thread_ts: Optional[str] = None


class SlackBotBase(ABC):
    """Base interface for Slack bot"""
    
    @abstractmethod
    async def handle_event(self, event: SlackEvent) -> bool:
        """Handle Slack event"""
        pass
    
    @abstractmethod
    async def send_message(
        self,
        channel: str,
        text: str,
        thread_ts: Optional[str] = None
    ) -> bool:
        """Send message to Slack"""
        pass
    
    @abstractmethod
    async def process_command(self, command: str, message: SlackMessage) -> str:
        """Process bot command"""
        pass

