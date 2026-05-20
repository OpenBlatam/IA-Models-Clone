"""
Onyxbot Module
Slack bot integration
"""

from .slack.base import (
    SlackEvent,
    SlackMessage,
    SlackBotBase
)
from .slack.service import SlackBotService

__all__ = [
    "SlackEvent",
    "SlackMessage",
    "SlackBotBase",
    "SlackBotService",
]

