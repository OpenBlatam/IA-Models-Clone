"""
Webhooks Factory Module
Factory functions for creating webhook manager instances
"""

from .manager_factory import create_webhook_manager, get_default_webhook_manager
from ..config import WebhookConfig

__all__ = ["create_webhook_manager", "get_default_webhook_manager", "WebhookConfig"]






