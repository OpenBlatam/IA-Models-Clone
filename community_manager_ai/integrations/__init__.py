"""Social media platform integrations"""

from .base_platform import SocialPlatform
from .factory import get_platform_handler, register_platform

__all__ = [
    "SocialPlatform",
    "get_platform_handler",
    "register_platform",
]




