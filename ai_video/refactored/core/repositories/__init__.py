from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from .base_repository import BaseRepository
from .template_repository import TemplateRepository
from .avatar_repository import AvatarRepository
from .video_repository import VideoRepository
from .script_repository import ScriptRepository
from .user_repository import UserRepository
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Repository Interfaces
====================

This module contains repository interfaces for the AI Video system.
Repositories define the contract for data access and persistence.
"""


__all__ = [
    "BaseRepository",
    "TemplateRepository",
    "AvatarRepository",
    "VideoRepository", 
    "ScriptRepository",
    "UserRepository",
] 