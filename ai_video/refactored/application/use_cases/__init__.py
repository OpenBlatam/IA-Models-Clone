from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from .template_use_cases import (
from .avatar_use_cases import (
from .video_use_cases import (
from .script_use_cases import (
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Application Use Cases
====================

This module contains all use cases for the AI Video system.
Use cases represent the business logic and orchestration of domain operations.
"""

    ListTemplatesUseCase,
    GetTemplateUseCase,
    CreateTemplateUseCase,
    UpdateTemplateUseCase,
    DeleteTemplateUseCase,
)
    ListAvatarsUseCase,
    GetAvatarUseCase,
    CreateAvatarUseCase,
    UpdateAvatarUseCase,
    DeleteAvatarUseCase,
    GenerateAvatarPreviewUseCase,
)
    CreateVideoUseCase,
    GetVideoUseCase,
    ListVideosUseCase,
    UpdateVideoUseCase,
    DeleteVideoUseCase,
    ProcessVideoUseCase,
)
    GenerateScriptUseCase,
    GetScriptUseCase,
    UpdateScriptUseCase,
    OptimizeScriptUseCase,
)

__all__ = [
    # Template use cases
    "ListTemplatesUseCase",
    "GetTemplateUseCase", 
    "CreateTemplateUseCase",
    "UpdateTemplateUseCase",
    "DeleteTemplateUseCase",
    
    # Avatar use cases
    "ListAvatarsUseCase",
    "GetAvatarUseCase",
    "CreateAvatarUseCase",
    "UpdateAvatarUseCase", 
    "DeleteAvatarUseCase",
    "GenerateAvatarPreviewUseCase",
    
    # Video use cases
    "CreateVideoUseCase",
    "GetVideoUseCase",
    "ListVideosUseCase",
    "UpdateVideoUseCase",
    "DeleteVideoUseCase",
    "ProcessVideoUseCase",
    
    # Script use cases
    "GenerateScriptUseCase",
    "GetScriptUseCase",
    "UpdateScriptUseCase",
    "OptimizeScriptUseCase",
] 