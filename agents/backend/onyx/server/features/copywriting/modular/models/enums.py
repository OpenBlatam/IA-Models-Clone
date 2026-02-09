from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from enum import Enum
from typing import Any, List, Dict, Optional
import logging
import asyncio
# -*- coding: utf-8 -*-
"""
Enums - Enumeraciones para modelos
==================================

Definiciones de enumeraciones para tipos de datos.
"""


class ToneType(Enum):
    """Tipos de tono disponibles para copywriting"""
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    URGENT = "urgent"
    FRIENDLY = "friendly"
    TECHNICAL = "technical"
    CREATIVE = "creative"

class LanguageType(Enum):
    """Idiomas disponibles"""
    SPANISH = "es"
    ENGLISH = "en"
    FRENCH = "fr"
    PORTUGUESE = "pt"
    ITALIAN = "it"

class UseCaseType(Enum):
    """Casos de uso disponibles"""
    GENERAL = "general"
    PRODUCT_LAUNCH = "product_launch"
    PROMOTION = "promotion"
    TECH_LAUNCH = "tech_launch"
    B2B = "b2b"
    SOCIAL_MEDIA = "social_media"
    EMAIL_MARKETING = "email_marketing"
    BLOG_POST = "blog_post"
    LANDING_PAGE = "landing_page" 