from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from .routes import router, create_app
from .middleware import setup_middleware
from .serializers import AnalysisRequestSerializer, AnalysisResponseSerializer
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
🎯 API MODULE - Presentation Layer
=================================

Capa de presentación para el motor NLP modular.
Incluye endpoints REST, middleware y serialización.
"""


__all__ = [
    'router',
    'create_app', 
    'setup_middleware',
    'AnalysisRequestSerializer',
    'AnalysisResponseSerializer'
] 