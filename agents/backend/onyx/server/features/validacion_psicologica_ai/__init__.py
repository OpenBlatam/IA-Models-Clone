"""
Validación Psicológica AI
==========================

Sistema de validación psicológica basado en IA que:
- Conecta con múltiples redes sociales del usuario
- Analiza contenido y comportamiento
- Genera informes de validación psicológica
- Almacena historial de validaciones en base de datos
"""

__version__ = "1.0.0"
__author__ = "Blatam Academy"
__license__ = "Proprietary"
__description__ = "AI-powered psychological validation system with social media analysis"

# Try to import components with error handling
try:
    from .models import (
        PsychologicalValidation,
        SocialMediaConnection,
        SocialMediaPlatform,
        ValidationReport,
        PsychologicalProfile,
    )
except ImportError:
    PsychologicalValidation = None
    SocialMediaConnection = None
    SocialMediaPlatform = None
    ValidationReport = None
    PsychologicalProfile = None

try:
    from .schemas import (
        ValidationCreate,
        ValidationRead,
        SocialMediaConnectRequest,
        ValidationReportResponse,
        PsychologicalProfileResponse,
    )
except ImportError:
    ValidationCreate = None
    ValidationRead = None
    SocialMediaConnectRequest = None
    ValidationReportResponse = None
    PsychologicalProfileResponse = None

try:
    from .service import PsychologicalValidationService
except ImportError:
    PsychologicalValidationService = None

__all__ = [
    "PsychologicalValidation",
    "SocialMediaConnection",
    "SocialMediaPlatform",
    "ValidationReport",
    "PsychologicalProfile",
    "ValidationCreate",
    "ValidationRead",
    "SocialMediaConnectRequest",
    "ValidationReportResponse",
    "PsychologicalProfileResponse",
    "PsychologicalValidationService",
]
