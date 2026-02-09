"""
Documentación OpenAPI Mejorada
===============================
Configuración y documentación OpenAPI/Swagger
"""

from typing import Dict, Any

# Configuración de OpenAPI
OPENAPI_TAGS = [
    {
        "name": "psychological-validation",
        "description": "Validación Psicológica AI - Sistema completo de análisis psicológico basado en redes sociales",
        "externalDocs": {
            "description": "Documentación completa",
            "url": "https://docs.example.com/psychological-validation"
        }
    },
    {
        "name": "social-media",
        "description": "Gestión de conexiones a redes sociales"
    },
    {
        "name": "reports",
        "description": "Generación y exportación de reportes"
    },
    {
        "name": "analytics",
        "description": "Análisis, predicciones y recomendaciones"
    },
    {
        "name": "notifications",
        "description": "Sistema de notificaciones en tiempo real"
    },
    {
        "name": "webhooks",
        "description": "Webhooks para integración con sistemas externos"
    },
    {
        "name": "batch",
        "description": "Procesamiento por lotes"
    },
    {
        "name": "feedback",
        "description": "Sistema de feedback de usuarios"
    },
    {
        "name": "dashboard",
        "description": "Dashboard y visualizaciones"
    },
    {
        "name": "versioning",
        "description": "Sistema de versionado de validaciones"
    },
    {
        "name": "backup",
        "description": "Backup y recuperación de datos"
    },
    {
        "name": "audit",
        "description": "Auditoría y logs de seguridad"
    },
    {
        "name": "health",
        "description": "Health checks y monitoreo"
    }
]

OPENAPI_INFO = {
    "title": "Validación Psicológica AI API",
    "description": """
## Sistema de Validación Psicológica basado en IA

Sistema completo para análisis psicológico que:
- Conecta con múltiples redes sociales del usuario
- Analiza contenido y comportamiento usando NLP avanzado
- Genera perfiles psicológicos detallados
- Crea reportes completos de validación
- Proporciona recomendaciones personalizadas
- Ofrece análisis predictivo y detección de anomalías

### Características Principales

- **Análisis Avanzado**: NLP, análisis de sentimientos, personalidad Big Five
- **Múltiples Plataformas**: Facebook, Twitter, Instagram, LinkedIn, TikTok, YouTube, etc.
- **Exportación**: JSON, Text, HTML, PDF, CSV
- **Tiempo Real**: Notificaciones WebSocket
- **Extensible**: Sistema de plugins
- **Seguro**: Encriptación avanzada, auditoría completa

### Autenticación

Todas las endpoints requieren autenticación mediante JWT token.
    """,
    "version": "1.6.0",
    "contact": {
        "name": "Blatam Academy",
        "email": "support@blatam.academy"
    },
    "license": {
        "name": "Proprietary",
        "url": "https://blatam.academy/license"
    }
}

OPENAPI_SERVERS = [
    {
        "url": "https://api.example.com/v1",
        "description": "Producción"
    },
    {
        "url": "https://staging-api.example.com/v1",
        "description": "Staging"
    },
    {
        "url": "http://localhost:8000",
        "description": "Desarrollo local"
    }
]

# Ejemplos de respuestas
RESPONSE_EXAMPLES = {
    "validation_detail": {
        "summary": "Ejemplo de respuesta de validación completa",
        "value": {
            "validation": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "user_id": "123e4567-e89b-12d3-a456-426614174001",
                "status": "completed",
                "connected_platforms": ["instagram", "twitter"],
                "created_at": "2024-01-01T10:00:00Z",
                "updated_at": "2024-01-01T10:05:00Z",
                "completed_at": "2024-01-01T10:05:00Z",
                "has_profile": True,
                "has_report": True
            },
            "profile": {
                "id": "123e4567-e89b-12d3-a456-426614174002",
                "personality_traits": {
                    "openness": 0.75,
                    "conscientiousness": 0.65,
                    "extraversion": 0.80,
                    "agreeableness": 0.70,
                    "neuroticism": 0.40
                },
                "confidence_score": 0.85
            },
            "report": {
                "id": "123e4567-e89b-12d3-a456-426614174003",
                "summary": "Análisis psicológico completado exitosamente...",
                "generated_at": "2024-01-01T10:05:00Z"
            }
        }
    },
    "error_response": {
        "summary": "Ejemplo de respuesta de error",
        "value": {
            "detail": "Validation not found",
            "error_code": "VALIDATION_NOT_FOUND"
        }
    }
}




