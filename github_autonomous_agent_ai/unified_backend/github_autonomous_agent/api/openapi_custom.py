"""
Customización de OpenAPI Schema para mejor documentación.
"""

from typing import Dict, Any
from fastapi.openapi.utils import get_openapi
from fastapi import FastAPI

from config.settings import settings


def custom_openapi(app: FastAPI) -> Dict[str, Any]:
    """
    Generar OpenAPI schema personalizado con ejemplos y mejor documentación.
    
    Args:
        app: Aplicación FastAPI
        
    Returns:
        OpenAPI schema personalizado
    """
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    
    # Agregar información adicional
    openapi_schema["info"]["contact"] = {
        "name": "GitHub Autonomous Agent Team",
        "email": "support@example.com",
        "url": "https://github.com/example/github-autonomous-agent"
    }
    
    openapi_schema["info"]["license"] = {
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT"
    }
    
    openapi_schema["info"]["x-logo"] = {
        "url": "https://github.com/example/github-autonomous-agent/logo.png"
    }
    
    # Agregar tags con descripciones
    openapi_schema["tags"] = [
        {
            "name": "agent",
            "description": "Endpoints para controlar el agente autónomo (iniciar, detener, pausar, reanudar)",
            "externalDocs": {
                "description": "Documentación completa",
                "url": "https://docs.example.com/agent"
            }
        },
        {
            "name": "tasks",
            "description": "Gestión de tareas: crear, listar, obtener y eliminar tareas",
            "externalDocs": {
                "description": "Guía de tareas",
                "url": "https://docs.example.com/tasks"
            }
        },
        {
            "name": "github",
            "description": "Interacción con GitHub: conectar repositorios, obtener información",
            "externalDocs": {
                "description": "GitHub Integration",
                "url": "https://docs.example.com/github"
            }
        },
        {
            "name": "llm",
            "description": "Servicios de LLM: generación de texto, análisis de código, documentación",
            "externalDocs": {
                "description": "LLM Service Guide",
                "url": "https://docs.example.com/llm"
            }
        },
        {
            "name": "batch",
            "description": "Operaciones en lote: crear, eliminar o actualizar múltiples tareas",
            "externalDocs": {
                "description": "Batch Operations",
                "url": "https://docs.example.com/batch"
            }
        },
        {
            "name": "monitoring",
            "description": "Monitoreo y métricas: métricas del sistema, alertas, performance",
            "externalDocs": {
                "description": "Monitoring Guide",
                "url": "https://docs.example.com/monitoring"
            }
        },
        {
            "name": "audit",
            "description": "Auditoría: eventos de sistema, acciones de usuarios, seguridad",
            "externalDocs": {
                "description": "Audit Logs",
                "url": "https://docs.example.com/audit"
            }
        },
        {
            "name": "notifications",
            "description": "Notificaciones: crear y gestionar notificaciones del sistema",
            "externalDocs": {
                "description": "Notifications",
                "url": "https://docs.example.com/notifications"
            }
        },
        {
            "name": "stats",
            "description": "Estadísticas: resúmenes, métricas de rendimiento, resúmenes de tareas",
            "externalDocs": {
                "description": "Statistics",
                "url": "https://docs.example.com/stats"
            }
        },
        {
            "name": "websocket",
            "description": "WebSocket: conexiones en tiempo real para actualizaciones",
            "externalDocs": {
                "description": "WebSocket API",
                "url": "https://docs.example.com/websocket"
            }
        }
    ]
    
    # Agregar ejemplos a schemas comunes
    if "components" in openapi_schema and "schemas" in openapi_schema["components"]:
        schemas = openapi_schema["components"]["schemas"]
        
        # Ejemplo para CreateTaskRequest
        if "CreateTaskRequest" in schemas:
            schemas["CreateTaskRequest"]["example"] = {
                "repository_owner": "octocat",
                "repository_name": "Hello-World",
                "instruction": "Add a new feature: user authentication",
                "metadata": {
                    "priority": "high",
                    "tags": ["feature", "auth"]
                }
            }
        
        # Ejemplo para TaskResponse
        if "TaskResponse" in schemas:
            schemas["TaskResponse"]["example"] = {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "repository_owner": "octocat",
                "repository_name": "Hello-World",
                "instruction": "Add a new feature: user authentication",
                "status": "pending",
                "created_at": "2024-01-01T12:00:00Z",
                "updated_at": "2024-01-01T12:00:00Z",
                "metadata": {
                    "priority": "high",
                    "tags": ["feature", "auth"]
                }
            }
        
        # Ejemplo para BatchCreateTasksRequest
        if "BatchCreateTasksRequest" in schemas:
            schemas["BatchCreateTasksRequest"]["example"] = {
                "tasks": [
                    {
                        "repository_owner": "octocat",
                        "repository_name": "Hello-World",
                        "instruction": "Fix bug in authentication"
                    },
                    {
                        "repository_owner": "octocat",
                        "repository_name": "Hello-World",
                        "instruction": "Add unit tests"
                    }
                ]
            }
    
    # Agregar servidores
    openapi_schema["servers"] = [
        {
            "url": "http://localhost:8030",
            "description": "Servidor de desarrollo local"
        },
        {
            "url": "https://api.example.com",
            "description": "Servidor de producción"
        }
    ]
    
    # Agregar seguridad
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "Token de autenticación Bearer"
        },
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "API Key para autenticación"
        }
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema



