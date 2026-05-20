"""
Testing Helpers
===============
Utilidades para testing.
"""

from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
import json
import asyncio
from unittest.mock import Mock, MagicMock, patch


def create_mock_response(data: Any, status_code: int = 200) -> Dict[str, Any]:
    """
    Crear respuesta mock.
    
    Args:
        data: Datos de la respuesta
        status_code: Código de estado
        
    Returns:
        Respuesta mock
    """
    return {
        "status_code": status_code,
        "data": data,
        "timestamp": datetime.now().isoformat()
    }


def create_mock_request(
    method: str = "POST",
    path: str = "/api/v1/coach",
    body: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Crear request mock.
    
    Args:
        method: Método HTTP
        path: Ruta
        body: Cuerpo de la request
        headers: Headers
        
    Returns:
        Request mock
    """
    return {
        "method": method,
        "path": path,
        "body": body or {},
        "headers": headers or {}
    }


def assert_response_structure(response: Dict[str, Any], required_fields: List[str]) -> bool:
    """
    Verificar estructura de respuesta.
    
    Args:
        response: Respuesta a verificar
        required_fields: Campos requeridos
        
    Returns:
        True si tiene todos los campos requeridos
    """
    for field in required_fields:
        if field not in response:
            return False
    return True


def generate_test_data(data_type: str, count: int = 1) -> List[Dict[str, Any]]:
    """
    Generar datos de prueba.
    
    Args:
        data_type: Tipo de datos (coaching, training_plan, etc.)
        count: Número de items a generar
        
    Returns:
        Lista de datos de prueba
    """
    templates = {
        "coaching": {
            "question": "How do I train my dog to sit?",
            "dog_breed": "Golden Retriever",
            "dog_age": "2 years",
            "training_goal": "obedience"
        },
        "training_plan": {
            "dog_breed": "German Shepherd",
            "dog_age": "1 year",
            "training_goals": ["obedience", "agility"]
        },
        "behavior_analysis": {
            "behavior_description": "Dog barks excessively when left alone",
            "dog_breed": "Border Collie",
            "frequency": "daily"
        }
    }
    
    template = templates.get(data_type, {})
    
    return [
        {**template, "id": i, "timestamp": datetime.now().isoformat()}
        for i in range(count)
    ]


def compare_responses(response1: Dict[str, Any], response2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Comparar dos respuestas.
    
    Args:
        response1: Primera respuesta
        response2: Segunda respuesta
        
    Returns:
        Diccionario con diferencias
    """
    differences = {
        "fields_only_in_1": [],
        "fields_only_in_2": [],
        "different_values": {}
    }
    
    all_keys = set(response1.keys()) | set(response2.keys())
    
    for key in all_keys:
        if key not in response1:
            differences["fields_only_in_2"].append(key)
        elif key not in response2:
            differences["fields_only_in_1"].append(key)
        elif response1[key] != response2[key]:
            differences["different_values"][key] = {
                "response1": response1[key],
                "response2": response2[key]
            }
    
    return differences


def create_mock_service(service_class: type, **methods) -> Mock:
    """
    Crear mock de servicio.
    
    Args:
        service_class: Clase del servicio
        **methods: Métodos mockeados
        
    Returns:
        Mock del servicio
    """
    mock_service = Mock(spec=service_class)
    
    for method_name, return_value in methods.items():
        if asyncio.iscoroutinefunction(getattr(service_class, method_name, None)):
            setattr(mock_service, method_name, AsyncMock(return_value=return_value))
        else:
            setattr(mock_service, method_name, Mock(return_value=return_value))
    
    return mock_service


class AsyncMock(MagicMock):
    """Mock para funciones async."""
    
    async def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)


def assert_async_response(response: Any, expected_keys: List[str]) -> bool:
    """
    Verificar respuesta async.
    
    Args:
        response: Respuesta a verificar
        expected_keys: Claves esperadas
        
    Returns:
        True si válida
    """
    if not isinstance(response, dict):
        return False
    
    return all(key in response for key in expected_keys)


def create_test_client_config(**overrides) -> Dict[str, Any]:
    """
    Crear configuración de test client.
    
    Args:
        **overrides: Valores a sobrescribir
        
    Returns:
        Configuración
    """
    default_config = {
        "openrouter_api_key": "test_key",
        "openrouter_model": "test_model",
        "debug": True,
        "request_timeout": 10,
        "max_retries": 1
    }
    
    default_config.update(overrides)
    return default_config


def mock_openrouter_response(content: str, status_code: int = 200) -> Dict[str, Any]:
    """
    Crear mock de respuesta de OpenRouter.
    
    Args:
        content: Contenido de la respuesta
        status_code: Código de estado
        
    Returns:
        Respuesta mock
    """
    return {
        "id": "test_id",
        "object": "chat.completion",
        "created": int(datetime.now().timestamp()),
        "model": "test_model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }
    }

