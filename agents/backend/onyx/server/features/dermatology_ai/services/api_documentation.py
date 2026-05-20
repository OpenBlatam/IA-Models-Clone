"""
Sistema de documentación API mejorada
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class HTTPMethod(str, Enum):
    """Métodos HTTP"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"


@dataclass
class APIExample:
    """Ejemplo de API"""
    description: str
    request: Dict
    response: Dict
    code: str = "python"  # python, curl, javascript, etc.


@dataclass
class APIEndpoint:
    """Endpoint de API"""
    path: str
    method: HTTPMethod
    description: str
    parameters: List[Dict]
    examples: List[APIExample]
    responses: Dict[int, str]  # status_code -> description


class APIDocumentation:
    """Documentación de API"""
    
    def __init__(self):
        """Inicializa la documentación"""
        self.endpoints: List[APIEndpoint] = []
        self._initialize_documentation()
    
    def _initialize_documentation(self):
        """Inicializa documentación de endpoints principales"""
        # Ejemplo: analyze-image
        self.endpoints.append(APIEndpoint(
            path="/dermatology/analyze-image",
            method=HTTPMethod.POST,
            description="Analiza una imagen de piel y proporciona métricas de calidad",
            parameters=[
                {
                    "name": "file",
                    "type": "file",
                    "required": True,
                    "description": "Imagen de piel (JPG, PNG)"
                },
                {
                    "name": "enhance",
                    "type": "boolean",
                    "required": False,
                    "description": "Mejorar imagen antes de analizar"
                }
            ],
            examples=[
                APIExample(
                    description="Análisis básico de imagen",
                    request={
                        "file": "skin_photo.jpg",
                        "enhance": False
                    },
                    response={
                        "success": True,
                        "quality_scores": {
                            "overall_score": 75.5,
                            "texture_score": 80.0
                        }
                    },
                    code="python"
                )
            ],
            responses={
                200: "Análisis completado exitosamente",
                400: "Error en la imagen proporcionada",
                500: "Error interno del servidor"
            }
        ))
    
    def get_endpoint_docs(self, path: str, method: HTTPMethod) -> Optional[APIEndpoint]:
        """Obtiene documentación de un endpoint"""
        for endpoint in self.endpoints:
            if endpoint.path == path and endpoint.method == method:
                return endpoint
        return None
    
    def get_all_docs(self) -> List[Dict]:
        """Obtiene toda la documentación"""
        return [
            {
                "path": e.path,
                "method": e.method.value,
                "description": e.description,
                "parameters": e.parameters,
                "examples": [
                    {
                        "description": ex.description,
                        "request": ex.request,
                        "response": ex.response,
                        "code": ex.code
                    }
                    for ex in e.examples
                ],
                "responses": e.responses
            }
            for e in self.endpoints
        ]
    
    def generate_openapi_spec(self) -> Dict:
        """Genera especificación OpenAPI"""
        return {
            "openapi": "3.0.0",
            "info": {
                "title": "Dermatology AI API",
                "version": "2.2.0",
                "description": "Sistema completo de análisis de piel"
            },
            "paths": {
                endpoint.path: {
                    endpoint.method.value.lower(): {
                        "summary": endpoint.description,
                        "parameters": endpoint.parameters,
                        "responses": {
                            str(code): {"description": desc}
                            for code, desc in endpoint.responses.items()
                        }
                    }
                }
                for endpoint in self.endpoints
            }
        }






