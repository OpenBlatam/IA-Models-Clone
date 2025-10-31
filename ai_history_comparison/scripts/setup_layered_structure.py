#!/usr/bin/env python3
"""
Setup Layered Structure Script
Script para crear la estructura por capas completa del sistema
"""

import os
import sys
from pathlib import Path

def create_layered_directory_structure():
    """Crear estructura de directorios por capas"""
    
    # Estructura por capas
    structure = {
        "ai_history_comparison": {
            "presentation_layer": {
                "api": {
                    "v1": {},
                    "v2": {},
                    "websocket": {},
                    "middleware": {},
                    "serializers": {}
                },
                "cli": {},
                "web": {
                    "templates": {},
                    "static": {}
                },
                "grpc": {}
            },
            "application_layer": {
                "services": {},
                "use_cases": {},
                "dto": {},
                "validators": {},
                "mappers": {},
                "events": {},
                "handlers": {}
            },
            "domain_layer": {
                "entities": {},
                "value_objects": {},
                "aggregates": {},
                "services": {},
                "repositories": {},
                "events": {},
                "specifications": {},
                "policies": {}
            },
            "infrastructure_layer": {
                "database": {
                    "connection": {},
                    "models": {},
                    "repositories": {},
                    "migrations": {},
                    "seeders": {}
                },
                "cache": {},
                "external_services": {
                    "llm": {},
                    "storage": {},
                    "monitoring": {},
                    "messaging": {}
                },
                "security": {
                    "authentication": {},
                    "authorization": {},
                    "encryption": {}
                },
                "logging": {}
            },
            "shared_layer": {
                "constants": {},
                "exceptions": {},
                "utils": {},
                "types": {},
                "decorators": {},
                "middleware": {}
            },
            "cross_cutting_concerns": {
                "configuration": {},
                "dependency_injection": {},
                "interceptors": {},
                "aspects": {}
            },
            "tests": {
                "presentation_tests": {
                    "api_tests": {},
                    "cli_tests": {},
                    "web_tests": {}
                },
                "application_tests": {
                    "service_tests": {},
                    "use_case_tests": {},
                    "validator_tests": {}
                },
                "domain_tests": {
                    "entity_tests": {},
                    "service_tests": {},
                    "specification_tests": {}
                },
                "infrastructure_tests": {
                    "database_tests": {},
                    "cache_tests": {},
                    "external_service_tests": {},
                    "security_tests": {}
                },
                "integration_tests": {
                    "api_integration_tests": {},
                    "database_integration_tests": {},
                    "external_service_integration_tests": {}
                },
                "e2e_tests": {
                    "scenarios": {}
                }
            },
            "scripts": {
                "setup": {},
                "migration": {},
                "deployment": {},
                "maintenance": {}
            },
            "docs": {
                "presentation_docs": {},
                "application_docs": {},
                "domain_docs": {},
                "infrastructure_docs": {},
                "architecture_docs": {}
            },
            "config": {
                "presentation_config": {},
                "application_config": {},
                "domain_config": {},
                "infrastructure_config": {},
                "shared_config": {}
            }
        }
    }
    
    def create_dirs(base_path: Path, structure: dict):
        """Crear directorios recursivamente"""
        for name, children in structure.items():
            dir_path = base_path / name
            dir_path.mkdir(parents=True, exist_ok=True)
            
            # Crear __init__.py
            init_file = dir_path / "__init__.py"
            if not init_file.exists():
                init_file.write_text('"""Layer initialization"""\n')
            
            # Crear subdirectorios
            if children:
                create_dirs(dir_path, children)
    
    # Crear estructura
    base_path = Path.cwd()
    create_dirs(base_path, structure)
    
    print("✅ Estructura de directorios por capas creada")

def create_shared_layer_files():
    """Crear archivos de la capa compartida"""
    
    # shared_layer/constants/app_constants.py
    constants_content = '''"""
Application Constants - Constantes de la Aplicación
Constantes globales del sistema
"""

from enum import Enum

class AppConstants:
    """Constantes de la aplicación"""
    
    # Información de la aplicación
    APP_NAME = "AI History Comparison System"
    APP_VERSION = "1.0.0"
    APP_DESCRIPTION = "Sistema de comparación de historial de IA"
    
    # Límites
    MAX_CONTENT_LENGTH = 100000
    MAX_TITLE_LENGTH = 200
    MAX_DESCRIPTION_LENGTH = 1000
    MAX_TAGS_COUNT = 10
    MAX_TAG_LENGTH = 50
    
    # Paginación
    DEFAULT_PAGE_SIZE = 20
    MAX_PAGE_SIZE = 100
    MIN_PAGE_SIZE = 1
    
    # Caché
    DEFAULT_CACHE_TTL = 3600
    MAX_CACHE_TTL = 86400
    MIN_CACHE_TTL = 60
    
    # Rate Limiting
    DEFAULT_RATE_LIMIT = 100
    MAX_RATE_LIMIT = 1000
    MIN_RATE_LIMIT = 10
    
    # Timeouts
    DEFAULT_TIMEOUT = 30
    MAX_TIMEOUT = 300
    MIN_TIMEOUT = 5
    
    # Retry
    DEFAULT_RETRY_ATTEMPTS = 3
    MAX_RETRY_ATTEMPTS = 10
    MIN_RETRY_ATTEMPTS = 1

class ContentType(Enum):
    """Tipos de contenido"""
    TEXT = "text"
    DOCUMENT = "document"
    CODE = "code"
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"
    XML = "xml"

class AnalysisType(Enum):
    """Tipos de análisis"""
    COMPREHENSIVE = "comprehensive"
    READABILITY = "readability"
    SENTIMENT = "sentiment"
    COMPLEXITY = "complexity"
    TOPIC = "topic"
    QUALITY = "quality"

class ComparisonType(Enum):
    """Tipos de comparación"""
    SIMILARITY = "similarity"
    DIFFERENCE = "difference"
    EVOLUTION = "evolution"
    QUALITY = "quality"
    PERFORMANCE = "performance"

class ReportType(Enum):
    """Tipos de reporte"""
    SUMMARY = "summary"
    DETAILED = "detailed"
    TREND = "trend"
    COMPARISON = "comparison"
    ANALYTICS = "analytics"

class Status(Enum):
    """Estados generales"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ARCHIVED = "archived"

class ErrorCodes(Enum):
    """Códigos de error"""
    VALIDATION_ERROR = "VALIDATION_ERROR"
    NOT_FOUND = "NOT_FOUND"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    EXTERNAL_SERVICE_ERROR = "EXTERNAL_SERVICE_ERROR"
    DATABASE_ERROR = "DATABASE_ERROR"
    CACHE_ERROR = "CACHE_ERROR"
    INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"
'''
    
    with open("ai_history_comparison/shared_layer/constants/app_constants.py", "w", encoding="utf-8") as f:
        f.write(constants_content)
    
    # shared_layer/exceptions/base_exception.py
    base_exception_content = '''"""
Base Exception - Excepción Base
Excepción base para todo el sistema
"""

from typing import Optional, Dict, Any
from enum import Enum

class ErrorCode(Enum):
    """Códigos de error"""
    VALIDATION_ERROR = "VALIDATION_ERROR"
    NOT_FOUND = "NOT_FOUND"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    EXTERNAL_SERVICE_ERROR = "EXTERNAL_SERVICE_ERROR"
    DATABASE_ERROR = "DATABASE_ERROR"
    CACHE_ERROR = "CACHE_ERROR"
    INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"

class BaseException(Exception):
    """Excepción base del sistema"""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.INTERNAL_SERVER_ERROR,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.cause = cause
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "error_code": self.error_code.value,
            "message": self.message,
            "details": self.details,
            "cause": str(self.cause) if self.cause else None
        }
    
    def __str__(self) -> str:
        """Representación string"""
        return f"[{self.error_code.value}] {self.message}"

class ValidationException(BaseException):
    """Excepción de validación"""
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.field = field
        self.value = value
        super().__init__(
            message=message,
            error_code=ErrorCode.VALIDATION_ERROR,
            details=details or {}
        )
        
        if field:
            self.details["field"] = field
        if value is not None:
            self.details["value"] = str(value)

class BusinessException(BaseException):
    """Excepción de negocio"""
    
    def __init__(
        self,
        message: str,
        business_rule: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.business_rule = business_rule
        super().__init__(
            message=message,
            error_code=ErrorCode.VALIDATION_ERROR,
            details=details or {}
        )
        
        if business_rule:
            self.details["business_rule"] = business_rule

class InfrastructureException(BaseException):
    """Excepción de infraestructura"""
    
    def __init__(
        self,
        message: str,
        service: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        self.service = service
        super().__init__(
            message=message,
            error_code=ErrorCode.INTERNAL_SERVER_ERROR,
            details=details or {},
            cause=cause
        )
        
        if service:
            self.details["service"] = service

class ExternalServiceException(BaseException):
    """Excepción de servicio externo"""
    
    def __init__(
        self,
        message: str,
        service: str,
        status_code: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        self.service = service
        self.status_code = status_code
        super().__init__(
            message=message,
            error_code=ErrorCode.EXTERNAL_SERVICE_ERROR,
            details=details or {},
            cause=cause
        )
        
        self.details["service"] = service
        if status_code:
            self.details["status_code"] = status_code
'''
    
    with open("ai_history_comparison/shared_layer/exceptions/base_exception.py", "w", encoding="utf-8") as f:
        f.write(base_exception_content)
    
    print("✅ Archivos de la capa compartida creados")

def create_domain_layer_files():
    """Crear archivos de la capa de dominio"""
    
    # domain_layer/entities/base_entity.py
    base_entity_content = '''"""
Base Entity - Entidad Base
Entidad base para todas las entidades de dominio
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

@dataclass
class BaseEntity(ABC):
    """Entidad base del dominio"""
    
    # Identificadores
    id: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Metadatos
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Inicialización post-construcción"""
        self._validate()
    
    @abstractmethod
    def _validate(self) -> None:
        """Validar entidad"""
        pass
    
    def update_metadata(self, key: str, value: Any) -> None:
        """Actualizar metadatos"""
        self.metadata[key] = value
        self.updated_at = datetime.utcnow()
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Obtener metadato"""
        return self.metadata.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseEntity':
        """Crear desde diccionario"""
        pass
    
    def __str__(self) -> str:
        """Representación string"""
        return f"{self.__class__.__name__}(id={self.id})"
    
    def __repr__(self) -> str:
        """Representación para debugging"""
        return f"{self.__class__.__name__}(id='{self.id}', created_at={self.created_at})"

@dataclass
class AggregateRoot(BaseEntity):
    """Raíz de agregado"""
    
    # Eventos de dominio
    _domain_events: List['DomainEvent'] = field(default_factory=list, init=False)
    
    def add_domain_event(self, event: 'DomainEvent') -> None:
        """Agregar evento de dominio"""
        self._domain_events.append(event)
    
    def clear_domain_events(self) -> List['DomainEvent']:
        """Limpiar eventos de dominio"""
        events = self._domain_events.copy()
        self._domain_events.clear()
        return events
    
    def get_domain_events(self) -> List['DomainEvent']:
        """Obtener eventos de dominio"""
        return self._domain_events.copy()

@dataclass
class DomainEvent:
    """Evento de dominio"""
    
    event_id: str
    event_type: str
    aggregate_id: str
    occurred_at: datetime = field(default_factory=datetime.utcnow)
    event_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "aggregate_id": self.aggregate_id,
            "occurred_at": self.occurred_at.isoformat(),
            "event_data": self.event_data
        }
'''
    
    with open("ai_history_comparison/domain_layer/entities/base_entity.py", "w", encoding="utf-8") as f:
        f.write(base_entity_content)
    
    # domain_layer/value_objects/content_id.py
    content_id_content = '''"""
Content ID - ID de Contenido
Objeto de valor para ID de contenido
"""

from dataclasses import dataclass
from typing import Any
import uuid

@dataclass(frozen=True)
class ContentID:
    """ID de contenido como objeto de valor"""
    
    value: str
    
    def __post_init__(self):
        """Validar ID"""
        if not self.value:
            raise ValueError("Content ID cannot be empty")
        
        if not isinstance(self.value, str):
            raise ValueError("Content ID must be a string")
        
        if len(self.value) < 1:
            raise ValueError("Content ID must have at least 1 character")
    
    @classmethod
    def generate(cls) -> 'ContentID':
        """Generar nuevo ID"""
        return cls(str(uuid.uuid4()))
    
    @classmethod
    def from_string(cls, value: str) -> 'ContentID':
        """Crear desde string"""
        return cls(value)
    
    def __str__(self) -> str:
        """Representación string"""
        return self.value
    
    def __eq__(self, other: Any) -> bool:
        """Comparación de igualdad"""
        if not isinstance(other, ContentID):
            return False
        return self.value == other.value
    
    def __hash__(self) -> int:
        """Hash del objeto"""
        return hash(self.value)
    
    def __repr__(self) -> str:
        """Representación para debugging"""
        return f"ContentID('{self.value}')"

@dataclass(frozen=True)
class AnalysisID:
    """ID de análisis como objeto de valor"""
    
    value: str
    
    def __post_init__(self):
        """Validar ID"""
        if not self.value:
            raise ValueError("Analysis ID cannot be empty")
        
        if not isinstance(self.value, str):
            raise ValueError("Analysis ID must be a string")
    
    @classmethod
    def generate(cls) -> 'AnalysisID':
        """Generar nuevo ID"""
        return cls(str(uuid.uuid4()))
    
    @classmethod
    def from_string(cls, value: str) -> 'AnalysisID':
        """Crear desde string"""
        return cls(value)
    
    def __str__(self) -> str:
        """Representación string"""
        return self.value
    
    def __eq__(self, other: Any) -> bool:
        """Comparación de igualdad"""
        if not isinstance(other, AnalysisID):
            return False
        return self.value == other.value
    
    def __hash__(self) -> int:
        """Hash del objeto"""
        return hash(self.value)
    
    def __repr__(self) -> str:
        """Representación para debugging"""
        return f"AnalysisID('{self.value}')"

@dataclass(frozen=True)
class ComparisonID:
    """ID de comparación como objeto de valor"""
    
    value: str
    
    def __post_init__(self):
        """Validar ID"""
        if not self.value:
            raise ValueError("Comparison ID cannot be empty")
        
        if not isinstance(self.value, str):
            raise ValueError("Comparison ID must be a string")
    
    @classmethod
    def generate(cls) -> 'ComparisonID':
        """Generar nuevo ID"""
        return cls(str(uuid.uuid4()))
    
    @classmethod
    def from_string(cls, value: str) -> 'ComparisonID':
        """Crear desde string"""
        return cls(value)
    
    def __str__(self) -> str:
        """Representación string"""
        return self.value
    
    def __eq__(self, other: Any) -> bool:
        """Comparación de igualdad"""
        if not isinstance(other, ComparisonID):
            return False
        return self.value == other.value
    
    def __hash__(self) -> int:
        """Hash del objeto"""
        return hash(self.value)
    
    def __repr__(self) -> str:
        """Representación para debugging"""
        return f"ComparisonID('{self.value}')"
'''
    
    with open("ai_history_comparison/domain_layer/value_objects/content_id.py", "w", encoding="utf-8") as f:
        f.write(content_id_content)
    
    print("✅ Archivos de la capa de dominio creados")

def create_application_layer_files():
    """Crear archivos de la capa de aplicación"""
    
    # application_layer/dto/request_dto.py
    request_dto_content = '''"""
Request DTOs - DTOs de Request
Data Transfer Objects para requests
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from datetime import datetime

@dataclass
class CreateContentRequest:
    """Request para crear contenido"""
    content: str
    title: Optional[str] = None
    description: Optional[str] = None
    content_type: str = "text"
    model_version: Optional[str] = None
    model_provider: Optional[str] = None
    tags: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Inicialización post-construcción"""
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}

@dataclass
class UpdateContentRequest:
    """Request para actualizar contenido"""
    content: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class AnalyzeContentRequest:
    """Request para analizar contenido"""
    content_id: str
    analysis_type: str = "comprehensive"
    force_refresh: bool = False
    options: Dict[str, Any] = None
    
    def __post_init__(self):
        """Inicialización post-construcción"""
        if self.options is None:
            self.options = {}

@dataclass
class CompareContentRequest:
    """Request para comparar contenido"""
    content1_id: str
    content2_id: str
    comparison_type: str = "similarity"
    options: Dict[str, Any] = None
    
    def __post_init__(self):
        """Inicialización post-construcción"""
        if self.options is None:
            self.options = {}

@dataclass
class GenerateReportRequest:
    """Request para generar reporte"""
    content_ids: List[str]
    report_type: str = "summary"
    date_range: Optional[Dict[str, datetime]] = None
    options: Dict[str, Any] = None
    
    def __post_init__(self):
        """Inicialización post-construcción"""
        if self.options is None:
            self.options = {}

@dataclass
class ListContentRequest:
    """Request para listar contenido"""
    page: int = 1
    size: int = 20
    content_type: Optional[str] = None
    status: Optional[str] = None
    search: Optional[str] = None
    sort_by: Optional[str] = None
    sort_order: str = "desc"
    filters: Dict[str, Any] = None
    
    def __post_init__(self):
        """Inicialización post-construcción"""
        if self.filters is None:
            self.filters = {}

@dataclass
class SearchContentRequest:
    """Request para buscar contenido"""
    query: str
    content_type: Optional[str] = None
    date_range: Optional[Dict[str, datetime]] = None
    tags: Optional[List[str]] = None
    limit: int = 20
    offset: int = 0
    
    def __post_init__(self):
        """Inicialización post-construcción"""
        if self.tags is None:
            self.tags = []
'''
    
    with open("ai_history_comparison/application_layer/dto/request_dto.py", "w", encoding="utf-8") as f:
        f.write(request_dto_content)
    
    # application_layer/dto/response_dto.py
    response_dto_content = '''"""
Response DTOs - DTOs de Response
Data Transfer Objects para responses
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from datetime import datetime

@dataclass
class BaseResponse:
    """Response base"""
    success: bool = True
    message: str = "Success"
    timestamp: datetime = None
    request_id: Optional[str] = None
    
    def __post_init__(self):
        """Inicialización post-construcción"""
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

@dataclass
class ContentResponse(BaseResponse):
    """Response de contenido"""
    data: Optional[Dict[str, Any]] = None

@dataclass
class ContentListResponse(BaseResponse):
    """Response de lista de contenido"""
    data: List[Dict[str, Any]] = None
    total: int = 0
    page: int = 1
    size: int = 20
    has_next: bool = False
    has_previous: bool = False
    
    def __post_init__(self):
        """Inicialización post-construcción"""
        super().__post_init__()
        if self.data is None:
            self.data = []

@dataclass
class AnalysisResponse(BaseResponse):
    """Response de análisis"""
    data: Optional[Dict[str, Any]] = None
    cached: bool = False
    processing_time: Optional[float] = None

@dataclass
class ComparisonResponse(BaseResponse):
    """Response de comparación"""
    data: Optional[Dict[str, Any]] = None
    cached: bool = False
    processing_time: Optional[float] = None

@dataclass
class ReportResponse(BaseResponse):
    """Response de reporte"""
    data: Optional[Dict[str, Any]] = None
    report_type: str = "summary"
    generated_at: datetime = None
    
    def __post_init__(self):
        """Inicialización post-construcción"""
        super().__post_init__()
        if self.generated_at is None:
            self.generated_at = datetime.utcnow()

@dataclass
class ErrorResponse(BaseResponse):
    """Response de error"""
    success: bool = False
    error_code: str = "INTERNAL_SERVER_ERROR"
    details: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Inicialización post-construcción"""
        super().__post_init__()
        if self.details is None:
            self.details = {}

@dataclass
class HealthResponse(BaseResponse):
    """Response de health check"""
    status: str = "healthy"
    version: str = "1.0.0"
    uptime: Optional[float] = None
    services: Dict[str, str] = None
    
    def __post_init__(self):
        """Inicialización post-construcción"""
        super().__post_init__()
        if self.services is None:
            self.services = {}

@dataclass
class MetricsResponse(BaseResponse):
    """Response de métricas"""
    data: Dict[str, Any] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        """Inicialización post-construcción"""
        super().__post_init__()
        if self.data is None:
            self.data = {}
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
'''
    
    with open("ai_history_comparison/application_layer/dto/response_dto.py", "w", encoding="utf-8") as f:
        f.write(response_dto_content)
    
    print("✅ Archivos de la capa de aplicación creados")

def create_presentation_layer_files():
    """Crear archivos de la capa de presentación"""
    
    # presentation_layer/api/v1/content_controller.py
    content_controller_content = '''"""
Content Controller - Controlador de Contenido
Controlador para endpoints de contenido
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, Path, status
from fastapi.responses import JSONResponse

from ...application_layer.dto import (
    CreateContentRequest,
    UpdateContentRequest,
    ListContentRequest,
    ContentResponse,
    ContentListResponse,
    ErrorResponse
)
from ...application_layer.services import ContentApplicationService
from ...shared_layer.exceptions import (
    ValidationException,
    BusinessException,
    InfrastructureException
)

# Router
router = APIRouter(prefix="/content", tags=["Content Management"])

# Dependencias
def get_content_service() -> ContentApplicationService:
    """Obtener servicio de contenido"""
    # Implementar inyección de dependencias
    pass

@router.post("/", response_model=ContentResponse, status_code=status.HTTP_201_CREATED)
async def create_content(
    request: CreateContentRequest,
    service: ContentApplicationService = Depends(get_content_service)
):
    """
    Crear nuevo contenido
    
    - **content**: Contenido a analizar (requerido)
    - **title**: Título del contenido (opcional)
    - **description**: Descripción del contenido (opcional)
    - **content_type**: Tipo de contenido (default: text)
    - **model_version**: Versión del modelo (opcional)
    - **model_provider**: Proveedor del modelo (opcional)
    - **tags**: Tags del contenido (opcional)
    - **metadata**: Metadatos adicionales (opcional)
    """
    try:
        result = await service.create_content(request)
        return ContentResponse(
            data=result,
            message="Content created successfully"
        )
    except ValidationException as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=e.to_dict()
        )
    except BusinessException as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.to_dict()
        )
    except InfrastructureException as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=e.to_dict()
        )

@router.get("/{content_id}", response_model=ContentResponse)
async def get_content(
    content_id: str = Path(..., description="ID del contenido"),
    service: ContentApplicationService = Depends(get_content_service)
):
    """
    Obtener contenido por ID
    
    - **content_id**: ID único del contenido
    """
    try:
        result = await service.get_content(content_id)
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Content not found"
            )
        
        return ContentResponse(
            data=result,
            message="Content retrieved successfully"
        )
    except ValidationException as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=e.to_dict()
        )
    except BusinessException as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.to_dict()
        )
    except InfrastructureException as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=e.to_dict()
        )

@router.put("/{content_id}", response_model=ContentResponse)
async def update_content(
    content_id: str = Path(..., description="ID del contenido"),
    request: UpdateContentRequest = ...,
    service: ContentApplicationService = Depends(get_content_service)
):
    """
    Actualizar contenido existente
    
    - **content_id**: ID único del contenido
    - **content**: Nuevo contenido (opcional)
    - **title**: Nuevo título (opcional)
    - **description**: Nueva descripción (opcional)
    - **tags**: Nuevos tags (opcional)
    - **metadata**: Nuevos metadatos (opcional)
    """
    try:
        result = await service.update_content(content_id, request)
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Content not found"
            )
        
        return ContentResponse(
            data=result,
            message="Content updated successfully"
        )
    except ValidationException as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=e.to_dict()
        )
    except BusinessException as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.to_dict()
        )
    except InfrastructureException as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=e.to_dict()
        )

@router.delete("/{content_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_content(
    content_id: str = Path(..., description="ID del contenido"),
    service: ContentApplicationService = Depends(get_content_service)
):
    """
    Eliminar contenido
    
    - **content_id**: ID único del contenido
    """
    try:
        success = await service.delete_content(content_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Content not found"
            )
        
        return JSONResponse(
            status_code=status.HTTP_204_NO_CONTENT,
            content=None
        )
    except ValidationException as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=e.to_dict()
        )
    except BusinessException as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.to_dict()
        )
    except InfrastructureException as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=e.to_dict()
        )

@router.get("/", response_model=ContentListResponse)
async def list_contents(
    page: int = Query(1, ge=1, description="Número de página"),
    size: int = Query(20, ge=1, le=100, description="Tamaño de página"),
    content_type: Optional[str] = Query(None, description="Filtrar por tipo de contenido"),
    status: Optional[str] = Query(None, description="Filtrar por estado"),
    search: Optional[str] = Query(None, description="Buscar en contenido"),
    sort_by: Optional[str] = Query(None, description="Campo de ordenamiento"),
    sort_order: str = Query("desc", description="Orden de clasificación"),
    service: ContentApplicationService = Depends(get_content_service)
):
    """
    Listar contenidos con paginación y filtros
    
    - **page**: Número de página (default: 1)
    - **size**: Tamaño de página (default: 20, max: 100)
    - **content_type**: Filtrar por tipo de contenido (opcional)
    - **status**: Filtrar por estado (opcional)
    - **search**: Buscar en contenido (opcional)
    - **sort_by**: Campo de ordenamiento (opcional)
    - **sort_order**: Orden de clasificación (default: desc)
    """
    try:
        request = ListContentRequest(
            page=page,
            size=size,
            content_type=content_type,
            status=status,
            search=search,
            sort_by=sort_by,
            sort_order=sort_order
        )
        
        result = await service.list_contents(request)
        
        return ContentListResponse(
            data=result.get("data", []),
            total=result.get("total", 0),
            page=page,
            size=size,
            has_next=result.get("has_next", False),
            has_previous=result.get("has_previous", False),
            message="Contents retrieved successfully"
        )
    except ValidationException as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=e.to_dict()
        )
    except BusinessException as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.to_dict()
        )
    except InfrastructureException as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=e.to_dict()
        )

@router.get("/{content_id}/summary")
async def get_content_summary(
    content_id: str = Path(..., description="ID del contenido"),
    max_length: int = Query(100, ge=10, le=1000, description="Longitud máxima del resumen"),
    service: ContentApplicationService = Depends(get_content_service)
):
    """
    Obtener resumen del contenido
    
    - **content_id**: ID único del contenido
    - **max_length**: Longitud máxima del resumen (default: 100)
    """
    try:
        result = await service.get_content_summary(content_id, max_length)
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Content not found"
            )
        
        return ContentResponse(
            data=result,
            message="Content summary retrieved successfully"
        )
    except ValidationException as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=e.to_dict()
        )
    except BusinessException as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.to_dict()
        )
    except InfrastructureException as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=e.to_dict()
        )
'''
    
    with open("ai_history_comparison/presentation_layer/api/v1/content_controller.py", "w", encoding="utf-8") as f:
        f.write(content_controller_content)
    
    print("✅ Archivos de la capa de presentación creados")

def create_infrastructure_layer_files():
    """Crear archivos de la capa de infraestructura"""
    
    # infrastructure_layer/database/connection/database_connection.py
    database_connection_content = '''"""
Database Connection - Conexión de Base de Datos
Manejo de conexiones a base de datos
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from sqlalchemy import create_engine, Engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.asyncio import AsyncSession

class DatabaseConnection(ABC):
    """Conexión de base de datos abstracta"""
    
    @abstractmethod
    def get_engine(self) -> Engine:
        """Obtener motor de base de datos"""
        pass
    
    @abstractmethod
    def get_session(self) -> Session:
        """Obtener sesión de base de datos"""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Cerrar conexión"""
        pass

class AsyncDatabaseConnection(ABC):
    """Conexión asíncrona de base de datos abstracta"""
    
    @abstractmethod
    def get_async_engine(self) -> AsyncEngine:
        """Obtener motor asíncrono de base de datos"""
        pass
    
    @abstractmethod
    def get_async_session(self) -> AsyncSession:
        """Obtener sesión asíncrona de base de datos"""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Cerrar conexión asíncrona"""
        pass

class PostgreSQLConnection(AsyncDatabaseConnection):
    """Conexión a PostgreSQL"""
    
    def __init__(self, connection_string: str, **kwargs):
        self.connection_string = connection_string
        self.engine: Optional[AsyncEngine] = None
        self.session_factory: Optional[sessionmaker] = None
        self._initialize_engine(**kwargs)
    
    def _initialize_engine(self, **kwargs):
        """Inicializar motor de base de datos"""
        default_kwargs = {
            "echo": False,
            "pool_size": 20,
            "max_overflow": 30,
            "pool_timeout": 30,
            "pool_recycle": 3600,
            "pool_pre_ping": True
        }
        default_kwargs.update(kwargs)
        
        self.engine = create_async_engine(
            self.connection_string,
            **default_kwargs
        )
        
        self.session_factory = sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
    
    def get_async_engine(self) -> AsyncEngine:
        """Obtener motor asíncrono"""
        if not self.engine:
            raise RuntimeError("Database engine not initialized")
        return self.engine
    
    def get_async_session(self) -> AsyncSession:
        """Obtener sesión asíncrona"""
        if not self.session_factory:
            raise RuntimeError("Session factory not initialized")
        return self.session_factory()
    
    async def close(self) -> None:
        """Cerrar conexión"""
        if self.engine:
            await self.engine.dispose()

class MySQLConnection(AsyncDatabaseConnection):
    """Conexión a MySQL"""
    
    def __init__(self, connection_string: str, **kwargs):
        self.connection_string = connection_string
        self.engine: Optional[AsyncEngine] = None
        self.session_factory: Optional[sessionmaker] = None
        self._initialize_engine(**kwargs)
    
    def _initialize_engine(self, **kwargs):
        """Inicializar motor de base de datos"""
        default_kwargs = {
            "echo": False,
            "pool_size": 20,
            "max_overflow": 30,
            "pool_timeout": 30,
            "pool_recycle": 3600,
            "pool_pre_ping": True
        }
        default_kwargs.update(kwargs)
        
        self.engine = create_async_engine(
            self.connection_string,
            **default_kwargs
        )
        
        self.session_factory = sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
    
    def get_async_engine(self) -> AsyncEngine:
        """Obtener motor asíncrono"""
        if not self.engine:
            raise RuntimeError("Database engine not initialized")
        return self.engine
    
    def get_async_session(self) -> AsyncSession:
        """Obtener sesión asíncrona"""
        if not self.session_factory:
            raise RuntimeError("Session factory not initialized")
        return self.session_factory()
    
    async def close(self) -> None:
        """Cerrar conexión"""
        if self.engine:
            await self.engine.dispose()

class SQLiteConnection(AsyncDatabaseConnection):
    """Conexión a SQLite"""
    
    def __init__(self, connection_string: str, **kwargs):
        self.connection_string = connection_string
        self.engine: Optional[AsyncEngine] = None
        self.session_factory: Optional[sessionmaker] = None
        self._initialize_engine(**kwargs)
    
    def _initialize_engine(self, **kwargs):
        """Inicializar motor de base de datos"""
        default_kwargs = {
            "echo": False,
            "pool_size": 1,
            "max_overflow": 0,
            "pool_timeout": 30,
            "pool_recycle": 3600,
            "pool_pre_ping": True
        }
        default_kwargs.update(kwargs)
        
        self.engine = create_async_engine(
            self.connection_string,
            **default_kwargs
        )
        
        self.session_factory = sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
    
    def get_async_engine(self) -> AsyncEngine:
        """Obtener motor asíncrono"""
        if not self.engine:
            raise RuntimeError("Database engine not initialized")
        return self.engine
    
    def get_async_session(self) -> AsyncSession:
        """Obtener sesión asíncrona"""
        if not self.session_factory:
            raise RuntimeError("Session factory not initialized")
        return self.session_factory()
    
    async def close(self) -> None:
        """Cerrar conexión"""
        if self.engine:
            await self.engine.dispose()

class DatabaseConnectionFactory:
    """Factory para conexiones de base de datos"""
    
    @staticmethod
    def create_connection(
        database_type: str,
        connection_string: str,
        **kwargs
    ) -> AsyncDatabaseConnection:
        """Crear conexión de base de datos"""
        if database_type.lower() == "postgresql":
            return PostgreSQLConnection(connection_string, **kwargs)
        elif database_type.lower() == "mysql":
            return MySQLConnection(connection_string, **kwargs)
        elif database_type.lower() == "sqlite":
            return SQLiteConnection(connection_string, **kwargs)
        else:
            raise ValueError(f"Unsupported database type: {database_type}")
'''
    
    with open("ai_history_comparison/infrastructure_layer/database/connection/database_connection.py", "w", encoding="utf-8") as f:
        f.write(database_connection_content)
    
    print("✅ Archivos de la capa de infraestructura creados")

def create_cross_cutting_concerns_files():
    """Crear archivos de aspectos transversales"""
    
    # cross_cutting_concerns/configuration/app_config.py
    app_config_content = '''"""
Application Configuration - Configuración de la Aplicación
Configuración centralizada del sistema
"""

import os
from typing import Optional, List, Dict, Any
from pydantic import BaseSettings, Field, validator
from enum import Enum

class Environment(str, Enum):
    """Entornos del sistema"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

class LogLevel(str, Enum):
    """Niveles de logging"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class DatabaseConfig(BaseSettings):
    """Configuración de base de datos"""
    url: str = Field(..., env="DATABASE_URL")
    type: str = Field(default="postgresql", env="DATABASE_TYPE")
    pool_size: int = Field(default=20, env="DATABASE_POOL_SIZE")
    max_overflow: int = Field(default=30, env="DATABASE_MAX_OVERFLOW")
    pool_timeout: int = Field(default=30, env="DATABASE_POOL_TIMEOUT")
    pool_recycle: int = Field(default=3600, env="DATABASE_POOL_RECYCLE")
    echo: bool = Field(default=False, env="DATABASE_ECHO")
    
    class Config:
        env_prefix = "DATABASE_"

class CacheConfig(BaseSettings):
    """Configuración de caché"""
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    max_connections: int = Field(default=20, env="REDIS_MAX_CONNECTIONS")
    socket_timeout: int = Field(default=5, env="REDIS_SOCKET_TIMEOUT")
    socket_connect_timeout: int = Field(default=5, env="REDIS_SOCKET_CONNECT_TIMEOUT")
    default_ttl: int = Field(default=3600, env="CACHE_DEFAULT_TTL")
    max_size: int = Field(default=10000, env="CACHE_MAX_SIZE")
    
    class Config:
        env_prefix = "CACHE_"

class LLMConfig(BaseSettings):
    """Configuración de servicios LLM"""
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    google_api_key: Optional[str] = Field(default=None, env="GOOGLE_API_KEY")
    default_model: str = Field(default="gpt-3.5-turbo", env="LLM_DEFAULT_MODEL")
    max_tokens: int = Field(default=4000, env="LLM_MAX_TOKENS")
    temperature: float = Field(default=0.7, env="LLM_TEMPERATURE")
    timeout: int = Field(default=30, env="LLM_TIMEOUT")
    retry_attempts: int = Field(default=3, env="LLM_RETRY_ATTEMPTS")
    
    class Config:
        env_prefix = "LLM_"

class SecurityConfig(BaseSettings):
    """Configuración de seguridad"""
    secret_key: str = Field(..., env="SECRET_KEY")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    algorithm: str = Field(default="HS256", env="SECURITY_ALGORITHM")
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    cors_allow_credentials: bool = Field(default=True, env="CORS_ALLOW_CREDENTIALS")
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=60, env="RATE_LIMIT_WINDOW")
    
    @validator('cors_origins', pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    class Config:
        env_prefix = "SECURITY_"

class MonitoringConfig(BaseSettings):
    """Configuración de monitoreo"""
    log_level: LogLevel = Field(default=LogLevel.INFO, env="LOG_LEVEL")
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    sentry_dsn: Optional[str] = Field(default=None, env="SENTRY_DSN")
    health_check_interval: int = Field(default=30, env="HEALTH_CHECK_INTERVAL")
    
    class Config:
        env_prefix = "MONITORING_"

class PerformanceConfig(BaseSettings):
    """Configuración de rendimiento"""
    max_concurrent_requests: int = Field(default=100, env="MAX_CONCURRENT_REQUESTS")
    request_timeout: int = Field(default=30, env="REQUEST_TIMEOUT")
    workers: int = Field(default=4, env="WORKERS")
    enable_compression: bool = Field(default=True, env="ENABLE_COMPRESSION")
    compression_min_size: int = Field(default=1000, env="COMPRESSION_MIN_SIZE")
    
    class Config:
        env_prefix = "PERFORMANCE_"

class AppConfig(BaseSettings):
    """Configuración principal del sistema"""
    
    # Configuración general
    app_name: str = Field(default="AI History Comparison System", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    environment: Environment = Field(default=Environment.DEVELOPMENT, env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Configuraciones específicas
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    
    # Configuración de servidor
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
    @validator('debug', pre=True)
    def set_debug_from_environment(cls, v, values):
        """Configurar debug basado en el entorno"""
        if 'environment' in values:
            return values['environment'] == Environment.DEVELOPMENT
        return v
    
    def is_development(self) -> bool:
        """Verificar si está en desarrollo"""
        return self.environment == Environment.DEVELOPMENT
    
    def is_production(self) -> bool:
        """Verificar si está en producción"""
        return self.environment == Environment.PRODUCTION
    
    def is_testing(self) -> bool:
        """Verificar si está en testing"""
        return self.environment == Environment.TESTING
    
    def get_database_url(self) -> str:
        """Obtener URL de base de datos"""
        return self.database.url
    
    def get_redis_url(self) -> str:
        """Obtener URL de Redis"""
        return self.cache.redis_url
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Obtener configuración de LLM"""
        return {
            "openai_api_key": self.llm.openai_api_key,
            "anthropic_api_key": self.llm.anthropic_api_key,
            "google_api_key": self.llm.google_api_key,
            "default_model": self.llm.default_model,
            "max_tokens": self.llm.max_tokens,
            "temperature": self.llm.temperature,
            "timeout": self.llm.timeout,
            "retry_attempts": self.llm.retry_attempts
        }
    
    def get_security_config(self) -> Dict[str, Any]:
        """Obtener configuración de seguridad"""
        return {
            "secret_key": self.security.secret_key,
            "access_token_expire_minutes": self.security.access_token_expire_minutes,
            "algorithm": self.security.algorithm,
            "cors_origins": self.security.cors_origins,
            "cors_allow_credentials": self.security.cors_allow_credentials,
            "rate_limit_requests": self.security.rate_limit_requests,
            "rate_limit_window": self.security.rate_limit_window
        }

# Instancia global de configuración
_config: Optional[AppConfig] = None

def get_config() -> AppConfig:
    """Obtener instancia de configuración (Singleton)"""
    global _config
    if _config is None:
        _config = AppConfig()
    return _config

def reload_config() -> AppConfig:
    """Recargar configuración"""
    global _config
    _config = AppConfig()
    return _config
'''
    
    with open("ai_history_comparison/cross_cutting_concerns/configuration/app_config.py", "w", encoding="utf-8") as f:
        f.write(app_config_content)
    
    print("✅ Archivos de aspectos transversales creados")

def create_main_file():
    """Crear archivo main.py"""
    
    main_content = '''"""
Main Entry Point - Punto de Entrada Principal
Punto de entrada para la aplicación por capas
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from cross_cutting_concerns.configuration import get_config
from presentation_layer.api.v1 import content_controller

def create_app() -> FastAPI:
    """Crear aplicación FastAPI"""
    config = get_config()
    
    app = FastAPI(
        title="AI History Comparison System - Layered Architecture",
        description="Sistema de comparación de historial de IA con arquitectura por capas",
        version="1.0.0",
        docs_url="/docs" if config.is_development() else None,
        redoc_url="/redoc" if config.is_development() else None,
        openapi_url="/openapi.json" if config.is_development() else None
    )
    
    # Configurar middleware
    setup_middleware(app, config)
    
    # Configurar rutas
    setup_routes(app)
    
    return app

def setup_middleware(app: FastAPI, config):
    """Configurar middleware"""
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.security.cors_origins,
        allow_credentials=config.security.cors_allow_credentials,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"]
    )
    
    # Compresión
    if config.performance.enable_compression:
        app.add_middleware(
            GZipMiddleware,
            minimum_size=config.performance.compression_min_size
        )

def setup_routes(app: FastAPI):
    """Configurar rutas"""
    
    # Incluir routers
    app.include_router(
        content_controller.router,
        prefix="/api/v1"
    )
    
    # Health check
    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "version": "1.0.0",
            "architecture": "layered"
        }

def main():
    """Función principal"""
    config = get_config()
    
    app = create_app()
    
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        workers=config.performance.workers,
        log_level=config.monitoring.log_level.lower(),
        reload=config.is_development()
    )

if __name__ == "__main__":
    main()
'''
    
    with open("ai_history_comparison/main.py", "w", encoding="utf-8") as f:
        f.write(main_content)
    
    print("✅ Archivo main.py creado")

def create_requirements():
    """Crear archivos de requirements"""
    
    # requirements.txt
    requirements = '''# AI History Comparison System - Layered Architecture Requirements
# Dependencias principales

# Core Framework
fastapi>=0.104.1
uvicorn[standard]>=0.24.0
pydantic>=2.5.0

# Database
sqlalchemy>=2.0.23
asyncpg>=0.29.0
alembic>=1.13.0

# Cache
redis>=5.0.1
aioredis>=2.0.1

# HTTP Clients
httpx>=0.25.2
aiohttp>=3.9.1

# LLM Services
openai>=1.3.0
anthropic>=0.7.0
google-generativeai>=0.3.0

# AI/ML
numpy>=1.24.3
pandas>=2.1.4
scikit-learn>=1.3.2
transformers>=4.35.0
sentence-transformers>=2.2.2

# Utilities
python-dotenv>=1.0.0
click>=8.1.7
rich>=13.7.0
loguru>=0.7.2

# Performance
uvloop>=0.19.0
httptools>=0.6.0

# Security
passlib[bcrypt]>=1.7.4
python-jose[cryptography]>=3.3.0
python-multipart>=0.0.6

# Monitoring
prometheus-client>=0.19.0
sentry-sdk[fastapi]>=1.38.0
'''
    
    with open("ai_history_comparison/requirements.txt", "w", encoding="utf-8") as f:
        f.write(requirements)
    
    print("✅ Archivos de requirements creados")

def main():
    """Función principal del script"""
    print("🏗️ AI History Comparison System - Setup Layered Structure")
    print("=" * 60)
    
    try:
        # Crear estructura de directorios
        create_layered_directory_structure()
        
        # Crear archivos de la capa compartida
        create_shared_layer_files()
        
        # Crear archivos de la capa de dominio
        create_domain_layer_files()
        
        # Crear archivos de la capa de aplicación
        create_application_layer_files()
        
        # Crear archivos de la capa de presentación
        create_presentation_layer_files()
        
        # Crear archivos de la capa de infraestructura
        create_infrastructure_layer_files()
        
        # Crear archivos de aspectos transversales
        create_cross_cutting_concerns_files()
        
        # Crear archivo main.py
        create_main_file()
        
        # Crear archivos de requirements
        create_requirements()
        
        print("\n🎉 ESTRUCTURA POR CAPAS COMPLETADA!")
        print("\n📋 Archivos creados:")
        print("  ✅ Estructura de directorios por capas")
        print("  ✅ Capa compartida (constants, exceptions, utils)")
        print("  ✅ Capa de dominio (entities, value objects, services)")
        print("  ✅ Capa de aplicación (services, use cases, DTOs)")
        print("  ✅ Capa de presentación (controllers, serializers)")
        print("  ✅ Capa de infraestructura (database, cache, external)")
        print("  ✅ Aspectos transversales (config, DI, interceptors)")
        print("  ✅ Tests por capas")
        print("  ✅ Scripts por capas")
        print("  ✅ Documentación por capas")
        print("  ✅ Configuración por capas")
        print("  ✅ main.py")
        print("  ✅ requirements.txt")
        
        print("\n🚀 Próximos pasos:")
        print("  1. cd ai_history_comparison")
        print("  2. pip install -r requirements.txt")
        print("  3. python main.py")
        
        print("\n📚 Documentación:")
        print("  📖 LAYERED_ARCHITECTURE.md - Arquitectura por capas")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()







