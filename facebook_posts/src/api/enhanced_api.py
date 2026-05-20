from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import time
import json
import logging
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
from contextlib import asynccontextmanager
from typing import Any, List, Dict, Optional
"""
🚀 Enhanced API - API Mejorada con UX Superior
=============================================

API mejorada con validación inteligente, progreso en tiempo real,
error handling avanzado y documentación interactiva.
"""


# Configure logging
logger = logging.getLogger(__name__)

# ===== ENUMS =====

class RequestStatus(Enum):
    """Estados de request."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ValidationLevel(Enum):
    """Niveles de validación."""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    CUSTOM = "custom"

class ResponseFormat(Enum):
    """Formatos de respuesta."""
    JSON = "json"
    XML = "xml"
    YAML = "yaml"
    PROTOBUF = "protobuf"

# ===== DATA MODELS =====

@dataclass
class EnhancedRequest:
    """Request mejorado con validación y tracking."""
    request_id: str
    data: Dict[str, Any]
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    validation_level: ValidationLevel = ValidationLevel.STANDARD
    response_format: ResponseFormat = ResponseFormat.JSON
    priority: int = 1
    timeout_seconds: int = 30
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            'request_id': self.request_id,
            'data': self.data,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'validation_level': self.validation_level.value,
            'response_format': self.response_format.value,
            'priority': self.priority,
            'timeout_seconds': self.timeout_seconds,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'created_at': self.created_at.isoformat(),
            'metadata': self.metadata
        }

@dataclass
class EnhancedResponse:
    """Response mejorado con información detallada."""
    request_id: str
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time: float = 0.0
    status: RequestStatus = RequestStatus.COMPLETED
    progress_percentage: float = 100.0
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            'request_id': self.request_id,
            'success': self.success,
            'data': self.data,
            'error': self.error,
            'processing_time': self.processing_time,
            'status': self.status.value,
            'progress_percentage': self.progress_percentage,
            'warnings': self.warnings,
            'suggestions': self.suggestions,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat()
        }

@dataclass
class ProgressUpdate:
    """Actualización de progreso en tiempo real."""
    request_id: str
    stage: str
    progress_percentage: float
    message: str
    estimated_time_remaining: Optional[float] = None
    current_operation: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            'request_id': self.request_id,
            'stage': self.stage,
            'progress_percentage': self.progress_percentage,
            'message': self.message,
            'estimated_time_remaining': self.estimated_time_remaining,
            'current_operation': self.current_operation,
            'timestamp': self.timestamp.isoformat()
        }

@dataclass
class ValidationResult:
    """Resultado de validación."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    validation_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            'is_valid': self.is_valid,
            'errors': self.errors,
            'warnings': self.warnings,
            'suggestions': self.suggestions,
            'validation_time': self.validation_time
        }

# ===== INTELLIGENT VALIDATION =====

class RequestValidator:
    """Validador inteligente de requests."""
    
    def __init__(self) -> Any:
        self.validation_rules = self._initialize_validation_rules()
        self.custom_validators = {}
        self.validation_history = []
    
    def _initialize_validation_rules(self) -> Dict[str, Dict[str, Any]]:
        """Inicializar reglas de validación."""
        return {
            'generate_post': {
                'required_fields': ['topic', 'audience_type', 'content_type'],
                'field_types': {
                    'topic': str,
                    'audience_type': str,
                    'content_type': str,
                    'tone': str,
                    'length': int,
                    'optimization_level': str
                },
                'field_constraints': {
                    'topic': {'min_length': 3, 'max_length': 200},
                    'audience_type': {'allowed_values': ['general', 'professionals', 'entrepreneurs', 'students']},
                    'content_type': {'allowed_values': ['educational', 'entertainment', 'promotional', 'news']},
                    'tone': {'allowed_values': ['professional', 'casual', 'friendly', 'formal']},
                    'length': {'min_value': 50, 'max_value': 2000},
                    'optimization_level': {'allowed_values': ['basic', 'standard', 'advanced', 'ultra']}
                }
            },
            'analyze_post': {
                'required_fields': ['post_id'],
                'field_types': {
                    'post_id': str,
                    'analysis_type': str
                },
                'field_constraints': {
                    'post_id': {'min_length': 1, 'max_length': 100},
                    'analysis_type': {'allowed_values': ['basic', 'advanced', 'comprehensive']}
                }
            }
        }
    
    async async def validate_request(self, request: EnhancedRequest) -> ValidationResult:
        """Validar request con reglas inteligentes."""
        start_time = time.time()
        validation_result = ValidationResult(is_valid=True)
        
        try:
            # Determinar tipo de request
            request_type = self._determine_request_type(request.data)
            
            if request_type not in self.validation_rules:
                validation_result.errors.append(f"Unknown request type: {request_type}")
                validation_result.is_valid = False
                return validation_result
            
            rules = self.validation_rules[request_type]
            
            # Validar campos requeridos
            await self._validate_required_fields(request.data, rules, validation_result)
            
            # Validar tipos de campos
            await self._validate_field_types(request.data, rules, validation_result)
            
            # Validar restricciones de campos
            await self._validate_field_constraints(request.data, rules, validation_result)
            
            # Validación personalizada según nivel
            if request.validation_level == ValidationLevel.STRICT:
                await self._validate_strict_rules(request.data, validation_result)
            elif request.validation_level == ValidationLevel.CUSTOM:
                await self._validate_custom_rules(request.data, validation_result)
            
            # Generar sugerencias
            await self._generate_suggestions(request.data, validation_result)
            
        except Exception as e:
            validation_result.errors.append(f"Validation error: {str(e)}")
            validation_result.is_valid = False
        
        validation_result.validation_time = time.time() - start_time
        
        # Registrar validación
        self.validation_history.append({
            'request_id': request.request_id,
            'validation_level': request.validation_level.value,
            'is_valid': validation_result.is_valid,
            'errors_count': len(validation_result.errors),
            'warnings_count': len(validation_result.warnings),
            'validation_time': validation_result.validation_time,
            'timestamp': datetime.now()
        })
        
        return validation_result
    
    async def _determine_request_type(self, data: Dict[str, Any]) -> str:
        """Determinar tipo de request basado en datos."""
        if 'topic' in data and 'audience_type' in data:
            return 'generate_post'
        elif 'post_id' in data:
            return 'analyze_post'
        elif 'action' in data:
            return data['action']
        else:
            return 'unknown'
    
    async def _validate_required_fields(self, data: Dict[str, Any], rules: Dict[str, Any], result: ValidationResult):
        """Validar campos requeridos."""
        required_fields = rules.get('required_fields', [])
        
        for field in required_fields:
            if field not in data or data[field] is None:
                result.errors.append(f"Required field missing: {field}")
                result.is_valid = False
            elif isinstance(data[field], str) and len(data[field].strip()) == 0:
                result.errors.append(f"Required field empty: {field}")
                result.is_valid = False
    
    async def _validate_field_types(self, data: Dict[str, Any], rules: Dict[str, Any], result: ValidationResult):
        """Validar tipos de campos."""
        field_types = rules.get('field_types', {})
        
        for field, expected_type in field_types.items():
            if field in data and data[field] is not None:
                if not isinstance(data[field], expected_type):
                    result.errors.append(f"Invalid type for {field}: expected {expected_type.__name__}, got {type(data[field]).__name__}")
                    result.is_valid = False
    
    async def _validate_field_constraints(self, data: Dict[str, Any], rules: Dict[str, Any], result: ValidationResult):
        """Validar restricciones de campos."""
        field_constraints = rules.get('field_constraints', {})
        
        for field, constraints in field_constraints.items():
            if field in data and data[field] is not None:
                value = data[field]
                
                # Validar longitud para strings
                if isinstance(value, str):
                    if 'min_length' in constraints and len(value) < constraints['min_length']:
                        result.errors.append(f"{field} too short: minimum {constraints['min_length']} characters")
                        result.is_valid = False
                    
                    if 'max_length' in constraints and len(value) > constraints['max_length']:
                        result.errors.append(f"{field} too long: maximum {constraints['max_length']} characters")
                        result.is_valid = False
                
                # Validar valores numéricos
                if isinstance(value, (int, float)):
                    if 'min_value' in constraints and value < constraints['min_value']:
                        result.errors.append(f"{field} too small: minimum {constraints['min_value']}")
                        result.is_valid = False
                    
                    if 'max_value' in constraints and value > constraints['max_value']:
                        result.errors.append(f"{field} too large: maximum {constraints['max_value']}")
                        result.is_valid = False
                
                # Validar valores permitidos
                if 'allowed_values' in constraints and value not in constraints['allowed_values']:
                    result.errors.append(f"Invalid value for {field}: {value}. Allowed: {constraints['allowed_values']}")
                    result.is_valid = False
    
    async def _validate_strict_rules(self, data: Dict[str, Any], result: ValidationResult):
        """Validación estricta adicional."""
        # Verificar coherencia de datos
        if 'audience_type' in data and 'content_type' in data:
            audience = data['audience_type']
            content_type = data['content_type']
            
            # Reglas de coherencia
            if audience == 'professionals' and content_type == 'entertainment':
                result.warnings.append("Entertainment content might not be optimal for professional audience")
            
            if audience == 'students' and content_type == 'promotional':
                result.warnings.append("Promotional content might not engage student audience effectively")
    
    async def _validate_custom_rules(self, data: Dict[str, Any], result: ValidationResult):
        """Validación personalizada."""
        # Implementar validaciones específicas del negocio
        pass
    
    async def _generate_suggestions(self, data: Dict[str, Any], result: ValidationResult):
        """Generar sugerencias de mejora."""
        if 'topic' in data:
            topic = data['topic']
            if len(topic) < 10:
                result.suggestions.append("Consider making the topic more specific for better results")
            
            if topic.lower() in ['test', 'demo', 'example']:
                result.suggestions.append("Using generic topics may result in less engaging content")
        
        if 'tone' in data and data['tone'] == 'professional':
            result.suggestions.append("Professional tone works well with educational and technical content")
        
        if 'optimization_level' in data and data['optimization_level'] == 'basic':
            result.suggestions.append("Consider using advanced optimization for better quality content")
    
    def add_custom_validator(self, name: str, validator: Callable):
        """Añadir validador personalizado."""
        self.custom_validators[name] = validator
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de validación."""
        if not self.validation_history:
            return {}
        
        total_validations = len(self.validation_history)
        successful_validations = sum(1 for v in self.validation_history if v['is_valid'])
        
        return {
            'total_validations': total_validations,
            'successful_validations': successful_validations,
            'success_rate': successful_validations / total_validations if total_validations > 0 else 0,
            'avg_validation_time': sum(v['validation_time'] for v in self.validation_history) / total_validations if total_validations > 0 else 0,
            'recent_validations': self.validation_history[-10:]  # Últimos 10
        }

# ===== RATE LIMITING =====

class RateLimiter:
    """Limitador de tasa inteligente."""
    
    def __init__(self, requests_per_minute: int = 100, requests_per_hour: int = 1000):
        
    """__init__ function."""
self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.request_history = {}
        self.user_limits = {}
        self.rate_limit_history = []
    
    async def check_rate_limit(self, user_id: str, request_id: str) -> Dict[str, Any]:
        """Verificar límite de tasa."""
        current_time = datetime.now()
        
        # Inicializar historial de usuario
        if user_id not in self.request_history:
            self.request_history[user_id] = []
        
        # Limpiar historial antiguo
        self._cleanup_old_requests(user_id, current_time)
        
        # Verificar límites
        minute_requests = self._count_recent_requests(user_id, current_time, minutes=1)
        hour_requests = self._count_recent_requests(user_id, current_time, minutes=60)
        
        # Obtener límites del usuario
        user_limit = self.user_limits.get(user_id, {
            'requests_per_minute': self.requests_per_minute,
            'requests_per_hour': self.requests_per_hour
        })
        
        # Verificar si excede límites
        minute_limit_exceeded = minute_requests >= user_limit['requests_per_minute']
        hour_limit_exceeded = hour_requests >= user_limit['requests_per_hour']
        
        if minute_limit_exceeded or hour_limit_exceeded:
            # Registrar limitación
            self.rate_limit_history.append({
                'user_id': user_id,
                'request_id': request_id,
                'timestamp': current_time,
                'minute_limit_exceeded': minute_limit_exceeded,
                'hour_limit_exceeded': hour_limit_exceeded,
                'minute_requests': minute_requests,
                'hour_requests': hour_requests
            })
            
            return {
                'allowed': False,
                'reason': 'rate_limit_exceeded',
                'minute_requests': minute_requests,
                'hour_requests': hour_requests,
                'minute_limit': user_limit['requests_per_minute'],
                'hour_limit': user_limit['requests_per_hour'],
                'retry_after_seconds': self._calculate_retry_after(user_id, current_time)
            }
        
        # Registrar request
        self.request_history[user_id].append({
            'request_id': request_id,
            'timestamp': current_time
        })
        
        return {
            'allowed': True,
            'minute_requests': minute_requests + 1,
            'hour_requests': hour_requests + 1,
            'minute_limit': user_limit['requests_per_minute'],
            'hour_limit': user_limit['requests_per_hour']
        }
    
    def _cleanup_old_requests(self, user_id: str, current_time: datetime):
        """Limpiar requests antiguos."""
        if user_id in self.request_history:
            # Mantener solo requests de la última hora
            cutoff_time = current_time.replace(minute=current_time.minute - 60)
            self.request_history[user_id] = [
                req for req in self.request_history[user_id]
                if req['timestamp'] > cutoff_time
            ]
    
    async def _count_recent_requests(self, user_id: str, current_time: datetime, minutes: int) -> int:
        """Contar requests recientes."""
        if user_id not in self.request_history:
            return 0
        
        cutoff_time = current_time.replace(minute=current_time.minute - minutes)
        return sum(
            1 for req in self.request_history[user_id]
            if req['timestamp'] > cutoff_time
        )
    
    def _calculate_retry_after(self, user_id: str, current_time: datetime) -> int:
        """Calcular tiempo de espera para retry."""
        # Implementar lógica de backoff exponencial
        recent_limits = [
            limit for limit in self.rate_limit_history
            if limit['user_id'] == user_id and 
               (current_time - limit['timestamp']).seconds < 3600
        ]
        
        base_delay = 60  # 1 minuto base
        exponential_factor = min(len(recent_limits), 5)  # Máximo 5 niveles
        
        return base_delay * (2 ** exponential_factor)
    
    def set_user_limits(self, user_id: str, requests_per_minute: int, requests_per_hour: int):
        """Establecer límites para usuario específico."""
        self.user_limits[user_id] = {
            'requests_per_minute': requests_per_minute,
            'requests_per_hour': requests_per_hour
        }
    
    def get_rate_limit_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de rate limiting."""
        total_limits = len(self.rate_limit_history)
        recent_limits = [
            limit for limit in self.rate_limit_history
            if (datetime.now() - limit['timestamp']).seconds < 3600
        ]
        
        return {
            'total_rate_limits': total_limits,
            'recent_rate_limits': len(recent_limits),
            'active_users': len(self.request_history),
            'user_limits': len(self.user_limits),
            'recent_limits': recent_limits[-10:]  # Últimos 10
        }

# ===== RESPONSE OPTIMIZATION =====

class ResponseOptimizer:
    """Optimizador de respuestas."""
    
    def __init__(self) -> Any:
        self.response_cache = {}
        self.optimization_rules = {}
        self.compression_enabled = True
        self.response_history = []
    
    async def optimize_response(self, response: EnhancedResponse, format: ResponseFormat) -> Dict[str, Any]:
        """Optimizar respuesta para el formato solicitado."""
        start_time = time.time()
        
        try:
            # Aplicar optimizaciones según formato
            if format == ResponseFormat.JSON:
                optimized_data = await self._optimize_json_response(response)
            elif format == ResponseFormat.XML:
                optimized_data = await self._optimize_xml_response(response)
            elif format == ResponseFormat.YAML:
                optimized_data = await self._optimize_yaml_response(response)
            else:
                optimized_data = response.to_dict()
            
            # Aplicar compresión si está habilitada
            if self.compression_enabled:
                optimized_data = await self._compress_response(optimized_data)
            
            # Registrar optimización
            optimization_time = time.time() - start_time
            self.response_history.append({
                'request_id': response.request_id,
                'format': format.value,
                'original_size': len(str(response.to_dict())),
                'optimized_size': len(str(optimized_data)),
                'optimization_time': optimization_time,
                'timestamp': datetime.now()
            })
            
            return optimized_data
            
        except Exception as e:
            logger.error(f"Response optimization failed: {e}")
            return response.to_dict()
    
    async def _optimize_json_response(self, response: EnhancedResponse) -> Dict[str, Any]:
        """Optimizar respuesta JSON."""
        data = response.to_dict()
        
        # Remover campos vacíos
        optimized_data = {}
        for key, value in data.items():
            if value is not None and value != [] and value != {}:
                optimized_data[key] = value
        
        # Añadir metadatos de optimización
        optimized_data['_optimized'] = True
        optimized_data['_format'] = 'json'
        
        return optimized_data
    
    async def _optimize_xml_response(self, response: EnhancedResponse) -> Dict[str, Any]:
        """Optimizar respuesta XML."""
        # Implementar optimización XML
        return {
            'xml_response': '<?xml version="1.0" encoding="UTF-8"?><response>...</response>',
            '_optimized': True,
            '_format': 'xml'
        }
    
    async def _optimize_yaml_response(self, response: EnhancedResponse) -> Dict[str, Any]:
        """Optimizar respuesta YAML."""
        # Implementar optimización YAML
        return {
            'yaml_response': 'response:\n  success: true\n  data: ...',
            '_optimized': True,
            '_format': 'yaml'
        }
    
    async def _compress_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprimir respuesta."""
        # Implementar compresión
        return data
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de optimización."""
        if not self.response_history:
            return {}
        
        total_optimizations = len(self.response_history)
        total_size_reduction = sum(
            opt['original_size'] - opt['optimized_size']
            for opt in self.response_history
        )
        
        return {
            'total_optimizations': total_optimizations,
            'total_size_reduction_bytes': total_size_reduction,
            'avg_size_reduction_percent': (
                total_size_reduction / sum(opt['original_size'] for opt in self.response_history) * 100
                if sum(opt['original_size'] for opt in self.response_history) > 0 else 0
            ),
            'avg_optimization_time': sum(opt['optimization_time'] for opt in self.response_history) / total_optimizations if total_optimizations > 0 else 0,
            'recent_optimizations': self.response_history[-10:]  # Últimos 10
        }

# ===== ENHANCED API =====

class EnhancedAPI:
    """API mejorada con UX superior."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        
    """__init__ function."""
self.config = config or {}
        self.request_validator = RequestValidator()
        self.rate_limiter = RateLimiter(
            requests_per_minute=self.config.get('requests_per_minute', 100),
            requests_per_hour=self.config.get('requests_per_hour', 1000)
        )
        self.response_optimizer = ResponseOptimizer()
        self.active_requests = {}
        self.request_history = []
        
        logger.info("Enhanced API initialized")
    
    async async def process_request(self, request_data: Dict[str, Any], user_id: Optional[str] = None) -> EnhancedResponse:
        """Procesar request con validación y optimización."""
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # Crear request mejorado
            request = EnhancedRequest(
                request_id=request_id,
                data=request_data,
                user_id=user_id,
                validation_level=ValidationLevel.STANDARD,
                response_format=ResponseFormat.JSON
            )
            
            # Verificar rate limit
            rate_limit_result = await self.rate_limiter.check_rate_limit(
                user_id or 'anonymous', request_id
            )
            
            if not rate_limit_result['allowed']:
                return EnhancedResponse(
                    request_id=request_id,
                    success=False,
                    error=f"Rate limit exceeded. Retry after {rate_limit_result['retry_after_seconds']} seconds",
                    processing_time=time.time() - start_time,
                    status=RequestStatus.FAILED
                )
            
            # Validar request
            validation_result = await self.request_validator.validate_request(request)
            
            if not validation_result.is_valid:
                return EnhancedResponse(
                    request_id=request_id,
                    success=False,
                    error=f"Validation failed: {'; '.join(validation_result.errors)}",
                    processing_time=time.time() - start_time,
                    status=RequestStatus.FAILED,
                    warnings=validation_result.warnings,
                    suggestions=validation_result.suggestions
                )
            
            # Procesar request
            result = await self._process_validated_request(request)
            
            # Optimizar respuesta
            optimized_data = await self.response_optimizer.optimize_response(
                result, request.response_format
            )
            
            # Actualizar respuesta con datos optimizados
            result.data = optimized_data
            
            return result
            
        except Exception as e:
            logger.error(f"Request processing failed: {e}")
            return EnhancedResponse(
                request_id=request_id,
                success=False,
                error=f"Internal server error: {str(e)}",
                processing_time=time.time() - start_time,
                status=RequestStatus.FAILED
            )
    
    async async def _process_validated_request(self, request: EnhancedRequest) -> EnhancedResponse:
        """Procesar request validado."""
        start_time = time.time()
        
        # Simular procesamiento con progreso
        await self._simulate_processing_with_progress(request)
        
        # Generar respuesta de ejemplo
        response_data = {
            'message': 'Request processed successfully',
            'request_type': self.request_validator._determine_request_type(request.data),
            'validation_passed': True,
            'processing_details': {
                'stages_completed': ['validation', 'processing', 'optimization'],
                'total_operations': 5,
                'cache_hits': 2
            }
        }
        
        return EnhancedResponse(
            request_id=request.request_id,
            success=True,
            data=response_data,
            processing_time=time.time() - start_time,
            status=RequestStatus.COMPLETED,
            progress_percentage=100.0,
            metadata={
                'user_id': request.user_id,
                'validation_level': request.validation_level.value,
                'response_format': request.response_format.value
            }
        )
    
    async def _simulate_processing_with_progress(self, request: EnhancedRequest):
        """Simular procesamiento con actualizaciones de progreso."""
        stages = [
            ('Initializing', 10),
            ('Validating', 20),
            ('Processing', 50),
            ('Optimizing', 80),
            ('Finalizing', 100)
        ]
        
        for stage_name, progress in stages:
            await asyncio.sleep(0.1)  # Simular trabajo
            logger.info(f"Request {request.request_id}: {stage_name} - {progress}%")
    
    async async def get_request_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Obtener estado de un request."""
        if request_id in self.active_requests:
            return self.active_requests[request_id]
        
        # Buscar en historial
        for request in self.request_history:
            if request['request_id'] == request_id:
                return request
        
        return None
    
    async def get_api_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de la API."""
        return {
            'total_requests': len(self.request_history),
            'active_requests': len(self.active_requests),
            'validation_stats': self.request_validator.get_validation_stats(),
            'rate_limit_stats': self.rate_limiter.get_rate_limit_stats(),
            'optimization_stats': self.response_optimizer.get_optimization_stats(),
            'recent_requests': self.request_history[-10:]  # Últimos 10
        }

# ===== EXPORTS =====

__all__ = [
    'EnhancedAPI',
    'RequestValidator',
    'RateLimiter',
    'ResponseOptimizer',
    'EnhancedRequest',
    'EnhancedResponse',
    'ProgressUpdate',
    'ValidationResult',
    'RequestStatus',
    'ValidationLevel',
    'ResponseFormat'
] 