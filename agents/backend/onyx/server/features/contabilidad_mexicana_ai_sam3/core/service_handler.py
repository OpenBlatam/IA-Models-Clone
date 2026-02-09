"""
Service Handler for Contador SAM3 Agent
========================================

Refactored with:
- ServiceHandlerRegistry for extensible handlers
- ServiceConfig dataclass for configuration
- ServiceResult dataclass for typed responses
- Pipeline pattern for request processing
"""

import logging
from typing import Dict, Any, Optional, List, Callable, Awaitable, Protocol
from datetime import datetime
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum

from .helpers import create_message
from .prompt_builder import PromptBuilder
from ..infrastructure.openrouter_client import OpenRouterClient
from ..infrastructure.truthgpt_client import TruthGPTClient

logger = logging.getLogger(__name__)


class ServiceType(Enum):
    """Service types."""
    CALCULAR_IMPUESTOS = "calcular_impuestos"
    ASESORIA_FISCAL = "asesoria_fiscal"
    GUIA_FISCAL = "guia_fiscal"
    TRAMITE_SAT = "tramite_sat"
    AYUDA_DECLARACION = "ayuda_declaracion"


@dataclass
class ServiceConfig:
    """Configuration for a service request."""
    service_type: ServiceType
    system_prompt_key: str
    response_key: str
    temperature: float = 0.3
    max_tokens: int = 4000
    include_timestamp: bool = False
    timestamp_key: str = "tiempo_calculo"
    
    @classmethod
    def for_calcular_impuestos(cls) -> "ServiceConfig":
        return cls(
            service_type=ServiceType.CALCULAR_IMPUESTOS,
            system_prompt_key="calculo_impuestos",
            response_key="resultado",
            temperature=0.3,
            include_timestamp=True,
            timestamp_key="tiempo_calculo",
        )
    
    @classmethod
    def for_asesoria_fiscal(cls) -> "ServiceConfig":
        return cls(
            service_type=ServiceType.ASESORIA_FISCAL,
            system_prompt_key="asesoria_fiscal",
            response_key="asesoria",
            temperature=0.5,
        )
    
    @classmethod
    def for_guia_fiscal(cls) -> "ServiceConfig":
        return cls(
            service_type=ServiceType.GUIA_FISCAL,
            system_prompt_key="guias_fiscales",
            response_key="guia",
            temperature=0.5,
            include_timestamp=True,
            timestamp_key="tiempo_generacion",
        )
    
    @classmethod
    def for_tramite_sat(cls) -> "ServiceConfig":
        return cls(
            service_type=ServiceType.TRAMITE_SAT,
            system_prompt_key="tramites_sat",
            response_key="informacion",
            temperature=0.3,
        )
    
    @classmethod
    def for_ayuda_declaracion(cls) -> "ServiceConfig":
        return cls(
            service_type=ServiceType.AYUDA_DECLARACION,
            system_prompt_key="declaraciones",
            response_key="guia",
            temperature=0.3,
        )


@dataclass
class ServiceResult:
    """Result from a service request."""
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    tokens_used: int = 0
    model: str = ""
    timestamp: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {**self.data}
        result["tokens_used"] = self.tokens_used
        result["model"] = self.model
        if self.timestamp:
            result["timestamp"] = self.timestamp
        if self.error:
            result["error"] = self.error
        return result
    
    @classmethod
    def from_response(
        cls,
        response: Dict[str, Any],
        config: ServiceConfig
    ) -> "ServiceResult":
        """Create from API response."""
        data = {
            config.response_key: response.get("response", ""),
        }
        
        return cls(
            success=True,
            data=data,
            tokens_used=response.get("tokens_used", 0),
            model=response.get("model", ""),
            timestamp=datetime.now().isoformat() if config.include_timestamp else None,
        )
    
    @classmethod
    def error_result(cls, error: str) -> "ServiceResult":
        """Create error result."""
        return cls(success=False, error=error)


class RequestMiddleware(Protocol):
    """Protocol for request middleware."""
    
    async def process(
        self, 
        prompt: str, 
        context: Dict[str, Any]
    ) -> str:
        """Process the request and return modified prompt."""
        ...


class TruthGPTMiddleware:
    """Middleware that optimizes prompts with TruthGPT."""
    
    def __init__(self, truthgpt_client: TruthGPTClient):
        self.client = truthgpt_client
    
    async def process(self, prompt: str, context: Dict[str, Any]) -> str:
        """Optimize prompt with TruthGPT."""
        return await self.client.optimize_query(prompt)


class BaseServiceHandler(ABC):
    """Abstract base class for service handlers."""
    
    @property
    @abstractmethod
    def service_type(self) -> ServiceType:
        """The service type this handler handles."""
        pass
    
    @property
    @abstractmethod
    def config(self) -> ServiceConfig:
        """Get the service configuration."""
        pass
    
    @abstractmethod
    def build_prompt(self, parameters: Dict[str, Any]) -> str:
        """Build the prompt for this service."""
        pass
    
    async def handle(
        self,
        parameters: Dict[str, Any],
        handler: "ServiceHandler"
    ) -> ServiceResult:
        """Handle the service request."""
        prompt = self.build_prompt(parameters)
        return await handler.execute_request(prompt, self.config)


class CalcularImpuestosHandler(BaseServiceHandler):
    """Handler for tax calculation service."""
    
    service_type = ServiceType.CALCULAR_IMPUESTOS
    config = ServiceConfig.for_calcular_impuestos()
    
    def build_prompt(self, parameters: Dict[str, Any]) -> str:
        return PromptBuilder.build_calculation_prompt(
            regimen=parameters.get("regimen"),
            tipo_impuesto=parameters.get("tipo_impuesto"),
            datos=parameters.get("datos", {}),
        )


class AsesoriaFiscalHandler(BaseServiceHandler):
    """Handler for fiscal advice service."""
    
    service_type = ServiceType.ASESORIA_FISCAL
    config = ServiceConfig.for_asesoria_fiscal()
    
    def build_prompt(self, parameters: Dict[str, Any]) -> str:
        return PromptBuilder.build_advice_prompt(
            pregunta=parameters.get("pregunta"),
            contexto=parameters.get("contexto"),
        )


class GuiaFiscalHandler(BaseServiceHandler):
    """Handler for fiscal guide service."""
    
    service_type = ServiceType.GUIA_FISCAL
    config = ServiceConfig.for_guia_fiscal()
    
    def build_prompt(self, parameters: Dict[str, Any]) -> str:
        return PromptBuilder.build_guide_prompt(
            tema=parameters.get("tema"),
            nivel_detalle=parameters.get("nivel_detalle", "completo"),
        )


class TramiteSATHandler(BaseServiceHandler):
    """Handler for SAT procedure service."""
    
    service_type = ServiceType.TRAMITE_SAT
    config = ServiceConfig.for_tramite_sat()
    
    def build_prompt(self, parameters: Dict[str, Any]) -> str:
        return PromptBuilder.build_procedure_prompt(
            tipo_tramite=parameters.get("tipo_tramite"),
            detalles=parameters.get("detalles"),
        )


class AyudaDeclaracionHandler(BaseServiceHandler):
    """Handler for declaration assistance service."""
    
    service_type = ServiceType.AYUDA_DECLARACION
    config = ServiceConfig.for_ayuda_declaracion()
    
    def build_prompt(self, parameters: Dict[str, Any]) -> str:
        return PromptBuilder.build_declaration_prompt(
            tipo_declaracion=parameters.get("tipo_declaracion"),
            periodo=parameters.get("periodo"),
            datos=parameters.get("datos"),
        )


class ServiceHandlerRegistry:
    """Registry for service handlers."""
    
    _handlers: Dict[ServiceType, BaseServiceHandler] = {}
    
    @classmethod
    def register(cls, handler: BaseServiceHandler):
        """Register a handler."""
        cls._handlers[handler.service_type] = handler
    
    @classmethod
    def get(cls, service_type: ServiceType) -> Optional[BaseServiceHandler]:
        """Get handler for service type."""
        return cls._handlers.get(service_type)
    
    @classmethod
    def register_defaults(cls):
        """Register default handlers."""
        cls.register(CalcularImpuestosHandler())
        cls.register(AsesoriaFiscalHandler())
        cls.register(GuiaFiscalHandler())
        cls.register(TramiteSATHandler())
        cls.register(AyudaDeclaracionHandler())


# Initialize default handlers
ServiceHandlerRegistry.register_defaults()


class ServiceHandler:
    """
    Handles service requests with common patterns.
    
    Refactored with:
    - Registry pattern for handlers
    - Middleware pipeline for request processing
    - ServiceConfig for typed configuration
    - ServiceResult for typed responses
    """
    
    def __init__(
        self,
        openrouter_client: OpenRouterClient,
        truthgpt_client: TruthGPTClient,
        system_prompts: Dict[str, str],
        config: Any
    ):
        self.openrouter_client = openrouter_client
        self.truthgpt_client = truthgpt_client
        self.system_prompts = system_prompts
        self.config = config
        
        # Middleware pipeline
        self._middleware: List[RequestMiddleware] = [
            TruthGPTMiddleware(truthgpt_client),
        ]
    
    def add_middleware(self, middleware: RequestMiddleware):
        """Add middleware to pipeline."""
        self._middleware.append(middleware)
    
    async def _apply_middleware(self, prompt: str, context: Dict[str, Any]) -> str:
        """Apply all middleware to prompt."""
        current_prompt = prompt
        for middleware in self._middleware:
            current_prompt = await middleware.process(current_prompt, context)
        return current_prompt
    
    async def execute_request(
        self,
        prompt: str,
        config: ServiceConfig,
        context: Optional[Dict[str, Any]] = None
    ) -> ServiceResult:
        """Execute a service request."""
        try:
            # Apply middleware
            optimized_prompt = await self._apply_middleware(
                prompt, context or {}
            )
            
            # Build messages
            messages = [
                create_message("system", self.system_prompts[config.system_prompt_key]),
                create_message("user", optimized_prompt),
            ]
            
            # Call OpenRouter
            response = await self.openrouter_client.chat_completion(
                model=self.config.openrouter.model,
                messages=messages,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
            
            return ServiceResult.from_response(response, config)
            
        except Exception as e:
            logger.error(f"Service request failed: {e}")
            return ServiceResult.error_result(str(e))
    
    async def handle(
        self,
        service_type: ServiceType,
        parameters: Dict[str, Any]
    ) -> ServiceResult:
        """Handle a service request using registered handler."""
        handler = ServiceHandlerRegistry.get(service_type)
        if not handler:
            return ServiceResult.error_result(f"Unknown service type: {service_type}")
        
        return await handler.handle(parameters, self)
    
    # === Convenience Methods (Backward Compatible) ===
    
    async def handle_service_request(
        self,
        service_type: str,
        prompt_builder_method: callable,
        system_prompt_key: str,
        response_key: str,
        temperature: float = 0.3,
        include_timestamp: bool = False,
        timestamp_key: str = "tiempo_calculo",
        *prompt_args,
        **prompt_kwargs
    ) -> Dict[str, Any]:
        """Handle a service request (backward compatible)."""
        prompt = prompt_builder_method(*prompt_args, **prompt_kwargs)
        optimized_prompt = await self.truthgpt_client.optimize_query(prompt)
        
        messages = [
            create_message("system", self.system_prompts[system_prompt_key]),
            create_message("user", optimized_prompt),
        ]
        
        response = await self.openrouter_client.chat_completion(
            model=self.config.openrouter.model,
            messages=messages,
            temperature=temperature,
            max_tokens=4000,
        )
        
        result = {
            response_key: response["response"],
            "tokens_used": response["tokens_used"],
            "model": response["model"],
        }
        
        if include_timestamp:
            result[timestamp_key] = datetime.now().isoformat()
        
        return result
    
    async def handle_calcular_impuestos(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tax calculation request."""
        result = await self.handle(ServiceType.CALCULAR_IMPUESTOS, parameters)
        return result.to_dict()
    
    async def handle_asesoria_fiscal(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle fiscal advice request."""
        result = await self.handle(ServiceType.ASESORIA_FISCAL, parameters)
        return result.to_dict()
    
    async def handle_guia_fiscal(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle fiscal guide request."""
        result = await self.handle(ServiceType.GUIA_FISCAL, parameters)
        return result.to_dict()
    
    async def handle_tramite_sat(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle SAT procedure request."""
        result = await self.handle(ServiceType.TRAMITE_SAT, parameters)
        return result.to_dict()
    
    async def handle_ayuda_declaracion(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle declaration assistance request."""
        result = await self.handle(ServiceType.AYUDA_DECLARACION, parameters)
        return result.to_dict()
