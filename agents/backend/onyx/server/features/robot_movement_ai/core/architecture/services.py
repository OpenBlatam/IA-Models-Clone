"""
Service Layer
=============

Capa de servicios que implementa la lógica de negocio.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .interfaces import (
    IRouteService,
    IRouteStrategy,
    IRouteRepository,
    IRouteModel,
    IInferenceEngine,
    RouteRequest,
    RouteResponse
)
from .domain import Route, RouteMetrics, RouteStatus
from .events import EventBus, TrainingEvent, InferenceEvent

logger = logging.getLogger(__name__)


class RouteService:
    """
    Servicio principal de routing.
    
    Implementa la lógica de negocio para encontrar rutas.
    """
    
    def __init__(
        self,
        strategy_registry: Dict[str, IRouteStrategy],
        repository: Optional[IRouteRepository] = None,
        event_bus: Optional[EventBus] = None
    ):
        """
        Inicializar servicio.
        
        Args:
            strategy_registry: Registro de estrategias
            repository: Repositorio de rutas (opcional)
            event_bus: Bus de eventos (opcional)
        """
        self.strategy_registry = strategy_registry
        self.repository = repository
        self.event_bus = event_bus
    
    def find_route(self, request: RouteRequest) -> RouteResponse:
        """
        Encontrar ruta.
        
        Args:
            request: Request de ruta
            
        Returns:
            Response con ruta encontrada
        """
        # Obtener estrategia
        strategy = self.strategy_registry.get(request.strategy)
        if not strategy:
            raise ValueError(f"Estrategia '{request.strategy}' no encontrada")
        
        # Emitir evento de inicio
        if self.event_bus:
            self.event_bus.emit("route_requested", {
                "request": request
            })
        
        try:
            # Encontrar ruta (requiere grafo - simplificado aquí)
            # En implementación real, obtener grafo del repositorio o contexto
            response = strategy.find_route(
                start=request.start_node,
                end=request.end_node,
                graph=None,  # Debe obtenerse del contexto
                constraints=request.constraints
            )
            
            # Guardar en repositorio si está disponible
            if self.repository:
                route_id = self.repository.save_route(response)
                response.metadata = response.metadata or {}
                response.metadata["route_id"] = route_id
            
            # Emitir evento de éxito
            if self.event_bus:
                self.event_bus.emit("route_found", {
                    "request": request,
                    "response": response
                })
            
            return response
        
        except Exception as e:
            logger.error(f"Error encontrando ruta: {e}")
            
            # Emitir evento de error
            if self.event_bus:
                self.event_bus.emit("route_error", {
                    "request": request,
                    "error": str(e)
                })
            
            raise
    
    def find_multiple_routes(
        self,
        requests: List[RouteRequest]
    ) -> List[RouteResponse]:
        """
        Encontrar múltiples rutas.
        
        Args:
            requests: Lista de requests
            
        Returns:
            Lista de responses
        """
        return [self.find_route(req) for req in requests]


class ModelService:
    """
    Servicio para gestión de modelos.
    """
    
    def __init__(
        self,
        model_registry: Dict[str, IRouteModel],
        inference_engine: Optional[IInferenceEngine] = None
    ):
        """
        Inicializar servicio.
        
        Args:
            model_registry: Registro de modelos
            inference_engine: Motor de inferencia (opcional)
        """
        self.model_registry = model_registry
        self.inference_engine = inference_engine
    
    def get_model(self, model_name: str) -> Optional[IRouteModel]:
        """Obtener modelo por nombre."""
        return self.model_registry.get(model_name)
    
    def predict(
        self,
        model_name: str,
        features: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Predecir usando modelo.
        
        Args:
            model_name: Nombre del modelo
            features: Features de entrada
            
        Returns:
            Predicciones
        """
        model = self.get_model(model_name)
        if not model:
            raise ValueError(f"Modelo '{model_name}' no encontrado")
        
        if self.inference_engine:
            return self.inference_engine.predict(model, features)
        else:
            return model.predict(features)


class TrainingService:
    """
    Servicio para entrenamiento de modelos.
    """
    
    def __init__(
        self,
        pipeline: Any,  # ITrainingPipeline
        event_bus: Optional[EventBus] = None
    ):
        """
        Inicializar servicio.
        
        Args:
            pipeline: Pipeline de entrenamiento
            event_bus: Bus de eventos (opcional)
        """
        self.pipeline = pipeline
        self.event_bus = event_bus
    
    def train(
        self,
        model: Any,
        train_data: Any,
        val_data: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Entrenar modelo.
        
        Args:
            model: Modelo a entrenar
            train_data: Datos de entrenamiento
            val_data: Datos de validación (opcional)
            config: Configuración (opcional)
            
        Returns:
            Resultados del entrenamiento
        """
        # Emitir evento de inicio
        if self.event_bus:
            self.event_bus.emit("training_started", {
                "model": model,
                "config": config
            })
        
        try:
            results = self.pipeline.train(
                model=model,
                train_data=train_data,
                val_data=val_data,
                config=config
            )
            
            # Emitir evento de finalización
            if self.event_bus:
                self.event_bus.emit("training_completed", {
                    "model": model,
                    "results": results
                })
            
            return results
        
        except Exception as e:
            logger.error(f"Error en entrenamiento: {e}")
            
            # Emitir evento de error
            if self.event_bus:
                self.event_bus.emit("training_error", {
                    "model": model,
                    "error": str(e)
                })
            
            raise


class InferenceService:
    """
    Servicio para inferencia.
    """
    
    def __init__(
        self,
        inference_engine: IInferenceEngine,
        event_bus: Optional[EventBus] = None
    ):
        """
        Inicializar servicio.
        
        Args:
            inference_engine: Motor de inferencia
            event_bus: Bus de eventos (opcional)
        """
        self.inference_engine = inference_engine
        self.event_bus = event_bus
    
    def predict(
        self,
        model: Any,
        input_data: Any,
        batch_size: Optional[int] = None
    ) -> Any:
        """
        Predecir.
        
        Args:
            model: Modelo
            input_data: Datos de entrada
            batch_size: Tamaño de batch (opcional)
            
        Returns:
            Predicciones
        """
        # Emitir evento
        if self.event_bus:
            self.event_bus.emit("inference_started", {
                "model": model,
                "input_shape": getattr(input_data, 'shape', None)
            })
        
        try:
            result = self.inference_engine.predict(
                model=model,
                input_data=input_data,
                batch_size=batch_size
            )
            
            # Emitir evento
            if self.event_bus:
                self.event_bus.emit("inference_completed", {
                    "model": model,
                    "result_shape": getattr(result, 'shape', None)
                })
            
            return result
        
        except Exception as e:
            logger.error(f"Error en inferencia: {e}")
            
            # Emitir evento
            if self.event_bus:
                self.event_bus.emit("inference_error", {
                    "model": model,
                    "error": str(e)
                })
            
            raise

