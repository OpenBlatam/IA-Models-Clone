"""
History Microservice - Microservicio de Historial
===============================================

Microservicio especializado en la gestión de historial de IA
con arquitectura hexagonal y event sourcing.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
import asyncio
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
import uuid

from ..core.domain.aggregates import HistoryAggregate
from ..core.domain.value_objects import ContentId, ModelType, QualityScore
from ..core.domain.events import HistoryCreatedEvent, HistoryUpdatedEvent, HistoryDeletedEvent
from ..core.application.commands import CreateHistoryCommand, UpdateHistoryCommand, DeleteHistoryCommand
from ..core.application.queries import GetHistoryQuery, ListHistoryQuery
from ..core.infrastructure.event_store import EventStore
from ..core.infrastructure.message_bus import MessageBus
from ..core.infrastructure.plugin_registry import PluginRegistry
from ..monitoring.metrics import MetricsCollector
from ..monitoring.tracing import DistributedTracer
from ..resilience.circuit_breaker import CircuitBreaker
from ..resilience.retry import RetryPolicy


class HistoryService:
    """
    Microservicio de historial con arquitectura hexagonal.
    
    Maneja la gestión completa del historial de IA con:
    - Event sourcing
    - CQRS
    - Plugin system
    - Circuit breakers
    - Distributed tracing
    - Metrics collection
    """
    
    def __init__(
        self,
        event_store: EventStore,
        message_bus: MessageBus,
        plugin_registry: PluginRegistry,
        metrics_collector: MetricsCollector,
        tracer: DistributedTracer,
        circuit_breaker: CircuitBreaker,
        retry_policy: RetryPolicy
    ):
        self.event_store = event_store
        self.message_bus = message_bus
        self.plugin_registry = plugin_registry
        self.metrics_collector = metrics_collector
        self.tracer = tracer
        self.circuit_breaker = circuit_breaker
        self.retry_policy = retry_policy
        
        # Configurar FastAPI
        self.app = FastAPI(
            title="History Microservice",
            description="Microservicio especializado en gestión de historial de IA",
            version="3.0.0"
        )
        self._setup_routes()
    
    def _setup_routes(self):
        """Configurar rutas del microservicio."""
        
        @self.app.post("/history")
        async def create_history(request: CreateHistoryRequest):
            """Crear nueva entrada de historial."""
            with self.tracer.start_span("create_history"):
                try:
                    command = CreateHistoryCommand(
                        command_id=str(uuid.uuid4()),
                        model_type=request.model_type,
                        content=request.content,
                        user_id=request.user_id,
                        session_id=request.session_id,
                        metadata=request.metadata,
                        assess_quality=request.assess_quality,
                        analyze_content=request.analyze_content
                    )
                    
                    result = await self._handle_create_history_command(command)
                    
                    # Métricas
                    self.metrics_collector.increment_counter("history_created")
                    
                    return result
                    
                except Exception as e:
                    self.metrics_collector.increment_counter("history_creation_failed")
                    raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/history/{entry_id}")
        async def get_history(entry_id: str):
            """Obtener entrada de historial."""
            with self.tracer.start_span("get_history"):
                try:
                    query = GetHistoryQuery(entry_id=ContentId(entry_id))
                    result = await self._handle_get_history_query(query)
                    
                    if not result:
                        raise HTTPException(status_code=404, detail="History entry not found")
                    
                    return result
                    
                except Exception as e:
                    raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/history")
        async def list_history(
            user_id: Optional[str] = None,
            model_type: Optional[str] = None,
            limit: int = 100,
            offset: int = 0
        ):
            """Listar entradas de historial."""
            with self.tracer.start_span("list_history"):
                try:
                    query = ListHistoryQuery(
                        user_id=user_id,
                        model_type=ModelType(model_type) if model_type else None,
                        limit=limit,
                        offset=offset
                    )
                    
                    result = await self._handle_list_history_query(query)
                    return result
                    
                except Exception as e:
                    raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.put("/history/{entry_id}")
        async def update_history(entry_id: str, request: UpdateHistoryRequest):
            """Actualizar entrada de historial."""
            with self.tracer.start_span("update_history"):
                try:
                    command = UpdateHistoryCommand(
                        command_id=str(uuid.uuid4()),
                        entry_id=ContentId(entry_id),
                        content=request.content,
                        metadata=request.metadata,
                        assess_quality=request.assess_quality,
                        analyze_content=request.analyze_content
                    )
                    
                    result = await self._handle_update_history_command(command)
                    
                    # Métricas
                    self.metrics_collector.increment_counter("history_updated")
                    
                    return result
                    
                except Exception as e:
                    self.metrics_collector.increment_counter("history_update_failed")
                    raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.delete("/history/{entry_id}")
        async def delete_history(entry_id: str):
            """Eliminar entrada de historial."""
            with self.tracer.start_span("delete_history"):
                try:
                    command = DeleteHistoryCommand(
                        command_id=str(uuid.uuid4()),
                        entry_id=ContentId(entry_id)
                    )
                    
                    await self._handle_delete_history_command(command)
                    
                    # Métricas
                    self.metrics_collector.increment_counter("history_deleted")
                    
                    return {"message": "History entry deleted successfully"}
                    
                except Exception as e:
                    self.metrics_collector.increment_counter("history_deletion_failed")
                    raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/health")
        async def health_check():
            """Health check del microservicio."""
            return {
                "status": "healthy",
                "service": "history",
                "timestamp": datetime.utcnow().isoformat(),
                "version": "3.0.0"
            }
    
    async def _handle_create_history_command(self, command: CreateHistoryCommand) -> Dict[str, Any]:
        """Manejar comando de creación de historial."""
        try:
            # Crear agregado
            aggregate = HistoryAggregate.create(
                model_type=command.model_type,
                content=command.content,
                user_id=command.user_id,
                session_id=command.session_id,
                metadata=command.metadata
            )
            
            # Guardar eventos
            events = aggregate.get_uncommitted_events()
            for event in events:
                await self.event_store.save_event(event)
            
            # Marcar eventos como confirmados
            aggregate.mark_events_as_committed()
            
            # Publicar eventos
            for event in events:
                await self.message_bus.publish(event)
            
            # Ejecutar plugins
            await self._execute_plugins("history_created", {
                "aggregate": aggregate,
                "command": command
            })
            
            return {
                "id": str(aggregate.id),
                "model_type": aggregate.model_type,
                "content": aggregate.content,
                "created_at": aggregate.created_at.isoformat(),
                "user_id": aggregate.user_id,
                "session_id": aggregate.session_id
            }
            
        except Exception as e:
            raise Exception(f"Failed to create history entry: {e}")
    
    async def _handle_get_history_query(self, query: GetHistoryQuery) -> Optional[Dict[str, Any]]:
        """Manejar query de obtención de historial."""
        try:
            # Obtener eventos del agregado
            events = await self.event_store.get_events_for_aggregate(query.entry_id)
            
            if not events:
                return None
            
            # Reconstruir agregado desde eventos
            aggregate = self._reconstruct_aggregate_from_events(events)
            
            return {
                "id": str(aggregate.id),
                "model_type": aggregate.model_type,
                "content": aggregate.content,
                "metadata": aggregate.metadata,
                "quality_score": aggregate.quality_score.to_dict() if aggregate.quality_score else None,
                "content_metrics": aggregate.content_metrics.to_dict() if aggregate.content_metrics else None,
                "created_at": aggregate.created_at.isoformat(),
                "updated_at": aggregate.updated_at.isoformat(),
                "user_id": aggregate.user_id,
                "session_id": aggregate.session_id
            }
            
        except Exception as e:
            raise Exception(f"Failed to get history entry: {e}")
    
    async def _handle_list_history_query(self, query: ListHistoryQuery) -> List[Dict[str, Any]]:
        """Manejar query de listado de historial."""
        try:
            # Obtener IDs de agregados con filtros
            aggregate_ids = await self.event_store.get_aggregate_ids_with_filters(
                user_id=query.user_id,
                model_type=query.model_type,
                limit=query.limit,
                offset=query.offset
            )
            
            results = []
            for aggregate_id in aggregate_ids:
                events = await self.event_store.get_events_for_aggregate(ContentId(aggregate_id))
                if events:
                    aggregate = self._reconstruct_aggregate_from_events(events)
                    results.append({
                        "id": str(aggregate.id),
                        "model_type": aggregate.model_type,
                        "content": aggregate.content[:100] + "..." if len(aggregate.content) > 100 else aggregate.content,
                        "created_at": aggregate.created_at.isoformat(),
                        "user_id": aggregate.user_id
                    })
            
            return results
            
        except Exception as e:
            raise Exception(f"Failed to list history entries: {e}")
    
    async def _handle_update_history_command(self, command: UpdateHistoryCommand) -> Dict[str, Any]:
        """Manejar comando de actualización de historial."""
        try:
            # Obtener agregado existente
            events = await self.event_store.get_events_for_aggregate(command.entry_id)
            if not events:
                raise Exception("History entry not found")
            
            aggregate = self._reconstruct_aggregate_from_events(events)
            
            # Actualizar agregado
            if command.content:
                aggregate.update_content(command.content)
            
            if command.metadata:
                aggregate.metadata.update(command.metadata)
            
            # Guardar nuevos eventos
            new_events = aggregate.get_uncommitted_events()
            for event in new_events:
                await self.event_store.save_event(event)
            
            # Marcar eventos como confirmados
            aggregate.mark_events_as_committed()
            
            # Publicar eventos
            for event in new_events:
                await self.message_bus.publish(event)
            
            # Ejecutar plugins
            await self._execute_plugins("history_updated", {
                "aggregate": aggregate,
                "command": command
            })
            
            return {
                "id": str(aggregate.id),
                "content": aggregate.content,
                "updated_at": aggregate.updated_at.isoformat()
            }
            
        except Exception as e:
            raise Exception(f"Failed to update history entry: {e}")
    
    async def _handle_delete_history_command(self, command: DeleteHistoryCommand) -> None:
        """Manejar comando de eliminación de historial."""
        try:
            # Obtener agregado existente
            events = await self.event_store.get_events_for_aggregate(command.entry_id)
            if not events:
                raise Exception("History entry not found")
            
            aggregate = self._reconstruct_aggregate_from_events(events)
            
            # Marcar como eliminado
            aggregate.delete()
            
            # Guardar evento de eliminación
            events = aggregate.get_uncommitted_events()
            for event in events:
                await self.event_store.save_event(event)
            
            # Marcar eventos como confirmados
            aggregate.mark_events_as_committed()
            
            # Publicar eventos
            for event in events:
                await self.message_bus.publish(event)
            
            # Ejecutar plugins
            await self._execute_plugins("history_deleted", {
                "aggregate": aggregate,
                "command": command
            })
            
        except Exception as e:
            raise Exception(f"Failed to delete history entry: {e}")
    
    def _reconstruct_aggregate_from_events(self, events: List) -> HistoryAggregate:
        """Reconstruir agregado desde eventos."""
        # Implementar reconstrucción desde eventos
        # Por simplicidad, asumimos que el primer evento es HistoryCreatedEvent
        if not events:
            raise Exception("No events found for aggregate")
        
        first_event = events[0]
        if isinstance(first_event, HistoryCreatedEvent):
            aggregate = HistoryAggregate(
                id=ContentId(first_event.aggregate_id),
                model_type=first_event.model_type,
                content=first_event.content,
                user_id=first_event.user_id,
                session_id=first_event.session_id,
                metadata=first_event.metadata,
                created_at=first_event.occurred_at
            )
            
            # Aplicar eventos restantes
            for event in events[1:]:
                if isinstance(event, HistoryUpdatedEvent):
                    aggregate.content = event.new_content
                    aggregate.updated_at = event.updated_at
                # Agregar más tipos de eventos según sea necesario
            
            return aggregate
        else:
            raise Exception("Invalid event sequence for aggregate reconstruction")
    
    async def _execute_plugins(self, event_type: str, context: Dict[str, Any]) -> None:
        """Ejecutar plugins registrados para el tipo de evento."""
        try:
            plugins = self.plugin_registry.get_plugins_for_event(event_type)
            for plugin in plugins:
                await plugin.execute(context)
        except Exception as e:
            # Log error but don't fail the operation
            print(f"Plugin execution failed: {e}")


# DTOs para la API
class CreateHistoryRequest(BaseModel):
    model_type: ModelType
    content: str = Field(..., min_length=1, max_length=50000)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    assess_quality: bool = False
    analyze_content: bool = True


class UpdateHistoryRequest(BaseModel):
    content: Optional[str] = Field(None, min_length=1, max_length=50000)
    metadata: Optional[Dict[str, Any]] = None
    assess_quality: bool = False
    analyze_content: bool = False




