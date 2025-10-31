"""
Advanced Usage Examples - Ejemplos de Uso Avanzado
================================================

Ejemplos avanzados de uso del sistema ultra modular con patrones complejos.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from ..core.plugin_system.plugin_manager import PluginManager
from ..core.extension_system.extension_manager import ExtensionManager, ExtensionPoint, ExtensionPointType
from ..core.middleware_system.middleware_pipeline import MiddlewarePipeline, MiddlewareType
from ..core.component_system.component_registry import ComponentManager, ComponentScope
from ..core.event_system.event_bus import EventBus, BaseEvent, EventType, EventPriority
from ..core.workflow_system.workflow_engine import WorkflowEngine, WorkflowDefinition, WorkflowStep, StepType

logger = logging.getLogger(__name__)


class AdvancedUsageExample:
    """Ejemplo de uso avanzado del sistema ultra modular."""
    
    def __init__(self):
        self.plugin_manager = None
        self.extension_manager = None
        self.middleware_pipeline = None
        self.component_manager = None
        self.event_bus = None
        self.workflow_engine = None
    
    async def initialize_systems(self):
        """Inicializar todos los sistemas."""
        try:
            logger.info("Initializing advanced usage example...")
            
            # Inicializar gestor de plugins
            self.plugin_manager = PluginManager("plugins/")
            await self.plugin_manager.initialize()
            
            # Inicializar gestor de extensiones
            self.extension_manager = ExtensionManager()
            await self.extension_manager.initialize()
            
            # Inicializar pipeline de middleware
            self.middleware_pipeline = MiddlewarePipeline("advanced")
            await self.middleware_pipeline.initialize()
            
            # Inicializar gestor de componentes
            self.component_manager = ComponentManager()
            await self.component_manager.initialize()
            
            # Inicializar bus de eventos
            self.event_bus = EventBus("advanced")
            await self.event_bus.initialize()
            
            # Inicializar motor de workflows
            self.workflow_engine = WorkflowEngine("advanced")
            await self.workflow_engine.initialize()
            
            logger.info("All systems initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing systems: {e}")
            raise
    
    async def shutdown_systems(self):
        """Cerrar todos los sistemas."""
        try:
            logger.info("Shutting down systems...")
            
            if self.workflow_engine:
                await self.workflow_engine.shutdown()
            
            if self.event_bus:
                await self.event_bus.shutdown()
            
            if self.component_manager:
                await self.component_manager.shutdown()
            
            if self.middleware_pipeline:
                await self.middleware_pipeline.shutdown()
            
            if self.extension_manager:
                await self.extension_manager.shutdown()
            
            if self.plugin_manager:
                await self.plugin_manager.shutdown()
            
            logger.info("All systems shutdown complete")
            
        except Exception as e:
            logger.error(f"Error shutting down systems: {e}")
    
    async def example_1_complex_plugin_system(self):
        """Ejemplo 1: Sistema complejo de plugins con dependencias."""
        logger.info("=== Example 1: Complex Plugin System ===")
        
        try:
            # Crear plugins personalizados
            await self._create_custom_plugins()
            
            # Instalar plugins con dependencias
            await self.plugin_manager.install_plugin("sentiment_analyzer")
            await self.plugin_manager.install_plugin("keyword_extractor")
            await self.plugin_manager.install_plugin("language_detector")
            
            # Activar plugins
            await self.plugin_manager.activate_plugin("sentiment_analyzer")
            await self.plugin_manager.activate_plugin("keyword_extractor")
            await self.plugin_manager.activate_plugin("language_detector")
            
            # Ejecutar hooks de plugins
            content = "This is a great example of advanced plugin usage!"
            results = await self.plugin_manager.execute_plugin_hook("pre_analysis", content)
            
            logger.info(f"Plugin analysis results: {results}")
            
            # Obtener estadísticas de plugins
            stats = await self.plugin_manager.get_execution_stats()
            logger.info(f"Plugin execution stats: {stats}")
            
        except Exception as e:
            logger.error(f"Error in complex plugin system example: {e}")
    
    async def example_2_advanced_extension_system(self):
        """Ejemplo 2: Sistema avanzado de extensiones con puntos personalizados."""
        logger.info("=== Example 2: Advanced Extension System ===")
        
        try:
            # Crear puntos de extensión personalizados
            await self.extension_manager.create_extension_point(
                "advanced_analysis",
                ExtensionPointType.CUSTOM,
                "Análisis avanzado con múltiples etapas",
                priority=200
            )
            
            await self.extension_manager.create_extension_point(
                "result_optimization",
                ExtensionPointType.TRANSFORMATION,
                "Optimización de resultados",
                priority=150
            )
            
            # Crear extensiones personalizadas
            await self._create_custom_extensions()
            
            # Adjuntar extensiones a puntos
            await self.extension_manager.attach_extension_to_point(
                "advanced_analyzer",
                "advanced_analysis"
            )
            
            await self.extension_manager.attach_extension_to_point(
                "result_optimizer",
                "result_optimization"
            )
            
            # Ejecutar extensiones
            from ..core.extension_system.extension_manager import ExtensionContext
            
            data = {"content": "Advanced extension example", "metadata": {"type": "test"}}
            context = ExtensionContext(data)
            
            # Ejecutar en cadena
            result1 = await self.extension_manager.execute_extensions("advanced_analysis", context)
            result2 = await self.extension_manager.execute_extensions("result_optimization", result1)
            
            logger.info(f"Extension chain results: {result2.data}")
            
            # Obtener estadísticas de extensiones
            stats = await self.extension_manager.get_execution_stats()
            logger.info(f"Extension execution stats: {stats}")
            
        except Exception as e:
            logger.error(f"Error in advanced extension system example: {e}")
    
    async def example_3_sophisticated_middleware_pipeline(self):
        """Ejemplo 3: Pipeline sofisticado de middleware con múltiples tipos."""
        logger.info("=== Example 3: Sophisticated Middleware Pipeline ===")
        
        try:
            # Crear middleware personalizado
            await self._create_custom_middleware()
            
            # Agregar middleware al pipeline en orden de prioridad
            await self.middleware_pipeline.add_middleware(
                self._create_auth_middleware(),
                MiddlewareType.AUTHENTICATION,
                priority=100
            )
            
            await self.middleware_pipeline.add_middleware(
                self._create_logging_middleware(),
                MiddlewareType.LOGGING,
                priority=90
            )
            
            await self.middleware_pipeline.add_middleware(
                self._create_validation_middleware(),
                MiddlewareType.VALIDATION,
                priority=80
            )
            
            await self.middleware_pipeline.add_middleware(
                self._create_transformation_middleware(),
                MiddlewareType.TRANSFORMATION,
                priority=70
            )
            
            await self.middleware_pipeline.add_middleware(
                self._create_metrics_middleware(),
                MiddlewareType.METRICS,
                priority=60
            )
            
            # Ejecutar pipeline
            from ..core.middleware_system.middleware_pipeline import MiddlewareContext
            
            request = {
                "user_id": "user123",
                "content": "Middleware pipeline example",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            context = MiddlewareContext(request=request)
            result = await self.middleware_pipeline.execute_pipeline(context)
            
            logger.info(f"Middleware pipeline result: {result.request}")
            logger.info(f"Pipeline metadata: {result.metadata}")
            
            # Obtener estadísticas del pipeline
            stats = await self.middleware_pipeline.get_pipeline_stats()
            logger.info(f"Middleware pipeline stats: {stats}")
            
        except Exception as e:
            logger.error(f"Error in sophisticated middleware pipeline example: {e}")
    
    async def example_4_component_dependency_injection(self):
        """Ejemplo 4: Inyección de dependencias compleja de componentes."""
        logger.info("=== Example 4: Complex Component Dependency Injection ===")
        
        try:
            # Crear componentes personalizados
            await self._create_custom_components()
            
            # Registrar componentes con dependencias
            await self.component_manager.register_component(
                "database_connection",
                self._create_database_connection_class(),
                scope=ComponentScope.SINGLETON
            )
            
            await self.component_manager.register_component(
                "cache_manager",
                self._create_cache_manager_class(),
                scope=ComponentScope.SINGLETON,
                dependencies=["database_connection"]
            )
            
            await self.component_manager.register_component(
                "repository",
                self._create_repository_class(),
                scope=ComponentScope.SINGLETON,
                dependencies=["database_connection", "cache_manager"]
            )
            
            await self.component_manager.register_component(
                "service",
                self._create_service_class(),
                scope=ComponentScope.TRANSIENT,
                dependencies=["repository", "cache_manager"]
            )
            
            # Obtener componente con inyección automática
            service = await self.component_manager.get_component("service")
            
            # Usar el servicio
            result = await service.process_data("test data")
            logger.info(f"Service result: {result}")
            
            # Obtener estadísticas de componentes
            stats = await self.component_manager.get_component_stats()
            logger.info(f"Component stats: {stats}")
            
            # Obtener grafo de dependencias
            dependency_graph = await self.component_manager.get_dependency_graph()
            logger.info(f"Dependency graph: {json.dumps(dependency_graph, indent=2)}")
            
            # Detectar dependencias circulares
            cycles = await self.component_manager.detect_circular_dependencies()
            if cycles:
                logger.warning(f"Circular dependencies detected: {cycles}")
            else:
                logger.info("No circular dependencies detected")
            
        except Exception as e:
            logger.error(f"Error in component dependency injection example: {e}")
    
    async def example_5_event_driven_architecture(self):
        """Ejemplo 5: Arquitectura dirigida por eventos con patrones avanzados."""
        logger.info("=== Example 5: Event-Driven Architecture ===")
        
        try:
            # Crear manejadores de eventos personalizados
            await self._create_custom_event_handlers()
            
            # Suscribirse a eventos
            subscription1 = await self.event_bus.subscribe("content_analysis", self._handle_content_analysis)
            subscription2 = await self.event_bus.subscribe("quality_assessment", self._handle_quality_assessment)
            subscription3 = await self.event_bus.subscribe("notification", self._handle_notification)
            
            # Publicar eventos en cadena
            await self.event_bus.publish("content_analysis", {
                "content": "Event-driven architecture example",
                "user_id": "user123",
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Esperar un poco para que se procesen los eventos
            await asyncio.sleep(1)
            
            # Publicar evento de alta prioridad
            high_priority_event = BaseEvent("critical_alert", {
                "message": "Critical system alert",
                "severity": "high"
            })
            high_priority_event.priority = EventPriority.CRITICAL
            
            await self.event_bus.publish("critical_alert", high_priority_event.data)
            
            # Esperar procesamiento
            await asyncio.sleep(1)
            
            # Obtener estadísticas de eventos
            stats = await self.event_bus.get_event_stats()
            logger.info(f"Event bus stats: {stats}")
            
            # Obtener estadísticas de suscripciones
            sub1_stats = await self.event_bus.get_subscription_stats(subscription1)
            sub2_stats = await self.event_bus.get_subscription_stats(subscription2)
            sub3_stats = await self.event_bus.get_subscription_stats(subscription3)
            
            logger.info(f"Subscription 1 stats: {sub1_stats}")
            logger.info(f"Subscription 2 stats: {sub2_stats}")
            logger.info(f"Subscription 3 stats: {sub3_stats}")
            
            # Cancelar suscripciones
            await self.event_bus.unsubscribe(subscription1)
            await self.event_bus.unsubscribe(subscription2)
            await self.event_bus.unsubscribe(subscription3)
            
        except Exception as e:
            logger.error(f"Error in event-driven architecture example: {e}")
    
    async def example_6_complex_workflow_orchestration(self):
        """Ejemplo 6: Orquestación compleja de workflows con múltiples pasos."""
        logger.info("=== Example 6: Complex Workflow Orchestration ===")
        
        try:
            # Crear definición de workflow compleja
            workflow_def = await self._create_complex_workflow_definition()
            
            # Registrar workflow
            await self.workflow_engine.register_workflow(workflow_def)
            
            # Iniciar múltiples instancias de workflow
            instance_ids = []
            for i in range(3):
                instance_id = await self.workflow_engine.start_workflow(
                    "complex_analysis_workflow",
                    {"batch_id": f"batch_{i}", "content": f"Content {i}"}
                )
                instance_ids.append(instance_id)
            
            # Monitorear progreso de workflows
            for instance_id in instance_ids:
                while True:
                    status = await self.workflow_engine.get_status(instance_id)
                    logger.info(f"Workflow {instance_id} status: {status}")
                    
                    if status in ["completed", "failed", "cancelled"]:
                        break
                    
                    await asyncio.sleep(2)
            
            # Obtener estadísticas de workflows
            stats = await self.workflow_engine.get_workflow_stats()
            logger.info(f"Workflow engine stats: {stats}")
            
            # Obtener instancias por estado
            completed_instances = await self.workflow_engine.list_workflow_instances()
            logger.info(f"All workflow instances: {completed_instances}")
            
        except Exception as e:
            logger.error(f"Error in complex workflow orchestration example: {e}")
    
    async def example_7_integrated_system_usage(self):
        """Ejemplo 7: Uso integrado de todos los sistemas."""
        logger.info("=== Example 7: Integrated System Usage ===")
        
        try:
            # Configurar integración entre sistemas
            
            # 1. Configurar eventos para plugins
            await self.plugin_manager.add_plugin_hook("plugin_installed", self._on_plugin_installed)
            await self.plugin_manager.add_plugin_hook("plugin_activated", self._on_plugin_activated)
            
            # 2. Configurar eventos para extensiones
            await self.extension_manager.add_hook("extension_executed", self._on_extension_executed)
            
            # 3. Configurar eventos para middleware
            await self.middleware_pipeline.add_middleware(
                self._create_event_middleware(),
                MiddlewareType.CUSTOM,
                priority=50
            )
            
            # 4. Configurar eventos para componentes
            await self.component_manager.register_component(
                "event_publisher",
                self._create_event_publisher_class(),
                scope=ComponentScope.SINGLETON
            )
            
            # 5. Ejecutar flujo integrado
            await self._execute_integrated_flow()
            
            # 6. Obtener estadísticas integradas
            integrated_stats = await self._get_integrated_stats()
            logger.info(f"Integrated system stats: {json.dumps(integrated_stats, indent=2)}")
            
        except Exception as e:
            logger.error(f"Error in integrated system usage example: {e}")
    
    # Métodos auxiliares para crear componentes personalizados
    
    async def _create_custom_plugins(self):
        """Crear plugins personalizados."""
        # Implementación de plugins personalizados
        pass
    
    async def _create_custom_extensions(self):
        """Crear extensiones personalizadas."""
        # Implementación de extensiones personalizadas
        pass
    
    async def _create_custom_middleware(self):
        """Crear middleware personalizado."""
        # Implementación de middleware personalizado
        pass
    
    async def _create_custom_components(self):
        """Crear componentes personalizados."""
        # Implementación de componentes personalizados
        pass
    
    async def _create_custom_event_handlers(self):
        """Crear manejadores de eventos personalizados."""
        # Implementación de manejadores de eventos personalizados
        pass
    
    async def _create_complex_workflow_definition(self) -> WorkflowDefinition:
        """Crear definición de workflow compleja."""
        # Implementación de workflow complejo
        pass
    
    # Métodos auxiliares para crear clases
    
    def _create_database_connection_class(self):
        """Crear clase de conexión a base de datos."""
        class DatabaseConnection:
            def __init__(self):
                self.connected = True
            
            async def connect(self):
                self.connected = True
            
            async def disconnect(self):
                self.connected = False
        
        return DatabaseConnection
    
    def _create_cache_manager_class(self):
        """Crear clase de gestor de caché."""
        class CacheManager:
            def __init__(self, db_connection):
                self.db_connection = db_connection
                self.cache = {}
            
            async def get(self, key):
                return self.cache.get(key)
            
            async def set(self, key, value):
                self.cache[key] = value
        
        return CacheManager
    
    def _create_repository_class(self):
        """Crear clase de repositorio."""
        class Repository:
            def __init__(self, db_connection, cache_manager):
                self.db_connection = db_connection
                self.cache_manager = cache_manager
            
            async def save(self, data):
                return f"Saved: {data}"
            
            async def find(self, id):
                return f"Found: {id}"
        
        return Repository
    
    def _create_service_class(self):
        """Crear clase de servicio."""
        class Service:
            def __init__(self, repository, cache_manager):
                self.repository = repository
                self.cache_manager = cache_manager
            
            async def process_data(self, data):
                result = await self.repository.save(data)
                await self.cache_manager.set("last_result", result)
                return result
        
        return Service
    
    def _create_event_publisher_class(self):
        """Crear clase de publicador de eventos."""
        class EventPublisher:
            def __init__(self):
                self.published_events = []
            
            async def publish(self, event_type, data):
                self.published_events.append({"type": event_type, "data": data})
        
        return EventPublisher
    
    # Métodos auxiliares para middleware
    
    def _create_auth_middleware(self):
        """Crear middleware de autenticación."""
        from ..core.interfaces.base_interfaces import IMiddleware
        
        class AuthMiddleware(IMiddleware):
            async def process_request(self, request):
                if isinstance(request, dict) and "user_id" in request:
                    request["authenticated"] = True
                return request
            
            async def process_response(self, response):
                return response
        
        return AuthMiddleware()
    
    def _create_logging_middleware(self):
        """Crear middleware de logging."""
        from ..core.interfaces.base_interfaces import IMiddleware
        
        class LoggingMiddleware(IMiddleware):
            async def process_request(self, request):
                logger.info(f"Processing request: {request}")
                return request
            
            async def process_response(self, response):
                logger.info(f"Processing response: {response}")
                return response
        
        return LoggingMiddleware()
    
    def _create_validation_middleware(self):
        """Crear middleware de validación."""
        from ..core.interfaces.base_interfaces import IMiddleware
        
        class ValidationMiddleware(IMiddleware):
            async def process_request(self, request):
                if isinstance(request, dict) and "content" in request:
                    if len(request["content"]) < 10:
                        raise ValueError("Content too short")
                return request
            
            async def process_response(self, response):
                return response
        
        return ValidationMiddleware()
    
    def _create_transformation_middleware(self):
        """Crear middleware de transformación."""
        from ..core.interfaces.base_interfaces import IMiddleware
        
        class TransformationMiddleware(IMiddleware):
            async def process_request(self, request):
                if isinstance(request, dict) and "content" in request:
                    request["content"] = request["content"].upper()
                return request
            
            async def process_response(self, response):
                return response
        
        return TransformationMiddleware()
    
    def _create_metrics_middleware(self):
        """Crear middleware de métricas."""
        from ..core.interfaces.base_interfaces import IMiddleware
        
        class MetricsMiddleware(IMiddleware):
            def __init__(self):
                self.request_count = 0
            
            async def process_request(self, request):
                self.request_count += 1
                if isinstance(request, dict):
                    request["request_count"] = self.request_count
                return request
            
            async def process_response(self, response):
                return response
        
        return MetricsMiddleware()
    
    def _create_event_middleware(self):
        """Crear middleware de eventos."""
        from ..core.interfaces.base_interfaces import IMiddleware
        
        class EventMiddleware(IMiddleware):
            async def process_request(self, request):
                # Publicar evento de request
                if self.event_bus:
                    await self.event_bus.publish("request_received", request)
                return request
            
            async def process_response(self, response):
                # Publicar evento de response
                if self.event_bus:
                    await self.event_bus.publish("response_sent", response)
                return response
        
        return EventMiddleware()
    
    # Métodos auxiliares para manejadores de eventos
    
    async def _handle_content_analysis(self, event):
        """Manejar evento de análisis de contenido."""
        logger.info(f"Handling content analysis event: {event.data}")
        return {"processed": True, "event_id": event.event_id}
    
    async def _handle_quality_assessment(self, event):
        """Manejar evento de evaluación de calidad."""
        logger.info(f"Handling quality assessment event: {event.data}")
        return {"quality_score": 0.85, "event_id": event.event_id}
    
    async def _handle_notification(self, event):
        """Manejar evento de notificación."""
        logger.info(f"Handling notification event: {event.data}")
        return {"notified": True, "event_id": event.event_id}
    
    # Métodos auxiliares para hooks
    
    async def _on_plugin_installed(self, plugin_name):
        """Hook para plugin instalado."""
        logger.info(f"Plugin installed: {plugin_name}")
        if self.event_bus:
            await self.event_bus.publish("plugin_installed", {"plugin_name": plugin_name})
    
    async def _on_plugin_activated(self, plugin_name):
        """Hook para plugin activado."""
        logger.info(f"Plugin activated: {plugin_name}")
        if self.event_bus:
            await self.event_bus.publish("plugin_activated", {"plugin_name": plugin_name})
    
    async def _on_extension_executed(self, extension_name, result):
        """Hook para extensión ejecutada."""
        logger.info(f"Extension executed: {extension_name}")
        if self.event_bus:
            await self.event_bus.publish("extension_executed", {
                "extension_name": extension_name,
                "result": result
            })
    
    # Métodos auxiliares para flujo integrado
    
    async def _execute_integrated_flow(self):
        """Ejecutar flujo integrado."""
        logger.info("Executing integrated flow...")
        
        # 1. Ejecutar middleware pipeline
        from ..core.middleware_system.middleware_pipeline import MiddlewareContext
        context = MiddlewareContext(request={"content": "Integrated flow test"})
        middleware_result = await self.middleware_pipeline.execute_pipeline(context)
        
        # 2. Ejecutar extensiones
        from ..core.extension_system.extension_manager import ExtensionContext
        ext_context = ExtensionContext(middleware_result.request)
        extension_result = await self.extension_manager.execute_extensions("pre_analysis", ext_context)
        
        # 3. Ejecutar plugins
        plugin_results = await self.plugin_manager.execute_plugin_hook("pre_analysis", extension_result.data)
        
        # 4. Publicar eventos
        await self.event_bus.publish("integrated_flow_completed", {
            "middleware_result": middleware_result.request,
            "extension_result": extension_result.data,
            "plugin_results": plugin_results
        })
        
        logger.info("Integrated flow completed successfully")
    
    async def _get_integrated_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas integradas."""
        stats = {
            "timestamp": datetime.utcnow().isoformat(),
            "systems": {}
        }
        
        # Estadísticas de plugins
        if self.plugin_manager:
            stats["systems"]["plugins"] = await self.plugin_manager.get_execution_stats()
        
        # Estadísticas de extensiones
        if self.extension_manager:
            stats["systems"]["extensions"] = await self.extension_manager.get_execution_stats()
        
        # Estadísticas de middleware
        if self.middleware_pipeline:
            stats["systems"]["middleware"] = await self.middleware_pipeline.get_pipeline_stats()
        
        # Estadísticas de componentes
        if self.component_manager:
            stats["systems"]["components"] = await self.component_manager.get_component_stats()
        
        # Estadísticas de eventos
        if self.event_bus:
            stats["systems"]["events"] = await self.event_bus.get_event_stats()
        
        # Estadísticas de workflows
        if self.workflow_engine:
            stats["systems"]["workflows"] = await self.workflow_engine.get_workflow_stats()
        
        return stats


async def run_advanced_examples():
    """Ejecutar todos los ejemplos avanzados."""
    example = AdvancedUsageExample()
    
    try:
        # Inicializar sistemas
        await example.initialize_systems()
        
        # Ejecutar ejemplos
        await example.example_1_complex_plugin_system()
        await example.example_2_advanced_extension_system()
        await example.example_3_sophisticated_middleware_pipeline()
        await example.example_4_component_dependency_injection()
        await example.example_5_event_driven_architecture()
        await example.example_6_complex_workflow_orchestration()
        await example.example_7_integrated_system_usage()
        
        logger.info("All advanced examples completed successfully!")
        
    except Exception as e:
        logger.error(f"Error running advanced examples: {e}")
    
    finally:
        # Cerrar sistemas
        await example.shutdown_systems()


if __name__ == "__main__":
    asyncio.run(run_advanced_examples())




