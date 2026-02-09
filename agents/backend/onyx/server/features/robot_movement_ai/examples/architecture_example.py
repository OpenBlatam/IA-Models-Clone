"""
Architecture Example
===================

Ejemplo completo de uso de la nueva arquitectura.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from core.architecture import (
    # Domain
    RouteRequest,
    RouteResponse,
    RouteMetrics,
    # Services
    RouteService,
    # Repositories
    RouteRepository,
    # Factories
    RouteModelFactory,
    StrategyFactory,
    # Builders
    RouteServiceBuilder,
    # Events
    EventBus,
    EventHandler,
    # Plugins
    PluginManager,
    RouteStrategyPlugin,
    # DI
    Container,
    register_service,
    resolve_service
)


# ============================================================================
# Ejemplo 1: Uso Básico con Builder Pattern
# ============================================================================

def example_builder_pattern():
    """Ejemplo de uso del builder pattern."""
    logger.info("=== Ejemplo 1: Builder Pattern ===")
    
    # Crear una estrategia simple (mock)
    class SimpleStrategy:
        def find_route(self, start, end, graph, constraints=None):
            return RouteResponse(
                route=[start, "intermediate", end],
                metrics={"distance": 10.0, "time": 5.0, "cost": 2.0},
                confidence=0.9
            )
        def get_name(self):
            return "simple"
    
    # Construir servicio usando builder
    repository = RouteRepository()
    event_bus = EventBus()
    strategy = SimpleStrategy()
    
    service = (RouteServiceBuilder()
        .add_strategy("simple", strategy)
        .with_repository(repository)
        .with_event_bus(event_bus)
        .build())
    
    # Usar servicio
    request = RouteRequest(
        start_node="A",
        end_node="B",
        strategy="simple"
    )
    
    response = service.find_route(request)
    logger.info(f"Ruta encontrada: {response.route}")
    logger.info(f"Métricas: {response.metrics}")


# ============================================================================
# Ejemplo 2: Event-Driven Architecture
# ============================================================================

def example_event_driven():
    """Ejemplo de arquitectura basada en eventos."""
    logger.info("\n=== Ejemplo 2: Event-Driven Architecture ===")
    
    event_bus = EventBus()
    
    # Crear handlers
    def on_route_requested(event):
        logger.info(f"Evento: Ruta solicitada - {event.data['request'].start_node} -> {event.data['request'].end_node}")
    
    def on_route_found(event):
        logger.info(f"Evento: Ruta encontrada - {event.data['response'].route}")
    
    def on_route_error(event):
        logger.error(f"Evento: Error - {event.data['error']}")
    
    # Suscribir handlers
    event_bus.subscribe("route_requested", EventHandler(on_route_requested))
    event_bus.subscribe("route_found", EventHandler(on_route_found))
    event_bus.subscribe("route_error", EventHandler(on_route_error))
    
    # Emitir eventos
    event_bus.emit("route_requested", {
        "request": RouteRequest(start_node="A", end_node="B")
    })
    
    event_bus.emit("route_found", {
        "response": RouteResponse(
            route=["A", "B"],
            metrics={"distance": 10.0},
            confidence=0.9
        )
    })
    
    # Ver historial
    history = event_bus.get_history(limit=5)
    logger.info(f"Historial de eventos: {len(history)} eventos")


# ============================================================================
# Ejemplo 3: Dependency Injection
# ============================================================================

def example_dependency_injection():
    """Ejemplo de inyección de dependencias."""
    logger.info("\n=== Ejemplo 3: Dependency Injection ===")
    
    # Registrar servicios
    repository = RouteRepository()
    register_service(RouteRepository, repository, singleton=True)
    
    # Resolver servicios
    resolved_repo = resolve_service(RouteRepository)
    logger.info(f"Repositorio resuelto: {resolved_repo is repository}")
    
    # Registrar con factory
    def create_event_bus():
        return EventBus()
    
    register_service(EventBus, factory=create_event_bus, singleton=False)
    
    # Resolver múltiples instancias (no singleton)
    bus1 = resolve_service(EventBus)
    bus2 = resolve_service(EventBus)
    logger.info(f"Instancias diferentes: {bus1 is not bus2}")


# ============================================================================
# Ejemplo 4: Plugin System
# ============================================================================

def example_plugin_system():
    """Ejemplo de sistema de plugins."""
    logger.info("\n=== Ejemplo 4: Plugin System ===")
    
    # Crear plugin
    class MyStrategyPlugin(RouteStrategyPlugin):
        def get_name(self):
            return "my_custom_strategy"
        
        def get_version(self):
            return "1.0.0"
        
        def initialize(self, config=None):
            logger.info("Plugin inicializado")
        
        def create_strategy(self):
            class CustomStrategy:
                def find_route(self, start, end, graph, constraints=None):
                    return RouteResponse(
                        route=[start, end],
                        metrics={"distance": 5.0},
                        confidence=1.0
                    )
                def get_name(self):
                    return "custom"
            return CustomStrategy()
    
    # Registrar plugin
    manager = PluginManager()
    plugin = MyStrategyPlugin()
    manager.register_plugin(plugin)
    manager.initialize_plugin("my_custom_strategy")
    
    # Usar plugin
    strategy_plugin = manager.get_strategy_plugin("my_custom_strategy")
    if strategy_plugin:
        strategy = strategy_plugin.create_strategy()
        logger.info(f"Estrategia creada: {strategy.get_name()}")
    
    # Listar plugins
    plugins = manager.list_plugins()
    logger.info(f"Plugins registrados: {plugins}")


# ============================================================================
# Ejemplo 5: Repository Pattern
# ============================================================================

def example_repository_pattern():
    """Ejemplo de patrón repository."""
    logger.info("\n=== Ejemplo 5: Repository Pattern ===")
    
    repository = RouteRepository()
    
    # Crear y guardar rutas
    response1 = RouteResponse(
        route=["A", "B", "C"],
        metrics={"distance": 10.0, "time": 5.0},
        confidence=0.9,
        metadata={"start_node": "A", "end_node": "C", "strategy": "shortest"}
    )
    
    route_id1 = repository.save_route(response1)
    logger.info(f"Ruta guardada con ID: {route_id1}")
    
    # Recuperar ruta
    retrieved = repository.get_route(route_id1)
    if retrieved:
        logger.info(f"Ruta recuperada: {retrieved.route}")
    
    # Buscar rutas
    routes = repository.find_routes(start="A", limit=10)
    logger.info(f"Rutas encontradas: {len(routes)}")
    
    # Contar rutas
    count = repository.count()
    logger.info(f"Total de rutas: {count}")


# ============================================================================
# Ejemplo 6: Factory Pattern
# ============================================================================

def example_factory_pattern():
    """Ejemplo de patrón factory."""
    logger.info("\n=== Ejemplo 6: Factory Pattern ===")
    
    # Registrar modelos
    class SimpleModel:
        def forward(self, x):
            return x
        def predict(self, features):
            return {"distance": 10.0}
    
    RouteModelFactory.register("simple", SimpleModel)
    
    # Crear modelo usando factory
    model = RouteModelFactory.create("simple")
    logger.info(f"Modelo creado: {type(model).__name__}")
    
    # Listar modelos disponibles
    models = RouteModelFactory.list_models()
    logger.info(f"Modelos disponibles: {models}")


# ============================================================================
# Ejemplo Completo: Integración
# ============================================================================

def example_complete_integration():
    """Ejemplo completo integrando todos los componentes."""
    logger.info("\n=== Ejemplo Completo: Integración ===")
    
    # 1. Crear componentes
    repository = RouteRepository()
    event_bus = EventBus()
    
    # 2. Configurar eventos
    def log_event(event):
        logger.info(f"Evento recibido: {event.type}")
    
    event_bus.subscribe_all(EventHandler(log_event))
    
    # 3. Crear estrategia
    class IntegratedStrategy:
        def find_route(self, start, end, graph, constraints=None):
            return RouteResponse(
                route=[start, end],
                metrics={"distance": 8.0, "time": 4.0, "cost": 1.5},
                confidence=0.95
            )
        def get_name(self):
            return "integrated"
    
    # 4. Construir servicio
    service = (RouteServiceBuilder()
        .add_strategy("integrated", IntegratedStrategy())
        .with_repository(repository)
        .with_event_bus(event_bus)
        .build())
    
    # 5. Usar servicio
    request = RouteRequest(
        start_node="START",
        end_node="END",
        strategy="integrated",
        constraints={"max_distance": 100}
    )
    
    response = service.find_route(request)
    
    logger.info(f"✅ Ruta: {response.route}")
    logger.info(f"✅ Métricas: {response.metrics}")
    logger.info(f"✅ Confianza: {response.confidence}")
    
    # 6. Verificar que se guardó
    saved_routes = repository.find_routes(start="START")
    logger.info(f"✅ Rutas guardadas: {len(saved_routes)}")


def main():
    """Ejecutar todos los ejemplos."""
    logger.info("Ejecutando ejemplos de arquitectura...\n")
    
    example_builder_pattern()
    example_event_driven()
    example_dependency_injection()
    example_plugin_system()
    example_repository_pattern()
    example_factory_pattern()
    example_complete_integration()
    
    logger.info("\n=== Todos los ejemplos completados ===")


if __name__ == "__main__":
    main()

