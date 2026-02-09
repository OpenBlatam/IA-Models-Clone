# Mejoras Adicionales v2

Este documento describe las mejoras adicionales implementadas en la arquitectura.

## Nuevas Funcionalidades

### 1. Sistema de Eventos (Domain Events)

**Ubicación**: `domain/events.py`, `infrastructure/events/`

**Eventos Implementados**:
- `VisualizationCreatedEvent`: Cuando se crea una visualización
- `VisualizationRetrievedEvent`: Cuando se recupera una visualización
- `ComparisonCreatedEvent`: Cuando se crea una comparación
- `BatchProcessingCompletedEvent`: Cuando se completa procesamiento por lotes

**Beneficios**:
- Desacoplamiento: Los use cases no necesitan conocer los handlers
- Extensibilidad: Fácil agregar nuevos handlers sin modificar código existente
- Observabilidad: Eventos pueden ser usados para métricas, logging, notificaciones

**Implementación**:
- `SimpleEventPublisher`: Publisher/subscriber en memoria
- `NullEventPublisher`: Null object pattern para deshabilitar eventos
- `MetricsEventHandler`: Handler que actualiza métricas
- `LoggingEventHandler`: Handler que registra eventos

### 2. Validadores de Dominio

**Ubicación**: `domain/validators.py`

**Validadores Implementados**:
- `SurgeryValidator`: Valida tipos de cirugía, intensidad, áreas objetivo
- `ImageDomainValidator`: Valida dimensiones y formatos de imágenes
- `VisualizationRequestValidator`: Valida requests completos

**Beneficios**:
- Validación centralizada de reglas de negocio
- Reutilizable en múltiples capas
- Fácil de testear

### 3. Use Cases Adicionales

**Nuevos Use Cases**:
- `CreateComparisonUseCase`: Crear comparaciones antes/después
- `ProcessBatchUseCase`: Procesar múltiples visualizaciones en lote

**Mejoras en Use Cases Existentes**:
- `CreateVisualizationUseCase`: Ahora incluye validación y eventos
- `GetVisualizationUseCase`: Ahora publica eventos

### 4. Interfaces Adicionales

**Nuevas Interfaces**:
- `IEventPublisher`: Para publicar eventos
- `IEventSubscriber`: Para suscribirse a eventos

## Arquitectura de Eventos

### Flujo de Eventos

```
Use Case
  ↓
Publish Event
  ↓
Event Publisher
  ↓
Event Handlers (múltiples)
  ├─> Metrics Handler
  ├─> Logging Handler
  └─> Custom Handlers (extensible)
```

### Ejemplo de Uso

```python
# En un use case
event = VisualizationCreatedEvent(
    visualization_id=visualization_id,
    surgery_type=surgery_type,
    intensity=intensity,
    processing_time=processing_time
)
await self.event_publisher.publish(event)

# Handlers se ejecutan automáticamente
# - MetricsEventHandler actualiza métricas
# - LoggingEventHandler registra el evento
```

## Validación de Dominio

### Ejemplo de Uso

```python
validator = VisualizationRequestValidator()
validator.validate(
    surgery_type=request.surgery_type,
    intensity=request.intensity,
    target_areas=request.target_areas,
    image=image,
    supported_formats=settings.supported_formats
)
```

### Reglas de Validación

1. **Intensidad**: Debe estar entre 0.0 y 1.0
2. **Tipo de Cirugía**: Debe ser un `SurgeryType` válido
3. **Áreas Objetivo**: Deben ser strings válidos
4. **Dimensiones de Imagen**: Entre MIN y MAX definidos
5. **Formato de Imagen**: Debe estar en formatos soportados

## Mejoras en Factories

### Event Publisher Factory

```python
def create_event_publisher() -> IEventPublisher:
    """Create event publisher with default handlers."""
    publisher = SimpleEventPublisher()
    _setup_event_handlers(publisher)  # Setup automático
    return publisher
```

### Setup Automático de Handlers

Los handlers se configuran automáticamente al crear el publisher:
- MetricsEventHandler
- LoggingEventHandler
- Fácil agregar más handlers

## Beneficios de las Mejoras

### 1. Desacoplamiento
- Use cases no conocen implementaciones de handlers
- Fácil cambiar o agregar handlers sin modificar use cases

### 2. Testabilidad
- Fácil mockear event publisher
- Validadores pueden testearse independientemente
- Use cases más simples y enfocados

### 3. Observabilidad
- Eventos proporcionan trazabilidad completa
- Métricas automáticas desde eventos
- Logging estructurado de eventos

### 4. Extensibilidad
- Agregar nuevos handlers: solo crear clase y suscribirse
- Agregar nuevos eventos: solo crear clase de evento
- No requiere modificar código existente

### 5. Validación Robusta
- Validación centralizada y reutilizable
- Reglas de negocio claramente definidas
- Fácil agregar nuevas reglas

## Estructura de Archivos

```
domain/
├── events.py              # Domain events
├── validators.py          # Domain validators
└── use_cases/
    ├── create_visualization.py  # Mejorado con eventos y validación
    ├── get_visualization.py      # Mejorado con eventos
    ├── create_comparison.py      # Nuevo
    └── process_batch.py          # Nuevo

infrastructure/
└── events/
    ├── event_publisher.py  # Implementación de publisher
    └── event_handlers.py   # Handlers de eventos

core/
├── interfaces.py          # IEventPublisher, IEventSubscriber
└── exceptions.py          # ValidationError agregada
```

## Próximos Pasos Sugeridos

1. **Event Store**: Persistir eventos para auditoría
2. **Event Sourcing**: Usar eventos como fuente de verdad
3. **CQRS**: Separar comandos y consultas
4. **Saga Pattern**: Para transacciones distribuidas
5. **Más Validadores**: Validación de reglas de negocio complejas
6. **Event Replay**: Reprocesar eventos para recuperación

## Configuración

Los eventos pueden habilitarse/deshabilitarse en settings:

```python
# config/settings.py
enable_events: bool = True  # Habilitar/deshabilitar eventos
```

Si está deshabilitado, se usa `NullEventPublisher` (no-op).

