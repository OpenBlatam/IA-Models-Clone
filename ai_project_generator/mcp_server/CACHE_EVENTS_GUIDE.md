# Guía de Caché Avanzado y Sistema de Eventos - MCP Server

## Resumen

Sistema de caché avanzado con múltiples estrategias y sistema de eventos/pub-sub para comunicación desacoplada.

## Caché Avanzado

### Estrategias de Caché

- **LRU (Least Recently Used)**: Elimina el menos recientemente usado
- **LFU (Least Frequently Used)**: Elimina el menos frecuentemente usado
- **FIFO (First In First Out)**: Elimina el más antiguo
- **TTL (Time To Live)**: Elimina entradas expiradas

### Uso Básico

```python
from mcp_server.utils.cache_advanced import AdvancedCache, CacheStrategy

# Crear caché con estrategia LRU
cache = AdvancedCache(
    max_size=1000,
    default_ttl=3600,  # 1 hora
    strategy=CacheStrategy.LRU
)

# Guardar valor
cache.set("user:123", {"name": "John", "email": "john@example.com"})

# Obtener valor
user = cache.get("user:123")
if user is None:
    # Valor no encontrado o expirado
    user = fetch_user_from_db(123)
    cache.set("user:123", user)
```

### Invalidación

```python
# Eliminar entrada específica
cache.delete("user:123")

# Invalidar por patrón
count = cache.invalidate_pattern("user:.*")

# Limpiar todo
cache.clear()
```

### Estadísticas

```python
stats = cache.get_stats()
print(f"Hit rate: {stats['hit_rate']:.2%}")
print(f"Size: {stats['size']}/{stats['max_size']}")
print(f"Evictions: {stats['evictions']}")

# Resetear estadísticas
cache.reset_stats()
```

### Generación de Claves

```python
from mcp_server.utils.cache_advanced import make_cache_key

# Crear clave a partir de argumentos
key = make_cache_key("user", user_id=123, operation="read")
cache.set(key, result)
```

## Sistema de Eventos

### Publicar Eventos

```python
from mcp_server.utils.event_system import get_event_bus, EventPriority

bus = get_event_bus()

# Publicar evento asíncrono
await bus.publish(
    "user.created",
    {"user_id": 123, "name": "John"},
    priority=EventPriority.HIGH,
    source="api"
)

# Publicar evento síncrono
bus.publish_sync(
    "user.updated",
    {"user_id": 123, "changes": {"email": "new@example.com"}}
)
```

### Suscribirse a Eventos

```python
from mcp_server.utils.event_system import get_event_bus, Event

bus = get_event_bus()

# Handler simple
def handle_user_created(event: Event):
    print(f"User created: {event.data['user_id']}")

bus.subscribe("user.created", handle_user_created)

# Handler con filtro
def handle_high_priority(event: Event) -> bool:
    return event.priority == EventPriority.HIGH

def handle_critical(event: Event):
    print(f"Critical event: {event.type}")

bus.subscribe(filter_func=handle_high_priority, handler=handle_critical)

# Handler asíncrono
async def async_handler(event: Event):
    await process_event(event)

bus.subscribe("user.created", async_handler)
```

### Decorador de Suscripción

```python
from mcp_server.utils.event_system import get_event_bus

bus = get_event_bus()

@bus.subscribe("user.created")
def on_user_created(event: Event):
    print(f"User created: {event.data}")

@bus.subscribe("user.updated", priority=10)
def on_user_updated(event: Event):
    print(f"User updated: {event.data}")
```

### Historial de Eventos

```python
# Obtener todos los eventos
history = bus.get_event_history()

# Filtrar por tipo
user_events = bus.get_event_history(event_type="user.created")

# Limitar cantidad
recent = bus.get_event_history(limit=100)

# Limpiar historial
bus.clear_history()
```

## Ejemplos de Uso

### Caché con Invalidación Inteligente

```python
from mcp_server.utils.cache_advanced import AdvancedCache, CacheStrategy

cache = AdvancedCache(strategy=CacheStrategy.LRU, max_size=500)

def get_user(user_id: int):
    key = f"user:{user_id}"
    user = cache.get(key)
    
    if user is None:
        user = fetch_user_from_db(user_id)
        cache.set(key, user, ttl=3600)
    
    return user

def update_user(user_id: int, changes: dict):
    # Actualizar en BD
    update_user_in_db(user_id, changes)
    
    # Invalidar caché
    cache.delete(f"user:{user_id}")
    
    # Invalidar todas las entradas relacionadas
    cache.invalidate_pattern(f"user:{user_id}:.*")
```

### Sistema de Eventos Completo

```python
from mcp_server.utils.event_system import (
    get_event_bus, Event, EventPriority
)

bus = get_event_bus()

# Suscribirse a múltiples eventos
@bus.subscribe("user.created")
@bus.subscribe("user.updated")
def handle_user_events(event: Event):
    print(f"User event {event.type}: {event.data}")

# Handler con prioridad
@bus.subscribe("order.completed", priority=10)
async def process_order(event: Event):
    order_id = event.data["order_id"]
    await send_confirmation_email(order_id)
    await update_inventory(event.data["items"])

# Publicar eventos
async def create_user(user_data: dict):
    user = save_user(user_data)
    
    await bus.publish(
        "user.created",
        {"user_id": user.id, **user_data},
        priority=EventPriority.NORMAL,
        source="user_service"
    )
    
    return user
```

### Integración Caché + Eventos

```python
from mcp_server.utils.cache_advanced import AdvancedCache
from mcp_server.utils.event_system import get_event_bus

cache = AdvancedCache()
bus = get_event_bus()

# Invalidar caché cuando se actualiza usuario
@bus.subscribe("user.updated")
def invalidate_user_cache(event: Event):
    user_id = event.data["user_id"]
    cache.delete(f"user:{user_id}")
    cache.invalidate_pattern(f"user:{user_id}:.*")

# Cachear resultado y publicar evento
def get_user_with_cache(user_id: int):
    key = f"user:{user_id}"
    user = cache.get(key)
    
    if user is None:
        user = fetch_user_from_db(user_id)
        cache.set(key, user)
        
        bus.publish_sync(
            "cache.miss",
            {"key": key, "user_id": user_id}
        )
    
    return user
```

## Próximos Pasos

1. Agregar persistencia de caché
2. Integrar con Redis para caché distribuido
3. Agregar más estrategias de caché
4. Mejorar sistema de eventos con retry
5. Agregar eventos distribuidos

