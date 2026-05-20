# Webhooks Module - Modular Architecture

## 📁 Estructura Modular

El sistema de webhooks ha sido refactorizado en una estructura modular para mejorar la mantenibilidad y escalabilidad:

```
webhooks/
├── __init__.py          # Exports públicos y funciones de conveniencia
├── models.py             # Modelos de datos (enums, dataclasses)
├── circuit_breaker.py    # Circuit breaker pattern
├── delivery.py           # Lógica de entrega de webhooks
└── manager.py            # Manager principal y orquestación
```

## 📦 Módulos

### `models.py`
**Responsabilidad**: Modelos de datos y tipos

- `WebhookEvent`: Enum de tipos de eventos
- `WebhookPayload`: Estructura del payload
- `WebhookEndpoint`: Configuración de endpoints
- `WebhookDelivery`: Registro de entregas

### `circuit_breaker.py`
**Responsabilidad**: Patrón Circuit Breaker

- Protección contra endpoints caídos
- Estados: closed, open, half_open
- Recovery automático
- Configurable (failure_threshold, timeout)

### `delivery.py`
**Responsabilidad**: Lógica de entrega

- `WebhookDeliveryService`: Servicio de entrega
- Generación de signatures HMAC
- Preparación de headers
- Cálculo de retry delays con jitter
- Manejo de timeouts y errores

### `manager.py`
**Responsabilidad**: Orquestación principal

- `WebhookManager`: Gestión de todo el sistema
- Worker pool management
- Queue management
- Métricas y estadísticas
- Endpoint registration

### `__init__.py`
**Responsabilidad**: API pública

- Exporta todas las clases y funciones públicas
- Funciones de conveniencia
- Instancia global de `webhook_manager`

## 🚀 Uso

### Importación

```python
# Importar desde el módulo webhooks
from webhooks import (
    WebhookEvent,
    WebhookEndpoint,
    send_webhook,
    register_webhook_endpoint,
    get_webhook_stats
)

# O desde sub-módulos específicos
from webhooks.models import WebhookPayload
from webhooks.circuit_breaker import CircuitBreaker
```

### Registrar Endpoint

```python
from webhooks import WebhookEndpoint, WebhookEvent

endpoint = WebhookEndpoint(
    id="my-endpoint",
    url="https://example.com/webhook",
    events=[WebhookEvent.ANALYSIS_COMPLETED],
    secret="your-secret-key",
    timeout=30,
    retry_count=3
)

register_webhook_endpoint(endpoint)
```

### Enviar Webhook

```python
from webhooks import send_webhook, WebhookEvent

result = await send_webhook(
    WebhookEvent.ANALYSIS_COMPLETED,
    {"analysis_id": "123", "status": "completed"},
    request_id="req-123",
    user_id="user-456"
)

print(result)  # {"status": "queued", "queued": 1, ...}
```

### Obtener Estadísticas

```python
from webhooks import get_webhook_stats

stats = get_webhook_stats()
print(f"Success rate: {stats['success_rate']}%")
print(f"Active endpoints: {stats['active_endpoints']}")
```

## 🔧 Ventajas de la Arquitectura Modular

### ✅ Separación de Responsabilidades
- Cada módulo tiene una responsabilidad única y clara
- Fácil de entender y mantener
- Testing más simple por módulo

### ✅ Reutilización
- Circuit breaker puede usarse independientemente
- Delivery service puede extenderse fácilmente
- Models pueden ser compartidos

### ✅ Escalabilidad
- Fácil agregar nuevos tipos de eventos
- Extender lógica de entrega sin afectar otros módulos
- Agregar nuevas métricas sin cambios masivos

### ✅ Testing
- Tests unitarios por módulo
- Mocks más fáciles
- Menor acoplamiento

### ✅ Mantenibilidad
- Código más organizado
- Fácil encontrar bugs
- Onboarding más rápido para nuevos desarrolladores

## 📝 Migración desde webhooks.py

El código antiguo sigue funcionando:

```python
# Antes (aún funciona)
from webhooks import send_webhook

# Ahora también puedes usar
from webhooks.manager import WebhookManager
from webhooks.delivery import WebhookDeliveryService
```

## 🔄 Extensibilidad

### Agregar Nuevo Tipo de Evento

```python
# En webhooks/models.py
class WebhookEvent(Enum):
    # ... eventos existentes
    NEW_EVENT = "new_event"
```

### Extender Delivery Service

```python
# Crear subclase en delivery.py
class CustomDeliveryService(WebhookDeliveryService):
    async def deliver(self, ...):
        # Lógica personalizada
        return await super().deliver(...)
```

### Agregar Nuevas Métricas

```python
# En manager.py
self._metrics["custom_metric"] = 0

def get_delivery_stats(self):
    return {
        # ... métricas existentes
        "custom_metric": self._metrics["custom_metric"]
    }
```

## ✅ Estado

- ✅ Modularización completa
- ✅ Backward compatible
- ✅ Bien documentado
- ✅ Type hints completos
- ✅ Listo para producción

---

**Versión**: 2.0.0  
**Fecha**: 2024






