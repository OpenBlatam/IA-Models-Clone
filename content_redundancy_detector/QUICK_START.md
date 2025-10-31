# 🚀 Quick Start - Sistema Listo para Usar

## ✅ Todo Está Listo

El sistema está **100% funcional** y listo para usar inmediatamente.

---

## 🎯 Usar el Sistema

### Opción 1: Import Simple (Recomendado)

```python
from webhooks import send_webhook, WebhookEvent

# Enviar webhook
result = await send_webhook(
    WebhookEvent.ANALYSIS_COMPLETED,
    {"data": "example"},
    request_id="req-123"
)
```

### Opción 2: Manager Directo

```python
from webhooks import webhook_manager, WebhookEvent

# Iniciar (si no está iniciado)
await webhook_manager.start()

# Enviar webhook
result = await webhook_manager.send_webhook(
    WebhookEvent.ANALYSIS_COMPLETED,
    {"data": "example"}
)
```

---

## ⚙️ Configuración Mínima (Opcional)

El sistema funciona **sin configuración** usando valores por defecto:

```bash
# .env (opcional)
REDIS_URL=redis://localhost:6379        # Para estado compartido
ENABLE_TRACING=true                     # OpenTelemetry
ENABLE_METRICS=true                     # Prometheus
```

**Sin configuración**: Usa in-memory storage (perfecto para desarrollo/local)

---

## 📝 Ejemplo Completo

```python
from fastapi import FastAPI
from webhooks import (
    webhook_manager,
    send_webhook,
    WebhookEvent,
    WebhookEndpoint
)

app = FastAPI()

@app.on_event("startup")
async def startup():
    await webhook_manager.start()
    
    # Registrar webhook endpoint
    endpoint = WebhookEndpoint(
        id="my-webhook",
        url="https://example.com/webhook",
        events=[WebhookEvent.ANALYSIS_COMPLETED]
    )
    webhook_manager.register_endpoint_sync(endpoint)

@app.post("/test")
async def test():
    # Enviar webhook
    result = await send_webhook(
        WebhookEvent.ANALYSIS_COMPLETED,
        {"message": "Test successful"}
    )
    return {"status": "ok", "webhook": result}

@app.on_event("shutdown")
async def shutdown():
    await webhook_manager.stop()
```

---

## ✅ Verificación Rápida

```python
# Verificar que funciona
from webhooks import webhook_manager

print("✅ Sistema listo:", webhook_manager is not None)
print("✅ Workers:", webhook_manager._max_workers)
print("✅ Storage:", type(webhook_manager._storage).__name__)
```

---

## 🎉 ¡Listo para Usar!

No necesitas configurar nada más. El sistema:
- ✅ Auto-detecta el entorno
- ✅ Selecciona storage apropiado
- ✅ Configura workers óptimos
- ✅ Funciona inmediatamente

**Solo importa y usa:**

```python
from webhooks import send_webhook, WebhookEvent
result = await send_webhook(WebhookEvent.ANALYSIS_COMPLETED, {})
```

---

**Estado**: ✅ **LISTO Y FUNCIONANDO**






