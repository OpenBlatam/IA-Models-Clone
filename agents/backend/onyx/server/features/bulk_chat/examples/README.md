# Ejemplos de Uso - Bulk Chat

## 📚 Ejemplos Disponibles

### 1. `example_usage.py` - Uso Programático Básico

Ejemplo de cómo usar el sistema programáticamente con el motor de chat.

```bash
python examples/example_usage.py
```

**Características:**
- Crear sesión de chat
- Iniciar chat continuo
- Pausar/reanudar
- Detener sesión

### 2. `api_example.py` - Uso de la API REST

Ejemplo de cómo usar la API REST del sistema.

```bash
# Primero inicia el servidor
python -m bulk_chat.main --llm-provider mock

# En otra terminal, ejecuta el ejemplo
python examples/api_example.py
```

**Características:**
- Crear sesión vía API
- Obtener mensajes
- Enviar mensajes
- Controlar sesión (pausar/reanudar/detener)

## 🚀 Ejemplos Rápidos

### Ejemplo 1: Chat Simple

```python
import asyncio
from bulk_chat.core.chat_engine import ContinuousChatEngine

async def simple_chat():
    engine = ContinuousChatEngine()
    session = await engine.create_session(
        initial_message="Hola, ¿cómo estás?",
        auto_continue=True
    )
    await engine.start_continuous_chat(session.session_id)
    await asyncio.sleep(10)  # Esperar respuestas
    await engine.stop_session(session.session_id)

asyncio.run(simple_chat())
```

### Ejemplo 2: Usando API REST

```python
import requests

# Crear sesión
response = requests.post(
    "http://localhost:8006/api/v1/chat/sessions",
    json={
        "initial_message": "Explícame sobre IA",
        "auto_continue": True
    }
)
session_id = response.json()["session_id"]

# Ver mensajes
messages = requests.get(
    f"http://localhost:8006/api/v1/chat/sessions/{session_id}/messages"
)
print(messages.json())
```

### Ejemplo 3: Múltiples Sesiones

```python
import asyncio
from bulk_chat.core.chat_engine import ContinuousChatEngine

async def multiple_sessions():
    engine = ContinuousChatEngine()
    
    sessions = []
    for i in range(3):
        session = await engine.create_session(
            initial_message=f"Sesión {i+1}: Hola",
            auto_continue=True
        )
        sessions.append(session)
        await engine.start_continuous_chat(session.session_id)
    
    # Esperar respuestas
    await asyncio.sleep(10)
    
    # Detener todas
    for session in sessions:
        await engine.stop_session(session.session_id)

asyncio.run(multiple_sessions())
```

### 3. `bulk_operations_example.py` - Operaciones Masivas

Ejemplo completo de cómo usar las operaciones masivas del sistema.

```bash
# Primero inicia el servidor
python -m bulk_chat.main --llm-provider mock

# En otra terminal, ejecuta el ejemplo
python examples/bulk_operations_example.py
```

**Características:**
- Crear múltiples sesiones en lote
- Pausar/reanudar sesiones masivamente
- Enviar mensajes a múltiples sesiones
- Analizar múltiples sesiones
- Exportar sesiones en lote
- Limpiar sesiones antiguas

## 📖 Más Información

- [README.md](../README.md) - Documentación completa
- [QUICK_START.md](../QUICK_START.md) - Guía de inicio rápido
- [COMMANDS.md](../COMMANDS.md) - Comandos útiles
- [BULK_OPERATIONS.md](../BULK_OPERATIONS.md) - Documentación de operaciones masivas

