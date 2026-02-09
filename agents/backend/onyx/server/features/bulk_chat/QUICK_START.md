# Quick Start Guide - Bulk Chat

## 🚀 Inicio Rápido en 3 Pasos

### 1. Instalar Dependencias

**Opción A: Instalación automática (recomendado)**
```bash
cd bulk_chat
python install.py
```

**Opción B: Instalación manual**
```bash
cd bulk_chat
pip install -r requirements.txt
```

### 2. Verificar Instalación (Recomendado)

```bash
# Verificar que todo esté listo
python verify_setup.py
```

### 3. Configurar (Opcional)

```bash
# Copiar archivo de ejemplo (si existe)
cp .env.example .env

# Editar .env con tu API key
# OPENAI_API_KEY=tu-api-key-aqui
```

**Nota**: Si no tienes API key, puedes usar el modo `mock` sin configuración.

### 4. Iniciar el Servidor

**Opción 1: Modo Mock (sin API key - recomendado para probar)**
```bash
python -m bulk_chat.main --llm-provider mock
```

**Opción 2: Con API key real**
```bash
# Configurar API key primero
export OPENAI_API_KEY=tu-api-key
python -m bulk_chat.main --llm-provider openai
```

**Opción 3: Usando el script de inicio**
```bash
python start.py
```

**Opción 4: Con opciones personalizadas**
```bash
python -m bulk_chat.main --llm-provider openai --llm-model gpt-4 --port 8006
```

## 📝 Ejemplo Rápido con cURL

```bash
# 1. Crear sesión y empezar chat
SESSION_ID=$(curl -X POST "http://localhost:8006/api/v1/chat/sessions" \
  -H "Content-Type: application/json" \
  -d '{"initial_message": "Hola, explícame sobre Python", "auto_continue": true}' \
  | jq -r '.session_id')

echo "Sesión creada: $SESSION_ID"

# 2. Esperar un poco para que genere respuestas
sleep 5

# 3. Ver mensajes
curl "http://localhost:8006/api/v1/chat/sessions/$SESSION_ID/messages" | jq

# 4. Pausar
curl -X POST "http://localhost:8006/api/v1/chat/sessions/$SESSION_ID/pause?reason=Usuario%20pausó"

# 5. Reanudar
curl -X POST "http://localhost:8006/api/v1/chat/sessions/$SESSION_ID/resume"

# 6. Detener
curl -X POST "http://localhost:8006/api/v1/chat/sessions/$SESSION_ID/stop"
```

## 🐍 Ejemplo con Python

```python
import asyncio
from bulk_chat.core.chat_engine import ContinuousChatEngine

async def quick_example():
    # Crear motor
    engine = ContinuousChatEngine()
    
    # Crear sesión
    session = await engine.create_session(
        initial_message="Explícame sobre machine learning",
        auto_continue=True
    )
    
    # Iniciar chat continuo
    await engine.start_continuous_chat(session.session_id)
    
    # Esperar respuestas
    await asyncio.sleep(10)
    
    # Ver mensajes
    for msg in session.messages:
        print(f"{msg.role}: {msg.content[:100]}")

asyncio.run(quick_example())
```

## ✅ Verificar que Funciona

```bash
# Health check
curl http://localhost:8006/health

# Debería responder:
# {"status":"healthy","service":"bulk_chat","active_sessions":0}
```

## 🎯 Próximos Pasos

- Lee el [README.md](README.md) completo para más detalles
- Revisa los [ejemplos](examples/) para casos de uso avanzados
- Configura tu proveedor de LLM en `.env`

¡Listo! Tu sistema de chat continuo está funcionando 🚀

