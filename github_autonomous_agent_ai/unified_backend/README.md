# Unified Backend

Backend construido para **Unified AI Model** - API completa para chat y generación de texto con IA.

## Quick Start

```powershell
cd backend\unified_backend
$env:PYTHONPATH = "."
python -m uvicorn main:app --host 0.0.0.0 --port 8080
```

O usa el script:
```cmd
start.bat
```

## API Endpoints

| Endpoint | Descripción |
|----------|-------------|
| `/docs` | Swagger UI - Documentación interactiva |
| `/api/v1/health` | Health check |
| `/api/v1/chat` | Chat con memoria de conversación |
| `/api/v1/chat/stream` | Chat con streaming |
| `/api/v1/generate` | Generación de texto |
| `/api/v1/generate/stream` | Generación con streaming |
| `/api/v1/generate/parallel` | Generación paralela multi-modelo |
| `/api/v1/conversations` | Manejo de conversaciones |
| `/api/v1/agents` | Agentes AI continuos |
| `/api/v1/models` | Modelos disponibles |
| `/api/v1/code/analyze` | Análisis de código |

## Configuración

Variables de entorno:

```env
# API Keys (al menos una requerida para LLM)
OPENROUTER_API_KEY=sk-or-v1-...
DEEPSEEK_API_KEY=sk-...

# Servidor
UNIFIED_AI_HOST=0.0.0.0
UNIFIED_AI_PORT=8080
```

## Estructura

```
unified_backend/
├── main.py              # Entry point principal
├── __init__.py          # Package init
├── start.bat            # Script de inicio Windows
│
├── unified_ai_model/    # Núcleo de la API
│   ├── main.py          # FastAPI app
│   ├── config.py        # Configuración
│   ├── api/             # Rutas y schemas
│   └── core/            # Servicios (LLM, Chat, Agents)
│
└── unified_core/        # Módulos adicionales (opcional)
```

## Características

- **Chat**: Conversaciones con memoria y contexto
- **Streaming**: Respuestas en tiempo real
- **Parallel Generation**: Múltiples modelos simultáneamente
- **Continuous Agents**: Agentes que procesan tareas continuamente
- **Code Analysis**: Análisis de código (bugs, performance, security)
