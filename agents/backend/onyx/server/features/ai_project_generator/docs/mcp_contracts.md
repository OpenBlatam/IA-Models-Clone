# MCP Contracts - Contratos de Entrada/Salida

## Introducción

Este documento define los "frames" de contexto estandarizados para el Model Context Protocol (MCP). Estos contratos aseguran que las respuestas del modelo sean consistentes y predecibles.

## ContextFrame

### Estructura

```json
{
  "frame_id": "uuid-v4",
  "content": "string (requerido, no vacío)",
  "context_type": "text|code|data|json|yaml",
  "max_tokens": 4096,
  "token_count": null,
  "metadata": {
    "timestamp": "ISO 8601",
    "source": "mcp|filesystem|database|api",
    "version": "1.0",
    "token_count": null,
    "encoding": "utf-8",
    "metadata": {}
  },
  "parent_frame_id": null,
  "related_frames": [],
  "fields": {}
}
```

### Campos

- **frame_id**: ID único del frame (UUID v4)
- **content**: Contenido del contexto (requerido, no puede estar vacío)
- **context_type**: Tipo de contexto:
  - `text`: Texto plano
  - `code`: Código fuente
  - `data`: Datos estructurados
  - `json`: JSON
  - `yaml`: YAML
- **max_tokens**: Límite máximo de tokens (1-100000, default: 4096)
- **token_count**: Conteo actual de tokens (calculado automáticamente)
- **metadata**: Metadata del frame
- **parent_frame_id**: ID del frame padre (para jerarquías)
- **related_frames**: IDs de frames relacionados
- **fields**: Campos adicionales personalizados

### Validaciones

1. `content` no puede estar vacío
2. `max_tokens` debe estar entre 1 y 100000
3. `token_count` se calcula automáticamente si es null (1 token ≈ 4 caracteres)

### Ejemplo

```python
from mcp_server.contracts import FrameSerializer

frame = FrameSerializer.create_context_frame(
    content="def hello(): print('Hello, World!')",
    context_type="code",
    source="filesystem",
    max_tokens=1000,
)
```

## PromptFrame

### Estructura

```json
{
  "prompt_id": "uuid-v4",
  "system_prompt": "string (opcional)",
  "user_prompt": "string (requerido, no vacío)",
  "context_frames": [],
  "temperature": 0.7,
  "max_tokens": 2048,
  "top_p": 1.0,
  "metadata": {
    "timestamp": "ISO 8601",
    "source": "mcp",
    "version": "1.0",
    "token_count": null,
    "encoding": "utf-8",
    "metadata": {}
  },
  "fields": {}
}
```

### Campos

- **prompt_id**: ID único del prompt (UUID v4)
- **system_prompt**: Prompt del sistema (opcional)
- **user_prompt**: Prompt del usuario (requerido, no puede estar vacío)
- **context_frames**: Lista de ContextFrames asociados
- **temperature**: Temperature para sampling (0.0-2.0, default: 0.7)
- **max_tokens**: Máximo de tokens en respuesta (default: 2048)
- **top_p**: Top-p sampling (0.0-1.0, default: 1.0)
- **metadata**: Metadata del prompt
- **fields**: Campos adicionales personalizados

### Validaciones

1. `user_prompt` no puede estar vacío
2. `temperature` debe estar entre 0.0 y 2.0
3. `top_p` debe estar entre 0.0 y 1.0
4. Total de tokens (contexto + respuesta) no debe exceder 8192

### Ejemplo

```python
from mcp_server.contracts import FrameSerializer, ContextFrame

# Crear contexto
context = FrameSerializer.create_context_frame(
    content="Project structure: src/, tests/, docs/",
    context_type="text",
)

# Crear prompt con contexto
prompt = FrameSerializer.create_prompt_frame(
    user_prompt="Generate a Python project structure",
    system_prompt="You are a helpful AI assistant",
    context_frames=[context],
    temperature=0.7,
)
```

## Serialización

### Formatos Soportados

1. **JSON**: Formato estándar con indentación
2. **Compact**: JSON sin espacios (para eficiencia)
3. **Base64**: Codificación base64 (para binarios)

### Uso

```python
from mcp_server.contracts import FrameSerializer, ContextFrame

# Crear frame
frame = FrameSerializer.create_context_frame(
    content="Hello, World!",
    context_type="text",
)

# Serializar
json_str = FrameSerializer.serialize_context_frame(frame, format="json")
compact_str = FrameSerializer.serialize_context_frame(frame, format="compact")
base64_str = FrameSerializer.serialize_context_frame(frame, format="base64")

# Deserializar
frame_restored = FrameSerializer.deserialize_context_frame(json_str, format="json")
```

## Límites de Tokens

### Recomendaciones

- **ContextFrame individual**: Máximo 4096 tokens
- **PromptFrame con contexto**: Máximo 8192 tokens totales
- **Múltiples frames**: Sumar tokens de todos los frames

### Cálculo de Tokens

```python
# Estimación simple: 1 token ≈ 4 caracteres
tokens = len(content) // 4

# Verificar límites
if frame.is_within_limits():
    # Procesar frame
    pass
```

## Mejores Prácticas

1. **Usar tipos de contexto apropiados**: `code` para código, `text` para texto, etc.
2. **Validar límites antes de enviar**: Usar `is_within_limits()`
3. **Incluir metadata útil**: Timestamp, source, version
4. **Relacionar frames cuando sea necesario**: Usar `parent_frame_id` y `related_frames`
5. **Serializar eficientemente**: Usar `compact` para producción, `json` para debugging

## Ejemplos Completos

Ver `examples/mcp_contracts_example.py` para ejemplos completos de uso.

