# Fase 31: Refactorización de Autonomous SAM3 Agent - Consolidación de Helpers

## Resumen

Esta fase refactoriza el módulo `autonomous_sam3_agent` para eliminar duplicación en construcción de mensajes, operaciones JSON, y construcción de outputs.

## Problemas Identificados

### 1. Duplicación en Construcción de Mensajes
- **Ubicación**: `agent_core.py`
- **Problema**: La construcción de mensajes para OpenRouter API estaba duplicada en múltiples lugares con estructura similar.
- **Impacto**: Código repetitivo, difícil de mantener, propenso a errores.

### 2. Duplicación en Operaciones JSON
- **Ubicación**: `agent_core.py`
- **Problema**: Lectura y escritura de archivos JSON con el mismo patrón repetido múltiples veces.
- **Impacto**: Código repetitivo, manejo de errores inconsistente.

### 3. Duplicación en Construcción de Outputs
- **Ubicación**: `agent_core.py`
- **Problema**: Múltiples métodos construyen outputs con estructura similar pero código duplicado.
- **Impacto**: Código repetitivo, difícil de mantener consistencia.

### 4. Duplicación en Extracción de Tool Calls
- **Ubicación**: `agent_core.py`
- **Problema**: Lógica de extracción y parsing de tool calls duplicada.
- **Impacto**: Código repetitivo, manejo de errores inconsistente.

## Soluciones Implementadas

### 1. Creación de `core/helpers.py`

Se creó un nuevo módulo `core/helpers.py` que centraliza todas las operaciones comunes:

```python
# Funciones de construcción de mensajes
def create_message(role: str, content: Any) -> Dict[str, Any]
def create_text_content(text: str) -> Dict[str, str]
def create_image_content(image_path: str) -> Dict[str, str]
def create_user_message_with_image(text: str, image_path: str) -> Dict[str, Any]
def create_tool_message(tool_call: Dict[str, Any]) -> Dict[str, Any]

# Funciones de JSON
def load_json_file(file_path: str) -> Dict[str, Any]
def save_json_file(data: Dict[str, Any], file_path: str, indent: int = 4) -> None

# Funciones de construcción de outputs
def create_output_structure(...) -> Dict[str, Any]
def filter_outputs_by_indices(outputs: Dict[str, Any], indices: List[int], offset: int = 0) -> Dict[str, Any]

# Funciones de parsing
def extract_tool_call_from_text(generated_text: str) -> Dict[str, Any]
```

### 2. Refactorización de `agent_core.py`

**Antes**:
```python
# Construcción de mensajes duplicada
messages = [
    {"role": "system", "content": system_prompt},
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": f"...",},
        ],
    },
]

# JSON duplicado
with open(output_json_path, "r") as f:
    sam3_outputs = json.load(f)

# Tool call parsing duplicado
if "<tool>" not in generated_text:
    raise ValueError(...)
tool_call_json_str = generated_text.split("<tool>")[-1]...
tool_call = json.loads(tool_call_json_str)
```

**Después**:
```python
from .helpers import (
    create_message,
    create_user_message_with_image,
    load_json_file,
    extract_tool_call_from_text,
    ...
)

# Construcción de mensajes usando helpers
messages = [
    create_message("system", system_prompt),
    create_user_message_with_image(text, image_path),
]

# JSON usando helper
sam3_outputs = load_json_file(output_json_path)

# Tool call parsing usando helper
tool_call = extract_tool_call_from_text(generated_text)
```

### 3. Refactorización de Métodos de Manejo de Tool Calls

**Antes**:
```python
messages.append({
    "role": "assistant",
    "content": [{"type": "text", "text": f"<tool>{json.dumps(tool_call)}</tool>"}],
})
messages.append({
    "role": "user",
    "content": [{"type": "text", "text": "..."}],
})

# Outputs duplicados
final_outputs = {
    "original_image_path": current_outputs["original_image_path"],
    "orig_img_h": current_outputs["orig_img_h"],
    "orig_img_w": current_outputs["orig_img_w"],
    "pred_boxes": [current_outputs["pred_boxes"][i - 1] for i in masks_to_keep],
    ...
}
```

**Después**:
```python
messages.append(create_tool_message(tool_call))
messages.append(create_message("user", [create_text_content("...")]))

# Outputs usando helper
return filter_outputs_by_indices(current_outputs, masks_to_keep, offset=0)
```

## Métricas

### Reducción de Código Duplicado
- **Líneas eliminadas**: ~60 líneas de código duplicado
- **Funciones consolidadas**: 11 funciones helper creadas
- **Archivos refactorizados**: 1 archivo principal (`agent_core.py`)

### Mejoras de Mantenibilidad
- **Punto único de cambio**: Todas las operaciones comunes ahora están centralizadas
- **Consistencia**: Garantiza que todos los mensajes y outputs se construyen de la misma manera
- **Testabilidad**: Los helpers pueden ser probados de forma independiente
- **Legibilidad**: Código más claro y expresivo

## Principios Aplicados

1. **DRY (Don't Repeat Yourself)**: Eliminación de código duplicado
2. **Single Responsibility Principle**: Cada helper tiene una responsabilidad única
3. **Reusabilidad**: Los helpers pueden ser utilizados en cualquier parte del código
4. **Mantenibilidad**: Cambios futuros solo requieren modificar un lugar

## Archivos Modificados

1. **`core/helpers.py`** (NUEVO): Módulo centralizado de helpers comunes
2. **`agent_core.py`**: Refactorizado para usar helpers en:
   - Construcción de mensajes (6 ocurrencias)
   - Operaciones JSON (4 ocurrencias)
   - Construcción de outputs (3 ocurrencias)
   - Extracción de tool calls (1 ocurrencia)
3. **`infrastructure/retry_helpers.py`** (NUEVO): Helper para retry con exponential backoff (preparado para futura refactorización)

## Compatibilidad

- ✅ **Backward Compatible**: Todas las funciones públicas mantienen su interfaz original
- ✅ **Sin Breaking Changes**: Los cambios son internos, no afectan la API pública

## Próximos Pasos (Opcionales)

1. Refactorizar `openrouter_client.py` para usar `retry_helpers.py` (requiere ajuste del diseño del helper)
2. Refactorizar `sam3_client.py` para usar helpers de JSON
3. Agregar tests unitarios para los nuevos helpers
4. Documentar casos de uso avanzados de los helpers

## Notas

- Las advertencias del linter sobre `cv2`, `torch`, `PIL`, y `sam3` son esperadas, ya que son dependencias opcionales
- Los helpers están diseñados para ser simples y eficientes
- Se mantiene la compatibilidad con la estructura de mensajes de OpenRouter API

