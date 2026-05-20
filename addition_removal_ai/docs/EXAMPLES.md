# Ejemplos de Uso - Addition Removal AI

## Ejemplos Básicos

### 1. Agregar Contenido Simple

```python
from addition_removal_ai.core.editor import ContentEditor

editor = ContentEditor()

# Agregar al final
result = await editor.add(
    content="Texto original",
    addition="Nuevo párrafo",
    position="end"
)

print(result["content"])
# Output: "Texto original\n\nNuevo párrafo"
```

### 2. Eliminar Contenido

```python
result = await editor.remove(
    content="Texto con elemento a eliminar",
    pattern="elemento a eliminar"
)

print(result["content"])
# Output: "Texto con "
```

### 3. Posicionamiento Automático con IA

```python
result = await editor.add(
    content="Documento largo con múltiples secciones...",
    addition="Nueva sección relevante",
    position="auto"  # IA determina la mejor posición
)
```

## Ejemplos Avanzados

### 4. Operaciones Batch

```python
# Agregar múltiples elementos
result = await editor.batch_add(
    content="Texto base",
    additions=[
        {"addition": "Introducción", "position": "start"},
        {"addition": "Conclusión", "position": "end"},
        {"addition": "Nota", "position": "end"}
    ]
)
```

### 5. Trabajar con Markdown

```python
markdown_content = """# Título Principal

## Sección 1
Contenido de la sección 1.

## Sección 2
Contenido de la sección 2.
"""

# Agregar nueva sección
result = await editor.add(
    content=markdown_content,
    addition="## Sección 3\nNuevo contenido.",
    position="end"
)
```

### 6. Trabajar con JSON

```python
json_content = '{"name": "John", "age": 30}'

# Agregar nuevo campo
result = await editor.add(
    content=json_content,
    addition='{"city": "New York"}',
    position="end"
)

# El resultado será JSON válido con el nuevo campo agregado
```

### 7. Calcular Diferencias

```python
from addition_removal_ai.core.diff import ContentDiff

diff = ContentDiff()
result = diff.compute_diff(
    original="Texto original",
    modified="Texto modificado"
)

print(result["summary"])
# Muestra estadísticas de cambios

# Obtener solo adiciones
additions = diff.get_additions(result)
print(additions)
```

### 8. Sistema Undo/Redo

```python
# Guardar estado antes de operación
editor.undo_redo.save_state("Estado inicial", "initial")

# Realizar operación
result = await editor.add("Estado inicial", "Nuevo contenido", "end")

# Deshacer
previous = editor.undo_redo.undo(result["content"])
print(previous["content"])  # "Estado inicial"

# Rehacer
next_state = editor.undo_redo.redo(previous["content"])
```

### 9. Obtener Métricas

```python
# Después de varias operaciones
stats = editor.metrics.get_stats()
print(stats["average_timings"])
print(stats["error_rate"])

# Métricas de un tipo específico
add_stats = editor.metrics.get_operation_stats("add")
print(add_stats["success_rate"])
```

### 10. Validación Semántica

```python
result = await editor.add(
    content="Documento sobre inteligencia artificial...",
    addition="Información sobre machine learning...",
    position="end"
)

# La validación semántica se incluye automáticamente
semantic = result["validation"]["semantic"]
print(f"Coherencia: {semantic.get('thematic_coherence', 0)}")
print(f"Fluidez: {semantic.get('fluency', 0)}")
```

## Ejemplos con API REST

### 11. Agregar vía API

```bash
curl -X POST http://localhost:8010/api/v1/add \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Texto original",
    "addition": "Nuevo contenido",
    "position": "end"
  }'
```

### 12. Operación Batch vía API

```bash
curl -X POST http://localhost:8010/api/v1/batch/add \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Base",
    "additions": [
      {"addition": "Primero", "position": "start"},
      {"addition": "Segundo", "position": "end"}
    ]
  }'
```

### 13. Calcular Diff vía API

```bash
curl -X POST http://localhost:8010/api/v1/diff \
  -H "Content-Type: application/json" \
  -d '{
    "original": "Texto original",
    "modified": "Texto modificado",
    "format": "html"
  }'
```

### 14. Obtener Métricas

```bash
curl http://localhost:8010/api/v1/metrics

# Métricas específicas
curl http://localhost:8010/api/v1/metrics/add
```

## Casos de Uso Reales

### 15. Edición de Documentos

```python
document = """# Reporte Anual

## Introducción
Este es el reporte anual...

## Resultados
Los resultados fueron...
"""

# Agregar nueva sección de conclusiones
result = await editor.add(
    content=document,
    addition="## Conclusiones\nLas conclusiones del análisis...",
    position="end"
)
```

### 16. Limpieza de Datos

```python
content_with_noise = """Texto útil.
[AD] Publicidad no deseada
Más texto útil.
[SPAM] Más spam
Texto final útil.
"""

# Eliminar elementos de spam
result = await editor.remove(
    content=content_with_noise,
    pattern="[AD]"
)

result = await editor.remove(
    content=result["content"],
    pattern="[SPAM]"
)
```

### 17. Actualización de Configuración JSON

```python
config = '{"host": "localhost", "port": 8000}'

# Agregar nueva configuración
result = await editor.add(
    content=config,
    addition='{"timeout": 30}',
    position="end"
)

# Resultado: {"host": "localhost", "port": 8000, "timeout": 30}
```






