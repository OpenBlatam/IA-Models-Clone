# Guía Rápida de Refactorización - Record Storage

## Checklist de Refactorización

### ✅ 1. Context Managers
**Reemplazar:**
```python
f = open(file_path, 'r')
data = f.read()
f.close()
```

**Por:**
```python
with open(file_path, 'r', encoding='utf-8') as f:
    data = f.read()
```

### ✅ 2. Indentación Correcta
**Asegurar que:**
- Todos los bloques de código están correctamente indentados
- Los métodos dentro de clases tienen indentación consistente
- Los bloques `if`, `for`, `try` tienen indentación apropiada

### ✅ 3. Manejo de Registros en Update
**Incorrecto:**
```python
for record in records:
    if record['id'] == record_id:
        record = updates  # ❌ Reemplaza todo
        break
```

**Correcto:**
```python
for i, record in enumerate(records):
    if record.get('id') == record_id:
        records[i].update(updates)  # ✅ Fusiona
        break
```

### ✅ 4. Validación de Entrada
**Agregar validación:**
```python
def update(self, record_id: str, updates: Dict[str, Any]) -> bool:
    if not isinstance(record_id, str) or not record_id:
        raise ValueError("record_id debe ser una cadena no vacía")
    
    if not isinstance(updates, dict):
        raise ValueError("updates debe ser un diccionario")
    # ... resto del código
```

### ✅ 5. Manejo de Errores
**Agregar try/except:**
```python
try:
    with open(self.file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
except json.JSONDecodeError as e:
    raise RuntimeError(f"JSON inválido: {e}") from e
except (IOError, OSError) as e:
    raise RuntimeError(f"Error al leer: {e}") from e
```

## Ejemplo de Uso

```python
from utils.record_storage import RecordStorage

# Crear instancia
storage = RecordStorage("data/records.json")

# Agregar registro
storage.add({"id": "1", "name": "Test", "value": 100})

# Leer todos
records = storage.read()

# Actualizar (fusiona campos)
storage.update("1", {"value": 200, "status": "active"})

# Obtener específico
record = storage.get("1")

# Eliminar
storage.delete("1")
```

## Errores Comunes a Evitar

1. ❌ **No usar context managers** - Puede causar leaks de recursos
2. ❌ **Reemplazar registro completo** - `record = updates` en lugar de `record.update(updates)`
3. ❌ **Indentación incorrecta** - Puede causar errores de lógica
4. ❌ **Sin validación** - Puede causar errores en tiempo de ejecución
5. ❌ **Sin manejo de errores** - Puede causar crashes inesperados

## Archivos Relacionados

- `record_storage.py` - Implementación completa refactorizada
- `record_storage_example.py` - Comparación antes/después
- `refactored_code_example.py` - Ejemplo directo de código
- `test_record_storage.py` - Pruebas y ejemplos de uso
- `RECORD_STORAGE_REFACTORING.md` - Documentación detallada


