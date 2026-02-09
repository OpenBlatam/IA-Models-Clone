# RecordStorage - Documentación de Uso

## Descripción

`RecordStorage` es una clase Python refactorizada que proporciona un sistema robusto para gestionar registros en archivos JSON. Implementa las mejores prácticas de Python para operaciones de archivo, manejo de errores y validación de entrada.

## Características

✅ **Context Managers**: Todas las operaciones de archivo usan `with` statements  
✅ **Manejo de Errores**: Validación completa y mensajes de error claros  
✅ **Actualizaciones Inteligentes**: Fusiona campos en lugar de reemplazar  
✅ **Type Hints**: Soporte completo de tipos para mejor IDE support  
✅ **Encoding UTF-8**: Soporte completo para caracteres Unicode  
✅ **Inicialización Automática**: Crea archivos si no existen  

## Instalación

No requiere instalación adicional. Solo necesita Python 3.7+ y la biblioteca estándar.

## Uso Básico

```python
from utils.record_storage import RecordStorage

# Crear instancia (crea el archivo si no existe)
storage = RecordStorage("data/records.json")

# Escribir registros
records = [
    {"id": "1", "name": "Alice", "age": 30},
    {"id": "2", "name": "Bob", "age": 25}
]
storage.write(records)

# Leer todos los registros
all_records = storage.read()
print(f"Total records: {len(all_records)}")

# Actualizar un registro (fusiona campos)
storage.update("1", {"age": 31})
# Resultado: {"id": "1", "name": "Alice", "age": 31}
```

## API Reference

### `__init__(file_path: str)`

Inicializa la instancia de RecordStorage.

**Parámetros:**
- `file_path` (str): Ruta al archivo JSON para almacenar registros

**Lanza:**
- `ValueError`: Si file_path es inválido
- `RuntimeError`: Si el archivo no puede ser inicializado

**Ejemplo:**
```python
storage = RecordStorage("my_data.json")
```

### `read() -> List[Dict[str, Any]]`

Lee todos los registros del archivo.

**Retorna:**
- `List[Dict[str, Any]]`: Lista de diccionarios de registros

**Lanza:**
- `RuntimeError`: Si el archivo no puede ser leído o contiene JSON inválido

**Ejemplo:**
```python
records = storage.read()
for record in records:
    print(record["name"])
```

### `write(records: List[Dict[str, Any]]) -> bool`

Escribe registros al archivo.

**Parámetros:**
- `records` (List[Dict[str, Any]]): Lista de diccionarios de registros

**Retorna:**
- `bool`: True si la escritura fue exitosa

**Lanza:**
- `ValueError`: Si records no es una lista o contiene items inválidos
- `RuntimeError`: Si el archivo no puede ser escrito

**Ejemplo:**
```python
new_records = [
    {"id": "3", "name": "Charlie", "age": 35}
]
storage.write(new_records)
```

### `update(record_id: str, updates: Dict[str, Any]) -> bool`

Actualiza un registro específico por ID.

**Parámetros:**
- `record_id` (str): ID del registro a actualizar
- `updates` (Dict[str, Any]): Diccionario de campos a actualizar

**Retorna:**
- `bool`: True si la actualización fue exitosa, False si el registro no fue encontrado

**Lanza:**
- `ValueError`: Si record_id o updates son inválidos
- `RuntimeError`: Si las operaciones de archivo fallan

**Ejemplo:**
```python
# Actualiza solo el campo 'age', preserva otros campos
success = storage.update("1", {"age": 32})
if success:
    print("Record updated successfully")
```

## Características Avanzadas

### Actualizaciones que Fusionan

A diferencia de reemplazar todo el registro, `update()` fusiona los campos:

```python
# Registro original
{"id": "1", "name": "Alice", "age": 30, "city": "NYC"}

# Actualizar solo 'age'
storage.update("1", {"age": 31})

# Resultado: {"id": "1", "name": "Alice", "age": 31, "city": "NYC"}
# ✅ Todos los campos originales se preservan
```

### Manejo de Errores

```python
try:
    storage.write([{"id": "1", "name": "Test"}])
except ValueError as e:
    print(f"Error de validación: {e}")
except RuntimeError as e:
    print(f"Error de archivo: {e}")
```

### Soporte Unicode

```python
# Funciona perfectamente con caracteres Unicode
unicode_records = [
    {"id": "1", "name": "José", "city": "São Paulo"},
    {"id": "2", "name": "李", "city": "北京"}
]
storage.write(unicode_records)
```

## Ejemplos Completos

Ver `examples/record_storage_usage.py` para ejemplos completos que incluyen:

- Uso básico
- Actualizaciones que fusionan
- Manejo de errores
- Seguridad de context managers
- Soporte Unicode
- Flujos de trabajo completos

## Mejores Prácticas

1. **Siempre usa try/except** para manejar errores
2. **Valida datos antes de escribir** para evitar errores
3. **Usa IDs únicos** para evitar conflictos
4. **Backup regular** de archivos JSON importantes
5. **No modifiques el ID** de un registro después de crearlo

## Comparación con Código Antiguo

### ❌ Antes (Problemático)
```python
def read(self):
    f = open(self.file_path, 'r')  # ❌ Sin context manager
    data = json.load(f)
    f.close()
    if 'records' in data:
    return data['records']  # ❌ Indentación incorrecta
    return []

def update(self, record_id, updates):
    records = self.read()
    for record in records:
        if record['id'] == record_id:
            record = updates  # ❌ Reemplaza todo
            break
    self.write(records)  # ❌ Indentación incorrecta
```

### ✅ Después (Refactorizado)
```python
def read(self) -> List[Dict[str, Any]]:
    try:
        with open(self.file_path, 'r', encoding='utf-8') as f:  # ✅ Context manager
            data = json.load(f)
        
        if not isinstance(data, dict) or 'records' not in data:
            return []
        
        records = data.get('records', [])
        if not isinstance(records, list):
            return []
        
        return records  # ✅ Indentación correcta
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON: {e}") from e

def update(self, record_id: str, updates: Dict[str, Any]) -> bool:
    records = self.read()
    
    for i, record in enumerate(records):
        if record.get('id') == record_id:
            records[i].update(updates)  # ✅ Fusiona en lugar de reemplazar
            break
    
    with open(self.file_path, 'w', encoding='utf-8') as f:  # ✅ Context manager
        json.dump({"records": records}, f, indent=2, ensure_ascii=False)
    
    return True  # ✅ Indentación correcta
```

## Testing

Ejecuta los tests con:

```bash
pytest tests/test_record_storage.py -v
```

Los tests cubren:
- ✅ Operaciones básicas (read/write/update)
- ✅ Validación de entrada
- ✅ Manejo de errores
- ✅ Fusionado de actualizaciones
- ✅ Soporte Unicode
- ✅ Seguridad de context managers

## Troubleshooting

### Error: "file_path must be a non-empty string"
**Solución**: Asegúrate de pasar un string no vacío al constructor.

### Error: "records must be a list"
**Solución**: Pasa una lista de diccionarios, no un diccionario solo.

### Error: "Invalid JSON in file"
**Solución**: El archivo JSON está corrupto. Restaura desde un backup o recrea el archivo.

### Error: "Error reading file"
**Solución**: Verifica permisos de archivo y que la ruta sea correcta.

## Contribuir

Al refactorizar código similar, consulta:
- `REFACTORING_GUIDE.md` - Guía de mejores prácticas
- `REFACTORING_CHECKLIST.md` - Checklist de refactorización
- `REFACTORING_COMPARISON.md` - Comparación antes/después

## Licencia

Parte del proyecto dermatology_ai.


