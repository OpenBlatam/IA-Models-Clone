# FileStorage - Guía Completa

## Descripción

`FileStorage` es una clase Python refactorizada que proporciona operaciones seguras de lectura, escritura y actualización de datos en archivos JSON. Implementa las mejores prácticas de Python incluyendo context managers, manejo de errores robusto y validación de entrada.

## Características Principales

✅ **Context Managers**: Todas las operaciones de archivo usan `with` statements  
✅ **Manejo de Errores**: Validación completa y excepciones descriptivas  
✅ **Type Hints**: Tipado completo para mejor IDE support  
✅ **Thread-Safe**: Variante disponible con locks para acceso concurrente  
✅ **Validación**: Validación de entrada en todos los métodos  
✅ **Documentación**: Docstrings completos y claros  

## Instalación

No requiere dependencias adicionales más allá de la biblioteca estándar de Python:

```python
from utils.file_storage import FileStorage
```

## Uso Básico

### Inicialización

```python
from utils.file_storage import FileStorage

storage = FileStorage("data/records.json")
```

### Escribir Datos

```python
data = [
    {"id": "1", "name": "Alice", "age": 30},
    {"id": "2", "name": "Bob", "age": 25}
]

storage.write(data)
```

### Leer Datos

```python
records = storage.read()
print(f"Total records: {len(records)}")
```

### Actualizar Registro

```python
success = storage.update("1", {"age": 31, "status": "active"})
if success:
    print("Record updated successfully")
```

### Agregar Registro

```python
storage.add({"id": "3", "name": "Charlie", "age": 28})
```

### Obtener Registro Específico

```python
record = storage.get("1")
if record:
    print(f"Found: {record['name']}")
```

### Eliminar Registro

```python
success = storage.delete("2")
if success:
    print("Record deleted")
```

## API Completa

### `__init__(file_path: str)`

Inicializa FileStorage con la ruta del archivo.

**Parámetros:**
- `file_path` (str): Ruta al archivo JSON

**Excepciones:**
- `ValueError`: Si file_path está vacío o no es string

### `write(data: List[Dict[str, Any]]) -> None`

Escribe una lista de diccionarios al archivo.

**Parámetros:**
- `data` (List[Dict]): Lista de diccionarios a escribir

**Excepciones:**
- `TypeError`: Si data no es una lista
- `ValueError`: Si los items no son diccionarios
- `IOError`: Si no se puede escribir el archivo

### `read() -> List[Dict[str, Any]]`

Lee datos del archivo.

**Retorna:**
- `List[Dict]`: Lista de diccionarios leídos

**Excepciones:**
- `json.JSONDecodeError`: Si el JSON es inválido
- `IOError`: Si no se puede leer el archivo

**Nota:** Retorna lista vacía si el archivo no existe.

### `update(record_id: str, updates: Dict[str, Any]) -> bool`

Actualiza un registro por ID.

**Parámetros:**
- `record_id` (str): ID del registro a actualizar
- `updates` (Dict): Diccionario con campos a actualizar

**Retorna:**
- `bool`: True si se actualizó, False si no se encontró

**Excepciones:**
- `TypeError`: Si record_id no es string o updates no es dict
- `ValueError`: Si record_id está vacío o updates está vacío
- `IOError`: Si fallan las operaciones de archivo

### `add(record: Dict[str, Any]) -> None`

Agrega un nuevo registro.

**Parámetros:**
- `record` (Dict): Diccionario del registro a agregar

**Excepciones:**
- `TypeError`: Si record no es un diccionario
- `ValueError`: Si record está vacío
- `IOError`: Si fallan las operaciones de archivo

### `delete(record_id: str) -> bool`

Elimina un registro por ID.

**Parámetros:**
- `record_id` (str): ID del registro a eliminar

**Retorna:**
- `bool`: True si se eliminó, False si no se encontró

**Excepciones:**
- `TypeError`: Si record_id no es string
- `ValueError`: Si record_id está vacío
- `IOError`: Si fallan las operaciones de archivo

### `get(record_id: str) -> Optional[Dict[str, Any]]`

Obtiene un registro específico por ID.

**Parámetros:**
- `record_id` (str): ID del registro a obtener

**Retorna:**
- `Optional[Dict]`: El registro encontrado o None

**Excepciones:**
- `TypeError`: Si record_id no es string
- `ValueError`: Si record_id está vacío
- `IOError`: Si fallan las operaciones de archivo

## Ejemplos de Uso

### Ejemplo 1: CRUD Completo

```python
from utils.file_storage import FileStorage

storage = FileStorage("data/users.json")

# Create
storage.write([
    {"id": "1", "name": "Alice", "email": "alice@example.com"},
    {"id": "2", "name": "Bob", "email": "bob@example.com"}
])

# Read
users = storage.read()
print(f"Total users: {len(users)}")

# Update
storage.update("1", {"email": "alice.new@example.com"})

# Get specific
user = storage.get("1")
print(f"User: {user['name']} - {user['email']}")

# Delete
storage.delete("2")
```

### Ejemplo 2: Manejo de Errores

```python
from utils.file_storage import FileStorage

storage = FileStorage("data/data.json")

try:
    # Intentar escribir datos inválidos
    storage.write("not a list")
except TypeError as e:
    print(f"Error de tipo: {e}")

try:
    # Intentar actualizar con ID inválido
    storage.update(123, {"key": "value"})
except TypeError as e:
    print(f"Error de tipo: {e}")

try:
    # Intentar actualizar registro inexistente
    success = storage.update("999", {"key": "value"})
    if not success:
        print("Registro no encontrado")
except IOError as e:
    print(f"Error de I/O: {e}")
```

### Ejemplo 3: Operaciones en Lote

```python
from utils.file_storage import FileStorage

storage = FileStorage("data/products.json")

# Cargar productos iniciales
products = [
    {"id": "1", "name": "Product A", "price": 10.99, "stock": 100},
    {"id": "2", "name": "Product B", "price": 20.99, "stock": 50},
    {"id": "3", "name": "Product C", "price": 30.99, "stock": 25}
]
storage.write(products)

# Actualizar múltiples productos
for product_id in ["1", "2"]:
    storage.update(product_id, {"stock": 0})

# Verificar cambios
for product_id in ["1", "2", "3"]:
    product = storage.get(product_id)
    if product:
        print(f"{product['name']}: Stock = {product['stock']}")
```

## Variantes Disponibles

### ThreadSafeFileStorage

Para acceso concurrente seguro:

```python
from utils.file_storage_variants import ThreadSafeFileStorage

storage = ThreadSafeFileStorage("data/concurrent.json")
```

### CachedFileStorage

Para reducir I/O con cache en memoria:

```python
from utils.file_storage_variants import CachedFileStorage

storage = CachedFileStorage("data/cached.json", cache_ttl=60.0)
```

### CompressedFileStorage

Para archivos grandes con compresión:

```python
from utils.file_storage_variants import CompressedFileStorage

storage = CompressedFileStorage("data/compressed.json.gz")
```

Ver `utils/file_storage_variants.py` para más variantes.

## Mejores Prácticas

### 1. Siempre Maneja Errores

```python
try:
    storage.write(data)
except (TypeError, ValueError) as e:
    print(f"Error de validación: {e}")
except IOError as e:
    print(f"Error de I/O: {e}")
```

### 2. Valida Datos Antes de Escribir

```python
def safe_write(storage, data):
    if not isinstance(data, list):
        raise ValueError("Data must be a list")
    
    if not all('id' in item for item in data):
        raise ValueError("All items must have 'id' field")
    
    storage.write(data)
```

### 3. Usa Type Hints

```python
from typing import List, Dict, Any

def process_records(storage: FileStorage) -> List[Dict[str, Any]]:
    records = storage.read()
    # Procesar registros
    return records
```

### 4. Crea Backups Importantes

```python
from utils.file_storage_variants import BackupFileStorage

storage = BackupFileStorage("data/important.json", max_backups=10)
# Los backups se crean automáticamente antes de escribir
```

## Testing

Ejecutar tests:

```bash
pytest tests/test_file_storage.py -v
```

O ejecutar un test específico:

```bash
pytest tests/test_file_storage.py::TestFileStorage::test_update_existing_record -v
```

## Troubleshooting

### Error: "File does not contain a valid list"

El archivo existe pero no contiene una lista JSON válida. Verifica el contenido del archivo.

### Error: "Failed to write to file"

Verifica permisos de escritura en el directorio y que haya espacio en disco.

### Error: "record_id must be a string"

Asegúrate de pasar el ID como string, no como número.

### Performance con Archivos Grandes

Para archivos grandes (>10MB), considera usar:
- `CompressedFileStorage` para reducir tamaño
- `CachedFileStorage` para reducir lecturas
- `IndexedFileStorage` para búsquedas rápidas

## Contribuir

Al agregar nuevas funcionalidades:

1. Mantén el uso de context managers
2. Agrega type hints
3. Documenta con docstrings
4. Escribe tests
5. Actualiza esta documentación

## Licencia

Parte del proyecto Addiction Recovery AI.

## Referencias

- [Documentación de Refactorización](docs/REFACTORING_FILE_STORAGE.md)
- [Guía de Migración](docs/MIGRATION_GUIDE.md)
- [Mejores Prácticas](docs/BEST_PRACTICES_FILE_STORAGE.md)
- [Comparación Antes/Después](docs/BEFORE_AFTER_COMPARISON.md)


