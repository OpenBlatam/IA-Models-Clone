# Mejores Prácticas - FileStorage

## Principios Fundamentales

### 1. Siempre Usa Context Managers

✅ **Correcto:**
```python
with open(file_path, 'r') as f:
    data = json.load(f)
```

❌ **Incorrecto:**
```python
f = open(file_path, 'r')
data = json.load(f)
f.close()  # Puede no ejecutarse
```

### 2. Valida Entrada Antes de Procesar

✅ **Correcto:**
```python
def update(self, record_id: str, updates: Dict[str, Any]) -> bool:
    if not isinstance(record_id, str):
        raise TypeError("record_id must be a string")
    if not record_id:
        raise ValueError("record_id cannot be empty")
    # ... resto del código
```

❌ **Incorrecto:**
```python
def update(self, record_id, updates):
    # Sin validación
    records = self.read()
    # ... puede fallar silenciosamente
```

### 3. Maneja Errores Específicos

✅ **Correcto:**
```python
try:
    data = storage.read()
except FileNotFoundError:
    return []
except json.JSONDecodeError as e:
    logger.error(f"Invalid JSON: {e}")
    raise
except IOError as e:
    logger.error(f"IO Error: {e}")
    raise
```

❌ **Incorrecto:**
```python
try:
    data = storage.read()
except:  # Demasiado amplio
    return []
```

### 4. Usa Type Hints

✅ **Correcto:**
```python
def read(self) -> List[Dict[str, Any]]:
    """Read data from file"""
    pass
```

❌ **Incorrecto:**
```python
def read(self):
    """Read data from file"""
    pass
```

## Patrones de Uso

### Patrón 1: Operaciones Atómicas

```python
def atomic_operation(storage: FileStorage, record_id: str, updates: Dict):
    """Operación atómica con rollback"""
    try:
        # Leer estado actual
        records = storage.read()
        backup = records.copy()
        
        # Aplicar cambios
        for i, record in enumerate(records):
            if record.get('id') == record_id:
                records[i].update(updates)
                break
        
        # Validar
        if not validate(records):
            # Rollback
            storage.write(backup)
            return False
        
        # Commit
        storage.write(records)
        return True
        
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        return False
```

### Patrón 2: Transacciones Múltiples

```python
def batch_operations(storage: FileStorage, operations: List[Dict]):
    """Ejecutar múltiples operaciones como transacción"""
    records = storage.read()
    backup = records.copy()
    
    try:
        for op in operations:
            if op['type'] == 'update':
                # Aplicar update
                pass
            elif op['type'] == 'delete':
                # Aplicar delete
                pass
        
        # Validar todo
        if validate_all(records):
            storage.write(records)
            return True
        else:
            storage.write(backup)  # Rollback
            return False
            
    except Exception as e:
        storage.write(backup)  # Rollback
        raise
```

### Patrón 3: Cache con Validación

```python
class CachedFileStorage(FileStorage):
    def __init__(self, file_path: str):
        super().__init__(file_path)
        self._cache = None
        self._cache_time = None
    
    def read(self, use_cache: bool = True) -> List[Dict[str, Any]]:
        """Read with optional caching"""
        if use_cache and self._cache is not None:
            return self._cache
        
        data = super().read()
        self._cache = data
        self._cache_time = time.time()
        return data
    
    def invalidate_cache(self):
        """Invalidate cache"""
        self._cache = None
        self._cache_time = None
```

### Patrón 4: Logging de Operaciones

```python
import logging

logger = logging.getLogger(__name__)

class LoggedFileStorage(FileStorage):
    def write(self, data: List[Dict[str, Any]]) -> None:
        logger.info(f"Writing {len(data)} records to {self.file_path}")
        try:
            super().write(data)
            logger.info("Write successful")
        except Exception as e:
            logger.error(f"Write failed: {e}")
            raise
    
    def update(self, record_id: str, updates: Dict[str, Any]) -> bool:
        logger.debug(f"Updating record {record_id}")
        result = super().update(record_id, updates)
        if result:
            logger.info(f"Record {record_id} updated successfully")
        else:
            logger.warning(f"Record {record_id} not found")
        return result
```

## Optimizaciones

### 1. Lectura Lazy

```python
class LazyFileStorage(FileStorage):
    def __init__(self, file_path: str):
        super().__init__(file_path)
        self._data = None
        self._dirty = False
    
    def read(self) -> List[Dict[str, Any]]:
        if self._data is None:
            self._data = super().read()
        return self._data
    
    def write(self, data: List[Dict[str, Any]]) -> None:
        self._data = data
        self._dirty = True
    
    def flush(self):
        """Write pending changes to disk"""
        if self._dirty and self._data is not None:
            super().write(self._data)
            self._dirty = False
```

### 2. Batching de Escrituras

```python
class BatchedFileStorage(FileStorage):
    def __init__(self, file_path: str, batch_size: int = 10):
        super().__init__(file_path)
        self.batch_size = batch_size
        self.pending_writes = []
    
    def write(self, data: List[Dict[str, Any]]) -> None:
        self.pending_writes.append(data)
        
        if len(self.pending_writes) >= self.batch_size:
            self.flush()
    
    def flush(self):
        """Write all pending writes"""
        if self.pending_writes:
            # Aplicar todas las escrituras
            final_data = self.pending_writes[-1]
            super().write(final_data)
            self.pending_writes.clear()
```

## Seguridad

### 1. Validación de Paths

```python
def validate_file_path(file_path: str) -> bool:
    """Validar que el path es seguro"""
    # Prevenir path traversal
    if '..' in file_path:
        return False
    
    # Prevenir paths absolutos en ciertos contextos
    if os.path.isabs(file_path) and not file_path.startswith('/safe/dir'):
        return False
    
    return True
```

### 2. Sanitización de Datos

```python
def sanitize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitizar datos antes de guardar"""
    sanitized = {}
    
    for key, value in record.items():
        # Validar key
        if not isinstance(key, str) or len(key) > 100:
            continue
        
        # Sanitizar value
        if isinstance(value, str):
            value = value[:1000]  # Limitar longitud
        elif isinstance(value, (int, float)):
            # Validar rangos
            pass
        
        sanitized[key] = value
    
    return sanitized
```

### 3. Permisos de Archivo

```python
import stat

def set_secure_permissions(file_path: str):
    """Establecer permisos seguros"""
    os.chmod(file_path, stat.S_IRUSR | stat.S_IWUSR)  # Solo owner
```

## Testing

### 1. Tests Unitarios

```python
import pytest
import tempfile

def test_file_storage_basic():
    with tempfile.NamedTemporaryFile(delete=False) as f:
        temp_path = f.name
    
    try:
        storage = FileStorage(temp_path)
        
        # Test write
        storage.write([{"id": "1", "name": "Test"}])
        
        # Test read
        data = storage.read()
        assert len(data) == 1
        assert data[0]["name"] == "Test"
        
        # Test update
        storage.update("1", {"name": "Updated"})
        record = storage.get("1")
        assert record["name"] == "Updated"
        
    finally:
        os.unlink(temp_path)
```

### 2. Tests de Integración

```python
def test_concurrent_access():
    """Test acceso concurrente"""
    storage = FileStorage("test_concurrent.json")
    
    def worker(worker_id):
        for i in range(10):
            storage.add({"id": f"{worker_id}-{i}", "value": i})
    
    # Ejecutar múltiples workers
    threads = [threading.Thread(target=worker, args=(i,)) 
               for i in range(5)]
    
    for t in threads:
        t.start()
    
    for t in threads:
        t.join()
    
    # Verificar resultados
    records = storage.read()
    assert len(records) == 50
```

## Performance

### 1. Índices para Búsquedas Rápidas

```python
class IndexedFileStorage(FileStorage):
    def __init__(self, file_path: str):
        super().__init__(file_path)
        self._index = {}
        self._rebuild_index()
    
    def _rebuild_index(self):
        """Rebuild index from current data"""
        records = super().read()
        self._index = {r.get('id'): i for i, r in enumerate(records)}
    
    def get(self, record_id: str) -> Optional[Dict[str, Any]]:
        if record_id in self._index:
            records = super().read()
            return records[self._index[record_id]]
        return None
```

### 2. Compresión para Archivos Grandes

```python
import gzip
import json

class CompressedFileStorage(FileStorage):
    def read(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.file_path):
            return []
        
        try:
            with gzip.open(self.file_path, 'rt', encoding='utf-8') as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception as e:
            raise IOError(f"Failed to read: {e}")
    
    def write(self, data: List[Dict[str, Any]]) -> None:
        try:
            with gzip.open(self.file_path, 'wt', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            raise IOError(f"Failed to write: {e}")
```

## Resumen de Mejores Prácticas

1. ✅ **Siempre usa context managers** para operaciones de archivo
2. ✅ **Valida entrada** antes de procesar
3. ✅ **Maneja errores específicos** con try-except apropiados
4. ✅ **Usa type hints** para mejor documentación y IDE support
5. ✅ **Documenta con docstrings** claros y completos
6. ✅ **Escribe tests** para validar funcionalidad
7. ✅ **Considera performance** para operaciones frecuentes
8. ✅ **Implementa seguridad** en validación y permisos
9. ✅ **Usa logging** para debugging y monitoreo
10. ✅ **Mantén código simple** y legible


