# Quick Reference - FileStorage

## Importación Rápida

```python
from utils.file_storage import FileStorage
```

## Inicialización

```python
storage = FileStorage("path/to/file.json")
```

## Operaciones Básicas

### Escribir
```python
storage.write([{"id": "1", "name": "John"}])
```

### Leer
```python
records = storage.read()
```

### Actualizar
```python
success = storage.update("1", {"age": 30})
```

### Agregar
```python
storage.add({"id": "2", "name": "Jane"})
```

### Obtener
```python
record = storage.get("1")
```

### Eliminar
```python
success = storage.delete("1")
```

## Manejo de Errores

```python
try:
    storage.write(data)
except TypeError as e:
    print(f"Tipo inválido: {e}")
except ValueError as e:
    print(f"Valor inválido: {e}")
except IOError as e:
    print(f"Error de I/O: {e}")
```

## Validaciones Automáticas

- ✅ Tipo de datos
- ✅ Estructura de datos
- ✅ Existencia de archivos
- ✅ Validez de JSON
- ✅ Parámetros vacíos

## Características

- ✅ Context managers automáticos
- ✅ Cierre seguro de archivos
- ✅ Creación automática de directorios
- ✅ Encoding UTF-8
- ✅ Type hints completos


