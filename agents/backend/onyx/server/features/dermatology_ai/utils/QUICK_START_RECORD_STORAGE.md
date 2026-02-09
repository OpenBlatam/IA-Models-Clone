# Quick Start - RecordStorage

## Instalación Rápida

```python
from utils.record_storage import RecordStorage
```

## Ejemplo en 30 Segundos

```python
# 1. Crear instancia
storage = RecordStorage("my_data.json")

# 2. Escribir datos
storage.write([
    {"id": "1", "name": "Alice", "age": 30},
    {"id": "2", "name": "Bob", "age": 25}
])

# 3. Leer datos
records = storage.read()
print(records)

# 4. Actualizar registro
storage.update("1", {"age": 31})

# 5. Verificar actualización
updated = storage.read()
print(updated[0])  # {"id": "1", "name": "Alice", "age": 31}
```

## Características Clave

✅ **Seguro**: Context managers garantizan que los archivos se cierren  
✅ **Inteligente**: Las actualizaciones fusionan campos, no reemplazan  
✅ **Robusto**: Validación completa y manejo de errores  
✅ **Unicode**: Soporte completo para caracteres especiales  

## Casos de Uso Comunes

### Caso 1: Guardar Configuración

```python
storage = RecordStorage("config.json")
config = [
    {"id": "app", "theme": "dark", "language": "en"}
]
storage.write(config)
```

### Caso 2: Actualizar Parcialmente

```python
# Solo actualiza 'theme', preserva 'language'
storage.update("app", {"theme": "light"})
```

### Caso 3: Manejo de Errores

```python
try:
    storage.write([{"id": "1", "name": "Test"}])
except ValueError as e:
    print(f"Error: {e}")
```

## Más Información

- **Documentación Completa**: `README_RECORD_STORAGE.md`
- **Ejemplos**: `examples/record_storage_usage.py`
- **Tests**: `tests/test_record_storage.py`


