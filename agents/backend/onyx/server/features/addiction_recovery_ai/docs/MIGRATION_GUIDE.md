# Guía de Migración - FileStorage Refactorizado

## Resumen

Esta guía te ayudará a migrar código existente que usa operaciones de archivo sin context managers al nuevo `FileStorage` refactorizado.

## Paso 1: Identificar Código a Migrar

### Código Problemático Típico

```python
# ❌ ANTES - Código problemático
class OldStorage:
    def __init__(self, file_path):
        self.file_path = file_path
    
    def read(self):
        f = open(self.file_path, 'r')
        data = json.load(f)
        f.close()
        return data
    
    def write(self, data):
        f = open(self.file_path, 'w')
        json.dump(data, f)
        f.close()
    
    def update(self, record_id, updates):
        records = self.read()
        for record in records:
            if record['id'] == record_id:
                record.update(updates)
                break
        # ❌ FALTA: Escribir de vuelta
```

## Paso 2: Reemplazar con FileStorage

### Opción A: Reemplazo Directo

```python
# ✅ DESPUÉS - Usando FileStorage
from utils.file_storage import FileStorage

# Reemplazar instanciación
old_storage = OldStorage("data.json")
new_storage = FileStorage("data.json")

# Los métodos son compatibles
data = new_storage.read()
new_storage.write(data)
new_storage.update("1", {"key": "value"})
```

### Opción B: Wrapper para Compatibilidad

```python
# Wrapper para mantener compatibilidad con código existente
class StorageWrapper:
    def __init__(self, file_path):
        self._storage = FileStorage(file_path)
    
    def read(self):
        return self._storage.read()
    
    def write(self, data):
        self._storage.write(data)
    
    def update(self, record_id, updates):
        return self._storage.update(record_id, updates)
```

## Paso 3: Actualizar Llamadas

### Antes
```python
storage = OldStorage("data.json")
records = storage.read()
for record in records:
    if record['id'] == "1":
        record['status'] = 'active'
        break
storage.write(records)  # Manual
```

### Después
```python
storage = FileStorage("data.json")
records = storage.read()
for record in records:
    if record['id'] == "1":
        record['status'] = 'active'
        break
storage.write(records)  # O mejor aún:
storage.update("1", {"status": "active"})  # Automático
```

## Paso 4: Manejo de Errores

### Antes
```python
try:
    f = open("data.json", 'r')
    data = json.load(f)
    f.close()
except:
    pass  # ❌ Silencia errores
```

### Después
```python
try:
    storage = FileStorage("data.json")
    data = storage.read()
except IOError as e:
    print(f"Error reading file: {e}")
except json.JSONDecodeError as e:
    print(f"Invalid JSON: {e}")
except ValueError as e:
    print(f"Validation error: {e}")
```

## Paso 5: Validación de Entrada

### Antes
```python
def update_record(record_id, updates):
    # ❌ Sin validación
    records = read_records()
    for record in records:
        if record['id'] == record_id:
            record.update(updates)
            break
```

### Después
```python
def update_record(record_id, updates):
    storage = FileStorage("data.json")
    
    # ✅ Validación automática
    try:
        success = storage.update(record_id, updates)
        if not success:
            print(f"Record {record_id} not found")
    except TypeError as e:
        print(f"Invalid input: {e}")
    except ValueError as e:
        print(f"Validation error: {e}")
```

## Checklist de Migración

### Pre-Migración
- [ ] Identificar todos los archivos que usan operaciones de archivo
- [ ] Documentar el comportamiento actual
- [ ] Crear backups de datos importantes
- [ ] Revisar dependencias

### Durante Migración
- [ ] Reemplazar `open()` sin context managers
- [ ] Actualizar llamadas a métodos
- [ ] Agregar manejo de errores apropiado
- [ ] Validar entrada de datos
- [ ] Corregir indentación

### Post-Migración
- [ ] Ejecutar tests
- [ ] Verificar funcionalidad
- [ ] Revisar logs de errores
- [ ] Actualizar documentación
- [ ] Entrenar al equipo

## Ejemplos de Migración por Escenario

### Escenario 1: Lectura Simple

**Antes:**
```python
f = open("data.json", 'r')
data = json.load(f)
f.close()
```

**Después:**
```python
storage = FileStorage("data.json")
data = storage.read()
```

### Escenario 2: Escritura Simple

**Antes:**
```python
f = open("data.json", 'w')
json.dump(data, f)
f.close()
```

**Después:**
```python
storage = FileStorage("data.json")
storage.write(data)
```

### Escenario 3: Actualización de Registro

**Antes:**
```python
f = open("data.json", 'r')
records = json.load(f)
f.close()

for record in records:
    if record['id'] == "1":
        record.update({"status": "active"})
        break

f = open("data.json", 'w')
json.dump(records, f)
f.close()
```

**Después:**
```python
storage = FileStorage("data.json")
storage.update("1", {"status": "active"})
```

### Escenario 4: Operaciones Múltiples

**Antes:**
```python
# Leer
f = open("data.json", 'r')
records = json.load(f)
f.close()

# Modificar
records.append({"id": "2", "name": "Bob"})

# Escribir
f = open("data.json", 'w')
json.dump(records, f)
f.close()
```

**Después:**
```python
storage = FileStorage("data.json")
storage.add({"id": "2", "name": "Bob"})
```

## Manejo de Errores Específicos

### Error: Archivo No Encontrado

**Antes:**
```python
try:
    f = open("data.json", 'r')
    data = json.load(f)
    f.close()
except FileNotFoundError:
    data = []
```

**Después:**
```python
storage = FileStorage("data.json")
data = storage.read()  # Retorna [] automáticamente si no existe
```

### Error: JSON Inválido

**Antes:**
```python
try:
    f = open("data.json", 'r')
    data = json.load(f)
    f.close()
except json.JSONDecodeError:
    data = []
```

**Después:**
```python
storage = FileStorage("data.json")
try:
    data = storage.read()
except json.JSONDecodeError as e:
    print(f"Invalid JSON: {e}")
    data = []
```

### Error: Permisos

**Antes:**
```python
try:
    f = open("data.json", 'w')
    json.dump(data, f)
    f.close()
except PermissionError:
    print("No permission")
```

**Después:**
```python
storage = FileStorage("data.json")
try:
    storage.write(data)
except IOError as e:
    if "Permission" in str(e):
        print("No permission")
    else:
        print(f"Error: {e}")
```

## Testing Post-Migración

### Test Básico
```python
def test_migration():
    storage = FileStorage("test.json")
    
    # Test write
    storage.write([{"id": "1", "name": "Test"}])
    
    # Test read
    data = storage.read()
    assert len(data) == 1
    
    # Test update
    storage.update("1", {"name": "Updated"})
    record = storage.get("1")
    assert record["name"] == "Updated"
    
    # Cleanup
    os.remove("test.json")
```

## Recursos Adicionales

- Ver `examples/file_storage_example.py` para ejemplos básicos
- Ver `examples/file_storage_advanced_example.py` para patrones avanzados
- Ver `tests/test_file_storage.py` para casos de prueba
- Ver `docs/BEFORE_AFTER_COMPARISON.md` para comparaciones detalladas

## Soporte

Si encuentras problemas durante la migración:
1. Revisa los logs de errores
2. Consulta la documentación
3. Ejecuta los tests
4. Verifica que los datos se mantienen correctamente


