# Guía de Refactorización - Mejores Prácticas para Operaciones de Archivo

## Resumen

Esta guía documenta las mejoras realizadas en el código para seguir las mejores prácticas de Python, especialmente en el manejo de archivos, manejo de errores y validación de entrada.

## Problemas Identificados y Soluciones

### 1. Uso de Context Managers (`with` statement)

#### ❌ Problema
```python
def read(self):
    f = open(self.file_path, 'r')
    data = json.load(f)
    f.close()
    return data
```

**Problemas:**
- Si ocurre una excepción antes de `f.close()`, el archivo puede no cerrarse
- No hay garantía de que el archivo se cierre en todos los casos
- Código más propenso a errores

#### ✅ Solución
```python
def read(self) -> List[Dict[str, Any]]:
    with open(self.file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data
```

**Beneficios:**
- El archivo se cierra automáticamente, incluso si ocurre una excepción
- Código más limpio y seguro
- Cumple con las mejores prácticas de Python

### 2. Corrección de Indentación

#### ❌ Problema
```python
def read(self):
    data = json.load(f)
    if 'records' in data:
    return data['records']  # ❌ Indentación incorrecta
    return []
```

#### ✅ Solución
```python
def read(self) -> List[Dict[str, Any]]:
    with open(self.file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if 'records' in data:
        return data['records']  # ✅ Indentación correcta
    return []
```

### 3. Manejo Correcto de Actualizaciones

#### ❌ Problema
```python
def update(self, record_id, updates):
    records = self.read()
    for record in records:
        if record['id'] == record_id:
            record = updates  # ❌ Reemplaza todo el registro
            break
    self.write(records)  # ❌ Indentación incorrecta
```

**Problemas:**
- Reemplaza todo el registro en lugar de fusionar actualizaciones
- Puede perder datos del registro original
- Indentación incorrecta puede causar que `write()` se ejecute incorrectamente

#### ✅ Solución
```python
def update(self, record_id: str, updates: Dict[str, Any]) -> bool:
    records = self.read()
    
    record_found = False
    for i, record in enumerate(records):
        if record.get('id') == record_id:
            original_id = record.get('id')
            records[i].update(updates)  # ✅ Fusiona actualizaciones
            if 'id' not in records[i] or records[i].get('id') != original_id:
                records[i]['id'] = original_id
            record_found = True
            break
    
    if not record_found:
        return False
    
    # ✅ Indentación correcta - write después del loop
    with open(self.file_path, 'w', encoding='utf-8') as f:
        json.dump({"records": records}, f, indent=2, ensure_ascii=False)
    
    return True
```

**Beneficios:**
- Fusiona actualizaciones en lugar de reemplazar
- Preserva el ID original del registro
- Manejo correcto del flujo de control

### 4. Validación de Entrada y Manejo de Errores

#### ❌ Problema
```python
def write(self, records):
    f = open(self.file_path, 'w')
    json.dump({"records": records}, f)
    f.close()
```

**Problemas:**
- No valida que `records` sea una lista
- No valida que cada elemento sea un diccionario
- No maneja errores de escritura
- No especifica encoding

#### ✅ Solución
```python
def write(self, records: List[Dict[str, Any]]) -> bool:
    if not isinstance(records, list):
        raise ValueError("records must be a list")
    
    for i, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Item at index {i} is not a dictionary")
    
    try:
        data = {"records": records}
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except (IOError, OSError) as e:
        raise RuntimeError(f"Error writing file: {e}") from e
    except (TypeError, ValueError) as e:
        raise RuntimeError(f"Error serializing data: {e}") from e
```

**Beneficios:**
- Validación completa de tipos
- Mensajes de error claros y específicos
- Encoding explícito (UTF-8)
- Manejo robusto de excepciones

## Ejemplo Completo Refactorizado

Ver el archivo `utils/record_storage.py` para una implementación completa que incluye:

- ✅ Context managers para todas las operaciones de archivo
- ✅ Indentación correcta en todos los métodos
- ✅ Manejo correcto de actualizaciones (fusión en lugar de reemplazo)
- ✅ Validación completa de entrada
- ✅ Manejo robusto de errores con mensajes claros
- ✅ Type hints para mejor claridad del código
- ✅ Documentación completa con docstrings
- ✅ Inicialización automática de archivos

## Mejores Prácticas Aplicadas

1. **Context Managers**: Siempre usar `with open(...) as f:` para operaciones de archivo
2. **Encoding Explícito**: Especificar `encoding='utf-8'` en todas las operaciones de archivo
3. **Validación de Entrada**: Validar tipos y valores antes de procesar
4. **Manejo de Errores**: Usar try/except con excepciones específicas y mensajes claros
5. **Type Hints**: Agregar type hints para mejorar la claridad y el soporte del IDE
6. **Documentación**: Incluir docstrings completos para todos los métodos
7. **Path Handling**: Usar `pathlib.Path` para manejo de rutas más robusto

## Checklist de Refactorización

Al refactorizar código con operaciones de archivo, asegúrate de:

- [ ] Reemplazar `f = open(...)` con `with open(...) as f:`
- [ ] Eliminar todas las llamadas a `.close()` manuales
- [ ] Agregar `encoding='utf-8'` a todas las operaciones de archivo
- [ ] Validar tipos de entrada antes de procesar
- [ ] Agregar manejo de errores con try/except
- [ ] Verificar indentación correcta en todos los métodos
- [ ] Agregar type hints donde sea apropiado
- [ ] Agregar docstrings para documentación
- [ ] Probar el código refactorizado

## Referencias

- [Python File I/O Documentation](https://docs.python.org/3/tutorial/inputoutput.html#reading-and-writing-files)
- [PEP 8 Style Guide](https://pep8.org/)
- [Python Type Hints](https://docs.python.org/3/library/typing.html)


