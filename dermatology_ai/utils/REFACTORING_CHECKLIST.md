# Checklist de Refactorización - Operaciones de Archivo

Use este checklist cuando refactorice código que maneja operaciones de archivo.

## Pre-Refactorización

- [ ] Identificar todos los archivos que necesitan refactorización
- [ ] Revisar el código actual para entender su funcionalidad
- [ ] Identificar dependencias y referencias al código
- [ ] Crear backup o branch antes de refactorizar

## Context Managers

- [ ] Reemplazar `f = open(...)` con `with open(...) as f:`
- [ ] Eliminar todas las llamadas a `.close()` manuales
- [ ] Verificar que todos los archivos se abren dentro de context managers
- [ ] Asegurar que el código dentro del `with` bloque es correcto

## Indentación

- [ ] Verificar indentación en métodos `read()`
- [ ] Verificar indentación en métodos `write()`
- [ ] Verificar indentación en métodos `update()`
- [ ] Asegurar que el flujo de control es correcto
- [ ] Verificar que los bloques `if/for/while` están correctamente indentados

## Manejo de Actualizaciones

- [ ] Verificar que `update()` fusiona campos en lugar de reemplazar
- [ ] Asegurar que el ID original se preserva
- [ ] Verificar que los registros actualizados se guardan correctamente
- [ ] Asegurar que `write()` se llama después de todas las actualizaciones

## Validación de Entrada

- [ ] Validar tipos de parámetros en `__init__()`
- [ ] Validar tipos de parámetros en `read()`
- [ ] Validar tipos de parámetros en `write()`
- [ ] Validar tipos de parámetros en `update()`
- [ ] Validar valores (no vacíos, rangos válidos, etc.)
- [ ] Proporcionar mensajes de error claros y específicos

## Manejo de Errores

- [ ] Agregar try/except blocks donde sea necesario
- [ ] Usar excepciones específicas (IOError, OSError, ValueError, etc.)
- [ ] Proporcionar mensajes de error informativos
- [ ] Usar `raise ... from e` para preservar el traceback
- [ ] Manejar casos edge (archivo no existe, JSON inválido, etc.)

## Encoding

- [ ] Especificar `encoding='utf-8'` en todas las operaciones de archivo
- [ ] Usar `ensure_ascii=False` en `json.dump()` para soporte Unicode
- [ ] Verificar que los caracteres Unicode se manejan correctamente

## Type Hints

- [ ] Agregar type hints a todos los métodos
- [ ] Importar tipos necesarios de `typing`
- [ ] Especificar tipos de retorno
- [ ] Especificar tipos de parámetros

## Documentación

- [ ] Agregar docstrings a la clase
- [ ] Agregar docstrings a todos los métodos
- [ ] Documentar parámetros con `Args:`
- [ ] Documentar valores de retorno con `Returns:`
- [ ] Documentar excepciones con `Raises:`

## Testing

- [ ] Crear tests para casos básicos
- [ ] Crear tests para casos edge
- [ ] Crear tests para manejo de errores
- [ ] Crear tests para validación de entrada
- [ ] Verificar que los tests pasan
- [ ] Verificar cobertura de código

## Post-Refactorización

- [ ] Ejecutar linter y corregir errores
- [ ] Ejecutar tests y verificar que pasan
- [ ] Revisar el código refactorizado
- [ ] Actualizar documentación si es necesario
- [ ] Verificar que no se rompieron dependencias
- [ ] Actualizar ejemplos de uso si es necesario

## Verificación Final

- [ ] El código usa context managers en todas las operaciones de archivo
- [ ] La indentación es correcta en todos los métodos
- [ ] Las actualizaciones fusionan en lugar de reemplazar
- [ ] Hay validación de entrada en todos los métodos
- [ ] Hay manejo de errores apropiado
- [ ] El encoding está especificado (UTF-8)
- [ ] Hay type hints completos
- [ ] Hay documentación completa
- [ ] Los tests pasan
- [ ] No hay errores de linting

## Ejemplo de Código Refactorizado

```python
class RecordStorage:
    def __init__(self, file_path: str):
        if not file_path or not isinstance(file_path, str):
            raise ValueError("file_path must be a non-empty string")
        
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.file_path.exists():
            self._initialize_file()
    
    def read(self) -> List[Dict[str, Any]]:
        if not self.file_path.exists():
            return []
        
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, dict) or 'records' not in data:
                return []
            
            records = data.get('records', [])
            if not isinstance(records, list):
                return []
            
            return records
            
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON in file: {e}") from e
        except (IOError, OSError) as e:
            raise RuntimeError(f"Error reading file: {e}") from e
    
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
    
    def update(self, record_id: str, updates: Dict[str, Any]) -> bool:
        if not isinstance(record_id, str) or not record_id:
            raise ValueError("record_id must be a non-empty string")
        
        if not isinstance(updates, dict):
            raise ValueError("updates must be a dictionary")
        
        if not updates:
            return False
        
        try:
            records = self.read()
            
            record_found = False
            for i, record in enumerate(records):
                if not isinstance(record, dict):
                    continue
                
                if record.get('id') == record_id:
                    original_id = record.get('id')
                    records[i].update(updates)
                    if 'id' not in records[i] or records[i].get('id') != original_id:
                        records[i]['id'] = original_id
                    record_found = True
                    break
            
            if not record_found:
                return False
            
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump({"records": records}, f, indent=2, ensure_ascii=False)
            
            return True
            
        except (IOError, OSError) as e:
            raise RuntimeError(f"Error updating record: {e}") from e
        except (ValueError, TypeError) as e:
            raise
        except Exception as e:
            raise RuntimeError(f"Unexpected error: {e}") from e
```

## Recursos Adicionales

- Ver `REFACTORING_GUIDE.md` para guía detallada
- Ver `REFACTORING_COMPARISON.md` para comparación antes/después
- Ver `examples/record_storage_usage.py` para ejemplos de uso
- Ver `tests/test_record_storage.py` para ejemplos de testing


