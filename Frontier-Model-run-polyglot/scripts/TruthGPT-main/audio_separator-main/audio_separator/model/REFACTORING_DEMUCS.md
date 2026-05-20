# Refactorización de DemucsModel

## 📋 Resumen

Refactorización completa de `demucs_model.py` aplicando principios SOLID, DRY y mejores prácticas, siguiendo el mismo enfoque usado en el módulo de optimizers.

## 🎯 Mejoras Aplicadas

### 1. **Constantes Extraídas**

**Antes:**
```python
variant: str = "htdemucs"  # Valor hardcodeado
source_names = getattr(self.model, 'sources', ['vocals', 'drums', 'bass', 'other'])  # Duplicado
```

**Después:**
```python
# ════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════════

DEFAULT_VARIANT = "htdemucs"
DEFAULT_SOURCE_NAMES = ['vocals', 'drums', 'bass', 'other']
```

**Beneficios:**
- ✅ Valores centralizados y fáciles de cambiar
- ✅ Eliminación de valores hardcodeados
- ✅ Mejor mantenibilidad

### 2. **Método Helper para Source Names**

**Problema**: Lógica duplicada para obtener source names (líneas 75-76 y 122-123).

**Antes:**
```python
# En forward()
source_names = getattr(self.model, 'sources', 
                      [f'source_{i}' for i in range(self.num_sources)])

# En separate()
source_names = getattr(self.model, 'sources', 
                      ['vocals', 'drums', 'bass', 'other'])
```

**Después:**
```python
def _get_source_names(self) -> List[str]:
    """Get source names for the model."""
    if self._source_names is not None:
        return self._source_names
    
    if hasattr(self.model, 'sources'):
        return list(self.model.sources)
    
    # Fallback to default or generated names
    return DEFAULT_SOURCE_NAMES[:self.num_sources] if self.num_sources <= len(DEFAULT_SOURCE_NAMES) else [
        f'source_{i}' for i in range(self.num_sources)
    ]
```

**Beneficios:**
- ✅ Eliminación de duplicación
- ✅ Lógica centralizada
- ✅ Más fácil de mantener

### 3. **Métodos Helper Extraídos**

**Antes:**
```python
def separate(...):
    # ... código ...
    if output_dir is None:
        output_dir = Path(audio_path).parent / "separated"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ... más código ...
    
    # Map output files
    audio_name = Path(audio_path).stem
    result = {}
    source_names = getattr(self.model, 'sources', ...)
    for source_name in source_names:
        output_path = output_dir / self.variant / audio_name / f"{source_name}.wav"
        if output_path.exists():
            result[source_name] = str(output_path)
    return result
```

**Después:**
```python
def _prepare_output_dir(self, audio_path: str, output_dir: Optional[str]) -> Path:
    """Prepare output directory for separation results."""
    # Lógica clara y enfocada

def _map_output_files(self, output_dir: Path, audio_name: str) -> Dict[str, str]:
    """Map output files to source names."""
    # Lógica clara y enfocada

def separate(...):
    # Método principal más claro
    output_dir = self._prepare_output_dir(audio_path, output_dir)
    # ... separación ...
    result = self._map_output_files(output_dir, audio_name)
    return result
```

**Beneficios:**
- ✅ Métodos pequeños y enfocados (SRP)
- ✅ Más fácil de testear
- ✅ Más fácil de mantener

### 4. **Mejora en Manejo de Errores**

**Antes:**
```python
except ImportError:
    raise ImportError("demucs is not installed")
except Exception as e:
    raise RuntimeError(f"Error during separation: {str(e)}")
```

**Después:**
```python
except ImportError as e:
    raise AudioModelError(
        "demucs is not installed. Install it with: pip install demucs",
        component="DemucsModel",
        error_code="DEMUCS_NOT_INSTALLED"
    ) from e
except Exception as e:
    raise AudioModelError(
        f"Error during separation: {str(e)}",
        component="DemucsModel",
        error_code="SEPARATION_FAILED"
    ) from e
```

**Beneficios:**
- ✅ Excepciones de dominio específicas
- ✅ Información de error más rica
- ✅ Mejor debugging

### 5. **Caché de Source Names**

**Antes:**
```python
# Se calculaba cada vez que se necesitaba
source_names = getattr(self.model, 'sources', ...)
```

**Después:**
```python
self._source_names: Optional[List[str]] = None  # Caché

def _load_model(self):
    # ...
    if hasattr(self.model, 'sources'):
        self._source_names = list(self.model.sources)  # Guardar una vez
        self.num_sources = len(self._source_names)

def _get_source_names(self) -> List[str]:
    if self._source_names is not None:
        return self._source_names  # Usar caché
```

**Beneficios:**
- ✅ Mejor rendimiento (no recalcula)
- ✅ Consistencia (mismo resultado siempre)
- ✅ Menos llamadas a getattr

### 6. **Logging Mejorado**

**Antes:**
```python
# Sin logging
self.model = get_model(self.variant)
```

**Después:**
```python
logger.debug(f"Loading Demucs model variant: {self.variant}")
self.model = get_model(self.variant)
logger.debug(f"Model has {self.num_sources} sources: {self._source_names}")
```

**Beneficios:**
- ✅ Mejor debugging
- ✅ Trazabilidad
- ✅ Información útil para usuarios

## 📊 Métricas

| Aspecto | Antes | Después | Mejora |
|---------|-------|---------|--------|
| **Líneas de código** | 137 | ~180 | +legibilidad |
| **Métodos helper** | 0 | 3 | **+3** |
| **Constantes** | 0 | 2 | **+2** |
| **Duplicación** | 2 lugares | 0 | **-100%** |
| **Manejo de errores** | Genérico | Específico | **⬆️** |
| **Logging** | Mínimo | Completo | **⬆️** |

## ✅ Principios Aplicados

### DRY (Don't Repeat Yourself)
- ✅ Lógica de source_names centralizada
- ✅ Métodos helper reutilizables

### Single Responsibility Principle (SRP)
- ✅ `_get_source_names()` - Solo obtener source names
- ✅ `_prepare_output_dir()` - Solo preparar directorio
- ✅ `_map_output_files()` - Solo mapear archivos

### KISS (Keep It Simple, Stupid)
- ✅ Constantes en lugar de valores hardcodeados
- ✅ Métodos pequeños y claros

### Clean Code
- ✅ Nombres descriptivos
- ✅ Logging apropiado
- ✅ Manejo de errores específico

## 🎯 Estado Final

✅ **Constantes Extraídas**  
✅ **Duplicación Eliminada**  
✅ **Métodos Helper Creados**  
✅ **Manejo de Errores Mejorado**  
✅ **Logging Mejorado**  
✅ **Caché de Source Names**  
✅ **Código Más Limpio y Mantenible**  

## 📝 Archivos Modificados

1. **`demucs_model.py`**
   - ✅ Constantes extraídas: `DEFAULT_VARIANT`, `DEFAULT_SOURCE_NAMES`
   - ✅ Método helper: `_get_source_names()`
   - ✅ Método helper: `_prepare_output_dir()`
   - ✅ Método helper: `_map_output_files()`
   - ✅ Caché de source names: `self._source_names`
   - ✅ Manejo de errores mejorado con `AudioModelError`
   - ✅ Logging mejorado

## ✨ Conclusión

El código está ahora completamente refactorizado, siguiendo los mismos principios aplicados en el módulo de optimizers. El código es más limpio, mantenible y fácil de extender.

**Refactorización completa.** 🎉

