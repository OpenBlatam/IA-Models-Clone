# Guía de Extensibilidad - Audio Separator

## 📋 Resumen

Esta guía explica cómo extender el código refactorizado para agregar nuevas funcionalidades sin modificar código existente, siguiendo el principio Open/Closed.

---

## 🎯 Principios de Extensibilidad

### 1. Open/Closed Principle

**Regla**: El código debe estar **abierto para extensión** pero **cerrado para modificación**.

**Aplicación en el código refactorizado:**
- ✅ Agregar nuevos tipos de normalización sin modificar código existente
- ✅ Agregar nuevos pasos al pipeline sin modificar código existente
- ✅ Agregar nuevos tipos de separators sin modificar código existente

---

## 🔧 Cómo Agregar Nuevo Tipo de Normalización

### Paso 1: Extender `_normalize_audio()` en Base Class

**Archivo**: `base_processor.py`

```python
def _normalize_audio(
    self,
    audio: np.ndarray,
    check_clipping: bool = False,
    method: str = 'peak'  # ✅ Agregar parámetro method
) -> np.ndarray:
    """
    Normalize audio with different methods.
    
    ✅ Abierto para extensión: Nuevos métodos sin modificar código existente.
    """
    if method == 'peak':
        # ✅ Implementación existente (no modificar)
        max_val = np.abs(audio).max()
        if max_val > 0:
            if check_clipping:
                if max_val > 1.0:
                    audio = audio / max_val
            else:
                audio = audio / max_val
        return audio
    
    elif method == 'rms':
        # ✅ Nueva implementación (solo agregar)
        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 0:
            target_rms = 0.1
            audio = audio * (target_rms / rms)
        return audio
    
    elif method == 'lufs':
        # ✅ Otra nueva implementación (solo agregar)
        # Implementación de LUFS normalization
        # ...
        return audio
    
    else:
        logger.warning(f"Unknown normalization method: {method}, using peak")
        return self._normalize_audio(audio, check_clipping, method='peak')
```

**Beneficios:**
- ✅ No modifica código existente
- ✅ Fácil agregar nuevos métodos
- ✅ Backward compatible

---

### Paso 2: Usar en Preprocessor/Postprocessor

**Archivo**: `preprocessor.py` o `postprocessor.py`

```python
# ✅ Usar nuevo método sin modificar código existente
def _normalize(self, audio: np.ndarray) -> np.ndarray:
    return self._normalize_audio(audio, check_clipping=False, method='rms')  # ✅ Nuevo método
```

---

## 🔧 Cómo Agregar Nuevo Paso al Pipeline

### Paso 1: Crear Nuevo Processor

**Archivo**: `processor/audio_enhancer.py` (nuevo)

```python
class AudioEnhancer(BaseAudioProcessor):
    """
    ✅ Nuevo processor sin modificar código existente.
    """
    
    def process(self, audio: Union[np.ndarray, torch.Tensor], **kwargs):
        """Enhance audio quality."""
        # ... implementación ...
        return enhanced_audio
```

---

### Paso 2: Agregar al Pipeline (Opcional)

**Archivo**: `audio_separator.py`

```python
def _perform_separation_pipeline(self, audio_path):
    # ... pasos existentes ...
    
    # ✅ Agregar nuevo paso sin modificar código existente
    if self.enhancer:  # Opcional
        logger.debug("Enhancing separated sources")
        separated_audio = {
            name: self.enhancer.process(audio)
            for name, audio in separated_audio.items()
        }
    
    return separated_audio
```

**Beneficios:**
- ✅ Nuevo paso opcional
- ✅ No modifica flujo existente
- ✅ Fácil activar/desactivar

---

## 🔧 Cómo Agregar Nuevo Tipo de Separator

### Paso 1: Crear Nueva Clase

**Archivo**: `separator/custom_separator.py` (nuevo)

```python
from .base_separator import BaseSeparator

class CustomSeparator(BaseSeparator):
    """
    ✅ Nuevo separator sin modificar código existente.
    """
    
    def _do_initialize(self, **kwargs):
        """Initialize custom separator."""
        # ... implementación específica ...
    
    def separate(
        self,
        audio_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> Dict[str, str]:
        """Separate audio using custom method."""
        # ... implementación específica ...
        return results
```

**Beneficios:**
- ✅ No modifica código existente
- ✅ Usa base class (código común)
- ✅ Implementa solo detalles específicos

---

## 🔧 Cómo Agregar Nueva Estrategia de Separación

### Paso 1: Agregar Método Helper

**Archivo**: `audio_separator.py`

```python
def _try_custom_separation_method(
    self,
    audio_path: Path,
    output_dir: Path
) -> Optional[Dict[str, str]]:
    """
    ✅ Nueva estrategia sin modificar código existente.
    """
    # ... implementación ...
    return result
```

---

### Paso 2: Agregar a la Cadena de Estrategias

**Archivo**: `audio_separator.py`

```python
def separate_file(self, audio_path, output_dir, save_outputs=True):
    # ... código existente ...
    
    # ✅ Agregar nueva estrategia (sin modificar código existente)
    result = self._try_custom_separation_method(audio_path, output_dir)
    if result is not None:
        return result
    
    # ... resto del código sin cambios ...
```

**Beneficios:**
- ✅ Fácil agregar nuevas estrategias
- ✅ No modifica código existente
- ✅ Fallback automático

---

## 🔧 Cómo Agregar Nueva Validación

### Paso 1: Extender Validación en Base Class

**Archivo**: `base_processor.py`

```python
def validate_audio(
    self,
    audio: Union[np.ndarray, torch.Tensor],
    allow_empty: bool = False,
    check_clipping: bool = False  # ✅ Nuevo parámetro
) -> None:
    """
    ✅ Extendido sin modificar validación existente.
    """
    # ... validación existente ...
    
    # ✅ Agregar nueva validación
    if check_clipping:
        if isinstance(audio, np.ndarray):
            max_val = np.abs(audio).max()
            if max_val > 1.0:
                logger.warning(f"Audio may clip (max={max_val:.3f})")
```

**Beneficios:**
- ✅ Validación extensible
- ✅ Backward compatible
- ✅ Fácil agregar nuevas validaciones

---

## 📊 Ejemplos de Extensión Completa

### Ejemplo 1: Agregar Normalización RMS

```python
# 1. Extender base_processor.py
def _normalize_audio(self, audio, check_clipping=False, method='peak'):
    if method == 'peak':
        # ... código existente ...
    elif method == 'rms':  # ✅ Solo agregar
        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 0:
            audio = audio * (0.1 / rms)
        return audio

# 2. Usar en preprocessor
def _normalize(self, audio):
    return self._normalize_audio(audio, check_clipping=False, method='rms')

# ✅ Listo! Sin modificar código existente
```

---

### Ejemplo 2: Agregar Enhancement Step

```python
# 1. Crear nuevo processor
class AudioEnhancer(BaseAudioProcessor):
    def process(self, audio, **kwargs):
        # ... enhancement logic ...
        return enhanced

# 2. Agregar al pipeline (opcional)
def _perform_separation_pipeline(self, audio_path):
    # ... pasos existentes ...
    
    # ✅ Agregar enhancement (opcional)
    if hasattr(self, 'enhancer') and self.enhancer:
        separated_audio = {
            name: self.enhancer.process(audio)
            for name, audio in separated_audio.items()
        }
    
    return separated_audio

# ✅ Listo! Enhancement opcional sin modificar flujo existente
```

---

### Ejemplo 3: Agregar Nuevo Separator

```python
# 1. Crear nueva clase
class CustomSeparator(BaseSeparator):
    def _do_initialize(self, **kwargs):
        # ... inicialización específica ...
    
    def separate(self, audio_path, output_dir, **kwargs):
        # ... lógica de separación específica ...
        return results

# ✅ Listo! Nuevo separator sin modificar código existente
```

---

## ✅ Checklist de Extensión

### Para Agregar Nuevo Tipo de Normalización:
- [ ] Extender `_normalize_audio()` con nuevo método
- [ ] Agregar tests para nuevo método
- [ ] Actualizar documentación

### Para Agregar Nuevo Paso al Pipeline:
- [ ] Crear nuevo processor (si necesario)
- [ ] Agregar paso opcional al pipeline
- [ ] Agregar tests
- [ ] Actualizar documentación

### Para Agregar Nuevo Separator:
- [ ] Crear nueva clase heredando de `BaseSeparator`
- [ ] Implementar `_do_initialize()` y `separate()`
- [ ] Agregar tests
- [ ] Actualizar documentación

---

## 🎯 Mejores Prácticas para Extensión

### 1. Seguir el Patrón Existente

**✅ Bueno:**
```python
# Seguir el mismo patrón que otros métodos
def _normalize_audio(self, audio, check_clipping=False, method='peak'):
    if method == 'peak':
        # ... código existente ...
    elif method == 'new_method':  # ✅ Mismo patrón
        # ... nueva implementación ...
```

**❌ Malo:**
```python
# No seguir el patrón
def normalize_new_way(self, audio):  # ❌ Patrón diferente
    # ... implementación diferente ...
```

---

### 2. Mantener Backward Compatibility

**✅ Bueno:**
```python
# Parámetros opcionales con defaults
def _normalize_audio(self, audio, check_clipping=False, method='peak'):
    # ✅ method='peak' es default (backward compatible)
    # ✅ Nuevos métodos son opcionales
```

**❌ Malo:**
```python
# Cambiar signature existente
def _normalize_audio(self, audio, method):  # ❌ method requerido
    # ❌ Rompe código existente
```

---

### 3. Agregar Tests

**✅ Bueno:**
```python
# Agregar tests para nueva funcionalidad
def test_normalize_rms():
    processor = BaseAudioProcessor()
    audio = np.array([0.1, 0.2, 0.3])
    result = processor._normalize_audio(audio, method='rms')
    # ... assertions ...
```

---

## 🚀 Ejemplo Completo: Agregar Normalización RMS

### Paso 1: Extender Base Class

```python
# base_processor.py
def _normalize_audio(
    self,
    audio: np.ndarray,
    check_clipping: bool = False,
    method: str = 'peak'  # ✅ Agregar parámetro
) -> np.ndarray:
    if method == 'peak':
        # ✅ Código existente (no modificar)
        max_val = np.abs(audio).max()
        # ...
    elif method == 'rms':  # ✅ Solo agregar
        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 0:
            target_rms = 0.1
            audio = audio * (target_rms / rms)
            logger.debug(f"Normalized audio to RMS {target_rms}")
        return audio
    else:
        logger.warning(f"Unknown method {method}, using peak")
        return self._normalize_audio(audio, check_clipping, method='peak')
```

### Paso 2: Usar en Preprocessor

```python
# preprocessor.py
def _normalize(self, audio: np.ndarray) -> np.ndarray:
    # ✅ Usar nuevo método
    return self._normalize_audio(audio, check_clipping=False, method='rms')
```

### Paso 3: Agregar Tests

```python
# test_processor.py
def test_normalize_rms():
    processor = BaseAudioProcessor()
    audio = np.array([0.1, 0.2, 0.3])
    result = processor._normalize_audio(audio, method='rms')
    # ... assertions ...
```

### Paso 4: Verificar Funcionalidad

```python
# Verificar que funciona
preprocessor = AudioPreprocessor()
audio = np.array([0.1, 0.2, 0.3])
normalized = preprocessor._normalize(audio)  # ✅ Usa RMS normalization
```

---

## ✅ Resumen

### Ventajas de la Arquitectura Refactorizada para Extensión

1. ✅ **Fácil Agregar Nuevos Tipos**: Solo agregar al método base
2. ✅ **Fácil Agregar Nuevos Pasos**: Solo agregar al pipeline
3. ✅ **Fácil Agregar Nuevos Separators**: Solo crear nueva clase
4. ✅ **Sin Modificar Código Existente**: Principio Open/Closed
5. ✅ **Backward Compatible**: Parámetros opcionales
6. ✅ **Testeable**: Fácil agregar tests para nuevas funcionalidades

---

**🎊🎊🎊 Guía de Extensibilidad Completa. Código Listo para Crecimiento. 🎊🎊🎊**

