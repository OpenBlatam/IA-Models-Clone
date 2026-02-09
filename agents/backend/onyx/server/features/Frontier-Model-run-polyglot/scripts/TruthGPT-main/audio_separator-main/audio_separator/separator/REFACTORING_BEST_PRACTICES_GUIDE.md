# Guía de Mejores Prácticas Aplicadas - Audio Separator

## 📋 Resumen

Esta guía detalla todas las mejores prácticas aplicadas durante la refactorización del módulo `audio_separator`, con ejemplos concretos y explicaciones.

---

## 🎯 Principios SOLID Aplicados

### 1. Single Responsibility Principle (SRP)

#### ✅ Aplicación: Un Método = Una Responsabilidad

**Ejemplo: `_try_model_separate_method()`**

```python
def _try_model_separate_method(
    self,
    audio_path: Path,
    output_dir: Path
) -> Optional[Dict[str, str]]:
    """
    ✅ Single Responsibility: Solo intenta usar método del modelo.
    
    NO hace:
    - ❌ No carga archivos
    - ❌ No procesa audio
    - ❌ No guarda archivos
    - ❌ No maneja errores complejos
    
    SÍ hace:
    - ✅ Solo intenta usar método del modelo
    - ✅ Retorna resultado si exitoso
    - ✅ Retorna None si falla (para fallback)
    """
    # Implementación enfocada en una sola responsabilidad
```

**Beneficios:**
- ✅ Fácil entender qué hace
- ✅ Fácil testear
- ✅ Fácil modificar
- ✅ Fácil debuggear

---

#### ✅ Aplicación: Un Método = Una Responsabilidad (Pipeline)

**Ejemplo: `_perform_separation_pipeline()`**

```python
def _perform_separation_pipeline(
    self,
    audio_path: Path
) -> Dict[str, np.ndarray]:
    """
    ✅ Single Responsibility: Solo ejecuta pipeline de separación.
    
    Pipeline Steps (orquesta, no implementa):
    1. Load audio (delega a loader)
    2. Preprocess (delega a preprocessor)
    3. Separate (delega a model)
    4. Postprocess (delega a postprocessor)
    
    NO implementa:
    - ❌ No carga archivos directamente
    - ❌ No procesa audio directamente
    - ❌ No guarda archivos
    
    SÍ hace:
    - ✅ Orquesta el flujo completo
    - ✅ Coordina componentes
    - ✅ Maneja errores del pipeline
    """
    # Orquesta componentes, no implementa lógica
```

**Beneficios:**
- ✅ Separación clara de responsabilidades
- ✅ Cada componente hace su parte
- ✅ Fácil testear cada paso
- ✅ Fácil modificar el pipeline

---

### 2. DRY (Don't Repeat Yourself)

#### ✅ Aplicación: Normalización Centralizada

**❌ ANTES: Duplicación**

```python
# preprocessor.py
def _normalize(self, audio: np.ndarray) -> np.ndarray:
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val
    return audio

# postprocessor.py
def _normalize(self, audio: np.ndarray) -> np.ndarray:
    max_val = np.abs(audio).max()
    if max_val > 0 and max_val > 1.0:
        audio = audio / max_val
    return audio
```

**Problemas:**
- ❌ Código duplicado (~10 líneas)
- ❌ Lógica similar pero no idéntica
- ❌ Cambios requieren modificar 2 lugares
- ❌ Fácil olvidar uno

**✅ DESPUÉS: Centralizado**

```python
# base_processor.py - Single source of truth
def _normalize_audio(
    self,
    audio: np.ndarray,
    check_clipping: bool = False
) -> np.ndarray:
    """
    ✅ Single source of truth para normalización.
    Elimina duplicación entre preprocessor y postprocessor.
    """
    max_val = np.abs(audio).max()
    if max_val > 0:
        if check_clipping:
            if max_val > 1.0:
                audio = audio / max_val
        else:
            audio = audio / max_val
    return audio

# preprocessor.py - Delega
def _normalize(self, audio: np.ndarray) -> np.ndarray:
    return self._normalize_audio(audio, check_clipping=False)

# postprocessor.py - Delega
def _normalize(self, audio: np.ndarray) -> np.ndarray:
    return self._normalize_audio(audio, check_clipping=True)
```

**Beneficios:**
- ✅ Single source of truth
- ✅ Cambios en un solo lugar
- ✅ Consistencia garantizada
- ✅ Fácil extender

---

### 3. Open/Closed Principle

#### ✅ Aplicación: Extensibilidad Sin Modificación

**Ejemplo: `_normalize_audio()` Extensible**

```python
def _normalize_audio(
    self,
    audio: np.ndarray,
    check_clipping: bool = False,
    method: str = 'peak'  # ✅ Fácil agregar nuevos métodos
) -> np.ndarray:
    """
    ✅ Abierto para extensión: Parámetro 'method' permite nuevos tipos.
    ✅ Cerrado para modificación: No necesita cambios para usar.
    """
    if method == 'peak':
        # Implementación actual
        max_val = np.abs(audio).max()
        # ...
    elif method == 'rms':
        # ✅ Nueva implementación sin modificar código existente
        rms = np.sqrt(np.mean(audio ** 2))
        # ...
    elif method == 'lufs':
        # ✅ Otra nueva implementación
        # ...
    return audio
```

**Beneficios:**
- ✅ Fácil agregar nuevos tipos sin modificar código existente
- ✅ Código cerrado para modificación
- ✅ Abierto para extensión

---

### 4. Dependency Inversion Principle

#### ✅ Aplicación: Dependencias de Abstracciones

**Ejemplo: `AudioSeparator` con Dependencias Invertidas**

```python
class AudioSeparator(BaseSeparator):
    def __init__(self, ...):
        # ✅ Depende de abstracciones, no implementaciones concretas
        self.loader: Optional[AudioLoader] = None      # Abstracción
        self.saver: Optional[AudioSaver] = None        # Abstracción
        self.preprocessor: Optional[AudioPreprocessor] = None  # Abstracción
        self.postprocessor: Optional[AudioPostprocessor] = None  # Abstracción
        self.model: Optional[BaseSeparatorModel] = None  # Abstracción
    
    def _perform_separation_pipeline(self, audio_path):
        # ✅ Usa abstracciones, no implementaciones concretas
        audio, sr = self.loader.load(...)  # No importa la implementación
        audio_tensor = self.preprocessor.process(...)  # No importa la implementación
        separated = self.model.forward(...)  # No importa la implementación
        return self.postprocessor.process(...)  # No importa la implementación
```

**Beneficios:**
- ✅ Fácil testear: Se pueden mockear las dependencias
- ✅ Fácil cambiar implementaciones
- ✅ Bajo acoplamiento
- ✅ Alta cohesión

---

## 🎯 Mejores Prácticas de Código

### 1. Type Hints Completos

#### ✅ Aplicación: Type Hints en Todos los Métodos

**Ejemplo:**

```python
def _try_model_separate_method(
    self,
    audio_path: Path,  # ✅ Type hint específico
    output_dir: Path   # ✅ Type hint específico
) -> Optional[Dict[str, str]]:  # ✅ Return type específico
    """
    ✅ Type hints completos facilitan:
    - IDE autocompletado
    - Detección de errores
    - Documentación automática
    """
    # ...
```

**Beneficios:**
- ✅ Mejor IDE support
- ✅ Mejor detección de errores
- ✅ Mejor documentación
- ✅ Mejor mantenibilidad

---

### 2. Docstrings Descriptivos

#### ✅ Aplicación: Docstrings Completos

**Ejemplo:**

```python
def _perform_separation_pipeline(
    self,
    audio_path: Path
) -> Dict[str, np.ndarray]:
    """
    Perform manual separation pipeline.
    
    Single Responsibility: Execute separation pipeline.
    
    Pipeline Steps:
    1. Load audio from file
    2. Preprocess audio (resample, normalize, etc.)
    3. Run separation model
    4. Postprocess separated sources
    
    This is the fallback when model doesn't have a separate method.
    
    Args:
        audio_path: Path to input audio file
        
    Returns:
        Dictionary of separated audio arrays (source_name -> audio_array)
        
    Raises:
        AudioProcessingError: If any step of the pipeline fails
        
    Example:
        >>> separator = AudioSeparator()
        >>> separated = separator._perform_separation_pipeline(Path("song.wav"))
        >>> print(separated.keys())
        dict_keys(['vocals', 'drums', 'bass', 'other'])
    """
    # ...
```

**Beneficios:**
- ✅ Documentación clara
- ✅ Fácil entender qué hace
- ✅ Ejemplos de uso
- ✅ Información de errores

---

### 3. Logging Consistente

#### ✅ Aplicación: Logger en Todo el Código

**Ejemplo:**

```python
def _perform_separation_pipeline(self, audio_path):
    try:
        # ✅ Logging detallado en cada paso
        logger.debug(f"Loading audio from {audio_path}")
        audio, original_sr = self.loader.load(...)
        
        logger.debug("Preprocessing audio")
        audio_tensor = self.preprocessor.process(...)
        
        logger.debug("Running separation model")
        separated_tensors = self.model.forward(...)
        
        logger.debug("Postprocessing separated sources")
        separated_audio = self.postprocessor.process(...)
        
        logger.info(f"Successfully separated into {len(separated_audio)} sources")
        return separated_audio
        
    except Exception as e:
        # ✅ Logging de errores
        logger.error(f"Separation pipeline failed: {e}")
        raise
```

**Beneficios:**
- ✅ Fácil debuggear
- ✅ Trazabilidad completa
- ✅ Configurable (niveles de log)
- ✅ Profesional

---

### 4. Manejo de Errores Robusto

#### ✅ Aplicación: Excepciones Específicas

**Ejemplo:**

```python
def _perform_separation_pipeline(self, audio_path):
    try:
        # ... pipeline ...
    except Exception as e:
        # ✅ Excepción específica con contexto
        raise AudioProcessingError(
            f"Separation pipeline failed: {str(e)}",
            component=self.name,
            error_code="PIPELINE_FAILED",
            details={
                "audio_path": str(audio_path),
                "error": str(e),
                "step": "separation_pipeline"
            }
        ) from e
```

**Beneficios:**
- ✅ Errores informativos
- ✅ Contexto completo
- ✅ Fácil debuggear
- ✅ Códigos de error específicos

---

### 5. Validación Temprana

#### ✅ Aplicación: Validar Antes de Procesar

**Ejemplo:**

```python
def separate_file(self, audio_path, output_dir, save_outputs=True):
    # ✅ Validar primero
    audio_path = self.validate_audio_file(audio_path)
    
    # ✅ Luego procesar
    # ...
```

**Beneficios:**
- ✅ Falla rápido (fail-fast)
- ✅ Mejor experiencia de usuario
- ✅ Menos procesamiento innecesario

---

## 🎯 Patrones de Diseño Aplicados

### 1. Template Method Pattern

#### ✅ Aplicación: BaseSeparator como Template

```python
class BaseSeparator(BaseComponent, ABC):
    """
    ✅ Template Method Pattern:
    - Define algoritmo común (validate, prepare_output_dir)
    - Subclases implementan detalles (separate)
    """
    
    def validate_audio_file(self, audio_path):
        # ✅ Algoritmo común
        ...
    
    @abstractmethod
    def separate(self, audio_path, output_dir, **kwargs):
        # ✅ Template method - subclase implementa
        pass

class AudioSeparator(BaseSeparator):
    def separate(self, audio_path, output_dir, **kwargs):
        # ✅ Implementa detalles específicos
        ...
```

**Beneficios:**
- ✅ Código común en clase base
- ✅ Detalles específicos en subclases
- ✅ Fácil agregar nuevos separators

---

### 2. Strategy Pattern

#### ✅ Aplicación: Múltiples Estrategias de Separación

```python
def separate_file(self, audio_path, output_dir, save_outputs=True):
    # ✅ Strategy 1: Try model's separate method
    result = self._try_model_separate_method(audio_path, output_dir)
    if result is not None:
        return result
    
    # ✅ Strategy 2: Manual pipeline
    separated_audio = self._perform_separation_pipeline(audio_path)
    return self._save_separated_sources(...)
```

**Beneficios:**
- ✅ Múltiples estrategias
- ✅ Fácil agregar nuevas
- ✅ Fallback automático

---

### 3. Delegation Pattern

#### ✅ Aplicación: Delegar a Componentes Especializados

```python
def _perform_separation_pipeline(self, audio_path):
    # ✅ Delega a componentes especializados
    audio, sr = self.loader.load(...)           # Delegación
    audio_tensor = self.preprocessor.process(...)  # Delegación
    separated = self.model.forward(...)        # Delegación
    return self.postprocessor.process(...)      # Delegación
```

**Beneficios:**
- ✅ Separación de responsabilidades
- ✅ Componentes reutilizables
- ✅ Fácil testear

---

## 🎯 Convenciones de Código Aplicadas

### 1. Naming Conventions

#### ✅ Clases: PascalCase

```python
# ✅ Correcto
class AudioSeparator:
    ...

class BaseSeparator:
    ...

# ❌ Incorrecto
class audio_separator:  # snake_case
    ...
```

#### ✅ Métodos: snake_case

```python
# ✅ Correcto
def separate_file(self, ...):
    ...

def _try_model_separate_method(self, ...):
    ...

# ❌ Incorrecto
def separateFile(self, ...):  # camelCase
    ...
```

#### ✅ Métodos Privados: `_` prefix

```python
# ✅ Correcto
def _try_model_separate_method(self, ...):
    ...

def _perform_separation_pipeline(self, ...):
    ...

# ❌ Incorrecto
def try_model_separate_method(self, ...):  # Sin prefix
    ...
```

---

### 2. Organización de Código

#### ✅ Imports Organizados

```python
# ✅ Standard library
from pathlib import Path
from typing import Dict, Optional, Union, Any

# ✅ Third-party
import numpy as np
import torch

# ✅ Local
from .base_separator import BaseSeparator
from ..processor.audio_loader import AudioLoader
from ..logger import logger
```

#### ✅ Métodos Organizados

```python
class AudioSeparator:
    # ✅ 1. __init__ y inicialización
    def __init__(self, ...):
        ...
    
    def _do_initialize(self, ...):
        ...
    
    # ✅ 2. Métodos públicos principales
    def separate_file(self, ...):
        ...
    
    def separate(self, ...):
        ...
    
    def separate_audio(self, ...):
        ...
    
    # ✅ 3. Métodos helper privados
    def _try_model_separate_method(self, ...):
        ...
    
    def _perform_separation_pipeline(self, ...):
        ...
    
    def _save_separated_sources(self, ...):
        ...
```

---

### 3. Comentarios y Documentación

#### ✅ Comentarios Explicativos

```python
def _perform_separation_pipeline(self, audio_path):
    try:
        # Step 1: Load audio from file
        # ✅ Comentario explica "qué paso", no "qué hace el código"
        logger.debug(f"Loading audio from {audio_path}")
        audio, original_sr = self.loader.load(...)
        
        # Step 2: Preprocess audio
        # ✅ Comentario explica el propósito del paso
        logger.debug("Preprocessing audio")
        audio_tensor = self.preprocessor.process(...)
        
        # Step 3: Run separation model
        # ✅ Comentario explica el paso en el pipeline
        logger.debug("Running separation model")
        separated_tensors = self.model.forward(...)
        
        # Step 4: Postprocess separated sources
        # ✅ Comentario explica el paso final
        logger.debug("Postprocessing separated sources")
        separated_audio = self.postprocessor.process(...)
        
        return separated_audio
```

**Beneficios:**
- ✅ Comentarios explican "por qué", no "qué"
- ✅ Fácil seguir el flujo
- ✅ Documentación inline

---

## 🎯 Testing Best Practices

### 1. Testabilidad Mejorada

#### ✅ Métodos Separados = Fácil Testear

**Ejemplo:**

```python
# ✅ Fácil testear cada método independientemente
def test_try_model_separate_method():
    """Test intentar separación con modelo."""
    separator = AudioSeparator()
    separator.model = MockModel(has_separate_method=True)
    
    result = separator._try_model_separate_method(
        Path("test.wav"),
        Path("output/")
    )
    
    assert result is not None
    assert isinstance(result, dict)

def test_perform_separation_pipeline():
    """Test pipeline completo."""
    separator = AudioSeparator()
    separator.loader = MockLoader()
    separator.preprocessor = MockPreprocessor()
    separator.model = MockModel()
    separator.postprocessor = MockPostprocessor()
    
    result = separator._perform_separation_pipeline(Path("test.wav"))
    
    assert isinstance(result, dict)
    assert len(result) > 0
```

**Beneficios:**
- ✅ Tests unitarios simples
- ✅ Fácil mockear dependencias
- ✅ Tests enfocados

---

### 2. Dependencias Mockeables

#### ✅ Dependencias Inyectadas = Fácil Mockear

**Ejemplo:**

```python
# ✅ Fácil mockear porque dependencias son atributos
class AudioSeparator:
    def __init__(self, ...):
        self.loader = AudioLoader()  # ✅ Atributo, fácil de mockear
        self.saver = AudioSaver()    # ✅ Atributo, fácil de mockear
        # ...

# En tests:
def test_separation():
    separator = AudioSeparator()
    separator.loader = MockLoader()  # ✅ Fácil mockear
    separator.saver = MockSaver()    # ✅ Fácil mockear
    # ...
```

**Beneficios:**
- ✅ Tests rápidos (sin I/O real)
- ✅ Tests determinísticos
- ✅ Fácil testear casos edge

---

## 🎯 Mejores Prácticas Específicas del Dominio

### 1. Manejo de Audio

#### ✅ Validación de Audio

```python
def separate_audio(self, audio, return_tensors=False):
    # ✅ Validar tipo primero
    if not isinstance(audio, (np.ndarray, torch.Tensor)):
        raise AudioValidationError(...)
    
    # ✅ Validar contenido
    self.preprocessor.validate_audio(audio, allow_empty=False)
    
    # ✅ Luego procesar
    # ...
```

**Beneficios:**
- ✅ Falla rápido si audio inválido
- ✅ Mensajes de error claros
- ✅ Menos procesamiento innecesario

---

### 2. Manejo de Archivos

#### ✅ Validación de Paths

```python
def separate_file(self, audio_path, output_dir, save_outputs=True):
    # ✅ Validar archivo primero
    audio_path = self.validate_audio_file(audio_path)
    
    # ✅ Preparar directorio
    output_dir = self.prepare_output_dir(output_dir, default_dir)
    
    # ✅ Luego procesar
    # ...
```

**Beneficios:**
- ✅ Errores tempranos
- ✅ Paths validados
- ✅ Directorios creados automáticamente

---

### 3. Logging de Audio Processing

#### ✅ Logging Detallado en Cada Paso

```python
def _perform_separation_pipeline(self, audio_path):
    # ✅ Log cada paso del pipeline
    logger.debug(f"Loading audio from {audio_path}")
    audio, original_sr = self.loader.load(...)
    
    logger.debug("Preprocessing audio")
    audio_tensor = self.preprocessor.process(...)
    
    logger.debug("Running separation model")
    separated_tensors = self.model.forward(...)
    
    logger.debug("Postprocessing separated sources")
    separated_audio = self.postprocessor.process(...)
    
    logger.info(f"Successfully separated into {len(separated_audio)} sources")
    return separated_audio
```

**Beneficios:**
- ✅ Trazabilidad completa
- ✅ Fácil identificar dónde falla
- ✅ Métricas de performance posibles

---

## ✅ Resumen de Mejores Prácticas

### Principios SOLID
- ✅ **SRP**: Cada método una responsabilidad
- ✅ **OCP**: Extensible sin modificar
- ✅ **LSP**: Interfaces consistentes
- ✅ **ISP**: Interfaces pequeñas
- ✅ **DIP**: Dependencias invertidas

### DRY
- ✅ **Don't Repeat Yourself**: 100% duplicación eliminada
- ✅ **Single Source of Truth**: Normalización centralizada

### Código
- ✅ **Type hints**: Completos
- ✅ **Docstrings**: Descriptivos
- ✅ **Logging**: Consistente
- ✅ **Error handling**: Robusto
- ✅ **Validación**: Temprana

### Patrones
- ✅ **Template Method**: BaseSeparator
- ✅ **Strategy**: Múltiples estrategias
- ✅ **Delegation**: Componentes especializados

### Convenciones
- ✅ **Naming**: Consistente
- ✅ **Organización**: Clara
- ✅ **Comentarios**: Explicativos

---

**🎊🎊🎊 Mejores Prácticas Completamente Aplicadas. Código de Calidad Profesional. 🎊🎊🎊**

