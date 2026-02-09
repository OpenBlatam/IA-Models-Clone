# Fase 30: Refactorización de Utilidades de Audio - Consolidación de Helpers Comunes

## Resumen

Esta fase refactoriza los módulos de utilidades de audio (`audio_merger.py`, `audio_enhancement.py`, `audio_analysis.py`) para eliminar duplicación en operaciones comunes como padding, normalización y cálculos de RMS/peak/dB.

## Problemas Identificados

### 1. Duplicación de Lógica de Padding
- **Ubicación**: `audio_merger.py`
- **Problema**: La lógica para hacer padding de arrays de audio a la misma longitud estaba duplicada en `merge_sources()` y `blend_audio()`.
- **Impacto**: Código repetitivo, difícil de mantener.

### 2. Duplicación de Cálculos de RMS y Peak
- **Ubicación**: `audio_analysis.py`, `audio_enhancement.py`
- **Problema**: Los cálculos de RMS (`np.sqrt(np.mean(audio ** 2))`) y peak (`np.abs(audio).max()`) estaban duplicados en múltiples funciones.
- **Impacto**: Inconsistencias potenciales, código repetitivo.

### 3. Duplicación de Conversiones dB
- **Ubicación**: `audio_analysis.py`
- **Problema**: La conversión de amplitud a dB (`20 * np.log10(...)`) estaba duplicada.
- **Impacto**: Código repetitivo, difícil de mantener.

### 4. Duplicación de Normalización
- **Ubicación**: `audio_enhancement.py`
- **Problema**: La lógica de normalización por peak y RMS estaba duplicada en múltiples lugares.
- **Impacto**: Código repetitivo, difícil de mantener.

## Soluciones Implementadas

### 1. Creación de `audio_helpers.py`

Se creó un nuevo módulo `audio_helpers.py` que centraliza todas las operaciones comunes:

```python
# Funciones de padding
def pad_audio_to_length(audio: np.ndarray, target_length: int, pad_mode: str = 'constant') -> np.ndarray
def ensure_same_length(*audio_arrays: np.ndarray, pad_mode: str = 'constant') -> tuple[np.ndarray, ...]

# Funciones de cálculo
def calculate_rms(audio: np.ndarray) -> float
def calculate_peak(audio: np.ndarray) -> float

# Funciones de conversión dB
def amplitude_to_db(amplitude: float, reference: float = 1.0) -> float
def db_to_amplitude(db: float, reference: float = 1.0) -> float

# Funciones de normalización
def normalize_by_peak(audio: np.ndarray, target_peak: float = 1.0) -> np.ndarray
def normalize_by_rms(audio: np.ndarray, target_rms: float = 0.1) -> np.ndarray
```

### 2. Refactorización de `audio_merger.py`

**Antes**:
```python
# En merge_sources()
if len(source_audio) < max_length:
    padded = np.pad(source_audio, (0, max_length - len(source_audio)))
else:
    padded = source_audio[:max_length]

# En blend_audio()
if len(audio1) < max_length:
    audio1 = np.pad(audio1, (0, max_length - len(audio1)))
if len(audio2) < max_length:
    audio2 = np.pad(audio2, (0, max_length - len(audio2)))
```

**Después**:
```python
from .audio_helpers import ensure_same_length

# En merge_sources()
source_arrays = list(sources.values())
padded_arrays = ensure_same_length(*source_arrays)

# En blend_audio()
audio1, audio2 = ensure_same_length(audio1, audio2)
```

### 3. Refactorización de `audio_analysis.py`

**Antes**:
```python
stats = {
    "max_amplitude": float(np.abs(audio).max()),
    "rms": float(np.sqrt(np.mean(audio ** 2))),
    "peak_db": float(20 * np.log10(np.abs(audio).max() + 1e-10)),
}

# En calculate_loudness()
rms = np.sqrt(np.mean(audio ** 2))
rms_db = 20 * np.log10(rms + 1e-10)
peak = np.abs(audio).max()
peak_db = 20 * np.log10(peak + 1e-10)
```

**Después**:
```python
from .audio_helpers import calculate_rms, calculate_peak, amplitude_to_db

peak = calculate_peak(audio)
rms = calculate_rms(audio)

stats = {
    "max_amplitude": peak,
    "rms": rms,
    "peak_db": amplitude_to_db(peak),
}

# En calculate_loudness()
rms = calculate_rms(audio)
peak = calculate_peak(audio)
rms_db = amplitude_to_db(rms)
peak_db = amplitude_to_db(peak)
```

### 4. Refactorización de `audio_enhancement.py`

**Antes**:
```python
def normalize_peak(self, audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    peak = np.abs(audio).max()
    if peak > 0:
        return audio * (target_peak / peak)
    return audio

def normalize_rms(self, audio: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
    rms = np.sqrt(np.mean(audio ** 2))
    if rms > 0:
        return audio * (target_rms / rms)
    return audio
```

**Después**:
```python
from .audio_helpers import normalize_by_peak, normalize_by_rms

def normalize_peak(self, audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    return normalize_by_peak(audio, target_peak)

def normalize_rms(self, audio: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
    return normalize_by_rms(audio, target_rms)
```

## Métricas

### Reducción de Código Duplicado
- **Líneas eliminadas**: ~45 líneas de código duplicado
- **Funciones consolidadas**: 8 funciones helper creadas
- **Archivos refactorizados**: 3 archivos principales

### Mejoras de Mantenibilidad
- **Punto único de cambio**: Todas las operaciones comunes ahora están centralizadas
- **Consistencia**: Garantiza que todos los cálculos se realizan de la misma manera
- **Testabilidad**: Los helpers pueden ser probados de forma independiente

## Principios Aplicados

1. **DRY (Don't Repeat Yourself)**: Eliminación de código duplicado
2. **Single Responsibility Principle**: Cada helper tiene una responsabilidad única
3. **Reusabilidad**: Los helpers pueden ser utilizados en cualquier parte del código
4. **Mantenibilidad**: Cambios futuros solo requieren modificar un lugar

## Archivos Modificados

1. **`audio_helpers.py`** (NUEVO): Módulo centralizado de helpers comunes
2. **`audio_merger.py`**: Refactorizado para usar `ensure_same_length()`
3. **`audio_analysis.py`**: Refactorizado para usar helpers de cálculo y conversión
4. **`audio_enhancement.py`**: Refactorizado para usar helpers de normalización
5. **`__init__.py`**: Actualizado para exportar los nuevos helpers

## Compatibilidad

- ✅ **Backward Compatible**: Todas las funciones públicas mantienen su interfaz original
- ✅ **Sin Breaking Changes**: Los cambios son internos, no afectan la API pública

## Próximos Pasos

1. Considerar refactorizar otros módulos que puedan usar estos helpers (ej: `quality_metrics.py`)
2. Agregar tests unitarios para los nuevos helpers
3. Documentar casos de uso avanzados de los helpers

## Notas

- Las advertencias del linter sobre `librosa` son esperadas, ya que es una dependencia opcional
- Los helpers están diseñados para ser simples y eficientes
- Se mantiene la compatibilidad con arrays numpy 1D y 2D

