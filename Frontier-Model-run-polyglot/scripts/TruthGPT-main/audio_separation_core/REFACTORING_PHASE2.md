# Refactorización Fase 2 - Optimizaciones Adicionales

## 📋 Resumen

Continuación de la refactorización con mejoras adicionales en configuraciones, separadores concretos y mixers.

## 🔄 Cambios Realizados

### 1. Configuraciones Simplificadas

**Problema**: Configuraciones con muchos parámetros que rara vez se usan.

**Solución**: Crear `config_simplified.py` con solo parámetros esenciales.

**Antes** (`SeparationConfig`):
```python
@dataclass
class SeparationConfig(AudioConfig):
    model_type: str = "spleeter"
    model_path: Optional[str] = None
    components: List[str] = field(default_factory=lambda: ["vocals", "accompaniment"])
    use_gpu: bool = True
    batch_size: int = 1
    overlap: float = 0.25
    segment_length: Optional[int] = None
    post_process: bool = True
    model_params: Dict[str, Any] = field(default_factory=dict)
```

**Después** (`config_simplified.py`):
```python
@dataclass
class SeparationConfig(AudioConfig):
    model_type: str = "spleeter"
    components: List[str] = field(default_factory=lambda: ["vocals", "accompaniment"])
    use_gpu: bool = True
    model_path: Optional[str] = None  # Solo si se necesita modelo personalizado
```

**Razón**: YAGNI - Eliminar parámetros que no se usan siempre. Si se necesitan, se pueden agregar después.

### 2. SpleeterSeparator Refactorizado

**Problema**: 
- Lógica de determinación de modelo duplicada
- Mapeo de componentes inline
- Construcción de rutas repetitiva

**Solución**: Consolidar en métodos helper y constantes.

**Antes** (183 líneas):
```python
def _load_model(self, **kwargs):
    # Lógica inline para determinar modelo
    components = self._config.components
    if len(components) == 2 and "vocals" in components:
        self._model_name = "spleeter:2stems"
    elif len(components) == 4:
        self._model_name = "spleeter:4stems"
    # ... más lógica inline

def _perform_separation(self, ...):
    # Mapeo inline
    spleeter_mapping = {
        "vocals": "vocals",
        "accompaniment": "accompaniment",
        # ...
    }
    # Construcción de rutas inline
    for component in components:
        spleeter_name = spleeter_mapping.get(component, component)
        output_file = output_dir / input_stem / f"{spleeter_name}.wav"
        # ... más código inline
```

**Después** (150 líneas, -18%):
```python
# Constantes de clase
SPLEETER_COMPONENT_MAP = {...}
MODEL_BY_COMPONENT_COUNT = {...}

def _load_model(self, **kwargs):
    self._model_name = self._determine_model_name()  # Método helper
    return Separator(self._model_name)

def _perform_separation(self, ...):
    self._model.separate_to_file(...)
    return self._build_output_paths(...)  # Método helper

def _determine_model_name(self) -> str:
    """Lógica consolidada."""
    if self._config.model_path:
        return self._config.model_path
    component_count = len(self._config.components)
    return self.MODEL_BY_COMPONENT_COUNT.get(component_count, "spleeter:2stems")

def _build_output_paths(self, ...) -> Dict[str, str]:
    """Construcción de rutas consolidada."""
    # Lógica clara y reutilizable
```

**Razón**: 
- DRY: Lógica consolidada en métodos helper
- Legibilidad: Constantes de clase más claras
- Mantenibilidad: Cambios en un solo lugar

### 3. SimpleMixer Refactorizado

**Problema**: Método `_perform_mixing()` muy largo (100+ líneas) con múltiples responsabilidades.

**Solución**: Dividir en métodos más pequeños y enfocados.

**Antes** (100+ líneas en un método):
```python
def _perform_mixing(self, ...):
    # Cargar archivos (30 líneas)
    for name, path in audio_files.items():
        y, sr = librosa.load(...)
        # ... procesamiento inline
        # ... resampleo inline
        # ... aplicación de volumen inline
    
    # Alinear longitudes (10 líneas)
    for name in audio_data:
        # ... padding inline
    
    # Mezclar (5 líneas)
    mixed = np.zeros(max_length)
    for y in audio_data.values():
        mixed = mixed + y
    
    # Normalizar (5 líneas)
    # Fade in/out (15 líneas)
    # Guardar (5 líneas)
```

**Después** (métodos pequeños y enfocados):
```python
def _perform_mixing(self, ...):
    # Paso 1: Cargar y procesar
    audio_data, sample_rate = self._load_and_process_files(...)
    
    # Paso 2: Alinear longitudes
    aligned_data = self._align_audio_lengths(audio_data, np)
    
    # Paso 3: Mezclar
    mixed = self._mix_tracks(aligned_data, np)
    
    # Paso 4: Post-procesamiento
    processed = self._post_process(mixed, sample_rate, np)
    
    # Paso 5: Guardar
    sf.write(str(output_path), processed, sample_rate)

def _load_and_process_files(self, ...):
    """Solo carga y procesa archivos."""
    # Lógica clara y enfocada

def _align_audio_lengths(self, ...):
    """Solo alinea longitudes."""
    # Lógica clara y enfocada

def _mix_tracks(self, ...):
    """Solo mezcla pistas."""
    # Lógica clara y enfocada

def _post_process(self, ...):
    """Solo post-procesamiento."""
    # Lógica clara y enfocada
```

**Razón**: 
- Single Responsibility: Cada método una responsabilidad
- Legibilidad: Métodos más cortos y claros
- Testeable: Métodos pequeños fáciles de testear

### 4. Utilidades de Audio Consolidadas

**Problema**: Imports y funciones duplicadas en múltiples lugares.

**Solución**: Crear `utils/audio_processing.py` con funciones reutilizables.

**Antes** (duplicado en múltiples archivos):
```python
# En simple_mixer.py
try:
    import librosa
    import soundfile as sf
    import numpy as np
except ImportError:
    raise ...

# En advanced_mixer.py
try:
    import librosa
    import soundfile as sf
    import numpy as np
except ImportError:
    raise ...
```

**Después** (consolidado):
```python
# En utils/audio_processing.py
def ensure_audio_libs():
    """Asegura que las librerías estén disponibles."""
    try:
        import librosa
        import soundfile as sf
        import numpy as np
        return librosa, sf, np
    except ImportError as e:
        raise ImportError(...) from e

# Uso en mixers
from ..utils.audio_processing import ensure_audio_libs
librosa, sf, np = ensure_audio_libs()
```

**Razón**: DRY - Una sola fuente de verdad para imports y funciones comunes.

## 📊 Métricas de Mejora Fase 2

| Métrica | Antes | Después | Mejora |
|---------|-------|---------|--------|
| **SpleeterSeparator** | 183 líneas | 150 líneas | **-18%** |
| **SimpleMixer** | 204 líneas | 180 líneas | **-12%** |
| **Método más largo** | 100+ líneas | 40 líneas | **-60%** |
| **Métodos helper** | 0 | 8 | **+8** |
| **Imports duplicados** | 3 lugares | 1 lugar | **-67%** |
| **Parámetros de config** | 15+ | 8 | **-47%** |

## 🎯 Principios Aplicados

### Single Responsibility Principle

**Antes**: `_perform_mixing()` hacía todo (cargar, procesar, alinear, mezclar, post-procesar)

**Después**: Cada método una responsabilidad:
- `_load_and_process_files()`: Solo carga
- `_align_audio_lengths()`: Solo alinea
- `_mix_tracks()`: Solo mezcla
- `_post_process()`: Solo post-procesa

### DRY

**Antes**: 
- Lógica de modelo duplicada
- Mapeo de componentes inline
- Imports duplicados

**Después**:
- Constantes de clase para mapeos
- Métodos helper para lógica común
- Utilidades compartidas para imports

### KISS

**Antes**: Configuraciones con 15+ parámetros, muchos no usados

**Después**: Configuraciones con solo parámetros esenciales

## ✅ Beneficios

1. **Código Más Legible**: Métodos más cortos y claros
2. **Menos Duplicación**: Lógica consolidada
3. **Más Mantenible**: Cambios en un solo lugar
4. **Más Testeable**: Métodos pequeños fáciles de testear
5. **Configuración Más Simple**: Menos parámetros innecesarios

## 📝 Ejemplos de Uso

### Configuración Simplificada

**Antes**:
```python
config = SeparationConfig(
    model_type="spleeter",
    use_gpu=True,
    batch_size=1,
    overlap=0.25,
    segment_length=None,
    post_process=True,
    model_params={}
)
```

**Después**:
```python
config = SeparationConfig(
    model_type="spleeter",
    use_gpu=True
)  # Solo lo esencial
```

### Separador Más Claro

**Antes**: Lógica inline difícil de seguir

**Después**: Métodos helper claros:
```python
model_name = self._determine_model_name()  # Claro qué hace
paths = self._build_output_paths(...)      # Claro qué hace
```

## 🔮 Próximos Pasos (Opcional)

1. Refactorizar `DemucsSeparator` y `LALALSeparator` de manera similar
2. Refactorizar `AdvancedMixer` dividiendo métodos largos
3. Agregar tests unitarios para métodos helper
4. Considerar simplificar aún más las configuraciones

## 📚 Archivos Creados/Modificados

### Nuevos
- ✅ `core/config_simplified.py` - Configuraciones simplificadas
- ✅ `separators/spleeter_separator_refactored.py` - Versión refactorizada
- ✅ `mixers/simple_mixer_refactored.py` - Versión refactorizada
- ✅ `utils/audio_processing.py` - Utilidades compartidas

### Documentación
- ✅ `REFACTORING_PHASE2.md` - Este documento

