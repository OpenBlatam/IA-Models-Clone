# Ultra Modular Architecture V5 - Maximum Granularity

## 🎯 Arquitectura Ultra-Granular con Módulos Especializados

Esta versión implementa el máximo nivel de modularidad con cada componente en su propio archivo especializado.

## 📁 Estructura de Micro-Módulos Especializados

```
micro_modules/
├── __init__.py
├── normalizers.py          # 5 normalizadores especializados
├── tokenizers.py           # 4 tokenizadores especializados
├── padders.py              # 5 estrategias de padding
├── augmenters.py           # 8 técnicas de augmentación
├── data_processors.py      # Re-exports + validators
├── model_components.py     # 4 componentes de modelo
├── training_components.py  # 4 componentes de entrenamiento
└── inference_components.py # 4 componentes de inferencia
```

## 🔧 Componentes Especializados

### 1. Normalizers (`normalizers.py`)

**5 Normalizadores Especializados:**
- `StandardNormalizer` - Normalización estándar (z-score)
- `MinMaxNormalizer` - Normalización min-max (0-1)
- `RobustNormalizer` - Normalización robusta (mediana/IQR)
- `UnitVectorNormalizer` - Normalización vector unitario (L2)
- `NormalizerFactory` - Factory para crear normalizadores

**Características:**
- Base class con interfaz común
- Soporte para fit/inverse
- Factory pattern

**Ejemplo:**
```python
from addiction_recovery_ai.core.layers.micro_modules.normalizers import (
    StandardNormalizer,
    NormalizerFactory
)

# Uso directo
normalizer = StandardNormalizer()
data = normalizer.normalize(torch.randn(10))

# O con factory
normalizer = NormalizerFactory.create('standard')
normalizer = NormalizerFactory.create('minmax', min_val=0, max_val=1)
normalizer = NormalizerFactory.create('robust')
```

### 2. Tokenizers (`tokenizers.py`)

**4 Tokenizadores Especializados:**
- `SimpleTokenizer` - Tokenización basada en palabras
- `CharacterTokenizer` - Tokenización a nivel de caracteres
- `HuggingFaceTokenizer` - Wrapper para HuggingFace
- `BPETokenizer` - Byte Pair Encoding
- `TokenizerFactory` - Factory para crear tokenizadores

**Características:**
- Base class con tokenize/detokenize
- Soporte para batch processing
- Vocabulario configurable

**Ejemplo:**
```python
from addiction_recovery_ai.core.layers.micro_modules.tokenizers import (
    SimpleTokenizer,
    CharacterTokenizer,
    TokenizerFactory
)

# Word-based
tokenizer = SimpleTokenizer()
tokenizer.build_vocab(["hello world", "goodbye world"])
tokens = tokenizer.tokenize("hello world")

# Character-based
char_tokenizer = CharacterTokenizer()
tokens = char_tokenizer.tokenize("hello")

# Factory
tokenizer = TokenizerFactory.create('simple')
```

### 3. Padders (`padders.py`)

**5 Estrategias de Padding:**
- `ZeroPadder` - Padding con ceros
- `RepeatPadder` - Repetir último valor
- `ReflectPadder` - Padding reflejado (mirror)
- `CircularPadder` - Padding circular (wrap around)
- `CustomPadder` - Padding personalizado
- `PadderFactory` - Factory para crear padders

**Características:**
- Soporte multi-dimensional
- Batch processing
- Estrategias configurables

**Ejemplo:**
```python
from addiction_recovery_ai.core.layers.micro_modules.padders import (
    ZeroPadder,
    RepeatPadder,
    ReflectPadder,
    PadderFactory
)

# Zero padding
padder = ZeroPadder(pad_value=0)
padded = padder.pad(sequence, target_length=20)

# Repeat padding
repeat_padder = RepeatPadder()
padded = repeat_padder.pad(sequence, target_length=20)

# Factory
padder = PadderFactory.create('zero', pad_value=0)
padder = PadderFactory.create('reflect')
```

### 4. Augmenters (`augmenters.py`)

**8 Técnicas de Augmentación:**
- `NoiseAugmenter` - Añadir ruido gaussiano
- `DropoutAugmenter` - Dropout aleatorio
- `ScaleAugmenter` - Escalado aleatorio
- `ShiftAugmenter` - Desplazamiento aleatorio
- `FlipAugmenter` - Volteo aleatorio
- `MixupAugmenter` - Mixup augmentation
- `CutoutAugmenter` - Cutout (zero out regions)
- `ComposeAugmenter` - Componer múltiples augmenters
- `AugmenterFactory` - Factory para crear augmenters

**Características:**
- Probabilidad configurable
- Batch augmentation
- Composición flexible

**Ejemplo:**
```python
from addiction_recovery_ai.core.layers.micro_modules.augmenters import (
    NoiseAugmenter,
    ScaleAugmenter,
    ComposeAugmenter,
    AugmenterFactory
)

# Noise augmentation
augmenter = NoiseAugmenter(noise_level=0.1, probability=0.5)
augmented = augmenter.augment(data)

# Compose multiple
composed = ComposeAugmenter([
    NoiseAugmenter(noise_level=0.1),
    ScaleAugmenter(scale_range=(0.9, 1.1))
])
augmented = composed.augment(data)

# Factory
augmenter = AugmenterFactory.create('noise', noise_level=0.1)
```

## 📊 Estadísticas de Modularidad V5

- **Archivos Especializados**: 8+ archivos
- **Componentes Individuales**: 30+ componentes
- **Factories**: 4 factories
- **Nivel de Granularidad**: Máximo
- **Separación de Concerns**: Máxima

## 🎯 Principios Aplicados

1. **Single File, Single Responsibility**: Cada archivo tiene un propósito específico
2. **Factory Pattern**: Factories para creación flexible
3. **Base Classes**: Interfaces comunes para cada categoría
4. **Composition**: Fácil componer componentes
5. **Extensibility**: Fácil agregar nuevos componentes

## 🔧 Uso Completo

```python
from addiction_recovery_ai.core.layers.micro_modules import (
    # Normalizers
    StandardNormalizer,
    NormalizerFactory,
    # Tokenizers
    SimpleTokenizer,
    TokenizerFactory,
    # Padders
    ZeroPadder,
    PadderFactory,
    # Augmenters
    NoiseAugmenter,
    ComposeAugmenter,
    AugmenterFactory
)

# 1. Normalización
normalizer = NormalizerFactory.create('standard')
data = normalizer.normalize(torch.randn(10))

# 2. Tokenización
tokenizer = TokenizerFactory.create('simple')
tokens = tokenizer.tokenize("hello world")

# 3. Padding
padder = PadderFactory.create('zero')
padded = padder.pad(sequence, 20)

# 4. Augmentación
augmenter = AugmenterFactory.create('noise', noise_level=0.1)
augmented = augmenter.augment(data)
```

## 🚀 Ventajas de V5

1. **Máxima Separación**: Cada componente en su propio archivo
2. **Fácil Navegación**: Fácil encontrar componentes específicos
3. **Máxima Reutilización**: Componentes ultra-específicos
4. **Fácil Testing**: Archivos pequeños fáciles de testear
5. **Máxima Extensibilidad**: Fácil agregar nuevos componentes

## 📈 Comparación de Versiones

| Aspecto | V3 | V4 | V5 |
|---------|----|----|----|
| Archivos por Categoría | 1 | 1 | 4+ |
| Componentes por Archivo | 5-9 | 5-9 | 1-8 |
| Granularidad | Alta | Muy Alta | Máxima |
| Separación | Media | Alta | Máxima |

## 🎓 Mejores Prácticas V5

1. **Importar desde Módulos Especializados**
   ```python
   from .normalizers import StandardNormalizer
   from .tokenizers import SimpleTokenizer
   ```

2. **Usar Factories para Flexibilidad**
   ```python
   normalizer = NormalizerFactory.create('standard')
   ```

3. **Componer Componentes**
   ```python
   composed = ComposeAugmenter([NoiseAugmenter(), ScaleAugmenter()])
   ```

4. **Extender Base Classes**
   ```python
   class CustomNormalizer(NormalizerBase):
       def _compute_stats(self, data):
           # Custom implementation
           pass
   ```

---

**Version**: 3.8.0  
**Status**: Maximum Granularity ✅  
**Modularity Level**: Extreme ⭐⭐⭐⭐⭐  
**Last Updated**: 2025



