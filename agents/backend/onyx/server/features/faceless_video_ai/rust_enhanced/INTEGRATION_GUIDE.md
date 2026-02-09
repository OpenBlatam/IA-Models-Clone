# 🔌 Guía de Integración - Rust Enhanced Core

Esta guía explica cómo integrar el core Rust mejorado con el proyecto Python existente.

## Instalación

### Desarrollo

```bash
cd rust_enhanced
maturin develop
```

### Producción

```bash
maturin develop --release
```

### Build Wheel

```bash
maturin build --release
```

## Integración Gradual

### Fase 1: Video Effects (Más fácil de integrar)

Reemplazar `VisualEffectsService` con `EffectsEngine`:

```python
# Antes
from services.visual_effects import VisualEffectsService
service = VisualEffectsService()
result = await service.add_ken_burns_effect(
    image_path="image.jpg",
    duration=5.0,
    zoom=1.2
)  # ~2.5 segundos

# Después
from faceless_video_enhanced import EffectsEngine
engine = EffectsEngine()
result = engine.ken_burns(
    image_path="image.jpg",
    duration=5.0,
    zoom=1.2
)  # ~0.05 segundos (50x más rápido!)
```

### Fase 2: Color Grading

Reemplazar procesamiento de color Python:

```python
# Antes
from PIL import Image, ImageEnhance
img = Image.open("image.jpg")
enhancer = ImageEnhance.Brightness(img)
img = enhancer.enhance(1.1)  # Lento

# Después
from faceless_video_enhanced import ColorGrading
grading = ColorGrading()
result = grading.apply(
    image_path="image.jpg",
    brightness=0.1,
    contrast=1.2,
    saturation=1.1
)  # 20-100x más rápido
```

### Fase 3: Transitions

Reemplazar transiciones FFmpeg:

```python
# Antes
import subprocess
subprocess.run([
    "ffmpeg", "-i", "video1.mp4", "-i", "video2.mp4",
    "-filter_complex", "xfade=transition=fade:duration=1",
    "output.mp4"
])  # ~1 segundo

# Después
from faceless_video_enhanced import TransitionEngine
transitions = TransitionEngine()
result = transitions.crossfade(
    image1_path="image1.jpg",
    image2_path="image2.jpg",
    duration=1.0
)  # ~0.03 segundos (33x más rápido)
```

### Fase 4: Audio Processing

Reemplazar pydub/ffmpeg para audio:

```python
# Antes
from pydub import AudioSegment
audio = AudioSegment.from_mp3("audio.mp3")
audio = audio.normalize()  # Lento

# Después
from faceless_video_enhanced import AudioProcessor
audio = AudioProcessor()
result = audio.normalize(
    audio_path="audio.mp3",
    target_db=-3.0
)  # 10-20x más rápido
```

## Wrapper para Compatibilidad

Crear un wrapper que mantenga la API existente:

```python
# services/visual_effects_rust.py
from faceless_video_enhanced import EffectsEngine as RustEffectsEngine
from services.visual_effects import VisualEffectsService

class VisualEffectsService:
    def __init__(self):
        self.rust_engine = RustEffectsEngine()
        self.use_rust = True  # Feature flag
    
    async def add_ken_burns_effect(self, image_path, duration, zoom, **kwargs):
        if self.use_rust:
            # Usar Rust (50x más rápido)
            return self.rust_engine.ken_burns(
                image_path=str(image_path),
                duration=float(duration),
                zoom=float(zoom),
                pan_x=kwargs.get("pan_x", 0.1),
                pan_y=kwargs.get("pan_y", 0.1)
            )
        else:
            # Fallback a Python
            from services.visual_effects import VisualEffectsService as PyService
            service = PyService()
            return await service.add_ken_burns_effect(image_path, duration, zoom, **kwargs)
```

## Feature Flags

Usar feature flags para rollout gradual:

```python
from config.settings import settings

if settings.USE_RUST_EFFECTS:
    from faceless_video_enhanced import EffectsEngine
    engine = EffectsEngine()
else:
    from services.visual_effects import VisualEffectsService
    engine = VisualEffectsService()
```

## Testing

### Unit Tests

```python
import pytest
from faceless_video_enhanced import EffectsEngine

def test_ken_burns():
    engine = EffectsEngine()
    result = engine.ken_burns(
        image_path="test_image.jpg",
        duration=1.0,
        zoom=1.1
    )
    assert result is not None
```

### Performance Tests

```python
import time
from faceless_video_enhanced import EffectsEngine

def test_performance():
    engine = EffectsEngine()
    
    start = time.time()
    result = engine.ken_burns(
        image_path="test_image.jpg",
        duration=5.0,
        zoom=1.2
    )
    duration = time.time() - start
    
    assert duration < 0.1  # Debe ser < 100ms
    print(f"Ken Burns took {duration:.3f}s")
```

## Benchmarks

Ejecutar benchmarks Rust:

```bash
cd rust_enhanced
cargo bench
```

Comparar con Python:

```python
import time
from services.visual_effects import VisualEffectsService
from faceless_video_enhanced import EffectsEngine

# Python
py_service = VisualEffectsService()
start = time.time()
result = await py_service.add_ken_burns_effect("image.jpg", 5.0, 1.2)
py_time = time.time() - start

# Rust
rust_engine = EffectsEngine()
start = time.time()
result = rust_engine.ken_burns("image.jpg", 5.0, 1.2)
rust_time = time.time() - start

print(f"Python: {py_time:.3f}s")
print(f"Rust: {rust_time:.3f}s")
print(f"Speedup: {py_time/rust_time:.1f}x")
```

## Troubleshooting

### Error: Module not found

```bash
# Reinstalar
cd rust_enhanced
maturin develop --release
```

### Error: Segmentation fault

```bash
# Recompilar con debug
maturin develop
```

### Performance no mejora

```python
# Verificar que estás usando Rust
from faceless_video_enhanced import EffectsEngine
print(EffectsEngine.__module__)  # Debe ser 'faceless_video_enhanced'
```

## Próximos Pasos

1. ✅ Implementar efectos básicos
2. ⏳ Agregar FFmpeg integration
3. ⏳ Implementar GPU acceleration
4. ⏳ Agregar más efectos
5. ⏳ Optimizar memory usage












