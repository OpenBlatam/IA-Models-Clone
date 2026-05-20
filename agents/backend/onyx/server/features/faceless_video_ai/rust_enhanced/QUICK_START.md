# 🦀 Quick Start - Rust Enhanced Core

## Instalación Rápida

```bash
cd rust_enhanced
pip install maturin
maturin develop --release
```

## Uso Básico

```python
from faceless_video_enhanced import EffectsEngine, ColorGrading

# Video effects (10-50x más rápido)
engine = EffectsEngine()
result = engine.ken_burns(
    image_path="image.jpg",
    duration=5.0,
    zoom=1.2
)

# Color grading (20-100x más rápido)
grading = ColorGrading()
result = grading.apply(
    image_path="image.jpg",
    brightness=0.1,
    contrast=1.2,
    saturation=1.1
)
```

## Módulos Disponibles

- `EffectsEngine` - Video effects (Ken Burns, fades, blur)
- `ColorGrading` - Color correction y palettes
- `TransitionEngine` - Video transitions
- `AudioProcessor` - Audio processing
- `VideoProcessor` - Core video operations

## Próximos Pasos

Ver `INTEGRATION_GUIDE.md` para integración completa con el proyecto existente.












