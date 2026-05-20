# Audio Separation Core

Sistema completo de separación y mezcla de audio con IA, basado en la arquitectura de `optimization_core`.

## 🎯 Características

- **Separación de Audio**: Separa audio de videos en componentes (voces, música, efectos, etc.)
- **Mezcla de Audio**: Mezcla componentes de audio con control de volúmenes y efectos
- **Procesamiento de Video**: Extrae audio de archivos de video
- **Múltiples Modelos**: Soporte para Spleeter, Demucs, y LALAL.AI
- **Arquitectura Modular**: Basada en interfaces y factories para fácil extensión
- **Alto Rendimiento**: Optimizado para procesamiento eficiente

## 📦 Instalación

```bash
# Instalar dependencias básicas
pip install librosa soundfile numpy scipy

# Para separación con Spleeter
pip install spleeter

# Para separación con Demucs
pip install demucs torch

# Para procesamiento de video (requiere ffmpeg)
# Windows: choco install ffmpeg
# Linux: sudo apt-get install ffmpeg
# macOS: brew install ffmpeg
```

## 🚀 Uso Rápido

### Separar Audio de un Video

```python
from audio_separation_core import separate_audio, process_video_audio

# Opción 1: Procesar video completo (extrae y separa)
result = process_video_audio(
    "video.mp4",
    output_dir="output",
    components=["vocals", "accompaniment"]
)

print(f"Audio extraído: {result['audio_path']}")
print(f"Voces: {result['separated']['vocals']}")
print(f"Acompañamiento: {result['separated']['accompaniment']}")

# Opción 2: Separar audio existente
separated = separate_audio(
    "audio.wav",
    output_dir="output",
    separator_type="demucs",  # o "spleeter", "lalal", "auto"
    components=["vocals", "drums", "bass", "other"]
)
```

### Mezclar Audio

```python
from audio_separation_core import mix_audio

# Mezclar componentes con volúmenes personalizados
mixed = mix_audio(
    {
        "vocals": "output/vocals.wav",
        "music": "output/accompaniment.wav"
    },
    "output/mixed.wav",
    mixer_type="advanced",
    volumes={
        "vocals": 0.8,
        "music": 0.6
    }
)
```

### Uso Avanzado

```python
from audio_separation_core import (
    create_audio_separator,
    create_audio_mixer,
    create_audio_processor,
    SeparationConfig,
    MixingConfig
)

# Crear separador con configuración personalizada
config = SeparationConfig(
    model_type="demucs",
    use_gpu=True,
    components=["vocals", "drums", "bass", "other"]
)
separator = create_audio_separator("demucs", config=config)

# Separar
results = separator.separate("audio.wav", "output")

# Crear mezclador avanzado
mixer_config = MixingConfig(
    mixer_type="advanced",
    apply_reverb=True,
    apply_eq=True,
    fade_in=0.5,
    fade_out=2.0
)
mixer = create_audio_mixer("advanced", config=mixer_config)

# Mezclar con efectos
mixed = mixer.mix(
    results,
    "output/final_mix.wav",
    volumes={"vocals": 0.9, "drums": 0.7, "bass": 0.8, "other": 0.6}
)
```

## 🏗️ Arquitectura

El sistema sigue la arquitectura de `optimization_core`:

```
audio_separation_core/
├── core/              # Interfaces, config, factories
├── separators/        # Implementaciones de separadores
├── mixers/            # Implementaciones de mezcladores
├── processors/        # Procesadores de audio/video
├── utils/             # Utilidades
└── specs/             # Especificaciones
```

### Componentes Principales

1. **Interfaces Core** (`core/interfaces.py`):
   - `IAudioComponent`: Interfaz base
   - `IAudioSeparator`: Interfaz para separadores
   - `IAudioMixer`: Interfaz para mezcladores
   - `IAudioProcessor`: Interfaz para procesadores

2. **Separadores**:
   - `SpleeterSeparator`: Usa Spleeter de Deezer
   - `DemucsSeparator`: Usa Demucs de Facebook Research
   - `LALALSeparator`: Usa LALAL.AI API

3. **Mezcladores**:
   - `SimpleMixer`: Mezcla básica con control de volúmenes
   - `AdvancedMixer`: Mezcla avanzada con efectos (reverb, EQ, compresor)

4. **Procesadores**:
   - `VideoAudioExtractor`: Extrae audio de videos usando ffmpeg
   - `AudioFormatConverter`: Convierte entre formatos
   - `AudioEnhancer`: Mejora calidad de audio

## 📚 Ejemplos

Ver la carpeta `examples/` para más ejemplos de uso.

## 🔧 Configuración

### Separación

```python
from audio_separation_core import SeparationConfig

config = SeparationConfig(
    model_type="demucs",        # "spleeter", "demucs", "lalal", "auto"
    use_gpu=True,                # Usar GPU si está disponible
    components=["vocals", "accompaniment"],
    overlap=0.25,                # Overlap entre chunks
    post_process=True            # Post-procesamiento para mejorar calidad
)
```

### Mezcla

```python
from audio_separation_core import MixingConfig

config = MixingConfig(
    mixer_type="advanced",       # "simple" o "advanced"
    default_volume=0.8,          # Volumen por defecto (0.0-1.0)
    normalize_output=True,       # Normalizar salida
    fade_in=0.5,                 # Fade in en segundos
    fade_out=2.0,                # Fade out en segundos
    apply_reverb=True,           # Aplicar reverb
    apply_eq=True,               # Aplicar EQ
    apply_compressor=True        # Aplicar compresor
)
```

## 🧪 Testing

```bash
# Ejecutar tests
python -m pytest tests/

# Con cobertura
python -m pytest tests/ --cov=audio_separation_core
```

## 📖 Documentación

Ver `specs/` para especificaciones detalladas de la arquitectura.

## 🤝 Contribuir

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📝 Licencia

Este proyecto está bajo la misma licencia que `optimization_core`.

## 🙏 Agradecimientos

- Basado en la arquitectura de `optimization_core`
- Modelos de separación: Spleeter (Deezer), Demucs (Facebook Research), LALAL.AI




