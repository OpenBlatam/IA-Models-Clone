# Music Generation Module

## Overview

Modular music generation system with unified interfaces for:
- Music generation models (Audiocraft, MusicGen, Stable Audio)
- Professional audio post-processing
- Voice synthesis and cloning

## Quick Start

### Basic Music Generation

```python
from core.music_generation import create_generator, AudioPostProcessor

# Create generator
generator = create_generator(
    generator_type="audiocraft",
    model_name="facebook/musicgen-large"
)

# Generate music
audio = generator.generate(
    prompt="upbeat electronic music with synthesizers",
    duration=30
)

# Post-process
processor = AudioPostProcessor(sample_rate=32000)
processed = processor.process_full_pipeline(audio)
```

### Voice Synthesis

```python
from core.music_generation import VoiceSynthesizer, VoiceCloner

# Synthesize voice
tts = VoiceSynthesizer()
audio = tts.synthesize("Your lyrics here", language="en")

# Clone voice
cloner = VoiceCloner()
cloned = cloner.clone_voice(
    text="Your lyrics",
    reference_audio="reference.wav"
)
```

## Modules

### 1. Generators (`generators.py`)
- `AudiocraftGenerator`: Meta's Audiocraft models
- `MusicGenHuggingFaceGenerator`: MusicGen via Hugging Face
- `StableAudioGenerator`: Stability AI's Stable Audio
- `create_generator()`: Factory function

### 2. Post-Processing (`post_processing.py`)
- `AudioPostProcessor`: Full post-processing pipeline
- `TimeStretchProcessor`: Time stretching and pitch shifting
- Effects: Reverb, compression, noise reduction, normalization

### 3. Voice Synthesis (`voice_synthesis.py`)
- `VoiceSynthesizer`: Text-to-speech synthesis
- `VoiceCloner`: Voice cloning with XTTS

## Features

- **Unified Interface**: Same API for all generators
- **Professional Post-Processing**: Industry-standard effects
- **Voice Cloning**: Clone voices with minimal reference audio
- **Batch Processing**: Generate multiple tracks in parallel
- **Error Handling**: Robust error handling and logging
- **Modular Design**: Easy to extend and customize

## Módulos Avanzados

### 4. Analysis (`analysis.py`) ⭐ NEW
- `AudioAnalyzer`: Análisis completo de audio
  - Análisis de tempo (librosa, Essentia, Madmom)
  - Análisis de key (tonalidad)
  - Análisis de estructura musical
  - Análisis de armonía
- `BeatTracker`: Seguimiento de beats avanzado

### 5. Mixing (`mixing.py`) ⭐ NEW
- `AudioMixer`: Mezcla profesional
  - Mezcla de múltiples pistas
  - EQ de 3 bandas
  - Limitador
  - Panning estéreo
- `StemSeparator`: Separación de stems
  - Separar voces, batería, bajo, otros
  - Usa Demucs para separación

### 6. Optimization (`optimization.py`) ⭐ NEW
- `GPUOptimizer`: Optimización de GPU
  - Configuración óptima de cuDNN
  - Limpieza de caché GPU
- `MemoryOptimizer`: Optimización de memoria
  - Cuantización de modelos (8-bit, 4-bit)
  - Gradient checkpointing
- `BatchOptimizer`: Optimización de batch size
- `ModelCache`: Caché de modelos

### 7. Pipeline (`pipeline.py`) ⭐ NEW
- `MusicGenerationPipeline`: Pipeline completo
  - Generación + post-procesamiento + análisis
  - Separación de stems integrada
  - Batch processing
- `VoiceMusicPipeline`: Pipeline con voces
  - Generación de música + síntesis de voz
  - Mezcla automática de música y voces
  - Soporte para clonación de voz

### 8. Async (`async_generator.py`) ⭐ NEW
- `AsyncMusicGenerator`: Generación asíncrona
  - Generación no bloqueante
  - Batch processing asíncrono
  - Post-procesamiento asíncrono
- `AsyncPipeline`: Pipeline asíncrono completo
  - Pipeline completo sin bloqueo
  - Múltiples generaciones concurrentes

### 9. Export (`export.py`) ⭐ NEW
- `AudioExporter`: Exportación a múltiples formatos
  - WAV (lossless)
  - MP3 (compressed)
  - FLAC (lossless compressed)
  - OGG Vorbis
  - Exportación a múltiples formatos simultáneamente

## Ejemplos Avanzados

### Análisis de Audio

```python
from core.music_generation import AudioAnalyzer

analyzer = AudioAnalyzer(sample_rate=44100)
analysis = analyzer.full_analysis(audio)

print(f"Tempo: {analysis['tempo']}")
print(f"Key: {analysis['key']}")
print(f"Structure: {analysis['structure']}")
```

### Mezcla de Pistas

```python
from core.music_generation import AudioMixer

mixer = AudioMixer(sample_rate=44100)

# Mezclar múltiples pistas
tracks = [drums, bass, vocals, melody]
volumes = [0.8, 0.7, 1.0, 0.6]
panning = [-0.3, 0.0, 0.2, 0.3]

mixed = mixer.mix_tracks(tracks, volumes=volumes, panning=panning)

# Aplicar EQ
eqd = mixer.apply_eq(mixed, low_gain=2.0, mid_gain=0.0, high_gain=-1.0)

# Aplicar limitador
final = mixer.apply_limiter(eqd, threshold=0.95)
```

### Separación de Stems

```python
from core.music_generation import StemSeparator

separator = StemSeparator(model_name="htdemucs")
stems = separator.separate(audio, sample_rate=44100)

# Acceder a stems individuales
vocals = stems["vocals"]
drums = stems["drums"]
bass = stems["bass"]
other = stems["other"]
```

### Optimización

```python
from core.music_generation import (
    GPUOptimizer,
    MemoryOptimizer,
    get_model_cache
)

# Setup GPU
GPUOptimizer.setup_optimal_gpu_settings()

# Cuantizar modelo
optimized_model = MemoryOptimizer.quantize_model(model, bits=8)

# Usar caché
cache = get_model_cache()
cached_model = cache.get("musicgen-large")
if cached_model is None:
    cached_model = load_model()
    cache.set("musicgen-large", cached_model)
```

## Requirements

See `REALISTIC_MUSIC_LIBRARIES.md` for complete requirements.

Essential:
- `audiocraft>=1.4.0`
- `pedalboard>=0.9.0`
- `TTS>=0.22.0`
- `noisereduce>=3.1.0`
- `librosa>=0.10.2`
- `demucs>=4.0.0` (for stem separation)
- `essentia>=2.1b6` (optional, for advanced analysis)
- `madmom>=0.16.0` (optional, for beat tracking)

