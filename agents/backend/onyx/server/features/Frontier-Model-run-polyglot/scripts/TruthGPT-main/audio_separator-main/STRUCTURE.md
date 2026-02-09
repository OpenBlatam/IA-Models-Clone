# Audio Separator - Estructura del Proyecto

Este documento describe la estructura del proyecto `audio_separator-main`, que sigue el mismo patrón organizacional que `sam3-main`.

## Estructura de Directorios

```
audio_separator-main/
├── audio_separator/          # Módulo principal
│   ├── __init__.py
│   ├── model_builder.py      # Constructor de modelos
│   ├── model/                # Modelos de separación
│   │   ├── __init__.py
│   │   ├── base_separator.py      # Clase base abstracta
│   │   ├── demucs_model.py        # Modelo Demucs
│   │   ├── spleeter_model.py      # Modelo Spleeter
│   │   ├── lalal_model.py         # Modelo LALAL.AI (API)
│   │   └── hybrid_model.py        # Modelo híbrido (ensamble)
│   ├── processor/            # Procesadores de audio
│   │   ├── __init__.py
│   │   ├── audio_loader.py        # Carga de archivos de audio
│   │   ├── audio_saver.py         # Guardado de archivos de audio
│   │   ├── preprocessor.py        # Preprocesamiento
│   │   └── postprocessor.py       # Postprocesamiento
│   ├── separator/            # Separadores de alto nivel
│   │   ├── __init__.py
│   │   ├── audio_separator.py     # Separador principal
│   │   └── batch_separator.py     # Procesamiento por lotes
│   ├── eval/                 # Evaluación
│   │   ├── __init__.py
│   │   └── metrics.py             # Métricas (SDR, SIR, SAR, ISDR)
│   ├── train/                # Entrenamiento (placeholder)
│   │   └── __init__.py
│   └── utils/                # Utilidades
│       ├── __init__.py
│       └── audio_utils.py         # Funciones auxiliares
├── examples/                 # Ejemplos de uso
│   ├── basic_separation.py
│   └── batch_processing.py
├── scripts/                  # Scripts de utilidad
│   └── eval/
│       └── evaluate_separation.py
├── assets/                   # Recursos
│   ├── models/              # Modelos pre-entrenados
│   ├── examples/            # Archivos de audio de ejemplo
│   ├── configs/             # Archivos de configuración
│   └── README.md
├── README.md                 # Documentación principal
├── CONTRIBUTING.md           # Guía de contribución
├── LICENSE                   # Licencia MIT
├── pyproject.toml            # Configuración del proyecto
├── MANIFEST.in               # Archivos a incluir en el paquete
└── .gitignore               # Archivos ignorados por git
```

## Componentes Principales

### 1. Modelos (`audio_separator/model/`)

- **BaseSeparatorModel**: Clase base abstracta que define la interfaz para todos los modelos
- **DemucsModel**: Implementación del modelo Demucs (Meta AI Research)
- **SpleeterModel**: Implementación del modelo Spleeter (Deezer Research)
- **LalalModel**: Wrapper para la API de LALAL.AI
- **HybridSeparatorModel**: Modelo híbrido que combina múltiples modelos

### 2. Procesadores (`audio_separator/processor/`)

- **AudioLoader**: Carga archivos de audio en varios formatos
- **AudioSaver**: Guarda archivos de audio
- **AudioPreprocessor**: Preprocesa audio (resample, normalización, etc.)
- **AudioPostprocessor**: Postprocesa salidas del modelo

### 3. Separadores (`audio_separator/separator/`)

- **AudioSeparator**: Interfaz de alto nivel para separación de audio
- **BatchSeparator**: Procesamiento por lotes de múltiples archivos

### 4. Evaluación (`audio_separator/eval/`)

- **metrics.py**: Implementa métricas estándar:
  - SDR (Signal-to-Distortion Ratio)
  - SIR (Signal-to-Interference Ratio)
  - SAR (Signal-to-Artifacts Ratio)
  - ISDR (Improvement in Signal-to-Distortion Ratio)

### 5. Utilidades (`audio_separator/utils/`)

- Funciones auxiliares para procesamiento de audio
- Información de archivos de audio
- Conversión de formatos

## Uso Básico

```python
from audio_separator import AudioSeparator

# Inicializar separador
separator = AudioSeparator(model_type="demucs")

# Separar archivo de audio
results = separator.separate_file("song.mp3", output_dir="separated")
```

## Extensibilidad

El proyecto está diseñado para ser fácilmente extensible:

1. **Agregar nuevos modelos**: Crear una clase que herede de `BaseSeparatorModel`
2. **Agregar nuevos procesadores**: Implementar las interfaces correspondientes
3. **Agregar nuevas métricas**: Extender el módulo `eval/metrics.py`

## Comparación con sam3-main

Esta estructura sigue el mismo patrón organizacional que `sam3-main`:

- Módulo principal con subcarpetas organizadas por funcionalidad
- Separación clara entre modelos, procesadores, y utilidades
- Ejemplos y scripts en directorios separados
- Documentación completa (README, CONTRIBUTING, etc.)
- Configuración estándar (pyproject.toml, MANIFEST.in)

