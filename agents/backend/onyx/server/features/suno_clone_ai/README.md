# Suno Clone AI - Sistema de Generación de Música con IA

## 🚀 Overview

Suno Clone AI es un sistema completo de generación de música con inteligencia artificial que permite a los usuarios crear canciones mediante chat, similar a Suno AI. Los usuarios pueden describir en lenguaje natural lo que quieren y el sistema generará la canción correspondiente.

## ✨ Características Principales

### 🎵 Generación de Música
- **Generación desde Chat**: Los usuarios pueden escribir en lenguaje natural lo que quieren
- **Múltiples Modelos**: Soporte para MusicGen (small, medium, large)
- **Procesamiento Inteligente**: Extracción automática de género, mood, tempo e instrumentos
- **Mejora con IA**: Uso de OpenAI para mejorar prompts de generación
- **Caché Inteligente**: Sistema de caché para evitar regeneraciones innecesarias
- **Procesamiento Automático**: Normalización y fade in/out automáticos
- **🚀 Ultra-Rápido**: Optimizaciones avanzadas con torch.compile, mixed precision, y caché (hasta 5-10x más rápido)

### 💬 Chat Inteligente
- **Procesamiento de Lenguaje Natural**: Interpreta solicitudes en lenguaje natural
- **Historial de Conversación**: Mantiene contexto de conversaciones previas
- **Extracción Automática**: Identifica automáticamente:
  - Género musical
  - Estado de ánimo
  - Tempo/BPM
  - Instrumentos
  - Duración

### 🎛️ Control Avanzado
- **Parámetros Personalizables**: Control sobre duración, temperatura, top-k, top-p
- **Múltiples Formatos**: Generación en WAV con calidad configurable
- **Background Processing**: Generación asíncrona de canciones

### 📊 Gestión de Canciones
- **Almacenamiento**: Base de datos SQLite para metadata
- **Historial**: Guarda todas las canciones generadas
- **Descarga**: Endpoint para descargar archivos de audio
- **Búsqueda**: Filtrado por usuario, fecha, etc.
- **Edición Avanzada**: Reverb, EQ, cambio de tempo/pitch
- **Mezcla**: Combinar múltiples canciones
- **Análisis**: Análisis detallado de características de audio

### 📈 Analytics y Métricas
- **Estadísticas Generales**: Tracking completo del sistema
- **Métricas por Usuario**: Estadísticas individuales
- **Performance Tracking**: Tiempos de generación y uso
- **Prompts Populares**: Análisis de uso más frecuente

## 📦 Instalación

### Requisitos Previos
- Python 3.8+
- CUDA (opcional, para GPU)
- FFmpeg (para procesamiento de audio)

### Instalación de Dependencias

```bash
pip install -r requirements.txt
```

### Configuración

Crea un archivo `.env` en el directorio raíz:

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8020
DEBUG=False

# OpenAI (opcional, para mejorar prompts)
OPENAI_API_KEY=your-openai-api-key
OPENAI_MODEL=gpt-4

# Music Generation
MUSIC_MODEL=facebook/musicgen-medium
USE_GPU=True
MAX_AUDIO_LENGTH=300
DEFAULT_DURATION=30
SAMPLE_RATE=32000

# Storage
AUDIO_STORAGE_PATH=./storage/audio
DATABASE_URL=sqlite:///./suno_clone.db

# Security
SECRET_KEY=your-secret-key-change-in-production
```

## 🎯 Uso Rápido

### Iniciar el Servidor

```bash
python main.py
```

O con uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 8020
```

### Ejemplos de Uso

#### 1. Crear Canción desde Chat

```bash
curl -X POST "http://localhost:8020/suno/chat/create-song" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Quiero una canción de rock energética con guitarra y batería, 2 minutos de duración",
    "user_id": "user123"
  }'
```

#### 2. Generar Canción Directamente

```bash
curl -X POST "http://localhost:8020/suno/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Upbeat electronic music with synthesizers",
    "duration": 30,
    "genre": "electronic",
    "mood": "energetic"
  }'
```

#### 3. Listar Canciones

```bash
curl "http://localhost:8020/suno/songs?user_id=user123"
```

#### 4. Descargar Canción

```bash
curl "http://localhost:8020/suno/songs/{song_id}/download" --output song.wav
```

## 📚 API Endpoints

### Chat
- `POST /suno/chat/create-song` - Crea canción desde mensaje de chat
- `GET /suno/chat/history/{user_id}` - Obtiene historial de chat

### Canciones
- `GET /suno/songs` - Lista todas las canciones
- `GET /suno/songs/{song_id}` - Obtiene información de una canción
- `GET /suno/songs/{song_id}/download` - Descarga archivo de audio
- `DELETE /suno/songs/{song_id}` - Elimina una canción
- `POST /suno/songs/{song_id}/edit` - Edita una canción con efectos
- `POST /suno/songs/mix` - Mezcla múltiples canciones
- `GET /suno/songs/{song_id}/analyze` - Analiza características de audio

### Generación
- `POST /suno/generate` - Genera canción desde prompt
- `GET /suno/generate/status/{task_id}` - Obtiene estado de generación

### Modelos
- `GET /suno/models` - Lista modelos disponibles
- `GET /suno/models/{model_id}` - Información de un modelo

### Métricas
- `GET /suno/metrics/stats` - Estadísticas generales del sistema
- `GET /suno/metrics/user/{user_id}` - Estadísticas de usuario

### Caché
- `GET /suno/cache/stats` - Estadísticas del caché
- `DELETE /suno/cache/clear` - Limpiar caché

## 🏗️ Arquitectura

```
suno_clone_ai/
├── api/                 # Endpoints de API
│   └── song_api.py     # Endpoints principales
├── core/               # Lógica de negocio
│   ├── music_generator.py    # Generador de música
│   ├── chat_processor.py     # Procesador de chat
│   ├── cache_manager.py      # Gestor de caché
│   ├── audio_processor.py    # Procesador de audio avanzado
│   └── error_handler.py      # Manejo centralizado de errores
├── services/            # Servicios
│   ├── song_service.py       # Gestión de canciones
│   └── metrics_service.py    # Servicio de métricas
├── config/             # Configuración
│   └── settings.py           # Settings
├── middleware/         # Middleware
│   ├── logging_middleware.py  # Logging
│   └── rate_limiter.py        # Rate limiting
├── utils/             # Utilidades
│   └── validators.py          # Validadores reutilizables
├── main.py            # Servidor principal
├── requirements.txt   # Dependencias
├── README.md          # Documentación principal
├── QUICK_START.md     # Guía rápida
└── ADVANCED_FEATURES.md # Funcionalidades avanzadas
```

## 🔧 Configuración Avanzada

### Modelos Disponibles

- **facebook/musicgen-small**: Modelo pequeño y rápido (~300MB)
- **facebook/musicgen-medium**: Modelo balanceado, por defecto (~1.5GB)
- **facebook/musicgen-large**: Modelo grande con mejor calidad (~3GB)

### Parámetros de Generación

- `temperature`: Controla la creatividad (default: 1.0)
- `top_k`: Número de tokens a considerar (default: 250)
- `top_p`: Nucleus sampling (default: 0.0)
- `cfg_coef`: Guidance scale (default: 3.0)

## 🚀 Despliegue

### Docker (Próximamente)

```bash
docker build -t suno-clone-ai .
docker run -p 8020:8020 suno-clone-ai
```

### Producción

Para producción, usar un servidor ASGI como Gunicorn:

```bash
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8020
```

## 📊 Performance

- **Generación Estándar**: ~10-30 segundos dependiendo del modelo y duración
- **Generación Ultra-Rápida**: ~2-6 segundos con todas las optimizaciones activadas (5-10x más rápido)
- **GPU**: Acelera significativamente la generación (requerido para máximo rendimiento)
- **Cache**: Redis opcional para caching de resultados + caché en memoria/disco integrado
- **Optimizaciones**: torch.compile, mixed precision (FP16), batch processing, async inference

### 🚀 Uso del Generador Ultra-Rápido

```python
from core.ultra_fast_generator import get_ultra_fast_generator

# Generador con todas las optimizaciones
generator = get_ultra_fast_generator(
    compile_mode="max-autotune",  # Máxima velocidad
    use_cache=True
)

# Generación rápida
audio = generator.generate_from_text(
    text="Upbeat electronic music",
    duration=30
)

# Generación asíncrona (para APIs)
import asyncio
audio = await generator.generate_async(
    text="Calm acoustic guitar",
    duration=30
)

# Generación por lotes (más eficiente)
texts = ["Rock song", "Jazz piece", "Electronic beat"]
audio_list = generator.generate_batch(texts, duration=30)
```

Ver [SPEED_OPTIMIZATIONS.md](SPEED_OPTIMIZATIONS.md) para más detalles sobre optimizaciones de velocidad.

Ver [ADVANCED_OPTIMIZATIONS.md](ADVANCED_OPTIMIZATIONS.md) para optimizaciones avanzadas adicionales (streaming, procesamiento paralelo, etc.).

Ver [ULTRA_OPTIMIZATIONS.md](ULTRA_OPTIMIZATIONS.md) para optimizaciones ultra avanzadas (ONNX, TensorRT, quantización, smart cache, etc.).

Ver [API_OPTIMIZATIONS.md](API_OPTIMIZATIONS.md) para optimizaciones de capa de API (serialización, compresión, queries, requests, etc.).

Ver [SYSTEM_OPTIMIZATIONS.md](SYSTEM_OPTIMIZATIONS.md) para optimizaciones de sistema (base de datos, almacenamiento, monitoreo, etc.).

Ver [FINAL_OPTIMIZATIONS.md](FINAL_OPTIMIZATIONS.md) para resumen completo de todas las optimizaciones (seguridad, escalabilidad, deployment, etc.).

Ver [COMPLETE_OPTIMIZATIONS.md](COMPLETE_OPTIMIZATIONS.md) para referencia completa de todas las optimizaciones implementadas.

Ver [EXECUTIVE_SUMMARY.md](EXECUTIVE_SUMMARY.md) para resumen ejecutivo de todas las optimizaciones y métricas de mejora.

Ver [ALL_OPTIMIZATIONS.md](ALL_OPTIMIZATIONS.md) para referencia completa de todos los 28+ módulos de optimización implementados.

Ver [ULTIMATE_OPTIMIZATIONS.md](ULTIMATE_OPTIMIZATIONS.md) para referencia ultimate de todos los 31+ módulos incluyendo costos, compliance y serverless.

## 🔒 Seguridad

- **Rate Limiting**: Protección contra abuso con límites configurables
- **Validación Exhaustiva**: Validación de todos los inputs con Pydantic
- **Sanitización**: Limpieza automática de datos peligrosos
- **Error Handling**: Manejo seguro de errores sin exponer información sensible
- **Health Checks**: Monitoreo completo del estado del sistema

## 🧪 Testing

```bash
pytest tests/
```

## 📝 Licencia

Ver LICENSE file para detalles.

## 🤝 Contribución

Contribuciones son bienvenidas! Por favor ver CONTRIBUTING.md para guías.

## 📧 Soporte

Para soporte, abrir un issue o contactar al equipo de desarrollo.

## 🆕 Roadmap

- [ ] Soporte para más modelos (MusicLM, AudioLM, etc.)
- [ ] Generación de letras con IA
- [ ] Mezcla y masterización automática
- [ ] Integración con servicios de streaming
- [ ] Frontend web interactivo
- [ ] API de WebSocket para streaming en tiempo real
- [ ] Soporte para múltiples voces
- [ ] Generación colaborativa

## 📖 Documentación Adicional

- [QUICK_START.md](QUICK_START.md) - Guía de inicio rápido
- [ADVANCED_FEATURES.md](ADVANCED_FEATURES.md) - Funcionalidades avanzadas (caché, edición, métricas)
- [IMPROVEMENTS.md](IMPROVEMENTS.md) - Mejoras implementadas (seguridad, validación, robustez)

