# Arquitectura del Sistema - Faceless Video AI

## 🏗️ Visión General

El sistema Faceless Video AI está diseñado con una arquitectura modular y escalable que permite generar videos completamente con IA a partir de scripts de texto.

## 📐 Arquitectura de Componentes

```
┌─────────────────────────────────────────────────────────────┐
│                    API Layer (FastAPI)                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Routes     │  │  Background  │  │   Upload     │     │
│  │              │  │    Tasks     │  │   Handler   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Orchestrator (VideoOrchestrator)               │
│         Coordina todo el pipeline de generación             │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   Script     │  │    Video    │  │    Audio     │
│  Processor   │  │  Generator  │  │  Generator   │
└──────────────┘  └──────────────┘  └──────────────┘
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Subtitle    │  │   Video      │  │   TTS        │
│  Generator   │  │  Compositor  │  │  Service    │
└──────────────┘  └──────────────┘  └──────────────┘
```

## 🔄 Flujo de Generación

### 1. Recepción de Request
- El usuario envía un `VideoGenerationRequest` con:
  - Script de texto
  - Configuración de video
  - Configuración de audio
  - Configuración de subtítulos

### 2. Procesamiento de Script
- **ScriptProcessor** divide el texto en segmentos
- Calcula timing para cada segmento
- Extrae keywords para generación de imágenes

### 3. Generación de Imágenes
- **VideoGenerator** crea imágenes con IA para cada segmento
- Usa prompts generados desde el texto
- Aplica estilo configurado (realistic, animated, etc.)

### 4. Generación de Audio
- **AudioGenerator** sintetiza voz del script completo
- Aplica configuración de voz, velocidad y pitch
- Opcionalmente agrega música de fondo

### 5. Generación de Subtítulos
- **SubtitleGenerator** crea subtítulos sincronizados
- Divide texto en líneas apropiadas
- Aplica estilos y animaciones

### 6. Composición Final
- **VideoCompositor** combina todos los elementos:
  - Secuencia de imágenes → video
  - Agrega audio
  - Superpone subtítulos
  - Exporta video final

## 📦 Componentes Principales

### Core Models (`core/models.py`)
- **VideoScript**: Modelo del script de entrada
- **VideoGenerationRequest**: Request completo
- **VideoGenerationResponse**: Respuesta con estado
- **VideoConfig**: Configuración de video
- **AudioConfig**: Configuración de audio
- **SubtitleConfig**: Configuración de subtítulos

### Services

#### ScriptProcessor
- Divide texto en segmentos
- Calcula timing
- Extrae keywords

#### VideoGenerator
- Genera imágenes con IA
- Crea secuencia de imágenes
- Maneja estilos y resoluciones

#### AudioGenerator
- Sintetiza voz (TTS)
- Agrega música de fondo
- Ajusta velocidad y pitch

#### SubtitleGenerator
- Genera subtítulos sincronizados
- Exporta formatos (SRT, VTT)
- Aplica estilos

#### VideoCompositor
- Compone video final
- Integra imágenes, audio y subtítulos
- Usa FFmpeg para procesamiento

#### VideoOrchestrator
- Coordina todo el pipeline
- Maneja estados y progreso
- Gestiona errores

## 🔌 Integraciones Externas

### Generación de Imágenes
- **OpenAI DALL-E**: `openai.images.generate()`
- **Stability AI**: API de Stable Diffusion
- **Hugging Face**: Modelos locales con Diffusers

### Text-to-Speech
- **Google TTS (gTTS)**: Gratis, sin API key
- **ElevenLabs**: Alta calidad, requiere API key
- **OpenAI TTS**: API de OpenAI
- **Azure Cognitive Services**: TTS de Microsoft

### Procesamiento de Video
- **FFmpeg**: Herramienta principal para composición
- Procesamiento de imágenes con Pillow
- Manipulación de audio con pydub (opcional)

## 🗄️ Almacenamiento

### Estructura de Directorios
```
/tmp/faceless_video/
├── images/          # Imágenes generadas
├── audio/           # Archivos de audio
├── subtitles/       # Archivos de subtítulos
└── output/          # Videos finales
```

### Gestión de Jobs
- Actualmente en memoria (dict)
- En producción: usar base de datos (PostgreSQL, MongoDB)
- Cache de imágenes para optimización

## 🔒 Seguridad y Rendimiento

### Seguridad
- Validación de inputs con Pydantic
- Sanitización de paths de archivos
- Límites de tamaño de archivos
- Timeouts para operaciones largas

### Rendimiento
- Procesamiento asíncrono
- Generación paralela de imágenes
- Cache de imágenes generadas
- Limpieza automática de archivos temporales

## 🚀 Escalabilidad

### Mejoras Futuras
1. **Queue System**: Redis/RabbitMQ para jobs
2. **Worker Pool**: Múltiples workers procesando en paralelo
3. **CDN**: Almacenamiento de videos en S3/CloudFront
4. **Database**: PostgreSQL para persistencia
5. **Caching**: Redis para cache de resultados
6. **Monitoring**: Prometheus + Grafana
7. **Load Balancing**: Múltiples instancias de API

## 📊 Monitoreo

### Métricas Clave
- Tiempo de generación por video
- Tasa de éxito/fallo
- Uso de recursos (CPU, memoria)
- Tamaño de archivos generados
- Latencia de API

### Logging
- Structured logging con niveles apropiados
- Tracking de jobs con IDs únicos
- Errores detallados para debugging

## 🔧 Configuración

### Variables de Entorno
- API keys para servicios externos
- Directorios de salida
- Límites y timeouts
- Configuraciones por defecto

Ver `config/settings.py` para detalles completos.

## 📝 Notas de Implementación

### Placeholders Actuales
- Generación de imágenes: Crea imágenes placeholder (gradientes)
- TTS: Crea archivos de audio placeholder
- **TODO**: Integrar servicios reales de IA

### Dependencias Críticas
- **FFmpeg**: Requerido para composición de video
- **Pillow**: Para procesamiento de imágenes
- **FastAPI**: Framework web
- **Pydantic**: Validación de datos

## 🎯 Próximos Pasos

1. Integrar servicios reales de IA
2. Implementar sistema de colas
3. Agregar base de datos
4. Implementar autenticación
5. Agregar tests unitarios e integración
6. Optimizar rendimiento
7. Agregar más estilos y opciones

