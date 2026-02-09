# 🏗️ Arquitectura - Social Media Identity Clone AI

## Visión General

Sistema modular para clonar identidades de perfiles de redes sociales y generar contenido auténtico basado en esa identidad.

## Componentes Principales

### 1. **Core Models** (`core/models.py`)
- `SocialProfile`: Perfil completo de una red social
- `IdentityProfile`: Perfil de identidad consolidado
- `ContentAnalysis`: Análisis de contenido extraído
- `GeneratedContent`: Contenido generado
- `VideoContent`, `PostContent`, `CommentContent`: Tipos de contenido

### 2. **Services** (`services/`)

#### `ProfileExtractor`
- Extrae perfiles completos de TikTok, Instagram, YouTube
- Consolida videos, posts y comentarios
- Maneja límites y paginación

#### `IdentityAnalyzer`
- Analiza contenido consolidado
- Construye perfil de identidad
- Crea base de conocimiento
- Usa IA para análisis profundo

#### `ContentGenerator`
- Genera contenido basado en identidad
- Soporta múltiples plataformas
- Mantiene coherencia con identidad original
- Usa prompts contextualizados

#### `VideoProcessor`
- Procesa videos y extrae transcripciones
- Integra con Whisper para transcripción
- Maneja diferentes formatos de video

### 3. **Connectors** (`connectors/`)

#### `TikTokConnector`
- Conecta con API de TikTok
- Extrae perfiles y videos
- Obtiene transcripciones

#### `InstagramConnector`
- Conecta con Instagram Graph API
- Extrae posts y comentarios
- Maneja autenticación OAuth2

#### `YouTubeConnector`
- Conecta con YouTube Data API v3
- Extrae canales y videos
- Obtiene transcripciones/captions

### 4. **Utils** (`utils/`)

#### `TextProcessor`
- Análisis básico de texto
- Extracción de hashtags y menciones
- Análisis de sentimiento básico

#### `VideoTranscriber`
- Transcripción de videos
- Soporte para múltiples servicios
- Manejo de diferentes formatos

### 5. **API** (`api/`)

#### `main.py`
- FastAPI application
- Configuración de CORS
- Health checks

#### `routes.py`
- Endpoints REST
- `/extract-profile`: Extraer perfil
- `/build-identity`: Construir identidad
- `/generate-content`: Generar contenido
- `/identity/{id}`: Obtener identidad

## Flujo de Datos

```
1. Extracción
   └─> ProfileExtractor
       ├─> TikTokConnector
       ├─> InstagramConnector
       └─> YouTubeConnector

2. Análisis
   └─> IdentityAnalyzer
       ├─> Consolida contenido
       ├─> Analiza con IA (OpenAI)
       └─> Construye knowledge base

3. Generación
   └─> ContentGenerator
       ├─> Usa IdentityProfile
       ├─> Genera con IA (OpenAI)
       └─> Mantiene coherencia
```

## Configuración

### Variables de Entorno

- **API Keys**: OpenAI, TikTok, Instagram, YouTube
- **Modelos**: LLM, Embedding, Transcription
- **Límites**: Videos, posts, comentarios por perfil
- **Generación**: Temperature, max length
- **Servidor**: Host, port, debug

### Almacenamiento

- **Base de Datos**: SQLite por defecto (configurable)
- **Storage**: Archivos locales (configurable)
- **Cache**: Redis (opcional)

## Seguridad

- ✅ Autenticación requerida para APIs
- ✅ Rate limiting
- ✅ Validación de inputs
- ✅ Manejo seguro de datos personales
- ✅ Respeto a términos de servicio

## Escalabilidad

- ✅ Arquitectura modular
- ✅ Servicios independientes
- ✅ Soporte para Docker
- ✅ Preparado para Kubernetes
- ✅ Cache con Redis

## Próximas Mejoras

- [ ] Implementación real de conectores de APIs
- [ ] Base de datos para almacenamiento persistente
- [ ] Sistema de caché avanzado
- [ ] Análisis de imágenes y visuales
- [ ] Generación de imágenes con estilo
- [ ] Dashboard web
- [ ] Webhooks para notificaciones
- [ ] Scheduler de contenido




