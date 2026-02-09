# Music Analyzer AI - Sistema de Análisis Musical y Coaching

## 🚀 Inicio Rápido - Un Solo Comando

**Iniciar todo el sistema con Docker:**

```bash
# Windows
deployment\start.bat

# Linux/Mac
./deployment/start.sh

# Python (Multiplataforma)
python deployment/start.py

# Make (Linux/Mac)
make -C deployment start
```

**Detener todo:**

```bash
# Windows
deployment\stop.bat

# Linux/Mac
./deployment/stop.sh

# Make
make -C deployment stop
```

Ver `START.md` o `deployment/README_START.md` para más detalles.

---

## 🎵 Descripción

Sistema avanzado de IA para análisis musical que se conecta a Spotify y proporciona análisis detallado de canciones, incluyendo notas, tonalidad, tempo, estructura y coaching musical personalizado.

## ✨ Características Principales

### Análisis Musical
- **Análisis de Tonalidad**: Identifica la nota raíz, modo (mayor/menor) y escala
- **Análisis de Tempo**: Detecta BPM y categoriza el tempo
- **Análisis de Estructura**: Identifica secciones (intro, verse, chorus, bridge, outro)
- **Análisis Armónico**: Analiza progresiones de acordes y cambios de tonalidad
- **Análisis Técnico**: Energía, bailabilidad, valencia, acústica, etc.

### Coaching Musical
- **Rutas de Aprendizaje**: Guías paso a paso para aprender canciones
- **Ejercicios de Práctica**: Ejercicios personalizados basados en la canción
- **Insights Educativos**: Explicaciones sobre notas, escalas y acordes
- **Recomendaciones**: Sugerencias adaptadas al nivel de habilidad
- **Tips de Interpretación**: Consejos para mejorar la ejecución

### Integración con Spotify
- Búsqueda de canciones
- Obtención de características de audio
- Análisis de audio detallado
- Información completa de tracks
- **Recomendaciones de canciones similares** (NUEVO)
- **Análisis comparativo entre múltiples canciones** (NUEVO)

### Mejoras de Rendimiento
- **Sistema de Cache**: Cache inteligente para mejorar rendimiento
- **Manejo de Errores**: Excepciones personalizadas y manejo robusto
- **Validaciones**: Validación de inputs y sanitización de queries
- **Rate Limiting**: Protección contra abuso de API

## 📁 Estructura del Proyecto

```
music_analyzer_ai/
├── api/                    # Endpoints de API
│   ├── __init__.py
│   └── music_api.py
├── core/                   # Lógica principal
│   ├── __init__.py
│   └── music_analyzer.py
├── services/               # Servicios
│   ├── __init__.py
│   ├── spotify_service.py
│   └── music_coach.py
├── models/                 # Modelos de datos
│   ├── __init__.py
│   └── schemas.py
├── utils/                  # Utilidades
├── config/                 # Configuración
│   ├── __init__.py
│   └── settings.py
├── tests/                  # Tests
├── main.py                 # Servidor principal
├── requirements.txt
└── README.md
```

## 🔧 Instalación

### Prerrequisitos

- Python 3.8+
- pip
- Cuenta de Spotify Developer (para obtener Client ID y Client Secret)

### Configuración de Spotify

1. Ve a [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
2. Crea una nueva aplicación
3. Obtén tu **Client ID** y **Client Secret**
4. Configura las credenciales en las variables de entorno

### Instalación

```bash
cd music_analyzer_ai
pip install -r requirements.txt
```

### Variables de Entorno

Cree un archivo `.env` en la raíz del proyecto:

```env
# Spotify API
SPOTIFY_CLIENT_ID=tu_client_id_aqui
SPOTIFY_CLIENT_SECRET=tu_client_secret_aqui
SPOTIFY_REDIRECT_URI=http://localhost:8010/callback

# Server
HOST=0.0.0.0
PORT=8010

# Logging
LOG_LEVEL=INFO
```

## 🚀 Uso

### Iniciar Servidor

```bash
python main.py
```

El servidor estará disponible en `http://localhost:8010`

### Documentación API

Una vez iniciado el servidor, acceda a:
- **Swagger UI**: `http://localhost:8010/docs`
- **ReDoc**: `http://localhost:8010/redoc`

## 📖 Endpoints Principales

### 1. Buscar Canciones

```bash
POST /music/search
Content-Type: application/json

{
  "query": "Bohemian Rhapsody Queen",
  "limit": 5
}
```

**Respuesta:**
```json
{
  "success": true,
  "query": "Bohemian Rhapsody Queen",
  "results": [
    {
      "id": "4uLU6hMCjMI75M1A2tKUQC",
      "name": "Bohemian Rhapsody",
      "artists": ["Queen"],
      "album": "A Night At The Opera",
      "duration_ms": 355000,
      "preview_url": "...",
      "popularity": 85
    }
  ],
  "total": 1
}
```

### 2. Analizar Canción

```bash
POST /music/analyze
Content-Type: application/json

{
  "track_id": "4uLU6hMCjMI75M1A2tKUQC",
  "include_coaching": true
}
```

O buscar por nombre:

```bash
POST /music/analyze
Content-Type: application/json

{
  "track_name": "Bohemian Rhapsody",
  "include_coaching": true
}
```

**Respuesta:**
```json
{
  "success": true,
  "track_basic_info": {
    "name": "Bohemian Rhapsody",
    "artists": ["Queen"],
    "album": "A Night At The Opera",
    "duration_seconds": 355.0
  },
  "musical_analysis": {
    "key_signature": "Bb Major",
    "root_note": "Bb",
    "mode": "Major",
    "tempo": {
      "bpm": 72.0,
      "category": "Lento (Adagio)"
    },
    "time_signature": "4/4",
    "scale": {
      "name": "Bb major",
      "notes": ["Bb", "C", "D", "Eb", "F", "G", "A"]
    }
  },
  "technical_analysis": {
    "energy": {
      "value": 0.456,
      "description": "Energía moderada"
    },
    "danceability": {
      "value": 0.321,
      "description": "Poco bailable"
    }
  },
  "composition_analysis": {
    "structure": [...],
    "complexity": {
      "level": "Compleja",
      "score": 0.75
    }
  },
  "educational_insights": {
    "key_analysis": {
      "note": "Bb",
      "mode": "Major",
      "scale_notes": ["Bb", "C", "D", "Eb", "F", "G", "A"],
      "common_chords": ["Bbmaj", "Cm", "Dm", "Ebmaj", "Fmaj", "Gm", "Adim"]
    },
    "learning_points": [
      "La canción está en Bb Major",
      "Tempo: 72 BPM (Lento (Adagio))",
      "Compás: 4/4"
    ],
    "practice_suggestions": [
      "Practica la escala de Bb major: Bb, C, D, Eb, F, G, A",
      "Practica con metrónomo a 54 BPM primero"
    ]
  },
  "coaching": {
    "overview": {
      "summary": "Análisis de 'Bohemian Rhapsody'",
      "difficulty_level": "Advanced",
      "suitable_for": ["Avanzado", "Intermedio-Avanzado"]
    },
    "learning_path": [
      {
        "step": 1,
        "title": "Familiarización",
        "description": "Escucha la canción varias veces...",
        "duration": "15-30 minutos"
      }
    ],
    "practice_exercises": [...],
    "performance_tips": [...],
    "recommendations": {...}
  }
}
```

### 3. Obtener Coaching

```bash
POST /music/coaching
Content-Type: application/json

{
  "track_name": "Bohemian Rhapsody"
}
```

### 4. Obtener Información de Track

```bash
GET /music/track/{track_id}/info
```

### 5. Obtener Características de Audio

```bash
GET /music/track/{track_id}/audio-features
```

### 6. Obtener Análisis de Audio

```bash
GET /music/track/{track_id}/audio-analysis
```

### 7. Comparar Canciones (NUEVO)

```bash
POST /music/compare
Content-Type: application/json

{
  "track_ids": [
    "4uLU6hMCjMI75M1A2tKUQC",
    "3n3Ppam7vgaVa1LRUg9NSd"
  ]
}
```

**Respuesta:**
```json
{
  "success": true,
  "comparison": {
    "key_signatures": {
      "all_same": false,
      "keys": ["Bb Major", "C Major"],
      "most_common": "Bb Major"
    },
    "tempos": {
      "average": 75.5,
      "min": 72.0,
      "max": 80.0,
      "range": 8.0
    }
  },
  "similarities": [...],
  "differences": [...],
  "recommendations": [...]
}
```

### 8. Obtener Recomendaciones (NUEVO)

```bash
GET /music/track/{track_id}/recommendations?limit=20
```

**Respuesta:**
```json
{
  "success": true,
  "seed_track_id": "4uLU6hMCjMI75M1A2tKUQC",
  "recommendations": [
    {
      "id": "...",
      "name": "Canción Similar",
      "artists": ["Artista"],
      "popularity": 85
    }
  ],
  "total": 20
}
```

### 9. Estadísticas de Cache (NUEVO)

```bash
GET /music/cache/stats
```

### 10. Limpiar Cache (NUEVO)

```bash
DELETE /music/cache/clear?prefix=spotify
```

### 11. Exportar Análisis (NUEVO)

```bash
POST /music/export/{track_id}?format=json&include_coaching=true
```

Formatos disponibles: `json`, `text`, `markdown`

### 12. Historial de Análisis (NUEVO)

```bash
# Obtener historial
GET /music/history?user_id=user123&limit=50

# Estadísticas del historial
GET /music/history/stats?user_id=user123

# Eliminar entrada
DELETE /music/history/{analysis_id}?user_id=user123
```

### 13. Analytics y Métricas (NUEVO)

```bash
# Obtener estadísticas del sistema
GET /music/analytics

# Resetear analytics
POST /music/analytics/reset
```

### 14. Favoritos (NUEVO)

```bash
# Agregar a favoritos
POST /music/favorites?user_id=user123&track_id=xxx&track_name=Song&artists=Artist

# Obtener favoritos
GET /music/favorites?user_id=user123

# Eliminar de favoritos
DELETE /music/favorites/{track_id}?user_id=user123

# Estadísticas de favoritos
GET /music/favorites/stats?user_id=user123
```

### 15. Tags/Etiquetas (NUEVO)

```bash
# Agregar tags
POST /music/tags?resource_id=xxx&resource_type=track&tags=rock,energetic

# Obtener tags
GET /music/tags/{resource_id}?resource_type=track

# Eliminar tags
DELETE /music/tags?resource_id=xxx&resource_type=track&tags=rock

# Buscar por tags
GET /music/tags/search?tags=rock,energetic

# Tags populares
GET /music/tags/popular?limit=20
```

### 16. Webhooks (NUEVO)

```bash
# Registrar webhook
POST /music/webhooks?url=https://example.com/webhook&events=analysis.completed,coaching.generated

# Listar webhooks
GET /music/webhooks?user_id=user123

# Eliminar webhook
DELETE /music/webhooks/{webhook_id}
```

Eventos disponibles: `analysis.completed`, `analysis.failed`, `coaching.generated`, `comparison.completed`, `export.completed`

### 17. Autenticación (NUEVO)

```bash
# Registrar usuario
POST /music/auth/register?username=user&email=user@example.com&password=pass123

# Login
POST /music/auth/login?username=user&password=pass123

# Obtener usuario actual
GET /music/auth/me
Authorization: Bearer {token}
```

### 18. Playlists (NUEVO)

```bash
# Crear playlist
POST /music/playlists?user_id=user123&name=My Playlist&is_public=false

# Obtener playlists
GET /music/playlists?user_id=user123
GET /music/playlists?public_only=true

# Obtener playlist específica
GET /music/playlists/{playlist_id}

# Agregar canción a playlist
POST /music/playlists/{playlist_id}/tracks?track_id=xxx&track_name=Song&artists=Artist

# Eliminar canción de playlist
DELETE /music/playlists/{playlist_id}/tracks/{track_id}

# Eliminar playlist
DELETE /music/playlists/{playlist_id}?user_id=user123
```

### 19. Recomendaciones Inteligentes (NUEVO)

### 20. Machine Learning - Predicciones (NUEVO)

```bash
# Predecir género
POST /music/ml/predict/genre?track_id=xxx

# Predecir emoción
POST /music/ml/predict/emotion?track_id=xxx

# Similitud en batch
POST /music/ml/similarity/batch
Content-Type: application/json

{
  "track_ids": ["xxx", "yyy", "zzz"],
  "target_track_id": "target_xxx"
}

# Agrupar por género
POST /music/ml/cluster/genres
Content-Type: application/json

{
  "track_ids": ["xxx", "yyy", "zzz"]
}

# Agrupar por emoción
POST /music/ml/cluster/emotions
Content-Type: application/json

{
  "track_ids": ["xxx", "yyy", "zzz"]
}

# Análisis ML avanzado
POST /music/ml/analyze/advanced?track_id=xxx&include_predictions=true&include_clustering=true

# Análisis Comprehensivo ML (NUEVO v2.1.0)
POST /music/ml/analyze-comprehensive?track_id=xxx

# Comparar Tracks ML (NUEVO v2.1.0)
POST /music/ml/compare-tracks?comparison_type=all
Content-Type: application/json

{
  "track_ids": ["xxx", "yyy", "zzz"]
}

# Predicción Multi-Tarea (NUEVO v2.1.0)
POST /music/ml/predict/multi-task?track_id=xxx

# Información del Pipeline ML (NUEVO v2.1.0)
GET /music/ml/pipeline/info

# Análisis de Estilo Musical (NUEVO v2.1.0)
POST /music/ml/analyze/style?track_id=xxx

# Predicción de Era Musical (NUEVO v2.1.0)
POST /music/ml/predict/era?track_id=xxx

# Análisis de Influencias (NUEVO v2.1.0)
POST /music/ml/analyze/influences?track_id=xxx

# Calcular Diversidad Musical (NUEVO v2.1.0)
POST /music/ml/diversity/calculate
Content-Type: application/json

{
  "track_ids": ["xxx", "yyy", "zzz", ...]
}
```

### 21. Dashboard de Métricas (NUEVO)

```bash
# Dashboard completo
GET /music/dashboard?user_id=user123
```

### 22. Notificaciones (NUEVO)

```bash
# Obtener notificaciones
GET /music/notifications?user_id=user123&unread_only=false&limit=50

# Marcar como leída
PUT /music/notifications/{notification_id}/read?user_id=user123

# Marcar todas como leídas
PUT /music/notifications/read-all?user_id=user123

# Eliminar notificación
DELETE /music/notifications/{notification_id}?user_id=user123

# Estadísticas de notificaciones
GET /music/notifications/stats?user_id=user123
```

```bash
# Recomendaciones por similitud, mood o género
POST /music/recommendations/intelligent?track_id=xxx&limit=10&method=similarity

# Métodos disponibles: similarity, mood, genre

# Generar playlist recomendada
POST /music/recommendations/playlist?playlist_length=20
Content-Type: application/json

{
  "genres": ["Rock", "Pop"],
  "moods": ["energetic", "happy"],
  "energy_range": [0.6, 1.0],
  "tempo_range": [100, 160],
  "seed_track_id": "xxx"
}
```

## 💻 Uso Programático

### Ejemplo Básico

```python
from services.spotify_service import SpotifyService
from core.music_analyzer import MusicAnalyzer
from services.music_coach import MusicCoach

# Inicializar servicios
spotify = SpotifyService()
analyzer = MusicAnalyzer()
coach = MusicCoach()

# Buscar canción
tracks = spotify.search_track("Bohemian Rhapsody", limit=1)
track_id = tracks[0]["id"]

# Obtener datos de Spotify
spotify_data = spotify.get_track_full_analysis(track_id)

# Analizar
analysis = analyzer.analyze_track(spotify_data)

# Obtener coaching
coaching = coach.generate_coaching_analysis(analysis)

print(f"Tonalidad: {analysis['musical_analysis']['key_signature']}")
print(f"Tempo: {analysis['musical_analysis']['tempo']['bpm']} BPM")
print(f"Dificultad: {coaching['overview']['difficulty_level']}")
```

## 🎯 Análisis Proporcionados

### Análisis Musical
- **Tonalidad**: Nota raíz y modo (mayor/menor)
- **Escala**: Notas de la escala principal
- **Tempo**: BPM y categorización
- **Compás**: Time signature
- **Cambios de Tonalidad**: Modulaciones en la canción
- **Cambios de Tempo**: Variaciones de velocidad
- **Género Musical**: Detección automática de género (Rock, Pop, Electronic, Hip-Hop, Jazz, Classical, etc.)
- **Análisis Armónico Avanzado**: Progresiones de acordes, cadencias (perfecta, plagal, media), patrones comunes
- **Análisis de Emociones**: Detección de emociones primarias y perfil emocional (8 emociones: happy, sad, energetic, calm, angry, romantic, nostalgic, mysterious)

### Análisis Técnico
- **Energía**: Nivel de intensidad
- **Bailabilidad**: Qué tan bailable es
- **Valencia**: Positividad/negatividad
- **Acústica**: Acústico vs electrónico
- **Instrumental**: Instrumental vs con voces
- **En Vivo**: En vivo vs estudio
- **Volumen**: Nivel de loudness

### Análisis de Composición
- **Estructura**: Secciones identificadas
- **Progresiones Armónicas**: Análisis de acordes
- **Complejidad**: Nivel de complejidad musical
- **Estilo**: Estilo de composición

### Coaching
- **Ruta de Aprendizaje**: Pasos para aprender la canción
- **Ejercicios de Práctica**: Ejercicios personalizados
- **Tips de Interpretación**: Consejos de ejecución
- **Recomendaciones**: Sugerencias por nivel

## 🔬 Notas Musicales y Escalas

El sistema identifica:
- **12 notas**: C, C#, D, D#, E, F, F#, G, G#, A, A#, B
- **Modos**: Major (Mayor) y Minor (Menor)
- **Escalas**: Major, Minor, Pentatonic, Blues, Dorian, Mixolydian
- **Acordes**: Acordes triada en la tonalidad

## 🎓 Niveles de Dificultad

- **Beginner**: Canciones simples, adecuadas para principiantes
- **Intermediate**: Canciones de dificultad moderada
- **Advanced**: Canciones complejas que requieren habilidades avanzadas

## 🔧 Configuración Avanzada

### Análisis Detallado

El sistema puede proporcionar diferentes niveles de detalle:
- **Basic**: Información esencial
- **Detailed**: Análisis completo (por defecto)
- **Expert**: Análisis muy detallado con todos los datos

## 🧪 Testing

```bash
# Ejecutar tests
pytest tests/

# Con cobertura
pytest --cov=music_analyzer_ai tests/
```

## 🆕 Mejoras Implementadas (v2.21.0)

### Meta-Learning y Análisis Conceptual
- ✅ **Meta-Learning**: Adaptación rápida del modelo a nuevas tareas (MAML-like)
- ✅ **Few-Shot Learning**: Aprendizaje efectivo con solo 2+ ejemplos
- ✅ **Análisis de Causalidad**: Relaciones causales entre características musicales
- ✅ **Explicabilidad Avanzada**: Múltiples métodos (Gradient, SHAP, LIME)
- ✅ **Análisis de Conceptos**: Detección de conceptos musicales en tracks
- ✅ **Adaptación Rápida**: Aprendizaje de nuevas tareas con pocos ejemplos
- ✅ **5 Nuevos Endpoints**: API completa para meta-learning y análisis conceptual

## 🆕 Mejoras Implementadas (v2.20.0)

### Calibración y Análisis Avanzado
- ✅ **Calibración de Modelos**: Calibración de probabilidades usando Isotonic Regression
- ✅ **Análisis de Incertidumbre**: Monte Carlo Dropout para medir incertidumbre
- ✅ **Active Learning**: Selección inteligente de muestras para etiquetar
- ✅ **Transfer Learning**: Análisis de transferibilidad entre dominios
- ✅ **Detección Adversarial**: Identificación de vulnerabilidades a adversarial examples
- ✅ **Múltiples Estrategias**: Uncertainty y diversity sampling para active learning
- ✅ **5 Nuevos Endpoints**: API completa para calibración y análisis avanzado

## 🆕 Mejoras Implementadas (v2.19.0)

### Optimización y Análisis Avanzado
- ✅ **Análisis de Confianza**: Evaluación de confianza en predicciones de género y emoción
- ✅ **Detección de Outliers**: Identificación de tracks anómalos con Z-score e Isolation
- ✅ **Sistema de Ensemble**: Combinación de hasta 5 modelos para mayor precisión
- ✅ **Batch Processing Avanzado**: Procesamiento optimizado con caching inteligente
- ✅ **Gestión de Cache**: Cache de embeddings y modelos con limpieza automática
- ✅ **Promedio Ponderado**: Agregación de predicciones con pesos configurables
- ✅ **6 Nuevos Endpoints**: API completa para optimización y análisis avanzado

## 🆕 Mejoras Implementadas (v2.18.0)

### Monitoreo y Producción Avanzada
- ✅ **Monitoreo en Producción**: Tracking automático de predicciones y métricas
- ✅ **Detección de Drift**: Identificación de cambios en distribución de datos
- ✅ **Detección de Degradación**: Verificación de degradación del modelo
- ✅ **Auto-Retraining**: Re-entrenamiento automático con validación de mejora
- ✅ **Métricas de Performance**: Latencia, throughput, P95, P99
- ✅ **Tracking de Errores**: Análisis de errores recientes
- ✅ **4 Nuevos Endpoints**: API completa para monitoreo y producción

## 🆕 Mejoras Implementadas (v2.17.0)

### Interpretabilidad y Producción
- ✅ **Interpretabilidad de Modelos**: Explicación de predicciones usando gradientes
- ✅ **A/B Testing**: Comparación de modelos en el mismo conjunto de prueba
- ✅ **Análisis de Robustez**: Evaluación de estabilidad ante perturbaciones
- ✅ **Versionado de Modelos**: Sistema completo de versionado con metadata
- ✅ **Feature Importance**: Identificación de características más importantes
- ✅ **Análisis de Estabilidad**: Medición de estabilidad de predicciones
- ✅ **5 Nuevos Endpoints**: API completa para interpretabilidad y producción

## 🆕 Mejoras Implementadas (v2.16.0)

### Fine-Tuning y Análisis Ético
- ✅ **Análisis de Tendencias Temporales**: Tracking de evolución de embeddings en el tiempo
- ✅ **Análisis de Bias y Fairness**: Evaluación de equidad en predicciones entre géneros
- ✅ **Generación de Reportes**: Reportes automáticos de entrenamiento con recomendaciones
- ✅ **Fine-Tuning Avanzado**: Ajuste fino de modelos pre-entrenados con opción de congelar encoder
- ✅ **Transfer Learning**: Aprovechamiento de modelos pre-entrenados
- ✅ **Detección de Overfitting**: Identificación automática de problemas de entrenamiento
- ✅ **4 Nuevos Endpoints**: API completa para fine-tuning y análisis ético

## 🆕 Mejoras Implementadas (v2.15.0)

### Análisis y Comparación Avanzada
- ✅ **Clustering de Tracks**: Agrupación de tracks usando embeddings (K-Means, DBSCAN)
- ✅ **Análisis de Importancia**: Identificación de características más importantes
- ✅ **Comparación de Modelos**: Compara múltiples modelos en el mismo conjunto de prueba
- ✅ **Exportación Avanzada**: Exporta resultados completos de entrenamiento
- ✅ **Reducción de Dimensionalidad**: PCA opcional para clustering
- ✅ **Análisis Estadístico**: Correlación de Pearson para importancia de características
- ✅ **4 Nuevos Endpoints**: API completa para análisis avanzado

## 🆕 Mejoras Implementadas (v2.14.0)

### Experimentación y Recomendaciones Avanzadas
- ✅ **Tracking de Experimentos**: Integración con WandB y TensorBoard para tracking
- ✅ **Optimización de Hiperparámetros**: Búsqueda grid de mejores hiperparámetros
- ✅ **Recomendaciones con Embeddings**: Sistema de recomendaciones basado en embeddings
- ✅ **Búsqueda de Similitud**: Encuentra tracks similares usando embeddings
- ✅ **Diversidad en Recomendaciones**: Balance entre similitud y diversidad
- ✅ **Múltiples Métricas**: Soporte para cosine similarity y euclidean distance
- ✅ **4 Nuevos Endpoints**: API completa para experimentación y recomendaciones

## 🆕 Mejoras Implementadas (v2.13.0)

### Evaluación y Entrenamiento Avanzado
- ✅ **Evaluación Avanzada**: Métricas detalladas (Accuracy, Precision, Recall, F1, Confusion Matrix)
- ✅ **Entrenamiento con Validación**: Conjuntos separados de train/val con early stopping
- ✅ **Early Stopping**: Detección automática de overfitting
- ✅ **Análisis de Embeddings**: Extracción de representaciones vectoriales de tracks
- ✅ **Guardado de Historial**: Persistencia de métricas de entrenamiento
- ✅ **Métricas de Regresión**: MSE, MAE, RMSE, R² para predicción de popularidad
- ✅ **4 Nuevos Endpoints**: API completa para evaluación y entrenamiento avanzado

## 🆕 Mejoras Implementadas (v2.12.0)

### Deep Learning y Transformers
- ✅ **Modelos Transformer Personalizados**: Arquitectura Transformer para análisis musical avanzado
- ✅ **Clasificación Multi-Tarea**: Predicción simultánea de género, emoción y popularidad
- ✅ **Sistema de Entrenamiento**: API completa para entrenar modelos personalizados
- ✅ **Análisis de Letras con DistilBERT**: Análisis de sentimiento usando modelos pre-entrenados
- ✅ **Gestión de Modelos**: Guardar y cargar modelos entrenados
- ✅ **Predicción en Batch**: Procesamiento eficiente de múltiples tracks
- ✅ **Mixed Precision Training**: Optimización para GPU con precisión mixta
- ✅ **8 Nuevos Endpoints**: API completa para deep learning

## 🆕 Mejoras Implementadas (v2.11.0)

### Nuevas Funcionalidades Avanzadas
- ✅ **Análisis de Sentimiento Mejorado**: Análisis línea por línea de sentimiento en letras con progresión
- ✅ **Sistema de Reportes Avanzado**: Generación de reportes comprehensivos y comparativos con insights
- ✅ **Análisis de Audio en Tiempo Real**: Análisis de streams de audio con tracking de tendencias y alertas
- ✅ **Resumen Ejecutivo**: Resúmenes de alto nivel para toma de decisiones
- ✅ **Insights Automáticos**: Generación automática de insights del análisis
- ✅ **Historial en Tiempo Real**: Buffer de historial de análisis en tiempo real

## 🆕 Mejoras Implementadas (v2.9.0)

### Nuevas Funcionalidades Avanzadas
- ✅ **Análisis de Estructura Avanzado**: Mapeo automático de secciones, análisis de transiciones, build-ups y drops
- ✅ **Sistema de Recomendaciones Mejorado**: Recomendaciones con ML avanzado y análisis multi-factor
- ✅ **Predicción de Éxito Mejorada**: Predicción multi-factor con análisis de atractivo comercial y tendencias
- ✅ **Análisis de Complejidad Estructural**: Cálculo avanzado de complejidad estructural
- ✅ **Identificación de Patrones**: Identificación automática de patrones estructurales comunes
- ✅ **Factores de Similitud**: Desglose detallado de factores de similitud en recomendaciones

## 🆕 Mejoras Implementadas (v2.6.0)

### Nuevas Funcionalidades Avanzadas
- ✅ **Análisis de Instrumentación**: Analiza instrumentos acústicos, electrónicos, voces y sección rítmica
- ✅ **Análisis de Textura**: Evalúa densidad y complejidad de timbre musical
- ✅ **Predicción de Tendencias de Géneros**: Predice qué géneros estarán en tendencia
- ✅ **Predicción de Tendencias de Emociones**: Predice emociones musicales en tendencia
- ✅ **Predicción de Tendencias de Características**: Predice cambios en energía, bailabilidad, valencia y tempo
- ✅ **Predicción del Próximo Gran Éxito**: Identifica tracks con alto potencial de éxito comercial
- ✅ **Análisis de Complejidad de Instrumentación**: Calcula complejidad general de la instrumentación

## 🆕 Mejoras Implementadas (v2.5.0)

### Nuevas Funcionalidades Avanzadas
- ✅ **Sistema de Descubrimiento Musical**: Descubre artistas similares, tracks underground y tracks frescos
- ✅ **Descubrimiento por Transición de Mood**: Encuentra tracks que transicionan entre diferentes moods
- ✅ **Análisis Detallado de Covers**: Compara covers con originales con análisis de fidelidad
- ✅ **Análisis Detallado de Remixes**: Analiza transformaciones y modificaciones en remixes
- ✅ **Búsqueda de Versiones**: Encuentra automáticamente covers y remixes de cualquier track
- ✅ **Identificación Automática**: Identifica tipo de cover/remix automáticamente
- ✅ **Análisis de Transformación**: Evalúa nivel de transformación en remixes

## 🆕 Mejoras Implementadas (v2.4.0)

### Nuevas Funcionalidades Avanzadas
- ✅ **Análisis Inteligente de Playlists**: Análisis completo de diversidad, coherencia y flujo
- ✅ **Sugerencias de Mejora**: Recomendaciones automáticas para mejorar playlists
- ✅ **Optimización de Orden**: Reordena tracks para mejor experiencia de escucha
- ✅ **Comparación de Artistas**: Compara hasta 5 artistas con análisis detallado
- ✅ **Análisis de Evolución**: Analiza cómo evoluciona la música de un artista
- ✅ **Detección de Cambios**: Identifica cambios de género y estilo a lo largo del tiempo
- ✅ **Análisis de Flujo**: Evalúa la progresión de energía en playlists

## 🆕 Mejoras Implementadas (v2.3.0)

### Nuevas Funcionalidades Avanzadas
- ✅ **Análisis Temporal**: Análisis de estructura temporal, progresión de energía y cambios de tempo
- ✅ **Detección de Build-ups y Drops**: Identificación automática de aumentos y caídas de energía
- ✅ **Análisis de Calidad**: Evaluación completa de calidad de producción musical
- ✅ **Recomendaciones Contextuales**: Recomendaciones basadas en contexto personalizado
- ✅ **Recomendaciones por Hora del Día**: Adaptadas a morning, afternoon, evening, night
- ✅ **Recomendaciones por Actividad**: Optimizadas para workout, study, party, relax, drive
- ✅ **Recomendaciones por Mood**: Basadas en happy, sad, energetic, calm, romantic
- ✅ **Análisis de Complejidad Temporal**: Cálculo de complejidad basada en cambios temporales

## 🆕 Mejoras Implementadas (v2.2.0)

### Nuevas Funcionalidades Avanzadas
- ✅ **Análisis de Tendencias**: Análisis de popularidad y tendencias de tracks y artistas
- ✅ **Predicción de Éxito Comercial**: Predicción basada en características musicales
- ✅ **Análisis de Patrones Rítmicos**: Análisis avanzado de densidad y consistencia rítmica
- ✅ **Análisis de Colaboraciones**: Detección y análisis de colaboraciones entre artistas
- ✅ **Red de Colaboraciones**: Análisis de redes de colaboración entre múltiples artistas
- ✅ **Comparación de Versiones**: Compara originales, covers y remixes
- ✅ **Sistema de Alertas Inteligentes**: Alertas de popularidad, oportunidades y colaboraciones
- ✅ **Gestión de Alertas**: Sistema completo con filtros, prioridades y gestión

## 🆕 Mejoras Implementadas (v2.1.0)

### Nuevas Funcionalidades ML Avanzadas
- ✅ **Análisis Comprehensivo**: Análisis ML completo con todas las predicciones y comparaciones
- ✅ **Comparación de Tracks ML**: Compara múltiples tracks usando ML (hasta 20 tracks)
- ✅ **Predicción Multi-Tarea**: Predicción simultánea de género, emoción, complejidad y dificultad
- ✅ **Pipeline Info**: Información detallada del pipeline de ML
- ✅ **Análisis de Estilo**: Detección completa del estilo musical
- ✅ **Predicción de Era**: Predicción de la era musical basada en características
- ✅ **Análisis de Influencias**: Detección de posibles influencias musicales
- ✅ **Cálculo de Diversidad**: Análisis de diversidad musical en conjuntos de tracks

## 🆕 Mejoras Implementadas (v2.0.0)

### Nuevas Funcionalidades
- ✅ **Sistema de Cache**: Cache inteligente con TTL configurable
- ✅ **Manejo de Errores Mejorado**: Excepciones personalizadas y códigos HTTP apropiados
- ✅ **Validaciones Robustas**: Validación de track IDs y sanitización de queries
- ✅ **Análisis Comparativo**: Compara hasta 10 canciones simultáneamente
- ✅ **Recomendaciones de Spotify**: Obtiene canciones similares usando la API de Spotify
- ✅ **Detección de Género**: Detecta género musical basado en características de audio (12 géneros)
- ✅ **Exportación de Análisis**: Exporta análisis en JSON, texto plano y Markdown
- ✅ **Sistema de Historial**: Guarda y gestiona historial de análisis realizados
- ✅ **Analytics y Métricas**: Tracking completo de uso, tiempos de respuesta y estadísticas
- ✅ **Rate Limiting**: Protección contra abuso con límite de 100 requests/minuto por IP
- ✅ **Sistema de Favoritos**: Guarda y gestiona canciones favoritas por usuario
- ✅ **Sistema de Tags**: Etiquetado de tracks, análisis y playlists con búsqueda avanzada
- ✅ **Webhooks**: Sistema de notificaciones en tiempo real para eventos
- ✅ **Análisis Armónico Avanzado**: Detección de progresiones, cadencias y patrones armónicos
- ✅ **Análisis de Emociones**: Detección de 8 emociones musicales (happy, sad, energetic, calm, angry, romantic, nostalgic, mysterious)
- ✅ **Sistema de Autenticación**: Registro, login y gestión de usuarios con JWT
- ✅ **Playlists Personalizadas**: Crear, gestionar y compartir playlists
- ✅ **Recomendaciones Inteligentes**: Sistema ML para recomendaciones por similitud, mood y género
- ✅ **Generación de Playlists**: Genera playlists automáticas basadas en preferencias
- ✅ **Machine Learning Avanzado**: Predicciones de género/emoción, clustering, similitud en batch
- ✅ **Dashboard de Métricas**: Dashboard completo con métricas del sistema y usuario
- ✅ **Sistema de Notificaciones**: Notificaciones personalizadas por usuario con prioridades
- ✅ **Gestión de Cache**: Endpoints para ver estadísticas y limpiar cache
- ✅ **Timeout y Retry**: Manejo de timeouts en peticiones a Spotify
- ✅ **Logging Mejorado**: Sistema de logging más detallado

### Mejoras de Rendimiento
- Cache en memoria para reducir llamadas a Spotify API
- Validación temprana de inputs
- Manejo eficiente de errores
- Timeout configurable en peticiones
- Rate limiting para proteger el sistema
- Tracking de métricas en tiempo real

## 📊 Estadísticas del Sistema

- **Total de Endpoints**: 82+
- **Endpoints ML**: 12+
- **Endpoints de Tendencias**: 4
- **Endpoints de Colaboraciones**: 3
- **Endpoints de Alertas**: 4
- **Endpoints Temporales**: 3
- **Endpoints de Calidad**: 1
- **Endpoints de Recomendaciones Contextuales**: 4
- **Endpoints de Análisis de Playlist**: 3
- **Endpoints de Análisis de Artistas**: 2
- **Endpoints de Descubrimiento**: 4
- **Endpoints de Covers/Remixes**: 3
- **Servicios Especializados**: 29+
- **Tipos de Análisis**: 26+
- **Emociones Detectadas**: 8
- **Géneros Soportados**: 12
- **Formatos de Exportación**: 3
- **Capacidades ML**: 11 tipos diferentes
- **Tipos de Alertas**: 4
- **Contextos de Recomendación**: 4 tipos
- **Métricas de Playlist**: 5 tipos
- **Tipos de Descubrimiento**: 4
- **Sistemas Integrados**: Autenticación, Playlists, Favoritos, Tags, Webhooks, Analytics, ML Avanzado, Dashboard, Notificaciones, Tendencias, Colaboraciones, Alertas, Análisis Temporal, Calidad, Recomendaciones Contextuales, Análisis de Playlist, Comparación de Artistas, Descubrimiento Musical, Análisis de Covers/Remixes

## 📊 Mejoras Futuras

- [ ] Análisis de audio local (sin Spotify)
- [ ] Detección de instrumentos
- [ ] Análisis de letras
- [ ] Comparación entre versiones
- [ ] Generación de partituras
- [ ] Integración con más servicios de música (YouTube, SoundCloud)
- [ ] Análisis de progresión temporal
- [ ] Sistema de seguimiento de progreso
- [ ] Dashboard web interactivo
- [ ] Machine Learning avanzado para recomendaciones
- [ ] Análisis de colaboraciones entre artistas
- [ ] Sistema de notificaciones push

## 🤝 Contribuir

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Cree una rama para su feature (`git checkout -b feature/AmazingFeature`)
3. Commit sus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abra un Pull Request

## 📝 Licencia

Este proyecto es parte de Blatam Academy.

## 📞 Soporte

Para soporte, contacte al equipo de Blatam Academy.

## 🔗 Enlaces Útiles

- [Spotify Web API Documentation](https://developer.spotify.com/documentation/web-api)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)

---

**Desarrollado con ❤️ por Blatam Academy**


## 🎵 Descripción

Sistema avanzado de IA para análisis musical que se conecta a Spotify y proporciona análisis detallado de canciones, incluyendo notas, tonalidad, tempo, estructura y coaching musical personalizado.

## ✨ Características Principales

### Análisis Musical
- **Análisis de Tonalidad**: Identifica la nota raíz, modo (mayor/menor) y escala
- **Análisis de Tempo**: Detecta BPM y categoriza el tempo
- **Análisis de Estructura**: Identifica secciones (intro, verse, chorus, bridge, outro)
- **Análisis Armónico**: Analiza progresiones de acordes y cambios de tonalidad
- **Análisis Técnico**: Energía, bailabilidad, valencia, acústica, etc.

### Coaching Musical
- **Rutas de Aprendizaje**: Guías paso a paso para aprender canciones
- **Ejercicios de Práctica**: Ejercicios personalizados basados en la canción
- **Insights Educativos**: Explicaciones sobre notas, escalas y acordes
- **Recomendaciones**: Sugerencias adaptadas al nivel de habilidad
- **Tips de Interpretación**: Consejos para mejorar la ejecución

### Integración con Spotify
- Búsqueda de canciones
- Obtención de características de audio
- Análisis de audio detallado
- Información completa de tracks
- **Recomendaciones de canciones similares** (NUEVO)
- **Análisis comparativo entre múltiples canciones** (NUEVO)

### Mejoras de Rendimiento
- **Sistema de Cache**: Cache inteligente para mejorar rendimiento
- **Manejo de Errores**: Excepciones personalizadas y manejo robusto
- **Validaciones**: Validación de inputs y sanitización de queries
- **Rate Limiting**: Protección contra abuso de API

## 📁 Estructura del Proyecto

```
music_analyzer_ai/
├── api/                    # Endpoints de API
│   ├── __init__.py
│   └── music_api.py
├── core/                   # Lógica principal
│   ├── __init__.py
│   └── music_analyzer.py
├── services/               # Servicios
│   ├── __init__.py
│   ├── spotify_service.py
│   └── music_coach.py
├── models/                 # Modelos de datos
│   ├── __init__.py
│   └── schemas.py
├── utils/                  # Utilidades
├── config/                 # Configuración
│   ├── __init__.py
│   └── settings.py
├── tests/                  # Tests
├── main.py                 # Servidor principal
├── requirements.txt
└── README.md
```

## 🔧 Instalación

### Prerrequisitos

- Python 3.8+
- pip
- Cuenta de Spotify Developer (para obtener Client ID y Client Secret)

### Configuración de Spotify

1. Ve a [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
2. Crea una nueva aplicación
3. Obtén tu **Client ID** y **Client Secret**
4. Configura las credenciales en las variables de entorno

### Instalación

```bash
cd music_analyzer_ai
pip install -r requirements.txt
```

### Variables de Entorno

Cree un archivo `.env` en la raíz del proyecto:

```env
# Spotify API
SPOTIFY_CLIENT_ID=tu_client_id_aqui
SPOTIFY_CLIENT_SECRET=tu_client_secret_aqui
SPOTIFY_REDIRECT_URI=http://localhost:8010/callback

# Server
HOST=0.0.0.0
PORT=8010

# Logging
LOG_LEVEL=INFO
```

## 🚀 Uso

### Iniciar Servidor

```bash
python main.py
```

El servidor estará disponible en `http://localhost:8010`

### Documentación API

Una vez iniciado el servidor, acceda a:
- **Swagger UI**: `http://localhost:8010/docs`
- **ReDoc**: `http://localhost:8010/redoc`

## 📖 Endpoints Principales

### 1. Buscar Canciones

```bash
POST /music/search
Content-Type: application/json

{
  "query": "Bohemian Rhapsody Queen",
  "limit": 5
}
```

**Respuesta:**
```json
{
  "success": true,
  "query": "Bohemian Rhapsody Queen",
  "results": [
    {
      "id": "4uLU6hMCjMI75M1A2tKUQC",
      "name": "Bohemian Rhapsody",
      "artists": ["Queen"],
      "album": "A Night At The Opera",
      "duration_ms": 355000,
      "preview_url": "...",
      "popularity": 85
    }
  ],
  "total": 1
}
```

### 2. Analizar Canción

```bash
POST /music/analyze
Content-Type: application/json

{
  "track_id": "4uLU6hMCjMI75M1A2tKUQC",
  "include_coaching": true
}
```

O buscar por nombre:

```bash
POST /music/analyze
Content-Type: application/json

{
  "track_name": "Bohemian Rhapsody",
  "include_coaching": true
}
```

**Respuesta:**
```json
{
  "success": true,
  "track_basic_info": {
    "name": "Bohemian Rhapsody",
    "artists": ["Queen"],
    "album": "A Night At The Opera",
    "duration_seconds": 355.0
  },
  "musical_analysis": {
    "key_signature": "Bb Major",
    "root_note": "Bb",
    "mode": "Major",
    "tempo": {
      "bpm": 72.0,
      "category": "Lento (Adagio)"
    },
    "time_signature": "4/4",
    "scale": {
      "name": "Bb major",
      "notes": ["Bb", "C", "D", "Eb", "F", "G", "A"]
    }
  },
  "technical_analysis": {
    "energy": {
      "value": 0.456,
      "description": "Energía moderada"
    },
    "danceability": {
      "value": 0.321,
      "description": "Poco bailable"
    }
  },
  "composition_analysis": {
    "structure": [...],
    "complexity": {
      "level": "Compleja",
      "score": 0.75
    }
  },
  "educational_insights": {
    "key_analysis": {
      "note": "Bb",
      "mode": "Major",
      "scale_notes": ["Bb", "C", "D", "Eb", "F", "G", "A"],
      "common_chords": ["Bbmaj", "Cm", "Dm", "Ebmaj", "Fmaj", "Gm", "Adim"]
    },
    "learning_points": [
      "La canción está en Bb Major",
      "Tempo: 72 BPM (Lento (Adagio))",
      "Compás: 4/4"
    ],
    "practice_suggestions": [
      "Practica la escala de Bb major: Bb, C, D, Eb, F, G, A",
      "Practica con metrónomo a 54 BPM primero"
    ]
  },
  "coaching": {
    "overview": {
      "summary": "Análisis de 'Bohemian Rhapsody'",
      "difficulty_level": "Advanced",
      "suitable_for": ["Avanzado", "Intermedio-Avanzado"]
    },
    "learning_path": [
      {
        "step": 1,
        "title": "Familiarización",
        "description": "Escucha la canción varias veces...",
        "duration": "15-30 minutos"
      }
    ],
    "practice_exercises": [...],
    "performance_tips": [...],
    "recommendations": {...}
  }
}
```

### 3. Obtener Coaching

```bash
POST /music/coaching
Content-Type: application/json

{
  "track_name": "Bohemian Rhapsody"
}
```

### 4. Obtener Información de Track

```bash
GET /music/track/{track_id}/info
```

### 5. Obtener Características de Audio

```bash
GET /music/track/{track_id}/audio-features
```

### 6. Obtener Análisis de Audio

```bash
GET /music/track/{track_id}/audio-analysis
```

### 7. Comparar Canciones (NUEVO)

```bash
POST /music/compare
Content-Type: application/json

{
  "track_ids": [
    "4uLU6hMCjMI75M1A2tKUQC",
    "3n3Ppam7vgaVa1LRUg9NSd"
  ]
}
```

**Respuesta:**
```json
{
  "success": true,
  "comparison": {
    "key_signatures": {
      "all_same": false,
      "keys": ["Bb Major", "C Major"],
      "most_common": "Bb Major"
    },
    "tempos": {
      "average": 75.5,
      "min": 72.0,
      "max": 80.0,
      "range": 8.0
    }
  },
  "similarities": [...],
  "differences": [...],
  "recommendations": [...]
}
```

### 8. Obtener Recomendaciones (NUEVO)

```bash
GET /music/track/{track_id}/recommendations?limit=20
```

**Respuesta:**
```json
{
  "success": true,
  "seed_track_id": "4uLU6hMCjMI75M1A2tKUQC",
  "recommendations": [
    {
      "id": "...",
      "name": "Canción Similar",
      "artists": ["Artista"],
      "popularity": 85
    }
  ],
  "total": 20
}
```

### 9. Estadísticas de Cache (NUEVO)

```bash
GET /music/cache/stats
```

### 10. Limpiar Cache (NUEVO)

```bash
DELETE /music/cache/clear?prefix=spotify
```

### 11. Exportar Análisis (NUEVO)

```bash
POST /music/export/{track_id}?format=json&include_coaching=true
```

Formatos disponibles: `json`, `text`, `markdown`

### 12. Historial de Análisis (NUEVO)

```bash
# Obtener historial
GET /music/history?user_id=user123&limit=50

# Estadísticas del historial
GET /music/history/stats?user_id=user123

# Eliminar entrada
DELETE /music/history/{analysis_id}?user_id=user123
```

### 13. Analytics y Métricas (NUEVO)

```bash
# Obtener estadísticas del sistema
GET /music/analytics

# Resetear analytics
POST /music/analytics/reset
```

### 14. Favoritos (NUEVO)

```bash
# Agregar a favoritos
POST /music/favorites?user_id=user123&track_id=xxx&track_name=Song&artists=Artist

# Obtener favoritos
GET /music/favorites?user_id=user123

# Eliminar de favoritos
DELETE /music/favorites/{track_id}?user_id=user123

# Estadísticas de favoritos
GET /music/favorites/stats?user_id=user123
```

### 15. Tags/Etiquetas (NUEVO)

```bash
# Agregar tags
POST /music/tags?resource_id=xxx&resource_type=track&tags=rock,energetic

# Obtener tags
GET /music/tags/{resource_id}?resource_type=track

# Eliminar tags
DELETE /music/tags?resource_id=xxx&resource_type=track&tags=rock

# Buscar por tags
GET /music/tags/search?tags=rock,energetic

# Tags populares
GET /music/tags/popular?limit=20
```

### 16. Webhooks (NUEVO)

```bash
# Registrar webhook
POST /music/webhooks?url=https://example.com/webhook&events=analysis.completed,coaching.generated

# Listar webhooks
GET /music/webhooks?user_id=user123

# Eliminar webhook
DELETE /music/webhooks/{webhook_id}
```

Eventos disponibles: `analysis.completed`, `analysis.failed`, `coaching.generated`, `comparison.completed`, `export.completed`

### 17. Autenticación (NUEVO)

```bash
# Registrar usuario
POST /music/auth/register?username=user&email=user@example.com&password=pass123

# Login
POST /music/auth/login?username=user&password=pass123

# Obtener usuario actual
GET /music/auth/me
Authorization: Bearer {token}
```

### 18. Playlists (NUEVO)

```bash
# Crear playlist
POST /music/playlists?user_id=user123&name=My Playlist&is_public=false

# Obtener playlists
GET /music/playlists?user_id=user123
GET /music/playlists?public_only=true

# Obtener playlist específica
GET /music/playlists/{playlist_id}

# Agregar canción a playlist
POST /music/playlists/{playlist_id}/tracks?track_id=xxx&track_name=Song&artists=Artist

# Eliminar canción de playlist
DELETE /music/playlists/{playlist_id}/tracks/{track_id}

# Eliminar playlist
DELETE /music/playlists/{playlist_id}?user_id=user123
```

### 19. Recomendaciones Inteligentes (NUEVO)

### 20. Machine Learning - Predicciones (NUEVO)

```bash
# Predecir género
POST /music/ml/predict/genre?track_id=xxx

# Predecir emoción
POST /music/ml/predict/emotion?track_id=xxx

# Similitud en batch
POST /music/ml/similarity/batch
Content-Type: application/json

{
  "track_ids": ["xxx", "yyy", "zzz"],
  "target_track_id": "target_xxx"
}

# Agrupar por género
POST /music/ml/cluster/genres
Content-Type: application/json

{
  "track_ids": ["xxx", "yyy", "zzz"]
}

# Agrupar por emoción
POST /music/ml/cluster/emotions
Content-Type: application/json

{
  "track_ids": ["xxx", "yyy", "zzz"]
}

# Análisis ML avanzado
POST /music/ml/analyze/advanced?track_id=xxx&include_predictions=true&include_clustering=true

# Análisis Comprehensivo ML (NUEVO v2.1.0)
POST /music/ml/analyze-comprehensive?track_id=xxx

# Comparar Tracks ML (NUEVO v2.1.0)
POST /music/ml/compare-tracks?comparison_type=all
Content-Type: application/json

{
  "track_ids": ["xxx", "yyy", "zzz"]
}

# Predicción Multi-Tarea (NUEVO v2.1.0)
POST /music/ml/predict/multi-task?track_id=xxx

# Información del Pipeline ML (NUEVO v2.1.0)
GET /music/ml/pipeline/info

# Análisis de Estilo Musical (NUEVO v2.1.0)
POST /music/ml/analyze/style?track_id=xxx

# Predicción de Era Musical (NUEVO v2.1.0)
POST /music/ml/predict/era?track_id=xxx

# Análisis de Influencias (NUEVO v2.1.0)
POST /music/ml/analyze/influences?track_id=xxx

# Calcular Diversidad Musical (NUEVO v2.1.0)
POST /music/ml/diversity/calculate
Content-Type: application/json

{
  "track_ids": ["xxx", "yyy", "zzz", ...]
}
```

### 21. Dashboard de Métricas (NUEVO)

```bash
# Dashboard completo
GET /music/dashboard?user_id=user123
```

### 22. Notificaciones (NUEVO)

```bash
# Obtener notificaciones
GET /music/notifications?user_id=user123&unread_only=false&limit=50

# Marcar como leída
PUT /music/notifications/{notification_id}/read?user_id=user123

# Marcar todas como leídas
PUT /music/notifications/read-all?user_id=user123

# Eliminar notificación
DELETE /music/notifications/{notification_id}?user_id=user123

# Estadísticas de notificaciones
GET /music/notifications/stats?user_id=user123
```

```bash
# Recomendaciones por similitud, mood o género
POST /music/recommendations/intelligent?track_id=xxx&limit=10&method=similarity

# Métodos disponibles: similarity, mood, genre

# Generar playlist recomendada
POST /music/recommendations/playlist?playlist_length=20
Content-Type: application/json

{
  "genres": ["Rock", "Pop"],
  "moods": ["energetic", "happy"],
  "energy_range": [0.6, 1.0],
  "tempo_range": [100, 160],
  "seed_track_id": "xxx"
}
```

## 💻 Uso Programático

### Ejemplo Básico

```python
from services.spotify_service import SpotifyService
from core.music_analyzer import MusicAnalyzer
from services.music_coach import MusicCoach

# Inicializar servicios
spotify = SpotifyService()
analyzer = MusicAnalyzer()
coach = MusicCoach()

# Buscar canción
tracks = spotify.search_track("Bohemian Rhapsody", limit=1)
track_id = tracks[0]["id"]

# Obtener datos de Spotify
spotify_data = spotify.get_track_full_analysis(track_id)

# Analizar
analysis = analyzer.analyze_track(spotify_data)

# Obtener coaching
coaching = coach.generate_coaching_analysis(analysis)

print(f"Tonalidad: {analysis['musical_analysis']['key_signature']}")
print(f"Tempo: {analysis['musical_analysis']['tempo']['bpm']} BPM")
print(f"Dificultad: {coaching['overview']['difficulty_level']}")
```

## 🎯 Análisis Proporcionados

### Análisis Musical
- **Tonalidad**: Nota raíz y modo (mayor/menor)
- **Escala**: Notas de la escala principal
- **Tempo**: BPM y categorización
- **Compás**: Time signature
- **Cambios de Tonalidad**: Modulaciones en la canción
- **Cambios de Tempo**: Variaciones de velocidad
- **Género Musical**: Detección automática de género (Rock, Pop, Electronic, Hip-Hop, Jazz, Classical, etc.)
- **Análisis Armónico Avanzado**: Progresiones de acordes, cadencias (perfecta, plagal, media), patrones comunes
- **Análisis de Emociones**: Detección de emociones primarias y perfil emocional (8 emociones: happy, sad, energetic, calm, angry, romantic, nostalgic, mysterious)

### Análisis Técnico
- **Energía**: Nivel de intensidad
- **Bailabilidad**: Qué tan bailable es
- **Valencia**: Positividad/negatividad
- **Acústica**: Acústico vs electrónico
- **Instrumental**: Instrumental vs con voces
- **En Vivo**: En vivo vs estudio
- **Volumen**: Nivel de loudness

### Análisis de Composición
- **Estructura**: Secciones identificadas
- **Progresiones Armónicas**: Análisis de acordes
- **Complejidad**: Nivel de complejidad musical
- **Estilo**: Estilo de composición

### Coaching
- **Ruta de Aprendizaje**: Pasos para aprender la canción
- **Ejercicios de Práctica**: Ejercicios personalizados
- **Tips de Interpretación**: Consejos de ejecución
- **Recomendaciones**: Sugerencias por nivel

## 🔬 Notas Musicales y Escalas

El sistema identifica:
- **12 notas**: C, C#, D, D#, E, F, F#, G, G#, A, A#, B
- **Modos**: Major (Mayor) y Minor (Menor)
- **Escalas**: Major, Minor, Pentatonic, Blues, Dorian, Mixolydian
- **Acordes**: Acordes triada en la tonalidad

## 🎓 Niveles de Dificultad

- **Beginner**: Canciones simples, adecuadas para principiantes
- **Intermediate**: Canciones de dificultad moderada
- **Advanced**: Canciones complejas que requieren habilidades avanzadas

## 🔧 Configuración Avanzada

### Análisis Detallado

El sistema puede proporcionar diferentes niveles de detalle:
- **Basic**: Información esencial
- **Detailed**: Análisis completo (por defecto)
- **Expert**: Análisis muy detallado con todos los datos

## 🧪 Testing

```bash
# Ejecutar tests
pytest tests/

# Con cobertura
pytest --cov=music_analyzer_ai tests/
```

## 🆕 Mejoras Implementadas (v2.6.0)

### Nuevas Funcionalidades Avanzadas
- ✅ **Predicción de Tendencias**: Predice tendencias de géneros, emociones y características
- ✅ **Análisis de Instrumentación**: Analiza instrumentación, textura y arreglos
- ✅ **Exportación Avanzada**: Exportación a CSV y reportes comprehensivos en Markdown
- ✅ **Análisis Acústico vs Eléctrico**: Identifica tipo de instrumentación
- ✅ **Análisis Instrumental vs Vocal**: Determina presencia de voces
- ✅ **Estimación de Instrumentos**: Estima instrumentos presentes en el track
- ✅ **Predicción de Tendencias Futuras**: Horizonte temporal configurable

## 🆕 Mejoras Implementadas (v2.5.0)

### Nuevas Funcionalidades Avanzadas
- ✅ **Sistema de Descubrimiento Musical**: Descubre artistas similares, tracks underground y tracks frescos
- ✅ **Descubrimiento por Transición de Mood**: Encuentra tracks que transicionan entre diferentes moods
- ✅ **Análisis Detallado de Covers**: Compara covers con originales con análisis de fidelidad
- ✅ **Análisis Detallado de Remixes**: Analiza transformaciones y modificaciones en remixes
- ✅ **Búsqueda de Versiones**: Encuentra automáticamente covers y remixes de cualquier track
- ✅ **Identificación Automática**: Identifica tipo de cover/remix automáticamente
- ✅ **Análisis de Transformación**: Evalúa nivel de transformación en remixes

## 🆕 Mejoras Implementadas (v2.4.0)

### Nuevas Funcionalidades Avanzadas
- ✅ **Análisis Inteligente de Playlists**: Análisis completo de diversidad, coherencia y flujo
- ✅ **Sugerencias de Mejora**: Recomendaciones automáticas para mejorar playlists
- ✅ **Optimización de Orden**: Reordena tracks para mejor experiencia de escucha
- ✅ **Comparación de Artistas**: Compara hasta 5 artistas con análisis detallado
- ✅ **Análisis de Evolución**: Analiza cómo evoluciona la música de un artista
- ✅ **Detección de Cambios**: Identifica cambios de género y estilo a lo largo del tiempo
- ✅ **Análisis de Flujo**: Evalúa la progresión de energía en playlists

## 🆕 Mejoras Implementadas (v2.3.0)

### Nuevas Funcionalidades Avanzadas
- ✅ **Análisis Temporal**: Análisis de estructura temporal, progresión de energía y cambios de tempo
- ✅ **Detección de Build-ups y Drops**: Identificación automática de aumentos y caídas de energía
- ✅ **Análisis de Calidad**: Evaluación completa de calidad de producción musical
- ✅ **Recomendaciones Contextuales**: Recomendaciones basadas en contexto personalizado
- ✅ **Recomendaciones por Hora del Día**: Adaptadas a morning, afternoon, evening, night
- ✅ **Recomendaciones por Actividad**: Optimizadas para workout, study, party, relax, drive
- ✅ **Recomendaciones por Mood**: Basadas en happy, sad, energetic, calm, romantic
- ✅ **Análisis de Complejidad Temporal**: Cálculo de complejidad basada en cambios temporales

## 🆕 Mejoras Implementadas (v2.2.0)

### Nuevas Funcionalidades Avanzadas
- ✅ **Análisis de Tendencias**: Análisis de popularidad y tendencias de tracks y artistas
- ✅ **Predicción de Éxito Comercial**: Predicción basada en características musicales
- ✅ **Análisis de Patrones Rítmicos**: Análisis avanzado de densidad y consistencia rítmica
- ✅ **Análisis de Colaboraciones**: Detección y análisis de colaboraciones entre artistas
- ✅ **Red de Colaboraciones**: Análisis de redes de colaboración entre múltiples artistas
- ✅ **Comparación de Versiones**: Compara originales, covers y remixes
- ✅ **Sistema de Alertas Inteligentes**: Alertas de popularidad, oportunidades y colaboraciones
- ✅ **Gestión de Alertas**: Sistema completo con filtros, prioridades y gestión

## 🆕 Mejoras Implementadas (v2.1.0)

### Nuevas Funcionalidades ML Avanzadas
- ✅ **Análisis Comprehensivo**: Análisis ML completo con todas las predicciones y comparaciones
- ✅ **Comparación de Tracks ML**: Compara múltiples tracks usando ML (hasta 20 tracks)
- ✅ **Predicción Multi-Tarea**: Predicción simultánea de género, emoción, complejidad y dificultad
- ✅ **Pipeline Info**: Información detallada del pipeline de ML
- ✅ **Análisis de Estilo**: Detección completa del estilo musical
- ✅ **Predicción de Era**: Predicción de la era musical basada en características
- ✅ **Análisis de Influencias**: Detección de posibles influencias musicales
- ✅ **Cálculo de Diversidad**: Análisis de diversidad musical en conjuntos de tracks

## 🆕 Mejoras Implementadas (v2.0.0)

### Nuevas Funcionalidades
- ✅ **Sistema de Cache**: Cache inteligente con TTL configurable
- ✅ **Manejo de Errores Mejorado**: Excepciones personalizadas y códigos HTTP apropiados
- ✅ **Validaciones Robustas**: Validación de track IDs y sanitización de queries
- ✅ **Análisis Comparativo**: Compara hasta 10 canciones simultáneamente
- ✅ **Recomendaciones de Spotify**: Obtiene canciones similares usando la API de Spotify
- ✅ **Detección de Género**: Detecta género musical basado en características de audio (12 géneros)
- ✅ **Exportación de Análisis**: Exporta análisis en JSON, texto plano y Markdown
- ✅ **Sistema de Historial**: Guarda y gestiona historial de análisis realizados
- ✅ **Analytics y Métricas**: Tracking completo de uso, tiempos de respuesta y estadísticas
- ✅ **Rate Limiting**: Protección contra abuso con límite de 100 requests/minuto por IP
- ✅ **Sistema de Favoritos**: Guarda y gestiona canciones favoritas por usuario
- ✅ **Sistema de Tags**: Etiquetado de tracks, análisis y playlists con búsqueda avanzada
- ✅ **Webhooks**: Sistema de notificaciones en tiempo real para eventos
- ✅ **Análisis Armónico Avanzado**: Detección de progresiones, cadencias y patrones armónicos
- ✅ **Análisis de Emociones**: Detección de 8 emociones musicales (happy, sad, energetic, calm, angry, romantic, nostalgic, mysterious)
- ✅ **Sistema de Autenticación**: Registro, login y gestión de usuarios con JWT
- ✅ **Playlists Personalizadas**: Crear, gestionar y compartir playlists
- ✅ **Recomendaciones Inteligentes**: Sistema ML para recomendaciones por similitud, mood y género
- ✅ **Generación de Playlists**: Genera playlists automáticas basadas en preferencias
- ✅ **Machine Learning Avanzado**: Predicciones de género/emoción, clustering, similitud en batch
- ✅ **Dashboard de Métricas**: Dashboard completo con métricas del sistema y usuario
- ✅ **Sistema de Notificaciones**: Notificaciones personalizadas por usuario con prioridades
- ✅ **Gestión de Cache**: Endpoints para ver estadísticas y limpiar cache
- ✅ **Timeout y Retry**: Manejo de timeouts en peticiones a Spotify
- ✅ **Logging Mejorado**: Sistema de logging más detallado

### Mejoras de Rendimiento
- Cache en memoria para reducir llamadas a Spotify API
- Validación temprana de inputs
- Manejo eficiente de errores
- Timeout configurable en peticiones
- Rate limiting para proteger el sistema
- Tracking de métricas en tiempo real

## 📊 Estadísticas del Sistema

- **Total de Endpoints**: 82+
- **Endpoints ML**: 12+
- **Endpoints de Tendencias**: 4
- **Endpoints de Colaboraciones**: 3
- **Endpoints de Alertas**: 4
- **Endpoints Temporales**: 3
- **Endpoints de Calidad**: 1
- **Endpoints de Recomendaciones Contextuales**: 4
- **Endpoints de Análisis de Playlist**: 3
- **Endpoints de Análisis de Artistas**: 2
- **Endpoints de Descubrimiento**: 4
- **Endpoints de Covers/Remixes**: 3
- **Servicios Especializados**: 29+
- **Tipos de Análisis**: 26+
- **Emociones Detectadas**: 8
- **Géneros Soportados**: 12
- **Formatos de Exportación**: 3
- **Capacidades ML**: 11 tipos diferentes
- **Tipos de Alertas**: 4
- **Contextos de Recomendación**: 4 tipos
- **Métricas de Playlist**: 5 tipos
- **Tipos de Descubrimiento**: 4
- **Sistemas Integrados**: Autenticación, Playlists, Favoritos, Tags, Webhooks, Analytics, ML Avanzado, Dashboard, Notificaciones, Tendencias, Colaboraciones, Alertas, Análisis Temporal, Calidad, Recomendaciones Contextuales, Análisis de Playlist, Comparación de Artistas, Descubrimiento Musical, Análisis de Covers/Remixes

## 📊 Mejoras Futuras

- [ ] Análisis de audio local (sin Spotify)
- [ ] Detección de instrumentos
- [ ] Análisis de letras
- [ ] Comparación entre versiones
- [ ] Generación de partituras
- [ ] Integración con más servicios de música (YouTube, SoundCloud)
- [ ] Análisis de progresión temporal
- [ ] Sistema de seguimiento de progreso
- [ ] Dashboard web interactivo
- [ ] Machine Learning avanzado para recomendaciones
- [ ] Análisis de colaboraciones entre artistas
- [ ] Sistema de notificaciones push

## 🤝 Contribuir

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Cree una rama para su feature (`git checkout -b feature/AmazingFeature`)
3. Commit sus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abra un Pull Request

## 📝 Licencia

Este proyecto es parte de Blatam Academy.

## 📞 Soporte

Para soporte, contacte al equipo de Blatam Academy.

## 🔗 Enlaces Útiles

- [Spotify Web API Documentation](https://developer.spotify.com/documentation/web-api)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)

---

**Desarrollado con ❤️ por Blatam Academy**

