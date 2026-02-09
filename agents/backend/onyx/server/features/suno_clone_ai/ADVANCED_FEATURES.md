# Suno Clone AI - Funcionalidades Avanzadas

## 🚀 Nuevas Funcionalidades

### 📦 Sistema de Caché

El sistema incluye un caché inteligente que almacena resultados de generación para evitar regenerar canciones idénticas.

**Características:**
- Almacenamiento persistente con diskcache
- Límite de 10GB por defecto
- TTL configurable
- Estadísticas de uso

**Uso:**
```python
from core.cache_manager import get_cache_manager

cache = get_cache_manager()
stats = cache.stats()
```

**Endpoints:**
- `GET /suno/cache/stats` - Ver estadísticas del caché
- `DELETE /suno/cache/clear` - Limpiar caché

### 🎚️ Procesamiento Avanzado de Audio

Sistema completo de edición y procesamiento de audio con múltiples efectos.

**Funcionalidades:**
- Normalización automática
- Fade in/out
- Eliminación de silencio
- Reverb
- Ecualización (EQ)
- Cambio de tempo (sin cambiar pitch)
- Cambio de pitch (sin cambiar tempo)
- Mezcla de múltiples pistas

**Ejemplo de Edición:**
```bash
curl -X POST "http://localhost:8020/suno/songs/{song_id}/edit" \
  -H "Content-Type: application/json" \
  -d '{
    "operations": [
      {"type": "reverb", "room_size": 0.7, "damping": 0.5},
      {"type": "eq", "low_gain": 2.0, "mid_gain": 0.0, "high_gain": -1.0}
    ],
    "fade_in": 1.0,
    "fade_out": 2.0,
    "normalize": true
  }'
```

**Ejemplo de Mezcla:**
```bash
curl -X POST "http://localhost:8020/suno/songs/mix" \
  -H "Content-Type: application/json" \
  -d '{
    "song_ids": ["song1", "song2", "song3"],
    "volumes": [1.0, 0.8, 0.6]
  }'
```

### 📊 Sistema de Métricas y Analytics

Tracking completo de uso y estadísticas del sistema.

**Métricas Incluidas:**
- Total de canciones generadas
- Tiempo promedio de generación
- Duración total de música generada
- Estadísticas diarias
- Prompts más populares
- Estadísticas por usuario

**Endpoints:**
- `GET /suno/metrics/stats?days=7` - Estadísticas generales
- `GET /suno/metrics/user/{user_id}?days=30` - Estadísticas de usuario

**Ejemplo:**
```bash
# Estadísticas generales (últimos 7 días)
curl "http://localhost:8020/suno/metrics/stats?days=7"

# Estadísticas de usuario
curl "http://localhost:8020/suno/metrics/user/user123?days=30"
```

### 🔍 Análisis de Audio

Análisis detallado de características de audio generado.

**Características Analizadas:**
- RMS (Root Mean Square)
- Peak level
- Zero crossing rate
- Spectral centroid
- Tempo estimado
- Duración

**Endpoint:**
- `GET /suno/songs/{song_id}/analyze`

**Ejemplo:**
```bash
curl "http://localhost:8020/suno/songs/{song_id}/analyze"
```

**Respuesta:**
```json
{
  "song_id": "abc123",
  "analysis": {
    "rms": 0.45,
    "peak": 0.98,
    "zero_crossing_rate": 0.12,
    "spectral_centroid": 2500.5,
    "tempo": 120.0,
    "duration": 30.0
  }
}
```

## ⚡ Optimizaciones de Performance

### Caché Inteligente
- Los resultados se cachean automáticamente
- Regeneraciones idénticas son instantáneas
- Reduce carga en modelos de IA

### Procesamiento Optimizado
- Uso de numba para operaciones numéricas críticas
- Normalización automática de audio
- Fade in/out aplicado por defecto

### Background Processing
- Generación asíncrona
- No bloquea la API
- Tracking de progreso

## 🎛️ Operaciones de Audio Disponibles

### Reverb
Aplica efecto de reverberación.

```json
{
  "type": "reverb",
  "room_size": 0.5,  // 0.0 - 1.0
  "damping": 0.5     // 0.0 - 1.0
}
```

### Ecualización (EQ)
Ajusta frecuencias bajas, medias y altas.

```json
{
  "type": "eq",
  "low_gain": 2.0,   // Ganancia en bajos (dB)
  "mid_gain": 0.0,   // Ganancia en medios (dB)
  "high_gain": -1.0  // Ganancia en agudos (dB)
}
```

### Cambio de Tempo
Cambia la velocidad sin afectar el pitch.

```json
{
  "type": "tempo",
  "factor": 1.2  // 1.0 = normal, >1.0 = más rápido, <1.0 = más lento
}
```

### Cambio de Pitch
Cambia el tono sin afectar el tempo.

```json
{
  "type": "pitch",
  "semitones": 2.0  // Semitonos a subir (positivo) o bajar (negativo)
}
```

## 📈 Casos de Uso Avanzados

### 1. Generar y Mejorar Automáticamente
```python
# Generar canción
response = requests.post(
    "http://localhost:8020/suno/generate",
    json={"prompt": "Rock song", "duration": 30}
)
song_id = response.json()["song_id"]

# Esperar generación
time.sleep(30)

# Aplicar mejoras automáticas
requests.post(
    f"http://localhost:8020/suno/songs/{song_id}/edit",
    json={
        "normalize": True,
        "fade_in": 0.5,
        "fade_out": 1.0,
        "trim_silence": True
    }
)
```

### 2. Crear Remix
```python
# Generar múltiples canciones
song_ids = []
for prompt in ["bass line", "drum beat", "melody"]:
    response = requests.post(
        "http://localhost:8020/suno/generate",
        json={"prompt": prompt, "duration": 30}
    )
    song_ids.append(response.json()["song_id"])

# Mezclar
requests.post(
    "http://localhost:8020/suno/songs/mix",
    json={
        "song_ids": song_ids,
        "volumes": [0.7, 1.0, 0.8]  # Bajo, batería, melodía
    }
)
```

### 3. Análisis de Calidad
```python
# Generar y analizar
response = requests.post(...)
song_id = response.json()["song_id"]

# Analizar
analysis = requests.get(
    f"http://localhost:8020/suno/songs/{song_id}/analyze"
).json()

# Verificar calidad
if analysis["analysis"]["rms"] < 0.1:
    print("Audio muy bajo, aplicar normalización")
```

## 🔧 Configuración Avanzada

### Caché
```env
# En .env
CACHE_TTL=3600  # Tiempo de vida del caché en segundos
CACHE_DIR=./cache/suno_clone  # Directorio de caché
```

### Procesamiento de Audio
```env
SAMPLE_RATE=32000  # Sample rate por defecto
AUTO_NORMALIZE=true  # Normalizar automáticamente
AUTO_FADE=true  # Aplicar fade automáticamente
```

## 📊 Monitoreo

### Ver Estadísticas del Sistema
```bash
# Estadísticas generales
curl "http://localhost:8020/suno/metrics/stats?days=7"

# Estadísticas de caché
curl "http://localhost:8020/suno/cache/stats"
```

### Limpiar Caché
```bash
curl -X DELETE "http://localhost:8020/suno/cache/clear"
```

## 🚀 Mejores Prácticas

1. **Usar Caché**: Los prompts similares se cachean automáticamente
2. **Normalizar Audio**: Siempre normalizar para consistencia
3. **Aplicar Fade**: Fade in/out mejora la calidad percibida
4. **Monitorear Métricas**: Revisar estadísticas regularmente
5. **Optimizar Prompts**: Prompts más específicos = mejor calidad

## 🔮 Próximas Funcionalidades

- [ ] Soporte para más formatos de audio (MP3, FLAC, etc.)
- [ ] Efectos adicionales (chorus, delay, distortion)
- [ ] Separación de stems (batería, bajo, melodía)
- [ ] Masterización automática
- [ ] Exportación a múltiples plataformas
- [ ] API de WebSocket para streaming en tiempo real

