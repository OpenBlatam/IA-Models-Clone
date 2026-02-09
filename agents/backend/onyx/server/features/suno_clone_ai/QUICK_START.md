# Suno Clone AI - Guía de Inicio Rápido

## 🚀 Inicio Rápido en 5 Minutos

### Paso 1: Instalación

```bash
# Clonar o navegar al directorio
cd suno_clone_ai

# Instalar dependencias
pip install -r requirements.txt
```

### Paso 2: Configuración Básica

Crea un archivo `.env`:

```env
API_PORT=8020
MUSIC_MODEL=facebook/musicgen-medium
USE_GPU=True
```

### Paso 3: Iniciar el Servidor

```bash
python main.py
```

El servidor estará disponible en `http://localhost:8020`

### Paso 4: Probar la API

#### Opción 1: Desde el Navegador

Abre `http://localhost:8020/docs` para ver la documentación interactiva de Swagger.

#### Opción 2: Desde la Terminal

```bash
# Crear una canción desde chat
curl -X POST "http://localhost:8020/suno/chat/create-song" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Una canción de pop alegre con piano",
    "user_id": "test_user"
  }'
```

Respuesta:
```json
{
  "song_id": "abc123...",
  "status": "processing",
  "message": "Canción en proceso de generación",
  "metadata": {
    "prompt": "...",
    "genre": "pop",
    "mood": "happy"
  }
}
```

#### Opción 3: Desde Python

```python
import requests

# Crear canción
response = requests.post(
    "http://localhost:8020/suno/chat/create-song",
    json={
        "message": "Rock energético con guitarra eléctrica",
        "user_id": "my_user"
    }
)

song_data = response.json()
song_id = song_data["song_id"]

# Verificar estado
import time
while True:
    status_response = requests.get(
        f"http://localhost:8020/suno/generate/status/{song_id}"
    )
    status = status_response.json()
    
    if status["status"] == "completed":
        print("¡Canción lista!")
        break
    elif status["status"] == "failed":
        print("Error en la generación")
        break
    
    time.sleep(2)

# Descargar canción
audio_response = requests.get(
    f"http://localhost:8020/suno/songs/{song_id}/download"
)

with open("mi_cancion.wav", "wb") as f:
    f.write(audio_response.content)
```

## 💡 Ejemplos de Uso

### Ejemplo 1: Canción Simple

```json
{
  "message": "Música relajante de jazz"
}
```

### Ejemplo 2: Canción con Especificaciones

```json
{
  "message": "Rock pesado con batería fuerte, 120 BPM, 2 minutos"
}
```

### Ejemplo 3: Canción con Instrumentos Específicos

```json
{
  "message": "Canción clásica con violín y piano, mood melancólico"
}
```

## 🎯 Casos de Uso Comunes

### 1. Generar Música de Fondo

```bash
curl -X POST "http://localhost:8020/suno/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Ambient background music, calm and peaceful",
    "duration": 60
  }'
```

### 2. Crear Música para Video

```bash
curl -X POST "http://localhost:8020/suno/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Upbeat electronic music for video background",
    "duration": 30,
    "genre": "electronic",
    "mood": "energetic"
  }'
```

### 3. Experimentar con Diferentes Géneros

```python
genres = ["rock", "jazz", "classical", "electronic", "hip hop"]

for genre in genres:
    response = requests.post(
        "http://localhost:8020/suno/generate",
        json={
            "prompt": f"{genre} music",
            "genre": genre,
            "duration": 20
        }
    )
    print(f"Generando {genre}...")
```

## 🔍 Verificar Estado

```bash
# Ver estado de generación
curl "http://localhost:8020/suno/generate/status/{song_id}"

# Listar todas las canciones
curl "http://localhost:8020/suno/songs"

# Ver información de una canción
curl "http://localhost:8020/suno/songs/{song_id}"
```

## ⚙️ Configuración Avanzada

### Usar Modelo Más Grande

En `.env`:
```env
MUSIC_MODEL=facebook/musicgen-large
```

### Ajustar Parámetros de Generación

```python
# En config/settings.py o .env
TEMPERATURE=1.2  # Más creativo
TOP_K=300        # Más opciones
CFG_COEF=4.0     # Más guiado
```

## 🐛 Troubleshooting

### Error: "Model not loaded"

- Verifica que tienes espacio en disco suficiente
- Asegúrate de tener conexión a internet para descargar el modelo
- Revisa los logs para más detalles

### Error: "CUDA out of memory"

- Usa un modelo más pequeño: `MUSIC_MODEL=facebook/musicgen-small`
- O desactiva GPU: `USE_GPU=False`

### Generación muy lenta

- Activa GPU si está disponible
- Usa modelo más pequeño
- Reduce la duración de la canción

## 🚀 Nuevas Características Mejoradas

### Generación Rápida Optimizada

```python
from suno_clone_ai.core import FastMusicGenerator

# Generador optimizado con caché y compilación
generator = FastMusicGenerator(use_cache=True, use_compile=True)

# Generación rápida (hasta 2x más rápido)
audio = generator.generate_from_text(
    "A happy upbeat song",
    duration=30
)

# Generación en batch
texts = ["Song 1", "Song 2", "Song 3"]
audios = generator.generate_batch(texts, duration=30)
```

### Procesamiento de Audio Avanzado

```python
from suno_clone_ai.core import AudioProcessor

processor = AudioProcessor(sample_rate=32000)

# Aplicar efectos
compressed = processor.apply_compressor(audio, threshold=-12.0, ratio=4.0)
reverb_audio = processor.apply_reverb(audio, room_size=0.7)
eq_audio = processor.apply_eq(audio, low_gain=2.0, high_gain=-1.0)

# Mezclar pistas
mixed = processor.mix_audio([audio1, audio2], volumes=[0.8, 0.6])

# Fade in/out
faded = processor.fade_in_out(audio, fade_in_duration=1.0)
```

### Sistema de Caché

```python
from suno_clone_ai.utils import MusicCache

cache = MusicCache(cache_dir="cache/music")

# Verificar caché antes de generar
cached = cache.get("A happy song", duration=30)
if cached is None:
    audio = generator.generate_from_text("A happy song", duration=30)
    cache.set("A happy song", 30, audio)
```

Ver [IMPROVEMENTS.md](IMPROVEMENTS.md) para más detalles sobre las mejoras.

## 📚 Próximos Pasos

1. Lee el [README.md](README.md) completo para más detalles
2. Explora las [mejoras](IMPROVEMENTS.md) del sistema
3. Explora la documentación de API en `/docs`
4. Experimenta con diferentes prompts y parámetros
5. Integra la API en tu aplicación

## 🆘 Ayuda

- Documentación completa: [README.md](README.md)
- Issues: Abre un issue en el repositorio
- Soporte: Contacta al equipo de desarrollo

