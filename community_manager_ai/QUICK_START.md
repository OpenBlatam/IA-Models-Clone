# Quick Start Guide - Community Manager AI

## 🚀 Inicio Rápido

### 1. Instalación

```bash
# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
cp .env.example .env
# Editar .env con tus credenciales
```

### 2. Ejecutar la API

```bash
python main.py
```

La API estará disponible en `http://localhost:8000`

### 3. Documentación de la API

Visita `http://localhost:8000/docs` para la documentación interactiva de Swagger.

## 📝 Ejemplos de Uso

### Programar un Post

```python
from community_manager_ai import CommunityManager
from datetime import datetime, timedelta

manager = CommunityManager()

# Programar post para mañana a las 10 AM
scheduled_time = datetime.now() + timedelta(days=1)
scheduled_time = scheduled_time.replace(hour=10, minute=0, second=0)

result = manager.schedule_post(
    content="¡Hola mundo desde Community Manager AI!",
    platforms=["facebook", "twitter", "instagram"],
    scheduled_time=scheduled_time
)

print(f"Post programado: {result['post_id']}")
```

### Publicar Inmediatamente

```python
results = manager.publish_now(
    content="Publicación inmediata",
    platforms=["facebook", "twitter"]
)

for platform, result in results.items():
    print(f"{platform}: {result['status']}")
```

### Agregar un Meme

```python
meme_id = manager.add_meme(
    image_path="path/to/meme.jpg",
    caption="Funny meme",
    tags=["funny", "tech"],
    category="humor"
)

print(f"Meme agregado: {meme_id}")
```

### Conectar una Plataforma

```python
# Conectar Facebook
success = manager.connect_platform(
    platform="facebook",
    credentials={
        "access_token": "your_access_token"
    }
)

if success:
    print("Facebook conectado exitosamente")
```

### Usar el Auto-Poster

```bash
# Ejecutar el script de auto-poster
python scripts/auto_post.py --interval 60 --max-posts 10
```

## 🔧 Scripts Disponibles

- `scripts/auto_post.py` - Publicación automática de posts programados
- `scripts/content_analyzer.py` - Análisis de contenido
- `scripts/engagement_tracker.py` - Rastreo de engagement

## 📚 Más Información

Ver `README.md` para documentación completa.




