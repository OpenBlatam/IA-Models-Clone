# 🚀 Quick Start Guide - Social Media Identity Clone AI

## Instalación Rápida

### 1. Instalar Dependencias

```bash
cd social_media_identity_clone_ai
pip install -r requirements.txt
```

### 2. Configurar Variables de Entorno

```bash
cp .env.example .env
# Editar .env con tus API keys
```

Variables de entorno necesarias:
- `OPENAI_API_KEY` - Requerido para análisis y generación
- `TIKTOK_API_KEY` - Opcional, para extracción de TikTok
- `INSTAGRAM_API_KEY` - Opcional, para extracción de Instagram
- `YOUTUBE_API_KEY` - Opcional, para extracción de YouTube

### 3. Ejecutar la API

```bash
# Opción 1: Directamente
python run_api.py

# Opción 2: Con uvicorn
uvicorn social_media_identity_clone_ai.api.main:app --host 0.0.0.0 --port 8030

# Opción 3: Con Docker
docker-compose up
```

## Uso Básico

### Usando la API

#### 1. Extraer Perfil

```bash
curl -X POST "http://localhost:8030/api/v1/extract-profile" \
  -H "Content-Type: application/json" \
  -d '{
    "platform": "tiktok",
    "username": "ejemplo_usuario"
  }'
```

#### 2. Construir Identidad

```bash
curl -X POST "http://localhost:8030/api/v1/build-identity" \
  -H "Content-Type: application/json" \
  -d '{
    "tiktok_username": "ejemplo_tiktok",
    "instagram_username": "ejemplo_instagram",
    "youtube_channel_id": "ejemplo_youtube"
  }'
```

#### 3. Generar Contenido

```bash
curl -X POST "http://localhost:8030/api/v1/generate-content" \
  -H "Content-Type: application/json" \
  -d '{
    "identity_profile_id": "identity_id_here",
    "platform": "instagram",
    "content_type": "post",
    "topic": "fitness",
    "style": "motivational"
  }'
```

### Usando Python

```python
import asyncio
from services.profile_extractor import ProfileExtractor
from services.identity_analyzer import IdentityAnalyzer
from services.content_generator import ContentGenerator

async def main():
    # Extraer perfiles
    extractor = ProfileExtractor()
    tiktok_profile = await extractor.extract_tiktok_profile("username")
    
    # Construir identidad
    analyzer = IdentityAnalyzer()
    identity = await analyzer.build_identity(tiktok_profile=tiktok_profile)
    
    # Generar contenido
    generator = ContentGenerator(identity_profile=identity)
    post = await generator.generate_instagram_post(topic="fitness")
    
    print(post.content)

asyncio.run(main())
```

## Documentación de API

Una vez que la API esté corriendo, visita:
- **Swagger UI**: http://localhost:8030/docs
- **ReDoc**: http://localhost:8030/redoc

## Notas Importantes

⚠️ **Limitaciones Actuales:**
- Los conectores de APIs de redes sociales requieren implementación real de las APIs oficiales
- Se requiere autenticación OAuth2 para Instagram
- YouTube requiere API key de Google Cloud
- TikTok puede requerir scraping (verificar términos de servicio)

✅ **Próximos Pasos:**
1. Implementar conectores reales a las APIs
2. Agregar almacenamiento de identidades en base de datos
3. Implementar sistema de caché para perfiles
4. Agregar más análisis de contenido

## Troubleshooting

### Error: "OpenAI client no disponible"
- Verifica que `OPENAI_API_KEY` esté configurado en `.env`

### Error: "Plataforma no soportada"
- Verifica que el nombre de la plataforma sea: `tiktok`, `instagram`, o `youtube`

### Error: "Error extrayendo perfil"
- Los conectores actuales son placeholders y requieren implementación real
- Verifica que tengas acceso a las APIs correspondientes




