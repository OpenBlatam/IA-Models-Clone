# 🚀 Inicio Rápido - Music Analyzer AI

## Iniciar Todo

### Windows

```cmd
python main.py
```

O con uvicorn directamente:

```cmd
uvicorn main:app --host 0.0.0.0 --port 8010 --reload
```

### Linux/Mac

```bash
python main.py
```

O con uvicorn directamente:

```bash
uvicorn main:app --host 0.0.0.0 --port 8010 --reload
```

### Docker (si está disponible)

```bash
# Windows
deployment\start.bat

# Linux/Mac
./deployment/start.sh

# Python (Multiplataforma)
python deployment/start.py
```

## Detener Todo

### Windows

```cmd
Ctrl+C
```

O si usas Docker:

```cmd
deployment\stop.bat
```

### Linux/Mac

```bash
Ctrl+C
```

O si usas Docker:

```bash
./deployment/stop.sh
```

## ¿Qué se Inicia?

Con un solo comando se inician:

- ✅ **Music Analyzer AI** - API principal
- ✅ **FastAPI Server** - Servidor ASGI
- ✅ **Spotify Integration** - Integración con Spotify API
- ✅ **Health Checks** - Endpoints de salud
- ✅ **API Documentation** - Swagger/OpenAPI docs
- ✅ **Gradio UI** - Interfaz web opcional (si está habilitada)
- ✅ **Redis** - Caché (si está configurado)
- ✅ **PostgreSQL** - Base de datos (opcional)

## URLs Disponibles

Una vez iniciado:

- 🌐 **API**: http://localhost:8010
- ❤️ **Health**: http://localhost:8010/health
- 📖 **Docs**: http://localhost:8010/docs
- 📘 **ReDoc**: http://localhost:8010/redoc
- 🎨 **Gradio UI**: http://localhost:7860 (si está habilitada)

## Configuración

El sistema usa variables de entorno. Crea un archivo `.env` en la raíz del proyecto:

```env
# Spotify API (REQUERIDO)
SPOTIFY_CLIENT_ID=tu_spotify_client_id_aqui
SPOTIFY_CLIENT_SECRET=tu_spotify_client_secret_aqui
SPOTIFY_REDIRECT_URI=http://localhost:8010/callback

# Server Configuration
HOST=0.0.0.0
PORT=8010
DEBUG=True

# Logging
LOG_LEVEL=INFO

# Redis (opcional, para caché)
REDIS_URL=redis://localhost:6379/0

# Database (opcional)
DATABASE_URL=postgresql://user:password@localhost:5432/music_analyzer
```

## Instalación Rápida

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Configurar variables de entorno
# Crear .env con tus credenciales de Spotify

# 3. Obtener credenciales de Spotify
# Ve a https://developer.spotify.com/dashboard
# Crea una aplicación y obtén Client ID y Client Secret

# 4. Iniciar servidor
python main.py
```

## Configuración de Spotify

1. Ve a [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
2. Crea una nueva aplicación
3. Obtén tu **Client ID** y **Client Secret**
4. Configura el **Redirect URI**: `http://localhost:8010/callback`
5. Agrega las credenciales a tu archivo `.env`

## Verificación

Para verificar que todo funciona:

```bash
# Health check
curl http://localhost:8010/health

# API docs
curl http://localhost:8010/docs

# Test de búsqueda (requiere credenciales de Spotify)
curl -X POST http://localhost:8010/music/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Bohemian Rhapsody", "limit": 5}'
```

## Más Información

- 📖 Ver `INSTALLATION_GUIDE.md` para instalación detallada
- 🏗️ Ver `ARCHITECTURE_QUICK_START.md` para arquitectura
- 📚 Ver `API_QUICK_START.md` para uso de la API
- 🔍 Ver `QUICK_REFERENCE.md` para referencia rápida

## Troubleshooting

### Puerto en uso

```bash
# Cambiar puerto en .env o directamente:
uvicorn main:app --port 8011
```

### Error de credenciales de Spotify

```bash
# Verificar que .env tiene las credenciales correctas
# Verificar que la aplicación de Spotify está activa
# Verificar que el Redirect URI está configurado correctamente
```

### Dependencias faltantes

```bash
pip install -r requirements.txt
```

### Errores de importación

```bash
# Asegúrate de estar en el directorio correcto
cd agents/backend/onyx/server/features/music_analyzer_ai
```

### Error: Spotify API no responde

```bash
# Verificar conexión a internet
# Verificar que las credenciales son válidas
# Verificar rate limits de Spotify API
```

## Próximos Pasos

1. ✅ Verificar que el servidor inicia correctamente
2. 📖 Revisar la documentación de la API en `/docs`
3. 🎵 Probar búsqueda de canciones
4. 🎼 Probar análisis de canciones
5. 🎓 Explorar coaching musical
6. 🔧 Configurar caché y base de datos si es necesario

## Endpoints Principales

- `POST /music/search` - Buscar canciones en Spotify
- `POST /music/analyze` - Analizar una canción
- `GET /music/analyze/{track_id}` - Analizar por ID
- `POST /music/coaching` - Obtener coaching musical
- `GET /music/recommendations/{track_id}` - Recomendaciones

---

**Última actualización**: 2025  
**Versión**: 2.21.0






