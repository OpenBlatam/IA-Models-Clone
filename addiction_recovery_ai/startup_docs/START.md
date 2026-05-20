# 🚀 Inicio Rápido - Addiction Recovery AI

## Iniciar Todo

### Windows

```cmd
python main.py
```

O con uvicorn directamente:

```cmd
uvicorn main:app --host 0.0.0.0 --port 8020 --reload
```

### Linux/Mac

```bash
python main.py
```

O con uvicorn directamente:

```bash
uvicorn main:app --host 0.0.0.0 --port 8020 --reload
```

### Docker (si está disponible)

```bash
docker-compose up -d
```

## Detener Todo

### Windows

```cmd
Ctrl+C
```

### Linux/Mac

```bash
Ctrl+C
```

### Docker

```bash
docker-compose down
```

## ¿Qué se Inicia?

Con un solo comando se inician:

- ✅ **Addiction Recovery AI** - API principal
- ✅ **FastAPI Server** - Servidor ASGI
- ✅ **Health Checks** - Endpoints de salud
- ✅ **API Documentation** - Swagger/OpenAPI docs
- ✅ **WebSocket Support** - Conexiones en tiempo real
- ✅ **GraphQL API** - API GraphQL opcional

## URLs Disponibles

Una vez iniciado:

- 🌐 **API**: http://localhost:8020
- ❤️ **Health**: http://localhost:8020/health
- 📖 **Docs**: http://localhost:8020/docs
- 📘 **ReDoc**: http://localhost:8020/redoc
- 🔌 **WebSocket**: ws://localhost:8020/ws
- 🎯 **GraphQL**: http://localhost:8020/graphql (si está habilitado)

## Configuración

El sistema usa variables de entorno. Crea un archivo `.env` en la raíz del proyecto:

```env
# Server Configuration
HOST=0.0.0.0
PORT=8020
DEBUG=True

# API Keys (si es necesario)
OPENAI_API_KEY=tu_api_key_aqui
ANTHROPIC_API_KEY=tu_api_key_aqui

# Database (si se usa)
DATABASE_URL=postgresql://user:password@localhost:5432/addiction_recovery

# Redis (si se usa para cache)
REDIS_URL=redis://localhost:6379/0

# Logging
LOG_LEVEL=INFO
```

## Instalación Rápida

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Configurar variables de entorno
cp .env.example .env
# Editar .env con tus credenciales

# 3. Iniciar servidor
python main.py
```

## Verificación

Para verificar que todo funciona:

```bash
# Health check
curl http://localhost:8020/health

# API docs
curl http://localhost:8020/docs
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
uvicorn main:app --port 8021
```

### Dependencias faltantes

```bash
pip install -r requirements.txt
```

### Errores de importación

```bash
# Asegúrate de estar en el directorio correcto
cd agents/backend/onyx/server/features/addiction_recovery_ai
```

## Próximos Pasos

1. ✅ Verificar que el servidor inicia correctamente
2. 📖 Revisar la documentación de la API en `/docs`
3. 🧪 Probar los endpoints principales
4. 🔧 Configurar variables de entorno según tus necesidades
5. 📊 Revisar los health checks y métricas

---

**Última actualización**: 2025  
**Versión**: 3.4.0






