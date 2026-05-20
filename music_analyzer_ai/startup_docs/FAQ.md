# ❓ Preguntas Frecuentes (FAQ) - Music Analyzer AI

## 🔧 Instalación y Configuración

### ¿Qué versión de Python necesito?

**R:** Python 3.8 o superior. Puedes verificar tu versión con:
```bash
python --version
```

### ¿Necesito una cuenta de Spotify Developer?

**R:** Sí, es obligatorio. Puedes crear una cuenta gratuita en:
https://developer.spotify.com/dashboard

### ¿Cuánto cuesta usar la API de Spotify?

**R:** La API de Spotify es gratuita para uso personal y desarrollo. Tiene límites de rate limiting que son generosos para la mayoría de casos de uso.

### ¿Puedo usar el sistema sin Redis?

**R:** Sí, Redis es opcional. El sistema funcionará sin él, pero el rendimiento será mejor con Redis habilitado para caché.

### ¿Necesito PostgreSQL?

**R:** No, PostgreSQL es opcional. Por defecto el sistema usa SQLite, que es suficiente para desarrollo y uso básico.

## 🚀 Uso y Funcionalidad

### ¿Cómo busco una canción?

**R:** Usa el endpoint `/music/search`:
```bash
curl -X POST http://localhost:8010/music/search \
  -H "Content-Type: application/json" \
  -d '{"query": "nombre de la canción", "limit": 5}'
```

### ¿Qué información puedo obtener de una canción?

**R:** Puedes obtener:
- Información básica (nombre, artista, álbum)
- Análisis musical (tonalidad, tempo, modo)
- Análisis técnico (energía, bailabilidad, valencia)
- Análisis de estructura (secciones, acordes)
- Coaching musical personalizado

### ¿Puedo analizar canciones que no están en Spotify?

**R:** No, el sistema requiere que las canciones estén disponibles en Spotify para poder analizarlas.

### ¿Cómo obtengo recomendaciones de canciones similares?

**R:** Usa el endpoint `/music/recommendations/{track_id}`:
```bash
curl http://localhost:8010/music/recommendations/4uLU6hMCjMI75M1A2tKUQC
```

## 🔐 Seguridad y Autenticación

### ¿Es seguro guardar mis credenciales de Spotify en .env?

**R:** Sí, siempre y cuando:
- No commitees el archivo `.env` al repositorio
- Uses variables de entorno del sistema en producción
- Consideres usar un secret manager (AWS Secrets Manager, etc.) para producción

### ¿Necesito autenticación para usar la API?

**R:** No, la mayoría de endpoints no requieren autenticación. Algunos endpoints avanzados pueden requerir tokens de Spotify.

### ¿Cómo protejo mi API de abuso?

**R:** El sistema incluye rate limiting por defecto. Puedes configurarlo en `.env`:
```env
RATE_LIMIT_MAX_REQUESTS=100
RATE_LIMIT_WINDOW_SECONDS=60
```

## 🐛 Problemas y Errores

### El servidor no inicia, ¿qué hago?

**R:** Revisa:
1. Que Python esté instalado correctamente
2. Que las dependencias estén instaladas: `pip install -r requirements.txt`
3. Que el puerto 8010 no esté en uso
4. Los logs para ver el error específico

### Recibo error 401 de Spotify, ¿por qué?

**R:** Esto generalmente significa:
- Las credenciales en `.env` son incorrectas
- El Client Secret ha expirado (regenera uno nuevo)
- El Redirect URI no coincide con el configurado en Spotify Dashboard

### ¿Por qué las búsquedas son lentas?

**R:** Puede ser por:
- Falta de caché (configura Redis)
- Rate limiting de Spotify
- Conexión a internet lenta
- El servidor está sobrecargado

### ¿Cómo veo los logs?

**R:** Los logs están en `logs/app.log` o puedes verlos en tiempo real:
```bash
tail -f logs/app.log  # Linux/Mac
type logs\app.log     # Windows
```

## 🔄 Actualización y Mantenimiento

### ¿Cómo actualizo el sistema?

**R:** 
```bash
# Actualizar dependencias
pip install -r requirements.txt --upgrade

# Si hay cambios en el código
git pull  # Si usas Git
```

### ¿Cómo hago backup de mis datos?

**R:** Si usas base de datos:
```bash
# SQLite
cp music_analyzer.db music_analyzer.db.backup

# PostgreSQL
pg_dump music_analyzer > backup.sql
```

### ¿Puedo usar el sistema en producción?

**R:** Sí, pero asegúrate de:
- Configurar variables de entorno de producción
- Usar un servidor web (Nginx) como reverse proxy
- Configurar SSL/TLS
- Habilitar logging y monitoreo
- Configurar backups

## 💻 Desarrollo

### ¿Cómo agrego un nuevo endpoint?

**R:** 
1. Crea el schema en `api/v1/schemas/`
2. Crea el controller en `api/v1/controllers/`
3. Registra la ruta en `api/v1/routes.py`
4. Crea el use case si es necesario en `application/use_cases/`

### ¿Cómo escribo tests?

**R:** 
```python
# tests/test_example.py
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
```

Ejecuta con: `pytest tests/`

### ¿Cómo contribuyo al proyecto?

**R:** 
1. Fork el repositorio
2. Crea una rama para tu feature
3. Haz tus cambios
4. Escribe tests
5. Crea un Pull Request

## 📊 Rendimiento

### ¿Cuántas requests por segundo puede manejar?

**R:** Depende de:
- Hardware del servidor
- Si Redis está configurado
- Complejidad de las requests
- Rate limits de Spotify

Típicamente: 50-200 requests/segundo con configuración estándar.

### ¿Cómo mejoro el rendimiento?

**R:**
1. Configura Redis para caché
2. Usa múltiples workers: `uvicorn main:app --workers 4`
3. Optimiza las queries a la base de datos
4. Usa un load balancer para distribuir carga

### ¿El sistema escala horizontalmente?

**R:** Sí, puedes:
- Usar múltiples instancias detrás de un load balancer
- Configurar Redis compartido para caché
- Usar una base de datos compartida (PostgreSQL)

## 🌐 Despliegue

### ¿Puedo desplegar en AWS/GCP/Azure?

**R:** Sí, el sistema es compatible con todos los principales proveedores de cloud. Ver [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) para detalles.

### ¿Necesito Docker?

**R:** No es obligatorio, pero es recomendado para:
- Consistencia entre entornos
- Facilidad de despliegue
- Aislamiento de dependencias

### ¿Cómo configuro SSL/TLS?

**R:** Usa un reverse proxy (Nginx) con Let's Encrypt:
```bash
sudo certbot --nginx -d tu-dominio.com
```

## 📚 Recursos Adicionales

### ¿Dónde encuentro más información?

**R:**
- **Documentación de API**: http://localhost:8010/docs
- **Spotify API Docs**: https://developer.spotify.com/documentation/web-api/
- **FastAPI Docs**: https://fastapi.tiangolo.com/

### ¿Hay una comunidad o soporte?

**R:** 
- Revisa los issues en el repositorio
- Consulta [TROUBLESHOOTING_GUIDE.md](TROUBLESHOOTING_GUIDE.md)
- Revisa los logs para errores específicos

## ❓ ¿Tu pregunta no está aquí?

Si tu pregunta no está en esta lista:

1. Revisa [TROUBLESHOOTING_GUIDE.md](TROUBLESHOOTING_GUIDE.md)
2. Consulta la documentación de la API en `/docs`
3. Revisa los logs del servidor
4. Busca en los issues del repositorio

---

**Última actualización**: 2025  
**Versión**: 2.21.0






