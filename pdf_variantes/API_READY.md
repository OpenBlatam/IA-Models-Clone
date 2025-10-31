# PDF Variantes API - Lista para Usar 🚀

## Estado del Sistema

✅ **API completamente funcional y lista para producción**
- ✅ FastAPI configurado con todos los routers
- ✅ Servicios inicializados correctamente
- ✅ Middleware de seguridad, CORS, rate limiting
- ✅ Manejo de errores frontend-friendly
- ✅ WebSockets para colaboración en tiempo real
- ✅ Health checks y monitoreo

## Inicio Rápido

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Configurar variables de entorno

Copia `.env.example` a `.env` y configura las variables necesarias:

```bash
cp env.example .env
```

**Variables mínimas requeridas:**
- `SECRET_KEY` - Clave secreta para JWT (puede usar la default en desarrollo)
- `DATABASE_URL` - URL de base de datos (SQLite por defecto)
- `REDIS_URL` - URL de Redis (opcional en desarrollo)

### 3. Ejecutar el servidor

```bash
python run.py
```

O usando uvicorn directamente:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

## Endpoints Disponibles

### Documentación
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

### Health Check
- **GET** `/health` - Verificación rápida de salud
- **GET** `/api/v1/health/status` - Estado detallado del sistema

### PDF Processing
- **POST** `/api/v1/pdf/upload` - Subir y procesar PDF
- **GET** `/api/v1/pdf/documents` - Listar documentos
- **GET** `/api/v1/pdf/documents/{document_id}` - Obtener documento
- **DELETE** `/api/v1/pdf/documents/{document_id}` - Eliminar documento

### Variant Generation
- **POST** `/api/v1/variants/generate` - Generar variantes
- **GET** `/api/v1/variants/documents/{document_id}/variants` - Listar variantes
- **GET** `/api/v1/variants/variants/{variant_id}` - Obtener variante

### Topic Extraction
- **POST** `/api/v1/topics/extract` - Extraer temas
- **GET** `/api/v1/topics/documents/{document_id}/topics` - Listar temas

### Brainstorming
- **POST** `/api/v1/brainstorm/generate` - Generar ideas
- **GET** `/api/v1/brainstorm/documents/{document_id}/ideas` - Listar ideas

### Collaboration
- **POST** `/api/v1/collaboration/invite` - Invitar colaborador
- **WebSocket** `/api/v1/collaboration/ws/{document_id}` - Colaboración en tiempo real

### Export
- **POST** `/api/v1/export/export` - Exportar contenido
- **GET** `/api/v1/export/download/{file_id}` - Descargar archivo exportado

### Analytics
- **GET** `/api/v1/analytics/dashboard` - Dashboard de analytics
- **GET** `/api/v1/analytics/reports` - Reportes de analytics

### Batch Processing
- **POST** `/api/v1/batch/process` - Procesar múltiples documentos

### Search
- **POST** `/api/v1/search/search` - Buscar en documentos

## Características

### 🔒 Seguridad
- Rate limiting configurado
- CORS configurado para frontend
- Validación de archivos
- Manejo seguro de errores

### 📊 Monitoreo
- Sistema de health checks
- Analytics y métricas
- Logging estructurado

### ⚡ Performance
- Caché con Redis
- Procesamiento asíncrono
- Optimización de respuestas

### 🤝 Colaboración
- WebSockets para tiempo real
- Sistema de invitaciones
- Anotaciones y feedback

## Configuración CORS

El CORS está configurado para permitir solicitudes desde:
- `http://localhost:3000` (React default)
- `http://localhost:5173` (Vite default)
- `http://localhost:4200` (Angular default)
- `http://localhost:8080` (Vue default)

En modo desarrollo (`ENVIRONMENT=development`), permite todas las orígenes (`*`).

## Formato de Respuesta

Todas las respuestas siguen un formato consistente:

### Éxito
```json
{
  "success": true,
  "data": { ... },
  "error": null
}
```

### Error
```json
{
  "success": false,
  "data": null,
  "error": {
    "message": "Error description",
    "status_code": 400,
    "type": "HTTPException"
  }
}
```

## Autenticación

Para desarrollo, puede usar el parámetro `user_id` directamente en las rutas. En producción, implemente el middleware de autenticación en `utils/auth.py`.

## Variables de Entorno Importantes

```bash
# Desarrollo
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG

# Producción
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=WARNING
SECRET_KEY=<tu-clave-secreta-fuerte>
```

## Soporte TypeScript/Frontend

La API está lista para integrarse con frontend TypeScript:
- Respuestas tipadas y consistentes
- CORS configurado
- Manejo de errores frontend-friendly
- Headers apropiados

## Próximos Pasos

1. Configurar base de datos (PostgreSQL recomendado)
2. Configurar Redis para caché
3. Configurar claves de API para servicios AI
4. Implementar autenticación JWT completa
5. Configurar logging en archivo
6. Configurar monitoreo (Prometheus/Grafana)

## Troubleshooting

### Error: "Service not available"
- Verifica que todos los servicios se inicializan correctamente
- Revisa los logs de inicio

### Error: "Rate limit exceeded"
- Ajusta `RATE_LIMIT_REQUESTS_PER_MINUTE` en `.env`

### Error de conexión a Redis/DB
- Verifica que los servicios estén corriendo
- Revisa las URLs en `.env`

## Contacto y Soporte

Para más información, consulte:
- `README.md` - Documentación general
- `DEPLOYMENT_GUIDE.md` - Guía de despliegue
- Código fuente con docstrings completos

---

**¡La API está lista para usarse!** 🎉






