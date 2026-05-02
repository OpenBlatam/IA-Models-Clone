# 🚀 BUL API - Quick Start para Frontend TypeScript

## ✅ Estado: Listo para usar

La API está completamente configurada y lista para consumirse desde un frontend TypeScript.

## 📁 Archivos Creados

### Backend (Python)
- ✅ `api_frontend_ready.py` - Servidor FastAPI principal con CORS configurado
- ✅ `start_api_frontend.bat` - Script de inicio para Windows
- ✅ `start_api_frontend.sh` - Script de inicio para Linux/Mac

### Frontend (TypeScript)
- ✅ `frontend_types.ts` - Tipos TypeScript para todos los endpoints
- ✅ `bul-api-client.ts` - Cliente TypeScript completo con métodos helper
- ✅ `example_frontend_usage.ts` - Ejemplos de uso
- ✅ `README_FRONTEND.md` - Documentación completa

## 🎯 Inicio Rápido

### 1. Iniciar el servidor

**Windows:**
```bash
start_api_frontend.bat
```

**Linux/Mac:**
```bash
chmod +x start_api_frontend.sh
./start_api_frontend.sh
```

**O manualmente:**
```bash
python api_frontend_ready.py --host 0.0.0.0 --port 8000
```

### 2. Verificar que funciona

Abre en el navegador:
- API: http://localhost:8000
- Documentación: http://localhost:8000/api/docs
- Health Check: http://localhost:8000/api/health

### 3. Usar en el frontend

```typescript
// 1. Copiar archivos TypeScript a tu proyecto frontend
// frontend/src/api/bul-api-client.ts
// frontend/src/api/frontend_types.ts

// 2. Importar y usar
import { createBULApiClient } from './api/bul-api-client';

const client = createBULApiClient({
  baseUrl: 'http://localhost:8000'
});

// 3. Generar documento
const document = await client.generateDocumentAndWait({
  query: 'Crear un plan de negocios',
  priority: 1
});

console.log(document.document.content);
```

## 📋 Endpoints Disponibles

### Sistema
- `GET /` - Info del sistema
- `GET /api/health` - Health check
- `GET /api/stats` - Estadísticas

### Documentos
- `POST /api/documents/generate` - Generar documento
- `GET /api/tasks/{task_id}/document` - Obtener documento
- `GET /api/documents` - Listar documentos

### Tareas
- `GET /api/tasks/{task_id}/status` - Estado de tarea
- `GET /api/tasks` - Listar tareas
- `DELETE /api/tasks/{task_id}` - Eliminar tarea
- `POST /api/tasks/{task_id}/cancel` - Cancelar tarea

## 🔧 Características

✅ **CORS configurado** - Listo para consumo desde cualquier frontend  
✅ **Tipos TypeScript** - Tipado completo para todos los endpoints  
✅ **Cliente completo** - Métodos helper para polling, espera, etc.  
✅ **Documentación** - Swagger UI en `/api/docs`  
✅ **Manejo de errores** - Timeouts y errores HTTP manejados  
✅ **Polling automático** - Método `waitForTaskCompletion` incluido  

## 📖 Ejemplos

Ver `example_frontend_usage.ts` para ejemplos completos de:
- Generación básica
- Polling automático
- Múltiples documentos
- Listar tareas/documentos
- Health check y estadísticas
- Cancelar tareas
- React Hook (simulado)

## 🛡️ Seguridad

### Desarrollo
- CORS permite todos los orígenes (`*`)
- Sin autenticación requerida

### Producción
Actualiza en `api_frontend_ready.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://tu-dominio.com"],  # Especificar dominios
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 📚 Documentación Completa

- `README_FRONTEND.md` - Guía completa con ejemplos
- `http://localhost:8000/api/docs` - Swagger UI interactivo
- `http://localhost:8000/api/redoc` - ReDoc

## ✅ Próximos Pasos

1. ✅ Servidor funcionando
2. ✅ Copiar archivos TypeScript al frontend
3. ✅ Configurar `baseUrl` en el cliente
4. ✅ Probar generación de documentos
5. ✅ Integrar en tu aplicación

## 🆘 Problemas Comunes

### "Connection refused"
- Verifica que el servidor esté corriendo
- Verifica el puerto (default: 8000)
- Verifica la URL en el cliente

### "CORS error"
- Verifica que CORS esté configurado (ya está por defecto)
- En producción, actualiza `allow_origins`

### "Timeout"
- Aumenta `timeout` en la configuración del cliente
- Verifica que el servidor esté respondiendo

## 🎉 ¡Listo!

La API está completamente funcional y lista para usar desde TypeScript.
































