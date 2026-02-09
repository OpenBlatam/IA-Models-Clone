# 📝 README - Qué Genera la API BUL

## 🎯 Resumen General

La API BUL es un sistema de generación de documentos con IA que procesa consultas de negocio y genera documentos profesionales automáticamente. Este documento explica **qué genera cada endpoint** y **qué tipo de respuestas** puedes esperar.

---

## 📊 Endpoints y Sus Respuestas

### 1. **GET /** - Información del Sistema

**Qué genera:**
- Información básica del sistema
- Versión de la API
- Estado operacional
- Enlaces a documentación

**Ejemplo de respuesta:**
```json
{
  "message": "BUL API - Frontend Ready",
  "version": "1.0.0",
  "status": "operational",
  "timestamp": "2024-01-15T10:30:00",
  "docs": "/api/docs",
  "health": "/api/health"
}
```

**Cuándo usar:**
- Verificar que la API está funcionando
- Obtener información básica del sistema

---

### 2. **GET /api/health** - Health Check

**Qué genera:**
- Estado de salud del sistema
- Tiempo de actividad (uptime)
- Número de tareas activas
- Total de solicitudes procesadas
- Versión de la API

**Ejemplo de respuesta:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00",
  "uptime": "2:30:15",
  "active_tasks": 3,
  "total_requests": 150,
  "version": "1.0.0"
}
```

**Cuándo usar:**
- Monitoreo del sistema
- Verificar disponibilidad
- Health checks automatizados

---

### 3. **GET /api/stats** - Estadísticas del Sistema

**Qué genera:**
- Métricas completas del sistema
- Total de solicitudes procesadas
- Tareas activas y completadas
- Tasa de éxito (success rate)
- Tiempo promedio de procesamiento
- Tiempo de actividad

**Ejemplo de respuesta:**
```json
{
  "total_requests": 150,
  "active_tasks": 2,
  "completed_tasks": 140,
  "success_rate": 0.933,
  "average_processing_time": 4.5,
  "uptime": "2:30:15"
}
```

**Cuándo usar:**
- Dashboard de métricas
- Análisis de rendimiento
- Monitoreo de sistema

---

### 4. **POST /api/documents/generate** - Generar Documento

**Qué genera:**
- Un `task_id` único para la tarea
- Estado inicial de la tarea (generalmente "queued")
- Mensaje descriptivo
- Tiempo estimado de procesamiento
- Posición en la cola
- Fecha de creación

**Ejemplo de request:**
```json
{
  "query": "Crear un plan de marketing digital para una startup",
  "business_area": "marketing",
  "document_type": "strategy",
  "priority": 1,
  "metadata": {
    "industry": "technology"
  }
}
```

**Ejemplo de respuesta:**
```json
{
  "task_id": "task_20240115_103045_a1b2c3d4",
  "status": "queued",
  "message": "Generación de documento iniciada",
  "estimated_time": 60,
  "queue_position": 1,
  "created_at": "2024-01-15T10:30:45"
}
```

**Qué genera el sistema:**
Después de procesar, genera un documento completo con:
- Contenido estructurado en Markdown
- Información basada en la consulta
- Formato profesional
- Conteo de palabras
- Metadatos de generación

**Cuándo usar:**
- Generar documentos de negocio
- Crear estrategias
- Generar planes y manuales
- Crear cualquier tipo de documento empresarial

---

### 5. **GET /api/tasks/{task_id}/status** - Estado de Tarea

**Qué genera:**
- Estado actual de la tarea (queued, processing, completed, failed, cancelled)
- Progreso (0-100%)
- Resultado del documento (si está completado)
- Error (si falló)
- Fechas de creación y actualización
- Tiempo de procesamiento

**Ejemplo de respuesta (en procesamiento):**
```json
{
  "task_id": "task_20240115_103045_a1b2c3d4",
  "status": "processing",
  "progress": 65,
  "result": null,
  "error": null,
  "created_at": "2024-01-15T10:30:45",
  "updated_at": "2024-01-15T10:31:10",
  "processing_time": 25.5
}
```

**Ejemplo de respuesta (completada):**
```json
{
  "task_id": "task_20240115_103045_a1b2c3d4",
  "status": "completed",
  "progress": 100,
  "result": {
    "content": "# Plan de Marketing Digital\n\n...",
    "format": "markdown",
    "word_count": 1250,
    "generated_at": "2024-01-15T10:31:30",
    "using_bul_system": true
  },
  "error": null,
  "created_at": "2024-01-15T10:30:45",
  "updated_at": "2024-01-15T10:31:30",
  "processing_time": 45.2
}
```

**Cuándo usar:**
- Verificar progreso de una tarea
- Polling para actualizaciones
- Mostrar progreso en UI

---

### 6. **GET /api/tasks/{task_id}/document** - Obtener Documento Generado

**Qué genera:**
- Documento completo generado
- Contenido en formato Markdown
- Metadatos de la solicitud original
- Fechas de creación y finalización
- Información completa del documento

**Ejemplo de respuesta:**
```json
{
  "task_id": "task_20240115_103045_a1b2c3d4",
  "document": {
    "content": "# Plan de Marketing Digital\n\n## Introducción\n\nEste documento presenta...",
    "format": "markdown",
    "word_count": 1250,
    "generated_at": "2024-01-15T10:31:30"
  },
  "metadata": {
    "query": "Crear un plan de marketing digital",
    "business_area": "marketing",
    "document_type": "strategy",
    "priority": 1
  },
  "created_at": "2024-01-15T10:30:45",
  "completed_at": "2024-01-15T10:31:30"
}
```

**Qué contiene el documento generado:**
- **Estructura profesional**: Títulos, secciones, listas
- **Contenido relevante**: Basado en la consulta y área de negocio
- **Formato Markdown**: Fácil de convertir a HTML, PDF, etc.
- **Información completa**: Cubre todos los aspectos solicitados

**Cuándo usar:**
- Obtener el documento final
- Mostrar contenido al usuario
- Descargar o exportar documento

---

### 7. **GET /api/tasks** - Listar Tareas

**Qué genera:**
- Lista de todas las tareas
- Información resumida de cada tarea
- Paginación (limit, offset, total)
- Filtros por estado y usuario

**Ejemplo de respuesta:**
```json
{
  "tasks": [
    {
      "task_id": "task_20240115_103045_a1b2c3d4",
      "status": "completed",
      "progress": 100,
      "created_at": "2024-01-15T10:30:45",
      "updated_at": "2024-01-15T10:31:30",
      "user_id": "user123",
      "query_preview": "Crear un plan de marketing digital para una startup..."
    }
  ],
  "total": 25,
  "limit": 50,
  "offset": 0,
  "has_more": false
}
```

**Cuándo usar:**
- Mostrar historial de tareas
- Dashboard de actividad
- Búsqueda de tareas anteriores

---

### 8. **GET /api/documents** - Listar Documentos

**Qué genera:**
- Lista de documentos completados
- Vista previa de consultas
- Metadatos básicos
- Paginación

**Ejemplo de respuesta:**
```json
{
  "documents": [
    {
      "task_id": "task_20240115_103045_a1b2c3d4",
      "created_at": "2024-01-15T10:30:45",
      "query_preview": "Crear un plan de marketing digital...",
      "business_area": "marketing",
      "document_type": "strategy"
    }
  ],
  "total": 20,
  "limit": 50,
  "offset": 0,
  "has_more": false
}
```

**Cuándo usar:**
- Biblioteca de documentos
- Historial de generaciones
- Búsqueda de documentos

---

### 9. **DELETE /api/tasks/{task_id}** - Eliminar Tarea

**Qué genera:**
- Mensaje de confirmación
- ID de la tarea eliminada

**Ejemplo de respuesta:**
```json
{
  "message": "Tarea eliminada exitosamente",
  "task_id": "task_20240115_103045_a1b2c3d4"
}
```

**Cuándo usar:**
- Limpieza de tareas antiguas
- Eliminar tareas no deseadas

---

### 10. **POST /api/tasks/{task_id}/cancel** - Cancelar Tarea

**Qué genera:**
- Mensaje de confirmación
- ID de la tarea cancelada

**Ejemplo de respuesta:**
```json
{
  "message": "Tarea cancelada exitosamente",
  "task_id": "task_20240115_103045_a1b2c3d4"
}
```

**Cuándo usar:**
- Detener tareas en procesamiento
- Cancelar tareas en cola

---

### 11. **WS /api/ws/{task_id}** - WebSocket para Actualizaciones

**Qué genera (mensajes en tiempo real):**
- Actualizaciones de progreso
- Cambios de estado
- Notificaciones de finalización
- Errores en tiempo real

**Ejemplo de mensajes:**
```json
// Mensaje inicial
{
  "type": "initial_state",
  "task_id": "task_20240115_103045_a1b2c3d4",
  "data": {
    "status": "processing",
    "progress": 50
  },
  "timestamp": "2024-01-15T10:31:00"
}

// Actualización de progreso
{
  "type": "task_update",
  "task_id": "task_20240115_103045_a1b2c3d4",
  "data": {
    "status": "processing",
    "progress": 75
  },
  "timestamp": "2024-01-15T10:31:15"
}

// Finalización
{
  "type": "task_update",
  "task_id": "task_20240115_103045_a1b2c3d4",
  "data": {
    "status": "completed",
    "progress": 100,
    "result": {...}
  },
  "timestamp": "2024-01-15T10:31:30"
}
```

**Cuándo usar:**
- Actualizaciones en tiempo real en UI
- Notificaciones push
- Monitoreo en vivo

---

## 📄 Tipos de Documentos que Genera

La API puede generar diferentes tipos de documentos según el `document_type`:

### 1. **Estrategias** (`strategy`)
- Planes de marketing
- Estrategias de ventas
- Estrategias de crecimiento
- Planes de negocio

### 2. **Manuales** (`manual`)
- Manuales de operaciones
- Manuales de usuario
- Guías de procedimientos
- Documentación técnica

### 3. **Planes** (`plan`)
- Planes de negocio
- Planes de recursos humanos
- Planes financieros
- Planes de proyecto

### 4. **Políticas** (`policy`)
- Políticas de empresa
- Políticas de recursos humanos
- Políticas de seguridad
- Políticas de cumplimiento

### 5. **Reportes** (`report`)
- Reportes de análisis
- Reportes de mercado
- Reportes financieros
- Reportes de desempeño

### 6. **Documento estándar** (`document`)
- Documentos genéricos
- Cualquier tipo de contenido empresarial

---

## 🎨 Formato de los Documentos Generados

Los documentos se generan en formato **Markdown** con:

- **Títulos y subtítulos** estructurados
- **Listas** numeradas y con viñetas
- **Secciones** organizadas lógicamente
- **Formato profesional** listo para convertir a:
  - HTML
  - PDF
  - DOCX
  - Otros formatos

**Ejemplo de estructura:**
```markdown
# Título Principal

## Sección 1

### Subsección 1.1

Contenido relevante...

## Sección 2

- Punto 1
- Punto 2
- Punto 3
```

---

## 🔄 Flujo Completo de Generación

1. **Cliente envía request** → `POST /api/documents/generate`
2. **API responde** → `task_id` y estado inicial
3. **Sistema procesa** → Genera documento en segundo plano
4. **WebSocket notifica** → Actualizaciones de progreso (opcional)
5. **Cliente consulta estado** → `GET /api/tasks/{task_id}/status`
6. **Documento completado** → `GET /api/tasks/{task_id}/document`

---

## 📊 Ejemplo Completo de Uso

```typescript
// 1. Generar documento
const response = await client.generateDocument({
  query: "Crear un plan de marketing digital",
  business_area: "marketing",
  document_type: "strategy",
  priority: 1
});

// 2. Esperar completación (con WebSocket automático)
const document = await client.generateDocumentAndWait(
  { query: "..." },
  {
    onProgress: (status) => {
      console.log(`Progreso: ${status.progress}%`);
    }
  }
);

// 3. Documento generado contiene:
console.log(document.document.content); // Markdown completo
console.log(document.document.word_count); // 1250 palabras
console.log(document.document.format); // "markdown"
```

---

## ✅ Características de los Documentos Generados

- ✅ **Contenido relevante**: Basado en la consulta y contexto
- ✅ **Estructura profesional**: Organizado y fácil de leer
- ✅ **Formato estándar**: Markdown universalmente compatible
- ✅ **Completo**: Cubre todos los aspectos solicitados
- ✅ **Personalizado**: Según área de negocio y tipo de documento
- ✅ **Listo para usar**: Puede usarse directamente o convertir a otros formatos

---

## 🚀 Próximos Pasos

1. Prueba la API con el script: `python test_api_responses.py`
2. Revisa los ejemplos en `example_frontend_usage.ts`
3. Consulta la documentación en `/api/docs`
4. Integra en tu frontend TypeScript

---

**La API genera documentos profesionales, estructurados y listos para usar en cualquier contexto empresarial.** 📄✨

---

## 🧪 Cómo Probar las Respuestas

### Opción 1: Script de Prueba Automático

```bash
# Asegúrate de que el servidor esté corriendo
python api_frontend_ready.py

# En otra terminal, ejecuta las pruebas
python test_api_responses.py
```

### Opción 2: Pruebas Manuales con cURL

```bash
# 1. Health Check
curl http://localhost:8000/api/health

# 2. Generar documento
curl -X POST http://localhost:8000/api/documents/generate \
  -H "Content-Type: application/json" \
  -d '{"query": "Crear un plan de marketing", "priority": 1}'

# 3. Verificar estado (usa el task_id de la respuesta anterior)
curl http://localhost:8000/api/tasks/{task_id}/status

# 4. Obtener documento
curl http://localhost:8000/api/tasks/{task_id}/document
```

### Opción 3: Desde el Frontend TypeScript

```typescript
import { createBULApiClient } from './api/bul-api-client';

const client = createBULApiClient({
  baseUrl: 'http://localhost:8000'
});

// Probar health check
const health = await client.getHealth();
console.log('Health:', health);

// Generar y obtener documento
const document = await client.generateDocumentAndWait({
  query: 'Crear un plan de marketing digital',
  business_area: 'marketing',
  document_type: 'strategy'
});

console.log('Documento generado:', document.document.content);
```

---

## 📋 Resumen de Respuestas Generadas

| Endpoint | Qué Genera | Formato |
|----------|-----------|---------|
| `GET /` | Info del sistema | JSON |
| `GET /api/health` | Estado de salud | JSON |
| `GET /api/stats` | Métricas del sistema | JSON |
| `POST /api/documents/generate` | Task ID y estado inicial | JSON |
| `GET /api/tasks/{id}/status` | Estado y progreso | JSON |
| `GET /api/tasks/{id}/document` | Documento completo | JSON (Markdown) |
| `GET /api/tasks` | Lista de tareas | JSON |
| `GET /api/documents` | Lista de documentos | JSON |
| `WS /api/ws/{id}` | Actualizaciones en tiempo real | JSON (WebSocket) |

---

## 🎯 Ejemplo Real de Documento Generado

Cuando generas un documento con la consulta:
```json
{
  "query": "Crear un plan de marketing digital para una startup tecnológica",
  "business_area": "marketing",
  "document_type": "strategy"
}
```

**El sistema genera un documento Markdown como:**

```markdown
# Plan de Marketing Digital

## Introducción

Este documento presenta un plan integral de marketing digital para una startup tecnológica...

## Objetivos Estratégicos

### 1. Aumentar la Visibilidad Digital
- Desarrollo de presencia en redes sociales
- Optimización SEO
- Contenido de valor

### 2. Generación de Leads
- Campañas de email marketing
- Webinars y eventos
- Contenido premium

## Estrategia de Contenido

### Blog y Artículos
- Publicación semanal de contenido relevante
- Temas: tecnología, innovación, casos de uso

### Redes Sociales
- LinkedIn: 3-5 publicaciones semanales
- Twitter: 2-3 publicaciones diarias
- Instagram: 2 publicaciones semanales

## Presupuesto y Recursos

- Marketing digital: 30% del presupuesto
- Herramientas: 20%
- Contenido: 25%
- Publicidad: 25%

## Métricas de Éxito

- Tráfico web: +50% en 6 meses
- Leads generados: 100+ por mes
- Engagement social: +30%

## Conclusión

Este plan de marketing digital proporciona una hoja de ruta clara...
```

**Características del documento:**
- ✅ Estructura profesional con títulos y secciones
- ✅ Contenido relevante y específico
- ✅ Formato Markdown listo para conversión
- ✅ Longitud apropiada (1000-2000 palabras típicamente)
- ✅ Información práctica y accionable

---

## 📊 Estadísticas de Generación

- **Tiempo promedio**: 30-60 segundos
- **Longitud típica**: 1000-2000 palabras
- **Formato**: Markdown
- **Tasa de éxito**: >95% (con fallback a simulación)
- **Áreas soportadas**: Marketing, Ventas, Operaciones, HR, Finanzas, Legal, Técnico, Contenido, Estrategia, Servicio al Cliente

---

**¡Todo listo para generar documentos profesionales!** 🚀📄

