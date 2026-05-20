# 📡 API Endpoints - AI Tutor Educacional

## Resumen Completo de Endpoints

### Core Endpoints (8)

#### Preguntas y Respuestas
- `POST /api/tutor/ask` - Hacer una pregunta al tutor
- `POST /api/tutor/explain` - Explicar un concepto
- `POST /api/tutor/exercises` - Generar ejercicios
- `POST /api/tutor/quiz` - Generar un quiz

#### Evaluación
- `POST /api/tutor/evaluate/answer` - Evaluar una respuesta
- `POST /api/tutor/evaluate/quiz` - Evaluar un quiz completo

#### Conversaciones
- `GET /api/tutor/conversation/{conversation_id}` - Obtener conversación
- `DELETE /api/tutor/conversation/{conversation_id}` - Eliminar conversación

### Recomendaciones (3)

- `GET /api/tutor/recommendations/learning-path/{student_id}` - Ruta de aprendizaje
- `GET /api/tutor/recommendations/practice/{student_id}` - Práctica recomendada
- `GET /api/tutor/recommendations/next-topic/{student_id}` - Siguiente tema

### Reportes (2)

- `GET /api/tutor/reports/student/{student_id}` - Reporte de estudiante
- `POST /api/tutor/reports/export` - Exportar reportes

### Gamificación (3)

- `GET /api/tutor/gamification/profile/{student_id}` - Perfil de gamificación
- `POST /api/tutor/gamification/action` - Registrar acción
- `GET /api/tutor/gamification/leaderboard` - Leaderboard

### Notificaciones (3)

- `GET /api/tutor/notifications/{student_id}` - Obtener notificaciones
- `POST /api/tutor/notifications/{student_id}/read/{notification_id}` - Marcar como leída
- `POST /api/tutor/notifications/{student_id}/read-all` - Marcar todas como leídas

### Dashboard (4)

- `GET /api/tutor/dashboard/overview` - Vista general
- `GET /api/tutor/dashboard/activity-timeline` - Timeline de actividad
- `GET /api/tutor/dashboard/engagement` - Métricas de engagement
- `GET /api/tutor/dashboard/insights` - Insights y análisis

### Autenticación (3)

- `POST /api/tutor/auth/register` - Registrar usuario
- `POST /api/tutor/auth/login` - Iniciar sesión
- `POST /api/tutor/auth/logout` - Cerrar sesión

### Base de Datos (3)

- `GET /api/tutor/database/students` - Listar estudiantes
- `GET /api/tutor/database/student/{student_id}/stats` - Estadísticas de estudiante
- `POST /api/tutor/database/backup` - Crear backup

### Webhooks (3)

- `POST /api/tutor/webhooks/register` - Registrar webhook
- `DELETE /api/tutor/webhooks/{webhook_id}` - Eliminar webhook
- `GET /api/tutor/webhooks` - Listar webhooks

### LMS (3)

- `GET /api/tutor/lms/courses` - Obtener cursos
- `POST /api/tutor/lms/sync/student` - Sincronizar estudiante
- `POST /api/tutor/lms/sync/grades` - Sincronizar calificaciones

### Sistema (3)

- `GET /api/tutor/metrics` - Métricas del sistema
- `DELETE /api/tutor/cache` - Limpiar cache
- `GET /api/tutor/health` - Health check

## Total: 35+ Endpoints

## Documentación Interactiva

Accede a la documentación completa en: `http://localhost:8000/docs`

---

**Versión:** 1.11.0  
**Última Actualización:** 2024-12-XX




