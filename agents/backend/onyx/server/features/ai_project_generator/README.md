# AI Project Generator 🚀

Generador automático de proyectos de IA que crea la estructura completa de backend y frontend basándose en una descripción del usuario. Funciona de forma continua sin parar.

## ✨ Características Principales

- ✅ **Generación Automática Inteligente**
  - Backend completo (FastAPI) con estructura modular
  - Frontend completo (React + TypeScript + Vite + Tailwind)
  - Detección automática de tipo de IA (chat, vision, audio, NLP, etc.)
  - Detección de características necesarias (auth, database, websocket, etc.)

- ✅ **Generación Continua**
  - Procesa proyectos automáticamente sin parar
  - Sistema de cola con prioridades
  - Persistencia de estado
  - Monitoreo en tiempo real

- ✅ **API REST Completa**
  - Endpoints para generar, listar, monitorear proyectos
  - Estadísticas y métricas
  - Gestión de cola

- ✅ **Código Inteligente**
  - Genera código según el tipo de IA detectado
  - Incluye WebSocket si es necesario
  - Soporte para file upload
  - Configuración de cache, queue, database según necesidad
  - Dependencias automáticas según características

- ✅ **Listo para Producción**
  - Docker y docker-compose incluidos
  - Tests automáticos
  - Documentación generada
  - Estructura profesional

## 📦 Instalación

```bash
cd ai_project_generator
pip install -r requirements.txt
```

## 🚀 CI/CD Pipeline

This project includes a complete CI/CD pipeline with GitHub Actions:

- **Continuous Integration**: Automated testing, linting, and security scanning
- **Continuous Deployment**: Automated deployment to AWS EC2
- **Release Management**: Automated release creation
- **Security Scanning**: Comprehensive security checks

See [CI_CD_README.md](CI_CD_README.md) for complete CI/CD documentation.

### Quick CI/CD Setup

1. **Configure GitHub Secrets**:
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`

2. **Enable Workflows**: Workflows are automatically enabled

3. **Install Pre-commit Hooks**:
   ```bash
   pip install pre-commit
   pre-commit install
   ```

4. **Test Locally**:
   ```bash
   ./scripts/ci_cd/test.sh all
   ./scripts/ci_cd/build.sh docker
   ```

## 🏃 Uso

### Iniciar el servidor

```bash
python main.py
```

El servidor estará disponible en `http://localhost:8020`

## 🚀 CI/CD Pipeline

Este proyecto incluye un pipeline completo de CI/CD con GitHub Actions:

### Características CI/CD

- ✅ **Continuous Integration**: Testing, linting y security scanning automatizados
- ✅ **Continuous Deployment**: Despliegue automatizado a AWS EC2
- ✅ **Release Management**: Creación automática de releases
- ✅ **Security Scanning**: Escaneo de seguridad completo
- ✅ **Pre-commit Hooks**: Validación antes de commit

### Quick Start CI/CD

```bash
# 1. Instalar pre-commit hooks
pip install pre-commit
pre-commit install

# 2. Ejecutar tests localmente
make test

# 3. Ejecutar CI completo
make ci

# 4. Build y deploy
make build
make deploy
```

Ver [CI_CD_README.md](CI_CD_README.md) para documentación completa.

### Generar un proyecto

```bash
curl -X POST "http://localhost:8020/api/v1/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Un sistema de chat con IA que responde preguntas sobre programación",
    "project_name": "programming_chat_ai",
    "author": "Blatam Academy"
  }'
```

### Ver estado del generador

```bash
curl "http://localhost:8020/api/v1/status"
```

### Ver cola de proyectos

```bash
curl "http://localhost:8020/api/v1/queue"
```

## 📚 API Endpoints

### Generación
- `POST /api/v1/generate` - Genera un nuevo proyecto
  - Body: `{description, project_name?, author?, version?, priority?, backend_framework?, frontend_framework?, generate_tests?, include_docker?, include_docs?}`

### Estado y Monitoreo
- `GET /api/v1/status` - Estado del generador continuo
- `GET /api/v1/project/{project_id}` - Estado de un proyecto específico
- `GET /api/v1/queue` - Cola de proyectos pendientes
- `GET /api/v1/stats` - Estadísticas del generador
- `GET /api/v1/projects` - Lista proyectos generados (con filtros)

### Control
- `POST /api/v1/start` - Inicia el generador continuo
- `POST /api/v1/stop` - Detiene el generador continuo
- `DELETE /api/v1/project/{project_id}` - Elimina proyecto de la cola

### Exportación y Validación
- `POST /api/v1/export/zip` - Exporta proyecto a ZIP
- `POST /api/v1/export/tar` - Exporta proyecto a TAR
- `POST /api/v1/validate` - Valida un proyecto generado

### Despliegue
- `POST /api/v1/deploy/generate` - Genera configuraciones de despliegue (Vercel, Netlify, Railway, Heroku)

### Clonado y Templates
- `POST /api/v1/clone` - Clona un proyecto existente
- `POST /api/v1/templates/save` - Guarda un template personalizado
- `GET /api/v1/templates/list` - Lista todos los templates
- `GET /api/v1/templates/{name}` - Obtiene un template específico
- `DELETE /api/v1/templates/{name}` - Elimina un template

### Búsqueda Avanzada
- `GET /api/v1/search` - Busca proyectos con filtros avanzados
- `GET /api/v1/search/stats` - Estadísticas de búsqueda

### Webhooks
- `POST /api/v1/webhooks/register` - Registra un webhook
- `GET /api/v1/webhooks` - Lista webhooks registrados
- `DELETE /api/v1/webhooks/{id}` - Desregistra un webhook

### Cache
- `POST /api/v1/cache/clear` - Limpia el cache
- `GET /api/v1/cache/stats` - Estadísticas del cache

### Rate Limiting
- `GET /api/v1/rate-limit` - Información de rate limit

### Autenticación
- `POST /api/v1/auth/register` - Registrar usuario
- `POST /api/v1/auth/login` - Autenticar usuario
- `POST /api/v1/auth/api-key` - Crear API key

### Métricas
- `GET /api/v1/metrics` - Métricas del sistema
- `GET /api/v1/metrics/prometheus` - Métricas en formato Prometheus

### Backup y Restore
- `POST /api/v1/backup/create` - Crear backup completo
- `GET /api/v1/backup/list` - Listar backups
- `POST /api/v1/backup/restore` - Restaurar backup
- `DELETE /api/v1/backup/{name}` - Eliminar backup

### Dashboard y UI
- `GET /dashboard` - Dashboard web interactivo
- `POST /api/v1/dashboard/generate` - Generar dashboard

### Health y Versiones
- `GET /health` - Health check básico
- `GET /health/detailed` - Health check detallado
- `GET /api/version` - Información de versiones de API

### Notificaciones
- `POST /api/v1/notifications/register` - Registrar canal de notificaciones
- `GET /api/v1/notifications/channels` - Listar canales

### Plugins
- `POST /api/v1/plugins/register` - Registrar plugin
- `GET /api/v1/plugins` - Listar plugins
- `POST /api/v1/plugins/{name}/enable` - Activar plugin
- `POST /api/v1/plugins/{name}/disable` - Desactivar plugin

### Eventos
- `GET /api/v1/events/history` - Historial de eventos
- `GET /api/v1/events/stats` - Estadísticas de eventos

### Logs
- `GET /api/v1/logs/stats` - Estadísticas de logs

### WebSocket
- `WS /ws` - WebSocket para actualizaciones en tiempo real
- `WS /ws/project/{project_id}` - WebSocket para suscribirse a un proyecto específico

### Batch Generation
- `POST /api/v1/generate/batch` - Generar múltiples proyectos en batch

### Performance
- `GET /api/v1/performance/stats` - Estadísticas de performance
- `GET /api/v1/performance/optimize` - Sugerencias de optimización

### Streaming en Tiempo Real
- `GET /api/v1/stream/events` - Eventos de streaming
- `GET /api/v1/stream/stats` - Estadísticas de streaming

### Analytics
- `GET /api/v1/analytics/trends` - Tendencias de analytics
- `GET /api/v1/analytics/top-ai-types` - Tipos de IA más populares
- `GET /api/v1/analytics/performance` - Reporte de performance
- `GET /api/v1/analytics/frameworks` - Uso de frameworks
- `GET /api/v1/analytics/authors` - Estadísticas por autor
- `GET /api/v1/analytics/report` - Reporte completo

### Recomendaciones
- `GET /api/v1/recommendations` - Recomendaciones inteligentes
- `GET /api/v1/recommendations/features` - Features recomendadas
- `GET /api/v1/recommendations/framework` - Framework recomendado

### Documentación
- `POST /api/v1/documentation/generate` - Generar documentación automática

### Alertas
- `POST /api/v1/alerts/rule` - Crear regla de alerta
- `POST /api/v1/alerts/trigger` - Disparar alerta
- `GET /api/v1/alerts` - Obtener alertas activas
- `POST /api/v1/alerts/{id}/acknowledge` - Reconocer alerta
- `GET /api/v1/alerts/stats` - Estadísticas de alertas

### Scheduling
- `POST /api/v1/scheduler/task` - Programar tarea
- `GET /api/v1/scheduler/tasks` - Listar tareas
- `GET /api/v1/scheduler/task/{id}` - Estado de tarea
- `POST /api/v1/scheduler/task/{id}/enable` - Habilitar tarea
- `POST /api/v1/scheduler/task/{id}/disable` - Deshabilitar tarea

### Import/Export Avanzado
- `POST /api/v1/export/advanced` - Exportar proyecto avanzado
- `POST /api/v1/import` - Importar proyecto

### Machine Learning
- `POST /api/v1/ml/predict/time` - Predecir tiempo de generación
- `POST /api/v1/ml/predict/success` - Predecir probabilidad de éxito
- `POST /api/v1/ml/train` - Entrenar modelo ML
- `GET /api/v1/ml/stats` - Estadísticas del modelo

### Optimización Automática
- `POST /api/v1/optimize/analyze` - Analizar proyecto
- `POST /api/v1/optimize/config` - Optimizar configuración
- `POST /api/v1/optimize/recommendations` - Recomendaciones de optimización

### Testing Avanzado
- `POST /api/v1/testing/run` - Ejecutar tests de proyecto

### Deployment Automático
- `POST /api/v1/deploy/vercel` - Desplegar a Vercel
- `POST /api/v1/deploy/netlify` - Desplegar a Netlify
- `POST /api/v1/deploy/railway` - Desplegar a Railway
- `GET /api/v1/deploy/history` - Historial de despliegues

### Performance Analysis
- `GET /api/v1/performance/analyze` - Analizar performance
- `GET /api/v1/performance/predict/{operation}` - Predecir tiempo
- `GET /api/v1/resources/stats` - Estadísticas de recursos

### Reportes Avanzados
- `POST /api/v1/reports/generate/project` - Generar reporte de proyecto
- `POST /api/v1/reports/generate/system` - Generar reporte del sistema
- `GET /api/v1/reports` - Listar reportes

### Monitoreo en Tiempo Real
- `POST /api/v1/monitoring/start` - Iniciar monitoreo
- `POST /api/v1/monitoring/stop` - Detener monitoreo
- `GET /api/v1/monitoring/metrics` - Métricas actuales
- `GET /api/v1/monitoring/history` - Historial de métricas
- `GET /api/v1/monitoring/alerts` - Alertas recientes

### Automatización
- `POST /api/v1/automation/create` - Crear automatización
- `GET /api/v1/automation/list` - Listar automatizaciones
- `GET /api/v1/automation/history` - Historial de ejecuciones

### Seguridad Avanzada
- `POST /api/v1/security/api-key/generate` - Generar API key
- `POST /api/v1/security/api-key/validate` - Validar API key
- `GET /api/v1/security/stats` - Estadísticas de seguridad

### Análisis de Calidad de Código
- `POST /api/v1/quality/analyze/file` - Analizar calidad de archivo
- `POST /api/v1/quality/analyze/project` - Analizar calidad de proyecto

### Sugerencias Inteligentes
- `POST /api/v1/suggestions/generate` - Generar sugerencias
- `POST /api/v1/suggestions/feedback` - Registrar feedback
- `GET /api/v1/suggestions/stats` - Estadísticas de sugerencias

### Benchmarking
- `POST /api/v1/benchmark/record` - Registrar benchmark
- `POST /api/v1/benchmark/compare` - Comparar proyectos
- `GET /api/v1/benchmark/leaderboard` - Leaderboard de benchmarks

### Métricas Avanzadas
- `POST /api/v1/metrics/record` - Registrar métrica
- `GET /api/v1/metrics/stats/{metric_name}` - Estadísticas de métrica
- `POST /api/v1/metrics/custom/create` - Crear métrica personalizada
- `GET /api/v1/metrics/summary` - Resumen de métricas

### System
- `GET /api/v1/system/info` - Información del sistema

## 🎯 Ejemplo de Uso

```python
import requests

# Generar proyecto
response = requests.post(
    "http://localhost:8020/api/v1/generate",
    json={
        "description": "Un analizador de imágenes con IA que detecta objetos",
        "project_name": "image_analyzer",
        "author": "Tu Nombre"
    }
)

project_id = response.json()["project_id"]
print(f"Proyecto en cola: {project_id}")

# Ver estado
status = requests.get(f"http://localhost:8020/api/v1/project/{project_id}")
print(status.json())
```

## 📁 Estructura Generada

Cada proyecto generado incluye:

```
proyecto/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── core/
│   │   ├── models/
│   │   ├── services/
│   │   └── utils/
│   ├── main.py
│   ├── requirements.txt
│   ├── Dockerfile
│   └── README.md
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── services/
│   │   └── utils/
│   ├── package.json
│   ├── vite.config.ts
│   ├── Dockerfile
│   └── README.md
├── docker-compose.yml
├── README.md
└── project_info.json
```

## 🔄 Generación Continua

El sistema funciona de forma continua:
1. Recibe descripciones de proyectos (vía API)
2. Analiza y extrae características automáticamente
3. Los agrega a una cola con prioridades
4. Los procesa automáticamente uno por uno
5. Genera backend y frontend completos con código inteligente
6. Todo listo para usar inmediatamente

## 🧠 Detección Inteligente

El sistema detecta automáticamente:

### Tipos de IA
- **Chat**: Conversacional, assistant, chatbot
- **Vision**: Imágenes, detección, reconocimiento, OCR
- **Audio**: Música, voz, transcripción, speech
- **NLP**: Texto, traducción, análisis de sentimiento
- **Video**: Streaming, procesamiento de video
- **Recommendation**: Sistemas de recomendación
- **Analytics**: Análisis de datos, reportes
- **Generation**: Generación de contenido
- **Classification**: Clasificación, categorización
- **QA**: Preguntas y respuestas

### Características
- **Auth**: Autenticación de usuarios
- **Database**: Base de datos (PostgreSQL, MySQL, MongoDB)
- **WebSocket**: Comunicación en tiempo real
- **File Upload**: Subida de archivos
- **Cache**: Redis, Memcached
- **Queue**: Tareas en background
- **Streaming**: Procesamiento en tiempo real

### Proveedores de Modelos
- OpenAI (GPT)
- Anthropic (Claude)
- Google (Gemini)
- HuggingFace (Transformers)
- Modelos locales (Llama, Mistral)

## 🛠️ Tecnologías Generadas

### Backend
- FastAPI
- Pydantic
- Uvicorn
- Estructura modular

### Frontend
- React 18
- TypeScript
- Vite
- Tailwind CSS
- React Router
- Axios

## 🚀 Funcionalidades Avanzadas

### ✨ Exportación de Proyectos
- Exporta proyectos a ZIP o TAR
- Incluye metadata del proyecto
- Filtra archivos innecesarios automáticamente

### ✅ Validación Automática
- Valida estructura de directorios
- Verifica archivos esenciales
- Comprueba configuración
- Valida sintaxis básica del código

### 🌐 Configuraciones de Despliegue
Genera configuraciones para:
- **Vercel**: `vercel.json` y `.vercelignore`
- **Netlify**: `netlify.toml`
- **Railway**: `railway.json`
- **Heroku**: `Procfile`, `runtime.txt`, `.slugignore`

### 📊 Metadata y Estadísticas
Cada proyecto incluye:
- Metadata completa en `project_metadata.json`
- Conteo de archivos por tipo
- Información de estructura
- Estadísticas del proyecto

### 🔄 Clonado de Proyectos
- Clona proyectos existentes con un solo comando
- Actualiza configuraciones automáticamente
- Excluye archivos innecesarios
- Mantiene historial de clonado

### 📝 Templates Personalizados
- Guarda configuraciones de proyectos como templates
- Reutiliza templates para generar proyectos similares
- Gestiona templates (crear, listar, eliminar)
- Templates reutilizables y personalizables

### 🔍 Búsqueda Avanzada
- Búsqueda por texto, tipo de IA, autor, fecha
- Filtros por características (tests, CI/CD)
- Estadísticas agregadas de proyectos
- Búsqueda rápida y eficiente

### ⚡ Cache Inteligente
- Cache automático de proyectos generados
- Reducción de tiempo de generación
- Expiración automática (7 días)
- Estadísticas de cache

### 🔔 Webhooks
- Notificaciones automáticas de eventos
- Eventos: project.queued, project.completed, project.failed
- Verificación con secret (HMAC)
- Gestión completa de webhooks

### 🛡️ Rate Limiting
- Protección contra abuso
- Límites configurables por endpoint
- Headers informativos (X-RateLimit-*)
- Middleware automático

### 🔐 Autenticación y Autorización
- Sistema de usuarios y roles
- Autenticación JWT
- API keys para acceso programático
- Roles: user, admin

### 📊 Métricas y Monitoreo
- Métricas en tiempo real
- Formato Prometheus compatible
- Tracking de requests, proyectos, cache
- Estadísticas de performance
- Uptime y disponibilidad

### 💾 Backup y Restore
- Backups automáticos completos
- Incluye proyectos, cache y cola
- Restauración fácil
- Gestión de múltiples backups

### 📊 Dashboard Web
- Dashboard interactivo en tiempo real
- Visualización de estadísticas
- Gráficos y métricas
- Actualización automática
- Accesible en `/dashboard`

### 🏥 Health Checks Avanzados
- Health check básico y detallado
- Verificación de sistema de archivos
- Monitoreo de memoria y disco
- Verificación de dependencias
- Estado completo del sistema

### 🔄 API Versioning
- Gestión de versiones de API
- Soporte para múltiples versiones
- Deprecación de versiones
- Información de versiones disponibles

### 🔔 Sistema de Notificaciones
- Notificaciones multi-canal
- Soporte para Slack, Discord, Telegram, Email
- Configuración por canal
- Prioridades de notificación

### 🔌 Sistema de Plugins
- Sistema extensible de plugins
- Registro dinámico de plugins
- Hooks personalizables
- Activación/desactivación de plugins

### 📡 Sistema de Eventos
- Eventos en tiempo real
- Historial de eventos
- Estadísticas de eventos
- Suscripción a eventos

### 📝 Logging Avanzado
- Logging estructurado
- Formato JSON opcional
- Estadísticas de logs
- Rotación automática

### 🔌 WebSocket en Tiempo Real
- Conexiones WebSocket
- Suscripciones por proyecto
- Actualizaciones en tiempo real
- Notificaciones push

### 📦 Generación en Batch
- Generar múltiples proyectos
- Procesamiento paralelo o secuencial
- Control de errores
- Hasta 50 proyectos por batch

### ⚡ Optimización de Performance Avanzada
- Cache inteligente LRU con TTL
- Procesamiento paralelo de proyectos
- Optimizador de generación con estadísticas
- Procesador inteligente de lotes
- Sugerencias automáticas de optimización
- Tracking de tiempos de respuesta
- Análisis de performance

### 📡 Streaming en Tiempo Real
- Sistema de eventos en tiempo real
- Streamers especializados (proyectos, cola, stats)
- Historial de eventos
- Suscripción a eventos
- Estadísticas de streaming

### 📊 Analytics Avanzado
- Motor de análisis completo
- Tendencias y estadísticas
- Reportes personalizables
- Análisis de performance
- Uso de frameworks
- Estadísticas por autor
- Tipos de IA más populares

### 🎯 Sistema de Recomendaciones Inteligentes
- Recomendaciones basadas en ML
- Features recomendadas por tipo de IA
- Framework recomendado
- Proyectos similares
- Aprendizaje continuo

### 📦 Sistema de Versionado
- Versionado semántico de proyectos
- Historial completo de versiones
- Restauración de versiones anteriores
- Comparación entre versiones
- Hash de integridad
- Metadata por versión

### 👥 Sistema de Colaboración
- Gestión de colaboradores
- Roles y permisos (owner, editor, viewer)
- Comentarios en proyectos
- Respuestas a comentarios
- Control de acceso granular

### 📚 Documentación Automática
- Generación automática de README.md
- Documentación de API
- CHANGELOG automático
- Templates personalizables
- Información completa del proyecto

### 🚨 Sistema de Alertas
- Alertas por niveles (info, warning, error, critical)
- Reglas de alerta configurables
- Historial de alertas
- Reconocimiento de alertas
- Estadísticas de alertas
- Notificaciones automáticas

### ⏰ Sistema de Scheduling
- Tareas programadas
- Ejecución automática
- Tipos: interval, cron, once
- Habilitar/deshabilitar tareas
- Historial de ejecuciones
- Estadísticas de tareas

### 📥📤 Import/Export Avanzado
- Exportación con opciones (dependencies, tests, docs)
- Múltiples formatos (zip, tar, tar.gz, tar.bz2, tar.xz)
- Importación de proyectos
- Validación automática
- Compresión configurable

### 🤖 Machine Learning
- Predicción de tiempo de generación
- Predicción de probabilidad de éxito
- Modelo entrenable con datos históricos
- Estadísticas del modelo
- Aprendizaje continuo

### ⚡ Optimización Automática
- Análisis automático de proyectos
- Sugerencias de optimización
- Optimización de configuración
- Recomendaciones inteligentes
- Score de optimización

### 🧪 Testing Avanzado
- Tests automáticos de backend (pytest)
- Tests automáticos de frontend (npm test)
- Ejecución de todos los tests
- Reportes JSON de resultados
- Timeout configurable

### 🚀 Deployment Automático
- Despliegue a Vercel
- Despliegue a Netlify
- Despliegue a Railway
- Historial de despliegues
- Configuración automática

### 📊 Performance Analysis
- Análisis avanzado de performance
- Predicción de tiempos de operaciones
- Monitoreo de recursos (CPU, memoria, disco)
- Métricas históricas
- Percentiles (P95, P99)

### 📋 Reportes Avanzados
- Reportes de proyectos personalizables
- Reportes del sistema (daily, weekly, monthly)
- Inclusión de estadísticas y timeline
- Almacenamiento persistente
- Listado y búsqueda de reportes

### 📡 Monitoreo en Tiempo Real
- Monitoreo continuo del sistema
- Recolección de métricas (CPU, memoria, disco)
- Alertas automáticas (CPU alto, memoria alta)
- Historial de métricas
- Control de inicio/detención

### 🤖 Motor de Automatización
- Automatizaciones basadas en triggers
- Múltiples tipos de triggers (project.created, scheduled, etc.)
- Múltiples acciones (run_tests, deploy, notify, etc.)
- Historial de ejecuciones
- Habilitar/deshabilitar automatizaciones

### 🔐 Seguridad Avanzada
- Generación y validación de API keys
- Control de acceso basado en permisos
- Rate limiting avanzado
- Bloqueo automático de IPs
- Protección contra intentos fallidos
- Estadísticas de seguridad
- Expiración de API keys

### 📊 Análisis de Calidad de Código
- Análisis de archivos Python (AST)
- Complejidad ciclomática
- Longitud de funciones y clases
- Detección de problemas de calidad
- Score de calidad (0-100)
- Análisis de proyectos completos
- Reportes detallados

### 🧠 Sugerencias Inteligentes
- Generación automática de sugerencias
- Análisis de descripción del proyecto
- Recomendaciones de frameworks
- Sugerencias de features
- Aprendizaje de preferencias del usuario
- Feedback y mejora continua
- Estadísticas de aceptación

### 📈 Sistema de Benchmarking
- Benchmarking de generación de proyectos
- Comparación de múltiples proyectos
- Leaderboard de performance
- Score de performance automático
- Métricas de tiempo y complejidad
- Análisis comparativo

### 📊 Métricas Avanzadas
- Sistema de métricas personalizables
- Estadísticas detalladas (min, max, avg, percentiles)
- Ventanas de tiempo configurables
- Métricas personalizadas
- Agregaciones automáticas
- Resumen completo de métricas

### 🔒 Seguridad Mejorada
- Security headers automáticos
- GZip compression
- Validaciones mejoradas
- Protección contra spam

## 📝 Notas

- Los proyectos se generan en el directorio `generated_projects/`
- La cola se guarda en `project_queue.json`
- El sistema detecta automáticamente el tipo de IA y genera código apropiado
- Los proyectos se validan automáticamente después de generarse
- Metadata se exporta automáticamente para cada proyecto

## 🎯 Ejemplo Completo con Todas las Características

```python
import requests

# 1. Generar proyecto
response = requests.post(
    "http://localhost:8020/api/v1/generate",
    json={
        "description": "Un sistema de chat con IA",
        "project_name": "chat_ai",
        "generate_tests": True,
        "include_cicd": True,
        "create_github_repo": True,
        "github_token": "ghp_...",
    }
)

project_id = response.json()["project_id"]
project_path = response.json()["project_info"]["project_dir"]

# 2. Validar proyecto
validation = requests.post(
    "http://localhost:8020/api/v1/validate",
    json={"project_path": project_path}
)
print(f"Proyecto válido: {validation.json()['valid']}")

# 3. Exportar a ZIP
export = requests.post(
    "http://localhost:8020/api/v1/export/zip",
    json={"project_path": project_path}
)
print(f"Exportado a: {export.json()['zip_path']}")

# 4. Generar configuraciones de despliegue
deploy = requests.post(
    "http://localhost:8020/api/v1/deploy/generate",
    json={
        "project_path": project_path,
        "platforms": ["vercel", "netlify"]
    }
)
print(f"Configuraciones generadas: {deploy.json()['generated']}")
```

## 👤 Autor

Blatam Academy

