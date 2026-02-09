# Changelog - Robot Maintenance AI

## [2.2.0] - Refactorización Mayor del Código (Fase 1 y 2)

### ✨ Refactorización

#### Separación de Responsabilidades
- **Schemas centralizados** (`api/schemas.py`): Todos los modelos Pydantic movidos a un módulo dedicado
- **Dependencias centralizadas** (`api/dependencies.py`): Gestión unificada de dependencias (tutor, rate limiter, conversation manager)
- **Excepciones personalizadas** (`api/exceptions.py`): Sistema de excepciones con códigos de error estandarizados
- **Respuestas estandarizadas** (`api/responses.py`): Helpers para crear respuestas API consistentes
- **Middleware de errores** (`middleware/error_handler.py`): Manejo centralizado de excepciones

#### Mejoras en `maintenance_api.py`
- Reducción de ~710 líneas a ~450 líneas (36% menos código)
- Eliminación de duplicación de validación
- Manejo de errores simplificado (delegado al middleware)
- Uso de dependencias inyectadas en lugar de variables globales
- Respuestas estandarizadas con helpers

#### Beneficios
- **Mantenibilidad**: Código más organizado y fácil de mantener
- **Testabilidad**: Dependencias inyectables facilitan testing
- **Consistencia**: Respuestas y errores estandarizados
- **Escalabilidad**: Estructura preparada para crecimiento futuro

### 📁 Archivos Nuevos

- `api/schemas.py` - Modelos Pydantic centralizados
- `api/dependencies.py` - Gestión de dependencias
- `api/exceptions.py` - Excepciones personalizadas
- `api/responses.py` - Helpers de respuestas
- `middleware/error_handler.py` - Middleware de manejo de errores

### 🔄 Cambios en Archivos Existentes

- `api/maintenance_api.py` - Refactorizado completamente para usar nuevos módulos (Fase 1)
- `core/maintenance_tutor.py` - Refactorizado para usar servicios especializados (Fase 2)
- `core/database.py` - Refactorizado con context managers y helpers (Fase 2)
- `CHANGELOG.md` - Este archivo actualizado

### 📊 Métricas de Mejora

- **Reducción de código**: `maintenance_api.py` de 710 a 450 líneas (36% reducción)
- **Reducción de código**: `maintenance_tutor.py` de 449 a ~320 líneas (29% reducción)
- **Eliminación de duplicación**: ~150 líneas de código duplicado eliminadas en `database.py`
- **Base Router**: Clase base disponible para reducir duplicación en 20+ routers de API
- **Routers refactorizados**: 22 routers principales refactorizados (maintenance, analytics, search, config, admin, monitoring, dashboard, reports, alerts, templates, validation, recommendations, incidents, batch, plugins, webhooks, export, audit, comparison, learning, notifications, sync, auth)
- **Eliminación adicional**: ~1,270 líneas de duplicación eliminadas en routers refactorizados
- **Mejora en mantenibilidad**: Separación clara de responsabilidades en servicios
- **Total líneas eliminadas**: ~2,110 líneas de código duplicado/innecesario

### 🔧 Mejoras Técnicas

#### Fase 1: API Layer
- Eliminación de variables globales (`tutor_instance`, `rate_limiter`, `conversation_manager`)
- Uso de dependency injection apropiada
- Middleware de errores para manejo centralizado
- Respuestas API consistentes en todos los endpoints
- Mejor separación de concerns

#### Fase 2: Core Layer
- **Servicios especializados**: Separación de OpenRouter y PromptBuilder en servicios dedicados
- **Context managers en database**: Manejo automático de conexiones con rollback en errores
- **Reducción de duplicación**: Helpers reutilizables para queries comunes
- **Mejor testabilidad**: Servicios pueden ser mockeados fácilmente

#### Fase 3: Base Router Class
- **BaseRouter class**: Clase base para todos los routers de API
- **Funcionalidad común**: Respuestas estandarizadas, logging, timing, dependencias
- **Reducción de duplicación**: Elimina código repetitivo en 20+ routers
- **Lazy loading**: Instancias de database y conversation_manager cargadas bajo demanda

#### Fase 4: Aplicación de BaseRouter
- **analytics_api.py**: Refactorizado usando BaseRouter (eliminadas ~80 líneas de duplicación)
- **search_api.py**: Refactorizado usando BaseRouter (eliminadas ~60 líneas de duplicación)
- **config_api.py**: Refactorizado usando BaseRouter (eliminadas ~50 líneas de duplicación)
- **admin_api.py**: Refactorizado usando BaseRouter (eliminadas ~60 líneas de duplicación)
- **monitoring_api.py**: Refactorizado usando BaseRouter (eliminadas ~70 líneas de duplicación)
- **dashboard_api.py**: Refactorizado usando BaseRouter (eliminadas ~60 líneas de duplicación)
- **reports_api.py**: Refactorizado usando BaseRouter (eliminadas ~50 líneas de duplicación)
- **alerts_api.py**: Refactorizado usando BaseRouter (eliminadas ~60 líneas de duplicación)
- **templates_api.py**: Refactorizado usando BaseRouter (eliminadas ~80 líneas de duplicación)
- **validation_api.py**: Refactorizado usando BaseRouter (eliminadas ~50 líneas de duplicación)
- **recommendations_api.py**: Refactorizado usando BaseRouter (eliminadas ~60 líneas de duplicación)
- **incidents_api.py**: Refactorizado usando BaseRouter (eliminadas ~70 líneas de duplicación)
- **batch_api.py**: Refactorizado usando BaseRouter (eliminadas ~60 líneas de duplicación)
- **plugins_api.py**: Refactorizado usando BaseRouter (eliminadas ~50 líneas de duplicación)
- **webhooks_api.py**: Refactorizado usando BaseRouter (eliminadas ~60 líneas de duplicación)
- **export_advanced_api.py**: Refactorizado usando BaseRouter (eliminadas ~50 líneas de duplicación)
- **audit_api.py**: Refactorizado usando BaseRouter (eliminadas ~60 líneas de duplicación)
- **comparison_api.py**: Refactorizado usando BaseRouter (eliminadas ~60 líneas de duplicación)
- **learning_api.py**: Refactorizado usando BaseRouter (eliminadas ~50 líneas de duplicación)
- **notifications_api.py**: Refactorizado usando BaseRouter (eliminadas ~30 líneas de duplicación)
- **sync_api.py**: Refactorizado usando BaseRouter (eliminadas ~50 líneas de duplicación)
- **auth_api.py**: Refactorizado usando BaseRouter (eliminadas ~20 líneas de duplicación, require_auth preservado)
- **maintenance_api.py**: Refactorizado usando BaseRouter (eliminado timing manual, ~30 líneas de duplicación eliminadas)
- **utils/validators.py**: Consolidadas funciones de validación duplicadas (creada función genérica `validate_in_list`)
- **utils/helpers.py**: Consolidada validación de sensor_data para usar validators (eliminada duplicación)
- **Mejoras**: Eliminación de try/catch duplicados, HTTPException manual, respuestas manuales, timing manual, validaciones duplicadas
- **Total Fase 4**: 22 routers refactorizados, ~1,270 líneas de duplicación eliminadas
- **Fase 5 (Utils)**: Consolidación de validaciones, ~20 líneas de duplicación eliminadas
- **Fase 6 (Middleware)**: Refactorización de error_handler.py (corregida indentación, eliminada duplicación de métricas y respuestas), ~40 líneas de duplicación eliminadas
- **Resumen Maestro**: Documentación completa de todas las 6 fases de refactorización
- **Fase 7 (Core Trainer)**: Refactorización de maintenance_trainer.py para usar OpenRouterService y PromptBuilder, eliminando ~76 líneas de duplicación (-28%)
- **Fase 8 (File Helpers)**: Creación de utils/file_helpers.py y refactorización de export_utils.py y backup_utils.py, eliminando ~22 líneas de duplicación
- **Fase 9 (Timestamps)**: Consolidación de timestamps en módulos core para usar get_iso_timestamp(), 6 archivos refactorizados
- **Fase 10 (Database & JSON)**: Refactorización de database.py para usar get_iso_timestamp() y creación de json_helpers.py para operaciones JSON seguras
- **Fase 11 (Final Consolidation)**: Consolidación final de timestamps en reports_api.py, completando la consolidación en toda la aplicación
- **Fase 12 (Date Helpers)**: Extensión de file_helpers.py con helpers de fecha/hora y refactorización de reports_api.py para usarlos
- **Fase 13 (Data Helpers)**: Creación de data_helpers.py con helpers de agregación y refactorización de reports_api.py y analytics_api.py
- **Fase 14 (Consolidation)**: Consolidación de data_helpers.py y aggregation_helpers.py en un solo módulo unificado
- **Fase 15 (Comparison API)**: Refactorización de comparison_api.py para usar helpers existentes y creación de helpers de filtrado por fecha
- **Fase 13 (Aggregation Helpers)**: Creación de aggregation_helpers.py con helpers de agregación y refactorización de reports_api.py para usarlos

### 📁 Archivos Nuevos (Fase 2)

- `core/services/__init__.py` - Módulo de servicios
- `core/services/openrouter_service.py` - Servicio para llamadas a OpenRouter API
- `core/services/prompt_builder.py` - Servicio para construcción de prompts

### 📁 Archivos Nuevos (Fase 3)

- `api/base_router.py` - Clase base para routers con funcionalidad común

## [2.1.0] - Auditoría, Plantillas y Validación Avanzada

### ✨ Nuevas Características

#### API de Auditoría
- Sistema completo de auditoría de actividades
- Registro de todas las acciones del sistema
- Filtrado avanzado de logs de auditoría
- Estadísticas de actividad
- Timeline de actividad para visualización
- Seguimiento de usuarios y recursos
- Análisis de patrones de uso

#### API de Plantillas
- Sistema de plantillas de mantenimiento
- Creación y gestión de plantillas reutilizables
- Plantillas por tipo de robot y mantenimiento
- Personalización de plantillas
- Plantillas populares por uso
- Categorización con tags
- Reutilización de procedimientos estándar

#### API de Validación Avanzada
- Validación de datos con reglas personalizables
- Validación por lotes
- Reglas de validación por tipo de dato
- Validación de sensor_data, maintenance_record, conversation
- Mensajes de error descriptivos
- Validación de tipos, rangos, patrones
- Reglas de longitud y requeridos

### 📁 Archivos Nuevos

- `api/audit_api.py` - Sistema completo de auditoría
- `api/templates_api.py` - Gestión de plantillas de mantenimiento
- `api/validation_api.py` - Validación avanzada de datos

### 🔧 Mejoras

#### Auditoría
- Registro completo de actividades del sistema
- Filtrado por acción, recurso, usuario y fecha
- Estadísticas y análisis de actividad
- Timeline para visualización

#### Plantillas
- Reutilización de procedimientos de mantenimiento
- Personalización de plantillas
- Seguimiento de uso
- Categorización avanzada

#### Validación
- Validación robusta de datos
- Reglas configurables
- Validación por lotes
- Soporte para múltiples tipos de datos

### 🔄 Cambios en Archivos Existentes

- `api/maintenance_api.py` - Nuevos routers agregados (audit, templates, validation), versión actualizada a 2.1.0
- `CHANGELOG.md` - Este archivo actualizado

## [2.0.0] - Configuración Dinámica y Monitoreo Avanzado

### ✨ Nuevas Características

#### API de Configuración Dinámica
- Gestión de configuración en tiempo de ejecución
- Actualización de configuración sin reinicio (cuando es posible)
- Validación de configuración antes de aplicar
- Secciones de configuración: openrouter, nlp, ml, cache, rate_limiter
- Máscara de valores sensibles (API keys)
- Validación de valores y rangos
- Indicación de requerimiento de reinicio

#### API de Monitoreo Avanzado
- Monitoreo detallado de salud del sistema
- Métricas de CPU, memoria y disco
- Historial de rendimiento
- Alertas del sistema automáticas
- Uso de recursos en tiempo real
- Métricas de proceso de la aplicación
- Alertas críticas y warnings

### 📁 Archivos Nuevos

- `api/config_api.py` - Gestión dinámica de configuración
- `api/monitoring_api.py` - Monitoreo avanzado del sistema

### 🔧 Mejoras

#### Configuración
- Actualización de configuración en tiempo de ejecución
- Validación robusta de valores
- Gestión de secciones de configuración
- Protección de valores sensibles

#### Monitoreo
- Monitoreo completo del sistema
- Alertas automáticas basadas en umbrales
- Historial de rendimiento
- Métricas de recursos en tiempo real

#### Dependencias
- Agregado `psutil>=5.9.0` para monitoreo del sistema

### 🔄 Cambios en Archivos Existentes

- `api/maintenance_api.py` - Nuevos routers agregados (config, monitoring), versión actualizada a 2.0.0
- `requirements.txt` - Agregado psutil para monitoreo
- `CHANGELOG.md` - Este archivo actualizado

## [1.9.0] - Dashboard en Tiempo Real, Webhooks y Exportación Avanzada

### ✨ Nuevas Características

#### API de Dashboard en Tiempo Real
- Dashboard completo con métricas en tiempo real
- Widgets especializados para visualización
- Métricas de actividad y rendimiento
- Estado de mantenimiento en tiempo real
- Gráficos de actividad temporal
- Distribución de robots
- Métricas de API y sistema

#### API de Webhooks
- Sistema completo de webhooks para integraciones externas
- Suscripción a eventos específicos
- Firma de webhooks con secretos
- Testing de webhooks
- Gestión de webhooks (crear, listar, eliminar)
- Eventos disponibles: maintenance, alerts, incidents, predictions, etc.
- Seguimiento de triggers y fallos

#### API de Exportación Avanzada
- Exportación en múltiples formatos (JSON, CSV, Excel)
- Exportación de conversaciones individuales
- Exportación de historial de mantenimiento
- Exportación de datos analíticos
- Filtrado por fecha y tipo de robot
- Descarga directa de archivos

### 📁 Archivos Nuevos

- `api/dashboard_api.py` - Dashboard en tiempo real con widgets
- `api/webhooks_api.py` - Sistema de webhooks para integraciones
- `api/export_advanced_api.py` - Exportación avanzada en múltiples formatos

### 🔧 Mejoras

#### Visualización
- Dashboard con múltiples widgets especializados
- Métricas en tiempo real actualizadas
- Gráficos de actividad temporal
- Visualización de distribución de datos

#### Integraciones
- Sistema de webhooks para eventos del sistema
- Integración con sistemas externos
- Notificaciones automáticas vía webhooks

#### Exportación
- Múltiples formatos de exportación
- Exportación filtrada y personalizada
- Descarga directa de archivos

### 🔄 Cambios en Archivos Existentes

- `api/maintenance_api.py` - Nuevos routers agregados (dashboard, webhooks, export_advanced), versión actualizada a 1.9.0
- `CHANGELOG.md` - Este archivo actualizado

## [1.8.0] - Comparación, Reportes Avanzados y Aprendizaje Continuo

### ✨ Nuevas Características

#### API de Comparación y Benchmarking
- Comparación de múltiples tipos de robots
- Benchmarking de métricas de mantenimiento
- Análisis de tendencias comparativas
- Rankings y estadísticas comparativas
- Análisis de intervalos de mantenimiento
- Distribución de tipos de mantenimiento

#### API de Reportes Avanzados
- Generación de reportes personalizados
- Reportes resumidos y detallados
- Reportes predictivos con pronósticos
- Reportes de análisis de costos
- Desglose mensual y por tipo
- Recomendaciones incluidas en reportes

#### API de Aprendizaje Continuo
- Sistema de feedback para mejorar modelos
- Entrenamiento y reentrenamiento de modelos
- Información de modelos ML
- Métricas de rendimiento de modelos
- Insights de aprendizaje y mejoras
- Seguimiento de precisión y deriva de modelos

### 📁 Archivos Nuevos

- `api/comparison_api.py` - Sistema de comparación y benchmarking
- `api/reports_api.py` - Generación de reportes avanzados
- `api/learning_api.py` - Sistema de aprendizaje continuo y mejora de modelos

### 🔧 Mejoras

#### Análisis Avanzado
- Comparación multi-robot con métricas detalladas
- Análisis de tendencias temporales
- Benchmarking automático
- Reportes con múltiples formatos y niveles de detalle

#### Machine Learning
- Sistema de feedback para aprendizaje supervisado
- Métricas de rendimiento en tiempo real
- Detección de deriva de modelos
- Insights automáticos de mejora

#### Integración
- Nuevos routers integrados en la aplicación principal
- Autenticación requerida para todos los endpoints
- Documentación automática en Swagger/OpenAPI

### 🔄 Cambios en Archivos Existentes

- `api/maintenance_api.py` - Nuevos routers agregados (comparison, reports, learning), versión actualizada a 1.8.0
- `CHANGELOG.md` - Este archivo actualizado

## [1.7.0] - Alertas, Recomendaciones e Incidencias

### ✨ Nuevas Características

#### Sistema de Alertas Inteligentes
- Creación automática de alertas basadas en análisis de sensores
- Reglas de alerta configurables
- Niveles de severidad (info, warning, error, critical)
- Análisis automático de salud del robot
- Sistema de reconocimiento de alertas
- Filtrado y búsqueda de alertas

#### Sistema de Recomendaciones
- Recomendaciones inteligentes de mantenimiento
- Análisis basado en historial y estado actual
- Recomendaciones de optimización
- Programación de mantenimiento basada en recomendaciones
- Priorización automática de recomendaciones
- Estimación de tiempo y costo

#### Gestión de Incidencias
- Sistema completo de tickets/incidencias
- Creación, actualización y resolución de incidencias
- Seguimiento de estado y prioridad
- Notas y comentarios en incidencias
- Estadísticas y resúmenes de incidencias
- Filtrado avanzado por tipo, estado y severidad

### 📁 Archivos Nuevos

- `api/alerts_api.py` - Sistema completo de alertas inteligentes
- `api/recommendations_api.py` - Sistema de recomendaciones de mantenimiento
- `api/incidents_api.py` - Gestión de incidencias y tickets

### 🔧 Mejoras

#### Integración
- Nuevos routers integrados en la aplicación principal
- Autenticación requerida para todos los endpoints
- Documentación automática en Swagger/OpenAPI

### 🔄 Cambios en Archivos Existentes

- `api/maintenance_api.py` - Nuevos routers agregados (alerts, recommendations, incidents), versión actualizada a 1.7.0
- `CHANGELOG.md` - Este archivo actualizado

## [1.6.0] - Analytics, Búsqueda y Operaciones por Lotes

### ✨ Nuevas Características

#### API de Analytics
- Dashboard de analytics con métricas completas
- Estadísticas de conversaciones y mensajes
- Análisis de rendimiento del sistema
- Tendencias y análisis temporal
- Distribución por tipo de robot y mantenimiento
- Actividad diaria y métricas de API

#### API de Búsqueda
- Búsqueda avanzada en conversaciones
- Búsqueda en registros de mantenimiento
- Sugerencias de búsqueda inteligentes
- Filtrado por tipo de robot y mantenimiento
- Paginación de resultados
- Búsqueda por contenido de mensajes

#### Operaciones por Lotes
- Procesamiento por lotes de preguntas
- Procesamiento por lotes de procedimientos
- Eliminación por lotes de conversaciones
- Exportación por lotes de conversaciones
- Procesamiento asíncrono con límite de concurrencia
- Estadísticas de éxito/fallo por lote

#### API de Plugins Mejorada
- Gestión completa de plugins vía API
- Registro y desregistro de plugins
- Ejecución de plugins con datos personalizados
- Gestión de hooks del sistema
- Listado de plugins y hooks disponibles
- Información detallada de plugins

#### Mejoras en Base de Datos
- Métodos adicionales para consultas por rango de fechas
- Obtención de todas las conversaciones y mensajes
- Eliminación de conversaciones
- Mejoras en el esquema de base de datos
- Índices optimizados para búsquedas

### 📁 Archivos Nuevos

- `api/analytics_api.py` - Endpoints de analytics y reportes
- `api/search_api.py` - Endpoints de búsqueda avanzada
- `api/batch_api.py` - Endpoints para operaciones por lotes
- `api/plugins_api.py` - API completa de gestión de plugins

### 🔧 Mejoras

#### Base de Datos
- Métodos `get_conversations_by_date_range()` y `get_messages_by_date_range()`
- Métodos `get_all_conversations()`, `get_all_messages()`, `get_all_maintenance_records()`
- Método `delete_conversation()` para eliminación
- Mejoras en el manejo de `created_at` en conversaciones

#### Integración
- Todos los nuevos routers integrados en la aplicación principal
- Autenticación requerida para endpoints de analytics, búsqueda y batch
- Documentación automática en Swagger/OpenAPI

### 🔄 Cambios en Archivos Existentes

- `api/maintenance_api.py` - Nuevos routers agregados (analytics, search, batch, plugins)
- `core/database.py` - Métodos adicionales para consultas y operaciones
- `CHANGELOG.md` - Este archivo actualizado

## [1.5.0] - Autenticación y Notificaciones

### ✨ Nuevas Características

#### Sistema de Autenticación
- Gestión de API keys para control de acceso
- Validación de API keys en endpoints protegidos
- Permisos por usuario (read, write, admin)
- Revocación de API keys
- Tokens de sesión con expiración

#### Sistema de Notificaciones
- Notificaciones para diferentes tipos de eventos
- Suscripción a tipos de notificaciones
- Gestión de notificaciones por usuario
- Marcado de notificaciones como leídas
- Limpieza de notificaciones

### 📁 Archivos Nuevos

- `core/auth.py` - Sistema de autenticación
- `core/notifications.py` - Sistema de notificaciones
- `api/auth_api.py` - Endpoints de autenticación
- `api/notifications_api.py` - Endpoints de notificaciones
- `docs/AUTHENTICATION.md` - Guía de autenticación

### 🔧 Mejoras

#### Integración
- Routers de autenticación y notificaciones integrados
- Endpoints protegidos con autenticación opcional
- Sistema de notificaciones listo para integración

### 🔄 Cambios en Archivos Existentes

- `api/maintenance_api.py` - Routers de auth y notifications agregados, versión actualizada
- `README.md` - Documentación actualizada
- `CHANGELOG.md` - Este archivo actualizado

## [1.4.0] - Persistencia y WebSockets

### ✨ Nuevas Características

#### Persistencia de Datos
- Base de datos SQLite para almacenar conversaciones
- Almacenamiento de mensajes con metadatos
- Registros de mantenimiento persistentes
- Historial de predicciones ML
- Índices optimizados para consultas rápidas

#### WebSockets
- Soporte WebSocket para actualizaciones en tiempo real
- Broadcast de mensajes a múltiples clientes
- Gestión de conexiones por conversación
- Sistema de ping/pong para mantener conexiones

#### Optimizaciones de Rendimiento
- Decoradores para medir tiempo de ejecución
- Procesamiento asíncrono por lotes
- Control de concurrencia con semáforos
- Logging de rendimiento automático

#### Utilidades de Seguridad
- Sanitización de HTML para prevenir XSS
- Validación de formato de API keys
- Prevención básica de SQL injection
- Generación de tokens CSRF

### 📁 Archivos Nuevos

- `core/database.py` - Capa de base de datos SQLite
- `api/websocket_api.py` - Endpoints WebSocket
- `utils/performance.py` - Utilidades de optimización
- `utils/security.py` - Utilidades de seguridad

### 🔧 Mejoras

#### Dependencias
- Agregado `websockets>=11.0.0` para soporte WebSocket

#### Integración
- WebSocket router integrado en la aplicación principal
- Base de datos disponible para persistencia opcional

### 🔄 Cambios en Archivos Existentes

- `api/maintenance_api.py` - WebSocket router agregado, versión actualizada
- `requirements.txt` - websockets agregado
- `utils/__init__.py` - Nuevas utilidades exportadas
- `README.md` - Documentación actualizada
- `CHANGELOG.md` - Este archivo actualizado

## [1.3.0] - Docker y Configuración Avanzada

### ✨ Nuevas Características

#### Docker Support
- Dockerfile optimizado para producción
- docker-compose.yml para desarrollo y producción
- Health checks integrados
- Volúmenes configurados para logs y datos
- .dockerignore para builds eficientes

#### Configuración YAML
- Soporte para archivos de configuración YAML
- Expansión automática de variables de entorno
- Archivo de ejemplo `config.yaml.example`
- Cargador de configuración flexible

#### Scripts de Inicio
- Script de inicio para Linux/Mac (`scripts/start.sh`)
- Script de inicio para Windows (`scripts/start.bat`)
- Validación automática de variables de entorno
- Creación automática de directorios necesarios

### 📁 Archivos Nuevos

- `Dockerfile` - Imagen Docker para el servicio
- `docker-compose.yml` - Orquestación con Docker Compose
- `.dockerignore` - Archivos excluidos del build
- `config/config.yaml.example` - Ejemplo de configuración YAML
- `utils/config_loader.py` - Cargador de configuración YAML
- `docs/DOCKER.md` - Guía completa de Docker
- `scripts/start.sh` - Script de inicio para Unix
- `scripts/start.bat` - Script de inicio para Windows

### 🔧 Mejoras

#### Dependencias
- Agregado `pyyaml>=6.0.0` para soporte YAML

#### Documentación
- Guía completa de Docker
- Ejemplos de despliegue
- Troubleshooting de Docker

### 🔄 Cambios en Archivos Existentes

- `requirements.txt` - PyYAML agregado
- `README.md` - Sección de Docker agregada
- `CHANGELOG.md` - Este archivo actualizado

## [1.2.0] - Mejoras de Producción

### ✨ Nuevas Características

#### Middleware de Logging
- Middleware automático para logging de todas las peticiones HTTP
- Información de timing (tiempo de procesamiento)
- Headers `X-Process-Time` en respuestas
- Logging estructurado de errores

#### Health Check Mejorado
- Health check detallado con información del sistema
- Estado de componentes (OpenRouter, NLP, ML, Cache)
- Información de versión de Python y plataforma
- Métricas integradas en health check

#### Exportación de Conversaciones
- Exportar conversaciones a JSON
- Exportar conversaciones a CSV
- Generación de reportes de mantenimiento
- Endpoints `/conversation/{id}/export/json` y `/conversation/{id}/export/csv`
- Endpoint `/conversation/{id}/report` para reportes

#### CORS Configurado
- Soporte CORS para integración frontend
- Configuración flexible de orígenes permitidos

### 🔧 Mejoras

#### Documentación OpenAPI
- Mejoras en la documentación automática de FastAPI
- Endpoints `/docs` y `/redoc` configurados
- OpenAPI schema mejorado

#### Estructura del Proyecto
- Nueva carpeta `middleware/` para middleware personalizado
- Utilidades de exportación en `utils/export_utils.py`

### 📁 Archivos Nuevos

- `middleware/__init__.py` - Inicialización de middleware
- `middleware/request_logging.py` - Middleware de logging
- `utils/export_utils.py` - Utilidades de exportación

### 🔄 Cambios en Archivos Existentes

- `api/maintenance_api.py` - Middleware agregado, health check mejorado, endpoints de exportación
- `README.md` - Documentación actualizada con nuevas características
- `CHANGELOG.md` - Este archivo actualizado

## [1.1.0] - Mejoras Avanzadas

### ✨ Nuevas Características

#### Rate Limiting
- Sistema de rate limiting con algoritmo token bucket
- Límite configurable: 100 requests/minuto por IP (por defecto)
- Headers `Retry-After` cuando se excede el límite
- Endpoints `/rate-limit/stats` y `/rate-limit/reset`
- Identificación de clientes por IP

#### Logging Mejorado
- Sistema de logging configurable
- Soporte para logging a archivo
- Formato estructurado con información detallada
- Configuración mediante variables de entorno (`LOG_LEVEL`, `LOG_FILE`)

#### Tests Unitarios
- Suite completa de tests para validadores (`test_validators.py`)
- Tests para sistema de caché (`test_cache_manager.py`)
- Tests para rate limiting (`test_rate_limiter.py`)
- Fixtures compartidos (`conftest.py`)
- Documentación de tests (`tests/README.md`)

#### Documentación API
- Referencia completa de la API (`docs/API_REFERENCE.md`)
- Ejemplos de uso para todos los endpoints
- Documentación de códigos de error
- Información sobre rate limiting y autenticación

### 🔧 Mejoras

#### Sistema de Métricas
- Tracking mejorado de todas las peticiones
- Estadísticas por endpoint
- Métricas de caché integradas
- Endpoint `/metrics` mejorado

#### Validación
- Validación completa en todos los modelos de request
- Validación de `robot_type`, `maintenance_type`, `difficulty`
- Validación estricta de `sensor_data`
- Mensajes de error más descriptivos

#### Manejo de Errores
- Códigos HTTP apropiados (400, 429, 503, 504, 500)
- Manejo específico de rate limiting (429)
- Logging detallado de errores
- Tracking de errores en métricas

#### Configuración
- Variables de entorno para configuración del servidor
- `HOST`, `PORT`, `LOG_LEVEL`, `LOG_FILE`
- Configuración flexible del logging

### 📦 Dependencias Agregadas

- `pytest>=7.4.0` - Framework de testing
- `pytest-cov>=4.1.0` - Cobertura de código
- `pytest-asyncio>=0.21.0` - Soporte para tests asíncronos

### 📁 Archivos Nuevos

- `utils/rate_limiter.py` - Sistema de rate limiting
- `utils/logger_config.py` - Configuración de logging
- `utils/metrics_decorator.py` - Decorador para métricas
- `tests/__init__.py` - Inicialización de tests
- `tests/test_validators.py` - Tests de validación
- `tests/test_cache_manager.py` - Tests de caché
- `tests/test_rate_limiter.py` - Tests de rate limiting
- `tests/conftest.py` - Fixtures de pytest
- `tests/README.md` - Documentación de tests
- `docs/API_REFERENCE.md` - Referencia de API
- `CHANGELOG.md` - Este archivo

### 🔄 Cambios en Archivos Existentes

- `api/maintenance_api.py` - Rate limiting agregado a endpoints
- `main.py` - Logging mejorado y configuración de entorno
- `requirements.txt` - Dependencias de testing agregadas
- `utils/__init__.py` - Nuevas utilidades exportadas
- `README.md` - Documentación actualizada con nuevas características

## [1.0.0] - Versión Inicial

### Características Principales

- Sistema tutor de IA para mantenimiento de robots
- Integración con OpenRouter
- Procesamiento NLP con spaCy y Transformers
- Machine Learning con scikit-learn
- API REST con FastAPI
- Sistema de caché con TTL y LRU
- Retry con backoff exponencial
- Validación de inputs
- Sistema de métricas
- Manejo de errores robusto

