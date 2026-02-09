# Bulk Chat - Sistema de Chat Continuo Proactivo

## 🚀 Descripción

**Bulk Chat** es un sistema de chat continuo y proactivo similar a ChatGPT, pero con la característica única de que **no se detiene automáticamente**. El chat genera respuestas de forma continua hasta que el usuario explícitamente lo pause.

### Características Principales

- ✅ **Chat Continuo**: Genera respuestas automáticamente sin detenerse
- ✅ **Control de Pausa**: El usuario puede pausar/reanudar el chat en cualquier momento
- ✅ **Streaming en Tiempo Real**: Soporte para Server-Sent Events (SSE)
- ✅ **Múltiples Sesiones**: Soporte para múltiples sesiones de chat simultáneas
- ✅ **Integración con LLMs**: Soporte para OpenAI, Anthropic y otros proveedores
- ✅ **API REST Completa**: Endpoints para control total del chat
- ✅ **Persistencia de Sesiones**: Guardado automático en JSON o Redis
- ✅ **Sistema de Métricas**: Monitoreo completo de rendimiento y uso
- ✅ **Rate Limiting**: Control de tasa de solicitudes para prevenir abuso
- ✅ **Retry Logic**: Reintentos automáticos con exponential backoff
- ✅ **Auto-guardado**: Guardado automático cada 30 segundos (configurable)
- ✅ **WebSockets**: Streaming bidireccional en tiempo real
- ✅ **Cache de Respuestas**: Reduce llamadas duplicadas al LLM
- ✅ **Sistema de Plugins**: Extensible con plugins personalizados
- ✅ **Análisis de Conversaciones**: Insights y estadísticas avanzadas
- ✅ **Exportación Multi-formato**: JSON, Markdown, CSV, HTML, TXT
- ✅ **Sistema de Plantillas**: Mensajes predefinidos con variables
- ✅ **Webhooks**: Notificaciones en tiempo real a sistemas externos
- ✅ **Autenticación JWT**: Sistema completo de auth con roles
- ✅ **Backups Automáticos**: Backups programados y restauración
- ✅ **Dashboard Web**: Interfaz web interactiva para monitoreo
- ✅ **Testing Framework**: Tests completos con pytest
- ✅ **Optimizador de Rendimiento**: Métricas P50/P95/P99 y detección de cuellos de botella
- ✅ **Logging Estructurado**: Logs JSON con contexto
- ✅ **Monitor de Salud**: Health checks avanzados del sistema
- ✅ **Cola de Tareas**: Procesamiento asíncrono en background
- ✅ **API GraphQL**: Consultas flexibles y eficientes
- ✅ **Sistema de Alertas**: Alertas y notificaciones avanzadas
- ✅ **Dashboard Mejorado**: Gráficos en tiempo real con Chart.js
- ✅ **Clustering Distribuido**: Escalabilidad horizontal con distribución automática
- ✅ **Feature Flags**: Control dinámico de características sin redesplegar
- ✅ **Versionado de API**: Gestión completa de versiones con deprecación
- ✅ **Analytics Avanzado**: Detección de patrones y análisis de comportamiento
- ✅ **Recomendaciones ML**: Sistema de recomendaciones con collaborative filtering
- ✅ **A/B Testing**: Framework completo para experimentos controlados
- ✅ **Sistema de Eventos**: Pub/sub para eventos en tiempo real
- ✅ **Seguridad Avanzada**: Audit logs, sanitización y validación
- ✅ **Internacionalización**: Soporte multi-idioma (i18n)
- ✅ **Workflows**: Sistema de automatización de tareas
- ✅ **Notificaciones Push**: Sistema de notificaciones en tiempo real
- ✅ **Integraciones**: Sistema de integraciones con servicios externos
- ✅ **Benchmarking**: Sistema de performance testing
- ✅ **Documentación Automática**: Generación de OpenAPI y Markdown
- ✅ **Monitoring Avanzado**: Sistema de métricas y alertas
- ✅ **Gestión de Secretos**: Sistema seguro de secretos y configuración
- ✅ **ML Optimizer**: Optimización basada en machine learning
- ✅ **Deployment Automático**: Sistema de deployment y rollback
- ✅ **Reportes Automatizados**: Generación automática de reportes
- ✅ **Gestión de Usuarios**: Sistema completo con roles y permisos
- ✅ **Búsqueda Avanzada**: Motor de búsqueda con índices y filtros
- ✅ **Cola de Mensajes**: Sistema de cola con prioridades y reintentos
- ✅ **Validación Avanzada**: Sistema de validación con reglas personalizadas

## 📋 Requisitos

- Python 3.8+
- FastAPI
- Un proveedor de LLM (OpenAI, Anthropic, etc.)

## 🔧 Instalación

### Opción 1: Instalación Automática (Recomendado)

```bash
# Instalación automática
python install.py

# Verificar instalación
python verify_setup.py
```

### Opción 2: Instalación Manual

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Verificar que todo esté listo
python verify_setup.py

# 3. Configurar variables de entorno (opcional)
# Copiar .env.example a .env y editar con tus API keys
```

**Nota**: Si no tienes una API key, puedes usar el modo `mock`:
```bash
python -m bulk_chat.main --llm-provider mock
```

### Scripts de Inicio Rápido

**Windows:**
```bash
start.bat
start.bat openai
```

**Linux/Mac:**
```bash
chmod +x start.sh
./start.sh
./start.sh openai
```

**Python (Multiplataforma):**
```bash
python run.py server
python run.py server --provider openai
```

Ver [COMMANDS.md](COMMANDS.md) para más comandos útiles.

## 🚀 Uso Rápido

### Iniciar el servidor

```bash
# Uso básico
python -m bulk_chat.main

# Con opciones personalizadas
python -m bulk_chat.main --host 0.0.0.0 --port 8006 --llm-provider openai --llm-model gpt-4
```

### Ejemplo de uso con cURL

```bash
# 1. Crear una sesión de chat
curl -X POST "http://localhost:8006/api/v1/chat/sessions" \
  -H "Content-Type: application/json" \
  -d '{
    "initial_message": "Hola, quiero que me expliques sobre inteligencia artificial",
    "auto_continue": true
  }'

# Respuesta:
# {
#   "session_id": "abc123...",
#   "state": "active",
#   "is_paused": false,
#   "message_count": 1,
#   "auto_continue": true
# }

# 2. El chat comenzará a generar respuestas automáticamente
# Puedes ver los mensajes en tiempo real:

curl "http://localhost:8006/api/v1/chat/sessions/{session_id}/messages"

# 3. Pausar el chat
curl -X POST "http://localhost:8006/api/v1/chat/sessions/{session_id}/pause" \
  -H "Content-Type: application/json" \
  -d '{"action": "pause", "reason": "Usuario pausó"}'

# 4. Reanudar el chat
curl -X POST "http://localhost:8006/api/v1/chat/sessions/{session_id}/resume"

# 5. Detener completamente el chat
curl -X POST "http://localhost:8006/api/v1/chat/sessions/{session_id}/stop"
```

### Ejemplo con Python

```python
import asyncio
from bulk_chat.core.chat_engine import ContinuousChatEngine
from bulk_chat.config.chat_config import ChatConfig

async def main():
    # Crear configuración
    config = ChatConfig()
    config.llm_provider = "openai"
    config.llm_model = "gpt-4"
    config.auto_continue = True
    config.response_interval = 2.0
    
    # Crear motor de chat
    engine = ContinuousChatEngine(
        llm_provider=config.get_llm_provider(),
        auto_continue=config.auto_continue,
        response_interval=config.response_interval,
    )
    
    # Crear sesión
    session = await engine.create_session(
        user_id="user123",
        initial_message="Hola, explícame sobre machine learning",
        auto_continue=True,
    )
    
    # Iniciar chat continuo
    await engine.start_continuous_chat(session.session_id)
    
    # El chat ahora generará respuestas automáticamente
    # Esperar un poco para ver las respuestas
    await asyncio.sleep(10)
    
    # Ver mensajes
    print(f"Mensajes generados: {len(session.messages)}")
    for msg in session.messages:
        print(f"{msg.role}: {msg.content[:100]}...")
    
    # Pausar el chat
    await engine.pause_session(session.session_id, "Pausado por usuario")
    
    # Reanudar
    await engine.resume_session(session.session_id)
    
    # Detener
    await engine.stop_session(session.session_id)

if __name__ == "__main__":
    asyncio.run(main())
```

## 📡 API Endpoints

### Sesiones

- `POST /api/v1/chat/sessions` - Crear nueva sesión
- `GET /api/v1/chat/sessions` - Listar todas las sesiones
- `GET /api/v1/chat/sessions/{session_id}` - Obtener información de sesión
- `DELETE /api/v1/chat/sessions/{session_id}` - Eliminar sesión

### Mensajes

- `POST /api/v1/chat/sessions/{session_id}/messages` - Enviar mensaje
- `GET /api/v1/chat/sessions/{session_id}/messages` - Obtener mensajes

### Control del Chat

- `POST /api/v1/chat/sessions/{session_id}/start` - Iniciar chat continuo
- `POST /api/v1/chat/sessions/{session_id}/pause` - Pausar chat
- `POST /api/v1/chat/sessions/{session_id}/resume` - Reanudar chat
- `POST /api/v1/chat/sessions/{session_id}/stop` - Detener chat

### Streaming

- `GET /api/v1/chat/sessions/{session_id}/stream` - Stream de respuestas (SSE)
- `WS /ws/chat/{session_id}` - WebSocket para streaming bidireccional

### Métricas

- `GET /api/v1/chat/sessions/{session_id}/metrics` - Métricas de sesión
- `GET /api/v1/chat/metrics` - Métricas globales del sistema
- `GET /api/v1/chat/rate-limit/{identifier}` - Estadísticas de rate limiting

### Cache

- `GET /api/v1/chat/cache/stats` - Estadísticas del cache
- `POST /api/v1/chat/cache/clear` - Limpiar cache

### Análisis

- `GET /api/v1/chat/sessions/{session_id}/analyze` - Analizar sesión
- `GET /api/v1/chat/sessions/{session_id}/summary` - Resumen de sesión

### Exportación

- `GET /api/v1/chat/sessions/{session_id}/export/{format}` - Exportar sesión (json, markdown, csv, html, txt)

### Plantillas

- `GET /api/v1/chat/templates` - Listar plantillas
- `POST /api/v1/chat/templates/{template_id}/render` - Renderizar plantilla

### Webhooks

- `POST /api/v1/chat/webhooks` - Registrar webhook
- `GET /api/v1/chat/webhooks` - Listar webhooks

### Autenticación

- `POST /api/v1/auth/register` - Registrar usuario
- `POST /api/v1/auth/login` - Iniciar sesión

### Backups

- `POST /api/v1/chat/backup/create` - Crear backup manual
- `GET /api/v1/chat/backup/list` - Listar backups
- `GET /api/v1/chat/backup/history` - Historial de backups

### Dashboard

- `GET /dashboard` - Dashboard web interactivo con gráficos

### GraphQL

- `POST /graphql` - Endpoint GraphQL para consultas flexibles

### Alertas

- `GET /api/v1/alerts` - Obtener alertas
- `POST /api/v1/alerts/{alert_id}/resolve` - Resolver alerta

### Feature Flags

- `GET /api/v1/feature-flags` - Listar feature flags
- `GET /api/v1/feature-flags/{flag_name}` - Obtener estado de flag
- `POST /api/v1/feature-flags/{flag_name}/enable` - Habilitar flag
- `POST /api/v1/feature-flags/{flag_name}/disable` - Deshabilitar flag

### Versionado de API

- `GET /api/versions` - Información de versiones disponibles

### Clustering

- `GET /api/v1/cluster/info` - Información del cluster

### Analytics Avanzado

- `GET /api/v1/analytics/patterns` - Patrones detectados
- `GET /api/v1/analytics/user/{user_id}/behavior` - Comportamiento de usuario
- `GET /api/v1/analytics/insights` - Insights generales

### Recomendaciones

- `GET /api/v1/recommendations/{user_id}` - Obtener recomendaciones
- `POST /api/v1/recommendations/interaction` - Registrar interacción

### A/B Testing

- `POST /api/v1/ab-testing/experiments` - Crear experimento
- `GET /api/v1/ab-testing/experiments/{experiment_id}/variant` - Obtener variante
- `GET /api/v1/ab-testing/experiments/{experiment_id}/stats` - Estadísticas

### Eventos

- `GET /api/v1/events/history` - Historial de eventos
- `GET /api/v1/events/subscribers` - Conteo de suscriptores

### Seguridad

- `GET /api/v1/security/audit-logs` - Logs de auditoría
- `GET /api/v1/security/stats` - Estadísticas de seguridad

### Internacionalización

- `GET /api/v1/i18n/translate` - Traducir clave
- `GET /api/v1/i18n/languages` - Idiomas soportados

### Workflows

- `POST /api/v1/workflows/execute` - Ejecutar workflow
- `GET /api/v1/workflows` - Listar workflows

### Notificaciones

- `POST /api/v1/notifications/send` - Enviar notificación
- `GET /api/v1/notifications/{user_id}` - Obtener notificaciones
- `POST /api/v1/notifications/{user_id}/read/{notification_id}` - Marcar como leída

### Integraciones

- `POST /api/v1/integrations/call` - Llamar integración
- `GET /api/v1/integrations` - Listar integraciones

### Benchmarking

- `POST /api/v1/benchmark/run` - Ejecutar benchmark
- `GET /api/v1/benchmark/results` - Obtener resultados

### Documentación

- `GET /api/v1/docs/openapi` - Especificación OpenAPI
- `GET /api/v1/docs/markdown` - Documentación Markdown
- `GET /api/v1/docs/endpoints` - Listar endpoints

### Monitoring

- `POST /api/v1/monitoring/metrics` - Registrar métrica
- `GET /api/v1/monitoring/metrics/{metric_name}/stats` - Estadísticas
- `GET /api/v1/monitoring/summary` - Resumen
- `GET /api/v1/monitoring/alerts` - Alertas

### Secretos

- `POST /api/v1/secrets/store` - Almacenar secreto
- `GET /api/v1/secrets/{secret_id}` - Obtener secreto
- `GET /api/v1/secrets` - Listar secretos

### ML Optimizer

- `POST /api/v1/ml-optimizer/record` - Registrar rendimiento
- `POST /api/v1/ml-optimizer/optimize` - Optimizar parámetro
- `POST /api/v1/ml-optimizer/predict` - Predecir rendimiento

### Deployment

- `POST /api/v1/deployment/deploy` - Ejecutar deployment
- `POST /api/v1/deployment/{deployment_id}/rollback` - Hacer rollback
- `GET /api/v1/deployment/current` - Versión actual
- `GET /api/v1/deployment` - Listar deployments

### Reportes

- `POST /api/v1/reports/generate` - Generar reporte
- `GET /api/v1/reports` - Listar reportes
- `GET /api/v1/reports/{report_id}` - Obtener reporte

### Gestión de Usuarios

- `POST /api/v1/users/register` - Registrar usuario
- `POST /api/v1/users/login` - Iniciar sesión
- `GET /api/v1/users` - Listar usuarios
- `GET /api/v1/users/{user_id}` - Obtener usuario

### Búsqueda

- `GET /api/v1/search` - Buscar items
- `POST /api/v1/search/index` - Indexar item
- `GET /api/v1/search/stats` - Estadísticas de búsqueda

### Cola de Mensajes

- `POST /api/v1/queue/enqueue` - Agregar mensaje a la cola
- `GET /api/v1/queue/stats` - Estadísticas de colas
- `GET /api/v1/queue/{queue_name}/size` - Tamaño de cola

### Validación

- `POST /api/v1/validation/validate` - Validar datos
- `POST /api/v1/validation/rules` - Registar regla de validación

### Throttling

- `POST /api/v1/throttle/configure` - Configurar throttling
- `GET /api/v1/throttle/status/{identifier}` - Estado de throttling

### Circuit Breaker

- `GET /api/v1/circuit-breaker/{identifier}/state` - Estado del circuit breaker
- `POST /api/v1/circuit-breaker/{identifier}/reset` - Resetear circuit breaker

### Optimizador Inteligente

- `POST /api/v1/optimizer/record-performance` - Registrar rendimiento
- `POST /api/v1/optimizer/analyze` - Analizar y obtener sugerencias
- `POST /api/v1/optimizer/apply/{suggestion_id}` - Aplicar optimización
- `GET /api/v1/optimizer/applied` - Optimizaciones aplicadas
- `GET /api/v1/optimizer/history` - Historial de optimizaciones

### Aprendizaje Adaptativo

- `POST /api/v1/learning/observe` - Observar resultado para aprendizaje
- `POST /api/v1/learning/predict` - Predecir mejor acción
- `GET /api/v1/learning/patterns` - Obtener patrones aprendidos

### Predicción de Demanda

- `POST /api/v1/demand/record` - Registrar demanda actual
- `POST /api/v1/demand/predict` - Predecir demanda futura
- `POST /api/v1/demand/predict-multiple` - Predecir demanda para múltiples recursos
- `GET /api/v1/demand/history/{resource_type}` - Historial de demanda
- `GET /api/v1/demand/forecasts` - Historial de pronósticos

### Health Checks Inteligentes

- `POST /api/v1/health/register-check` - Registar check de salud
- `POST /api/v1/health/register-metric` - Registrar métrica de salud
- `POST /api/v1/health/update-metric` - Actualizar valor de métrica
- `GET /api/v1/health/check` - Ejecutar todos los health checks
- `GET /api/v1/health/summary` - Resumen de salud

### Auto-Scaling Predictivo

- `POST /api/v1/scaling/evaluate` - Evaluar necesidad de scaling
- `POST /api/v1/scaling/apply` - Aplicar cambio de capacidad
- `GET /api/v1/scaling/history` - Historial de scaling
- `GET /api/v1/scaling/capacity/{resource_type}` - Capacidad actual

### Optimizador de Costos

- `POST /api/v1/costs/set-rate` - Establecer costo por unidad
- `POST /api/v1/costs/record` - Registrar costo
- `POST /api/v1/costs/analyze` - Analizar y obtener sugerencias
- `GET /api/v1/costs/summary` - Resumen de costos
- `GET /api/v1/costs/recent` - Costos recientes

### Alertas Inteligentes

- `POST /api/v1/alerts/record-metric` - Registrar métrica para alertas
- `POST /api/v1/alerts/register-handler` - Registrar handler de alertas
- `GET /api/v1/alerts/recent` - Alertas recientes
- `GET /api/v1/alerts/summary` - Resumen de alertas

### Observabilidad Avanzada

- `POST /api/v1/observability/start-trace` - Iniciar trace
- `POST /api/v1/observability/start-span` - Iniciar span
- `POST /api/v1/observability/end-span` - Finalizar span
- `POST /api/v1/observability/add-log` - Agregar log a span
- `POST /api/v1/observability/record-metric` - Registrar métrica
- `GET /api/v1/observability/trace/{trace_id}` - Obtener trace completo
- `GET /api/v1/observability/traces` - Obtener traces
- `GET /api/v1/observability/metrics` - Obtener métricas
- `GET /api/v1/observability/summary` - Resumen de observabilidad

### Balanceador de Carga Inteligente

- `POST /api/v1/load-balancer/add-node` - Agregar nodo
- `POST /api/v1/load-balancer/remove-node` - Remover nodo
- `POST /api/v1/load-balancer/select-node` - Seleccionar nodo
- `POST /api/v1/load-balancer/record-request` - Registrar petición
- `POST /api/v1/load-balancer/update-health` - Actualizar salud
- `GET /api/v1/load-balancer/stats` - Estadísticas de nodos
- `GET /api/v1/load-balancer/summary` - Resumen del balanceador

### Gestor de Recursos

- `POST /api/v1/resources/set-quota` - Establecer cuota
- `POST /api/v1/resources/allocate` - Asignar recurso
- `POST /api/v1/resources/release` - Liberar recurso
- `GET /api/v1/resources/quota` - Obtener cuota
- `GET /api/v1/resources/usage-history` - Historial de uso
- `POST /api/v1/resources/cleanup-expired` - Limpiar asignaciones expiradas

### Recuperación de Desastres

- `POST /api/v1/disaster-recovery/register-node` - Registrar nodo de replicación
- `POST /api/v1/disaster-recovery/create-point` - Crear punto de recuperación
- `POST /api/v1/disaster-recovery/verify-point` - Verificar punto de recuperación
- `POST /api/v1/disaster-recovery/restore` - Restaurar desde punto
- `POST /api/v1/disaster-recovery/failover` - Iniciar failover
- `GET /api/v1/disaster-recovery/points` - Obtener puntos de recuperación
- `GET /api/v1/disaster-recovery/failover-history` - Historial de failovers
- `GET /api/v1/disaster-recovery/status` - Estado de recuperación

### Seguridad Avanzada

- `POST /api/v1/security/add-rule` - Agregar regla de seguridad
- `POST /api/v1/security/record-event` - Registrar evento de seguridad
- `POST /api/v1/security/failed-auth` - Registrar autenticación fallida
- `GET /api/v1/security/check-blocked` - Verificar si está bloqueado
- `GET /api/v1/security/events` - Obtener eventos de seguridad
- `GET /api/v1/security/summary` - Resumen de seguridad
- `GET /api/v1/security/analyze/{source}` - Analizar comportamiento

### Optimizador Automático

- `POST /api/v1/optimizer/register-parameter` - Registrar parámetro
- `POST /api/v1/optimizer/record-performance` - Registrar rendimiento
- `POST /api/v1/optimizer/optimize` - Optimizar parámetro(s)
- `GET /api/v1/optimizer/parameters` - Estado de parámetros
- `GET /api/v1/optimizer/history` - Historial de optimizaciones
- `GET /api/v1/optimizer/summary` - Resumen de optimizaciones

### Controlador de Tasa Adaptativo

- `POST /api/v1/adaptive-rate/register` - Registrar identificador
- `POST /api/v1/adaptive-rate/record` - Registrar petición
- `POST /api/v1/adaptive-rate/check` - Verificar rate limit
- `GET /api/v1/adaptive-rate/{identifier}` - Obtener límite
- `GET /api/v1/adaptive-rate/{identifier}/history` - Historial de ajustes
- `GET /api/v1/adaptive-rate/summary` - Resumen del controlador

### Gestor Inteligente de Reintentos

- `POST /api/v1/retry/create` - Crear operación con reintentos
- `GET /api/v1/retry/{operation_id}` - Información de operación
- `GET /api/v1/retry/patterns/{operation_type}` - Patrones aprendidos
- `GET /api/v1/retry/summary` - Resumen del gestor

### Gestor de Locks Distribuidos

- `POST /api/v1/locks/acquire` - Adquirir lock
- `POST /api/v1/locks/{lock_id}/release` - Liberar lock
- `POST /api/v1/locks/{lock_id}/renew` - Renovar lock
- `GET /api/v1/locks/{lock_id}` - Información de lock
- `GET /api/v1/locks/resource/{resource_id}` - Lock de recurso
- `POST /api/v1/locks/cleanup` - Limpiar locks expirados
- `GET /api/v1/locks/summary` - Resumen del gestor

### Gestor de Pipelines de Datos

- `POST /api/v1/pipelines/create` - Crear pipeline
- `POST /api/v1/pipelines/{pipeline_id}/add-stage` - Agregar stage
- `POST /api/v1/pipelines/{pipeline_id}/execute` - Ejecutar pipeline
- `GET /api/v1/pipelines/{pipeline_id}` - Información de pipeline
- `GET /api/v1/pipelines/executions/{execution_id}` - Información de ejecución
- `POST /api/v1/pipelines/executions/{execution_id}/cancel` - Cancelar ejecución
- `GET /api/v1/pipelines/executions/history` - Historial de ejecuciones
- `GET /api/v1/pipelines/summary` - Resumen del gestor

### Programador de Eventos

- `POST /api/v1/scheduler/schedule` - Programar evento
- `POST /api/v1/scheduler/{event_id}/pause` - Pausar evento
- `POST /api/v1/scheduler/{event_id}/resume` - Reanudar evento
- `POST /api/v1/scheduler/{event_id}/cancel` - Cancelar evento
- `GET /api/v1/scheduler/{event_id}` - Información de evento
- `GET /api/v1/scheduler/history` - Historial de ejecuciones
- `GET /api/v1/scheduler/summary` - Resumen del scheduler

### Gestor de Degradación Gradual

- `POST /api/v1/degradation/register-service` - Registar servicio
- `POST /api/v1/degradation/register-fallback` - Registar fallback
- `POST /api/v1/degradation/add-rule` - Agregar regla de degradación
- `POST /api/v1/degradation/record-metric` - Registrar métrica
- `POST /api/v1/degradation/record-call` - Registrar llamada a servicio
- `GET /api/v1/degradation/service/{service_id}` - Salud de servicio
- `GET /api/v1/degradation/status` - Estado de degradación
- `GET /api/v1/degradation/history` - Historial de degradación
- `GET /api/v1/degradation/summary` - Resumen del gestor

### Sistema de Precalentamiento de Cache

- `POST /api/v1/cache-warmer/register-rule` - Registrar regla de precalentamiento
- `POST /api/v1/cache-warmer/record-access` - Registrar acceso a cache
- `POST /api/v1/cache-warmer/start` - Iniciar precalentamiento
- `POST /api/v1/cache-warmer/stop` - Detener precalentamiento
- `GET /api/v1/cache-warmer/patterns` - Patrones de acceso
- `GET /api/v1/cache-warmer/statistics` - Estadísticas de precalentamiento
- `GET /api/v1/cache-warmer/summary` - Resumen del warmer

### Sistema de Descarga de Carga

- `POST /api/v1/load-shedder/record-metric` - Registrar métrica de carga
- `POST /api/v1/load-shedder/add-rule` - Agregar regla de descarga
- `POST /api/v1/load-shedder/check-request` - Verificar si aceptar petición
- `GET /api/v1/load-shedder/statistics` - Estadísticas de carga
- `GET /api/v1/load-shedder/history` - Historial de descarga
- `GET /api/v1/load-shedder/summary` - Resumen del shedder

### Resolvedor de Conflictos

- `POST /api/v1/conflicts/register` - Registrar conflicto
- `POST /api/v1/conflicts/{conflict_id}/resolve` - Resolver conflicto
- `POST /api/v1/conflicts/register-rule` - Registrar regla de resolución
- `GET /api/v1/conflicts/{conflict_id}` - Información de conflicto
- `GET /api/v1/conflicts/pending` - Conflictos pendientes
- `GET /api/v1/conflicts/history` - Historial de resoluciones
- `GET /api/v1/conflicts/summary` - Resumen del resolvedor

### Máquina de Estados

- `POST /api/v1/state-machines/create` - Crear máquina de estados
- `POST /api/v1/state-machines/{machine_id}/add-transition` - Agregar transición
- `POST /api/v1/state-machines/{machine_id}/transition` - Realizar transición
- `GET /api/v1/state-machines/{machine_id}` - Información de máquina
- `GET /api/v1/state-machines/history` - Historial de estados
- `GET /api/v1/state-machines/summary` - Resumen del gestor

### Motor de Flujos de Trabajo V2

- `POST /api/v1/workflows-v2/create` - Crear workflow
- `POST /api/v1/workflows-v2/{workflow_id}/add-step` - Agregar step
- `POST /api/v1/workflows-v2/{workflow_id}/execute` - Ejecutar workflow
- `GET /api/v1/workflows-v2/{workflow_id}` - Información de workflow
- `GET /api/v1/workflows-v2/executions/{execution_id}` - Información de ejecución
- `POST /api/v1/workflows-v2/executions/{execution_id}/cancel` - Cancelar ejecución
- `GET /api/v1/workflows-v2/summary` - Resumen del motor

### Bus de Eventos

- `POST /api/v1/events/publish` - Publicar evento
- `POST /api/v1/events/subscribe` - Suscribirse a eventos
- `POST /api/v1/events/unsubscribe` - Desuscribirse de eventos
- `GET /api/v1/events/history` - Historial de eventos
- `GET /api/v1/events/subscriptions` - Suscripciones
- `GET /api/v1/events/summary` - Resumen del bus

### Gestor de Feature Toggles

- `POST /api/v1/feature-toggles/create` - Crear feature toggle
- `GET /api/v1/feature-toggles/{toggle_id}/check` - Verificar toggle
- `POST /api/v1/feature-toggles/{toggle_id}/update` - Actualizar toggle
- `GET /api/v1/feature-toggles/{toggle_id}` - Información de toggle
- `GET /api/v1/feature-toggles/{toggle_id}/statistics` - Estadísticas
- `GET /api/v1/feature-toggles/summary` - Resumen del gestor

### Limitador de Tasa V2

- `POST /api/v1/rate-limiter-v2/add-rule` - Agregar regla
- `POST /api/v1/rate-limiter-v2/check` - Verificar rate limit
- `GET /api/v1/rate-limiter-v2/status` - Estado de rate limiting
- `GET /api/v1/rate-limiter-v2/history` - Historial de bloqueos
- `GET /api/v1/rate-limiter-v2/summary` - Resumen del limitador

### Circuit Breaker V2

- `POST /api/v1/circuit-breakers-v2/create` - Crear circuit breaker
- `GET /api/v1/circuit-breakers-v2/{circuit_id}` - Información
- `POST /api/v1/circuit-breakers-v2/{circuit_id}/reset` - Resetear
- `GET /api/v1/circuit-breakers-v2/summary` - Resumen del gestor

### Optimizador Adaptativo

- `POST /api/v1/optimizer/register-parameter` - Registrar parámetro
- `POST /api/v1/optimizer/add-goal` - Agregar objetivo
- `POST /api/v1/optimizer/record-metric` - Registrar métrica
- `POST /api/v1/optimizer/start` - Iniciar optimización
- `POST /api/v1/optimizer/stop` - Detener optimización
- `GET /api/v1/optimizer/parameter/{parameter_id}` - Información de parámetro
- `GET /api/v1/optimizer/results` - Resultados de optimización
- `GET /api/v1/optimizer/summary` - Resumen del optimizador

### Verificador de Salud V2

- `POST /api/v1/health-v2/register-check` - Registrar health check
- `POST /api/v1/health-v2/{check_id}/run` - Ejecutar check manualmente
- `GET /api/v1/health-v2/overall` - Salud general
- `GET /api/v1/health-v2/{check_id}/history` - Historial de checks
- `GET /api/v1/health-v2/summary` - Resumen del verificador

### Auto Scaler

- `POST /api/v1/auto-scaler/add-rule` - Agregar regla de escalado
- `POST /api/v1/auto-scaler/record-metric` - Registrar métrica
- `POST /api/v1/auto-scaler/set-instances` - Establecer instancias manualmente
- `POST /api/v1/auto-scaler/start` - Iniciar auto-escalado
- `POST /api/v1/auto-scaler/stop` - Detener auto-escalado
- `GET /api/v1/auto-scaler/status` - Estado de escalado
- `GET /api/v1/auto-scaler/history` - Historial de escalado
- `GET /api/v1/auto-scaler/summary` - Resumen del escalador

### Procesador por Lotes

- `POST /api/v1/batch/add-item` - Agregar item a batch
- `POST /api/v1/batch/register-processor` - Registrar procesador
- `GET /api/v1/batch/queue-status` - Estado de cola(s)
- `GET /api/v1/batch/history` - Historial de batches
- `GET /api/v1/batch/summary` - Resumen del procesador

### Monitor de Rendimiento

- `POST /api/v1/performance/record-metric` - Registrar métrica
- `POST /api/v1/performance/record-latency` - Registrar latencia
- `POST /api/v1/performance/create-snapshot` - Crear snapshot
- `GET /api/v1/performance/summary` - Resumen de rendimiento
- `GET /api/v1/performance/metric/{metric_name}` - Historial de métrica
- `GET /api/v1/performance/monitor-summary` - Resumen del monitor

### Gestor de Colas

- `POST /api/v1/queues/create` - Crear cola
- `POST /api/v1/queues/{queue_name}/enqueue` - Encolar mensaje
- `POST /api/v1/queues/{queue_name}/dequeue` - Desencolar mensaje
- `POST /api/v1/queues/messages/{message_id}/ack` - Confirmar mensaje
- `POST /api/v1/queues/messages/{message_id}/nack` - Negar mensaje
- `GET /api/v1/queues/{queue_name}/status` - Estado de cola
- `GET /api/v1/queues/summary` - Resumen del gestor

### Gestor de Conexiones

- `POST /api/v1/connections/register` - Registrar conexión
- `POST /api/v1/connections/{connection_type}/acquire` - Adquirir conexión
- `POST /api/v1/connections/{connection_id}/release` - Liberar conexión
- `POST /api/v1/connections/{connection_id}/close` - Cerrar conexión
- `GET /api/v1/connections/{connection_id}` - Información de conexión
- `GET /api/v1/connections/type/{connection_type}` - Conexiones por tipo
- `GET /api/v1/connections/summary` - Resumen del gestor

### Gestor de Transacciones

- `POST /api/v1/transactions/begin` - Iniciar transacción
- `POST /api/v1/transactions/{transaction_id}/add-operation` - Agregar operación
- `POST /api/v1/transactions/{transaction_id}/commit` - Commit transacción
- `POST /api/v1/transactions/{transaction_id}/rollback` - Rollback transacción
- `GET /api/v1/transactions/{transaction_id}` - Información de transacción
- `GET /api/v1/transactions/history` - Historial de transacciones
- `GET /api/v1/transactions/summary` - Resumen del gestor

### Orquestador de Sagas

- `POST /api/v1/sagas/create` - Crear saga
- `POST /api/v1/sagas/{saga_id}/add-step` - Agregar step
- `POST /api/v1/sagas/{saga_id}/execute` - Ejecutar saga
- `GET /api/v1/sagas/{saga_id}` - Información de saga
- `GET /api/v1/sagas/history` - Historial de sagas
- `GET /api/v1/sagas/summary` - Resumen del orquestador

### Coordinador Distribuido

- `POST /api/v1/coordination/register-node` - Registrar nodo
- `POST /api/v1/coordination/propose` - Proponer valor para consenso
- `GET /api/v1/coordination/leader` - Información del líder
- `GET /api/v1/coordination/status` - Estado de coordinación
- `GET /api/v1/coordination/summary` - Resumen del coordinador

### Malla de Servicios

- `POST /api/v1/mesh/register-service` - Registrar servicio
- `POST /api/v1/mesh/register-instance` - Registrar instancia
- `GET /api/v1/mesh/service/{service_name}/instance` - Obtener instancia
- `POST /api/v1/mesh/instance/{instance_id}/status` - Actualizar estado
- `GET /api/v1/mesh/service/{service_name}/instances` - Instancias de servicio
- `GET /api/v1/mesh/summary` - Resumen de la malla

### Limitador Adaptativo

- `POST /api/v1/throttler/add-rule` - Agregar regla
- `POST /api/v1/throttler/record-metric` - Registrar métrica
- `POST /api/v1/throttler/check` - Verificar throttling
- `GET /api/v1/throttler/{rule_id}/status` - Estado de throttling
- `GET /api/v1/throttler/summary` - Resumen del limitador

### Gestor de Backpressure

- `POST /api/v1/backpressure/add-rule` - Agregar regla
- `POST /api/v1/backpressure/record-metric` - Registrar métrica
- `GET /api/v1/backpressure/{component_id}/level` - Nivel de backpressure
- `POST /api/v1/backpressure/{component_id}/check` - Verificar si aceptar
- `GET /api/v1/backpressure/status` - Estado de backpressure
- `GET /api/v1/backpressure/history` - Historial de backpressure
- `GET /api/v1/backpressure/summary` - Resumen del gestor

### Aprendizaje Federado

- `POST /api/v1/federated-learning/register-client` - Registrar cliente
- `POST /api/v1/federated-learning/start-round` - Iniciar ronda
- `POST /api/v1/federated-learning/submit-update` - Enviar actualización
- `GET /api/v1/federated-learning/global-model` - Obtener modelo global
- `GET /api/v1/federated-learning/round/{round_id}` - Estado de ronda
- `GET /api/v1/federated-learning/summary` - Resumen de aprendizaje

### Gestor de Conocimiento

- `POST /api/v1/knowledge/add` - Agregar entrada de conocimiento
- `POST /api/v1/knowledge/search` - Buscar conocimiento
- `POST /api/v1/knowledge/remove` - Remover entrada
- `POST /api/v1/knowledge/add-relationship` - Agregar relación
- `GET /api/v1/knowledge/related/{entry_id}` - Conocimiento relacionado
- `GET /api/v1/knowledge/stats` - Estadísticas de conocimiento

### Generador Automático

- `POST /api/v1/generator/register-template` - Registrar plantilla
- `POST /api/v1/generator/generate` - Generar artefacto
- `POST /api/v1/generator/generate-batch` - Generar múltiples artefactos
- `GET /api/v1/generator/templates` - Listar plantillas
- `GET /api/v1/generator/template/{template_id}` - Obtener plantilla
- `GET /api/v1/generator/history` - Historial de generaciones
- `GET /api/v1/generator/stats` - Estadísticas de generación

### Recomendador de Arquitectura

- `POST /api/v1/architecture/add-requirement` - Agregar requisito
- `POST /api/v1/architecture/recommend` - Recomendar arquitectura
- `GET /api/v1/architecture/recommendations` - Obtener recomendaciones
- `GET /api/v1/architecture/stats` - Estadísticas

### Gestor de MLOps

- `POST /api/v1/mlops/create-experiment` - Crear experimento
- `POST /api/v1/mlops/update-experiment` - Actualizar experimento
- `POST /api/v1/mlops/register-model` - Registar modelo
- `POST /api/v1/mlops/deploy-model` - Desplegar modelo
- `POST /api/v1/mlops/record-performance` - Registrar rendimiento
- `POST /api/v1/mlops/detect-drift` - Detectar drift
- `GET /api/v1/mlops/experiment/{experiment_id}` - Obtener experimento
- `GET /api/v1/mlops/model/{model_id}` - Obtener modelo
- `GET /api/v1/mlops/drift` - Detecciones de drift
- `GET /api/v1/mlops/summary` - Resumen de MLOps

### Gestor de Dependencias

- `POST /api/v1/dependencies/register` - Registrar dependencia
- `POST /api/v1/dependencies/check-updates` - Verificar actualizaciones
- `POST /api/v1/dependencies/scan-vulnerabilities` - Escanear vulnerabilidades
- `POST /api/v1/dependencies/update` - Actualizar dependencia
- `GET /api/v1/dependencies/tree/{dependency_name}` - Árbol de dependencias
- `GET /api/v1/dependencies/vulnerabilities` - Obtener vulnerabilidades
- `GET /api/v1/dependencies/summary` - Resumen de dependencias

### Gestor de CI/CD

- `POST /api/v1/cicd/register-template` - Registrar plantilla de pipeline
- `POST /api/v1/cicd/create-pipeline` - Crear pipeline
- `POST /api/v1/cicd/create-from-template` - Crear desde plantilla
- `POST /api/v1/cicd/run-pipeline` - Ejecutar pipeline
- `POST /api/v1/cicd/cancel-pipeline` - Cancelar pipeline
- `GET /api/v1/cicd/pipeline/{pipeline_id}` - Obtener pipeline
- `GET /api/v1/cicd/pipeline/{pipeline_id}/logs` - Logs de pipeline
- `GET /api/v1/cicd/summary` - Resumen de CI/CD

### Análisis de Calidad de Código

- `POST /api/v1/code-quality/analyze` - Analizar código
- `GET /api/v1/code-quality/report/{report_id}` - Obtener reporte
- `GET /api/v1/code-quality/summary` - Resumen de calidad

### Métricas de Negocio

- `POST /api/v1/business-metrics/record` - Registrar métrica
- `POST /api/v1/business-metrics/define-kpi` - Definir KPI
- `POST /api/v1/business-metrics/update-kpi` - Actualizar KPI
- `GET /api/v1/business-metrics/kpi/{kpi_id}` - Estado de KPI
- `POST /api/v1/business-metrics/create-funnel` - Crear embudo
- `POST /api/v1/business-metrics/record-funnel-stage` - Registrar stage
- `GET /api/v1/business-metrics/funnel/{funnel_id}` - Análisis de embudo
- `GET /api/v1/business-metrics/trend/{metric_name}` - Tendencias
- `GET /api/v1/business-metrics/summary` - Resumen de negocio

### Control de Versiones

- `POST /api/v1/version-control/create-branch` - Crear branch
- `POST /api/v1/version-control/switch-branch` - Cambiar de branch
- `POST /api/v1/version-control/commit` - Crear commit
- `POST /api/v1/version-control/merge` - Fusionar branch
- `GET /api/v1/version-control/commit/{commit_id}` - Obtener commit
- `GET /api/v1/version-control/branch/{branch_name}/history` - Historial de branch
- `GET /api/v1/version-control/file/{file_path}/history` - Historial de archivo
- `GET /api/v1/version-control/diff` - Diff entre commits
- `GET /api/v1/version-control/summary` - Resumen

### Analizador de Logs

- `POST /api/v1/logs/ingest` - Ingerir log
- `POST /api/v1/logs/search` - Buscar logs
- `GET /api/v1/logs/statistics` - Estadísticas de logs
- `GET /api/v1/logs/patterns` - Coincidencias de patrones
- `GET /api/v1/logs/summary` - Resumen de análisis

### Rendimiento de API

- `POST /api/v1/api-performance/record` - Registrar llamada
- `GET /api/v1/api-performance/endpoint` - Rendimiento de endpoint
- `GET /api/v1/api-performance/slow-endpoints` - Endpoints lentos
- `GET /api/v1/api-performance/error-endpoints` - Endpoints con errores
- `GET /api/v1/api-performance/summary` - Resumen de rendimiento

### Gestión Avanzada de Secretos

- `POST /api/v1/secrets/create` - Crear secreto
- `GET /api/v1/secrets/{secret_id}` - Obtener valor
- `GET /api/v1/secrets/{secret_id}/info` - Información de secreto
- `POST /api/v1/secrets/{secret_id}/rotate` - Rotar secreto
- `POST /api/v1/secrets/{secret_id}/revoke` - Revocar secreto
- `POST /api/v1/secrets/auto-rotate` - Rotación automática
- `GET /api/v1/secrets/access-log` - Log de accesos
- `GET /api/v1/secrets/summary` - Resumen de secretos

### Caché Inteligente

- `POST /api/v1/cache/get` - Obtener valor
- `POST /api/v1/cache/set` - Guardar valor
- `POST /api/v1/cache/invalidate` - Invalidar entrada
- `POST /api/v1/cache/invalidate-pattern` - Invalidar por patrón
- `POST /api/v1/cache/prefetch` - Pre-cargar entrada
- `GET /api/v1/cache/stats` - Estadísticas de caché
- `GET /api/v1/cache/patterns` - Patrones de acceso
- `POST /api/v1/cache/clear` - Limpiar caché

### Analizador de Sentimientos

- `POST /api/v1/sentiment/analyze` - Analizar sentimiento
- `POST /api/v1/sentiment/analyze-batch` - Analizar lote
- `POST /api/v1/sentiment/summary` - Resumen de sentimientos

### Gestor de Tareas

- `POST /api/v1/tasks/create` - Crear tarea
- `POST /api/v1/tasks/{task_id}/update-status` - Actualizar estado
- `POST /api/v1/tasks/{task_id}/update-progress` - Actualizar progreso
- `GET /api/v1/tasks/{task_id}` - Obtener tarea
- `GET /api/v1/tasks/status/{status}` - Tareas por estado
- `GET /api/v1/tasks/assignee/{assignee}` - Tareas por asignado
- `GET /api/v1/tasks/overdue` - Tareas vencidas
- `POST /api/v1/task-lists/create` - Crear lista
- `POST /api/v1/task-lists/{list_id}/add-task` - Agregar tarea
- `GET /api/v1/task-lists/{list_id}` - Obtener lista
- `GET /api/v1/tasks/summary` - Resumen

### Monitor de Recursos

- `GET /api/v1/resources/current` - Métricas actuales
- `GET /api/v1/resources/history/{resource_type}` - Historial
- `GET /api/v1/resources/statistics/{resource_type}` - Estadísticas
- `GET /api/v1/resources/alerts` - Alertas de recursos
- `POST /api/v1/resources/alerts/{alert_id}/resolve` - Resolver alerta
- `GET /api/v1/resources/summary` - Resumen del monitor

### Notificaciones Push

- `POST /api/v1/notifications/send` - Enviar notificación
- `POST /api/v1/notifications/subscribe` - Suscribir usuario
- `POST /api/v1/notifications/unsubscribe` - Desuscribir usuario
- `GET /api/v1/notifications/user/{user_id}` - Notificaciones de usuario
- `POST /api/v1/notifications/{notification_id}/read` - Marcar como leída
- `GET /api/v1/notifications/stats` - Estadísticas de notificaciones

### Sincronización Distribuida

- `POST /api/v1/sync/create-resource` - Crear recurso sincronizado
- `POST /api/v1/sync/update-resource` - Actualizar recurso
- `POST /api/v1/sync/sync-from-remote` - Sincronizar desde remoto
- `POST /api/v1/sync/resolve-conflict` - Resolver conflicto
- `GET /api/v1/sync/resource/{resource_id}` - Obtener recurso
- `GET /api/v1/sync/conflicts` - Obtener conflictos
- `GET /api/v1/sync/summary` - Resumen de sincronización

### Analizador de Queries

- `POST /api/v1/queries/record` - Registrar ejecución
- `GET /api/v1/queries/slow` - Queries lentas
- `GET /api/v1/queries/patterns` - Patrones de queries
- `GET /api/v1/queries/statistics` - Estadísticas de queries
- `GET /api/v1/queries/summary` - Resumen del analizador

### Gestor de Archivos

- `POST /api/v1/files/upload` - Subir archivo
- `GET /api/v1/files/{file_id}` - Descargar archivo
- `GET /api/v1/files/{file_id}/metadata` - Metadatos de archivo
- `GET /api/v1/files/{file_id}/versions` - Versiones de archivo
- `GET /api/v1/files/search` - Buscar archivos
- `DELETE /api/v1/files/{file_id}` - Eliminar archivo
- `POST /api/v1/files/{file_id}/restore` - Restaurar archivo
- `GET /api/v1/files/summary` - Resumen del gestor

### Compresión de Datos

- `POST /api/v1/compression/compress` - Comprimir datos
- `POST /api/v1/compression/decompress` - Descomprimir datos
- `POST /api/v1/compression/find-best` - Encontrar mejor algoritmo
- `GET /api/v1/compression/stats` - Estadísticas de compresión

### Backup Incremental

- `POST /api/v1/backup/create` - Crear backup
- `POST /api/v1/backup/restore` - Restaurar backup
- `GET /api/v1/backup/{backup_id}` - Información de backup
- `GET /api/v1/backup/list` - Listar backups
- `POST /api/v1/backup/create-set` - Crear conjunto de backups
- `GET /api/v1/backup/summary` - Resumen de backups

### Analizador de Red

- `POST /api/v1/network/record` - Registrar métrica
- `GET /api/v1/network/endpoint/{endpoint}` - Estadísticas de endpoint
- `GET /api/v1/network/slow-endpoints` - Endpoints lentos
- `GET /api/v1/network/events` - Eventos de red
- `GET /api/v1/network/summary` - Resumen de red

### Gestor de Configuraciones

- `POST /api/v1/config/register` - Registrar configuración
- `GET /api/v1/config/{config_id}` - Obtener configuración
- `GET /api/v1/config/{config_id}/history` - Historial de configuración
- `GET /api/v1/config/changes` - Cambios de configuración
- `POST /api/v1/config/{config_id}/rollback` - Revertir configuración
- `POST /api/v1/config/{config_id}/subscribe` - Suscribirse a cambios
- `GET /api/v1/config/summary` - Resumen del gestor

### Autenticación MFA

- `POST /api/v1/mfa/setup/totp` - Configurar TOTP
- `POST /api/v1/mfa/setup/sms` - Configurar SMS
- `POST /api/v1/mfa/setup/email` - Configurar Email
- `POST /api/v1/mfa/initiate` - Iniciar proceso MFA
- `POST /api/v1/mfa/verify` - Verificar código MFA
- `GET /api/v1/mfa/status/{user_id}` - Estado MFA del usuario

### Rate Limiter Avanzado

- `POST /api/v1/rate-limit/create-rule` - Crear regla
- `POST /api/v1/rate-limit/check` - Verificar rate limit
- `POST /api/v1/rate-limit/block` - Bloquear identificador
- `GET /api/v1/rate-limit/violations` - Violaciones
- `GET /api/v1/rate-limit/summary` - Resumen

### Analizador de Comportamiento

- `POST /api/v1/behavior/record` - Registar acción
- `GET /api/v1/behavior/profile/{user_id}` - Perfil de usuario
- `GET /api/v1/behavior/high-risk` - Usuarios de alto riesgo
- `GET /api/v1/behavior/anomalies` - Anomalías
- `GET /api/v1/behavior/summary` - Resumen

### Flujo de Eventos

- `POST /api/v1/events/publish` - Publicar evento
- `GET /api/v1/events` - Obtener eventos
- `POST /api/v1/events/subscribe` - Suscribirse
- `GET /api/v1/events/summary` - Resumen

### Analizador de Seguridad

- `POST /api/v1/security/analyze` - Analizar entrada
- `POST /api/v1/security/block` - Bloquear fuente
- `GET /api/v1/security/threats` - Obtener amenazas
- `POST /api/v1/security/threats/{threat_id}/resolve` - Resolver amenaza
- `GET /api/v1/security/summary` - Resumen

### Gestor de Sesiones

- `POST /api/v1/sessions/create` - Crear sesión
- `POST /api/v1/sessions/{session_id}/activity` - Actualizar actividad
- `POST /api/v1/sessions/{session_id}/status` - Actualizar estado
- `GET /api/v1/sessions/{session_id}` - Información de sesión
- `GET /api/v1/sessions/{session_id}/analytics` - Analíticas de sesión
- `GET /api/v1/sessions/active` - Sesiones activas
- `POST /api/v1/sessions/cleanup` - Limpiar sesiones
- `GET /api/v1/sessions/summary` - Resumen

### Métricas en Tiempo Real

- `POST /api/v1/metrics/record` - Registar métrica
- `GET /api/v1/metrics/aggregates/{metric_name}` - Agregados
- `GET /api/v1/metrics` - Obtener métricas
- `POST /api/v1/metrics/alerts/create` - Crear alerta
- `GET /api/v1/metrics/summary` - Resumen

### Optimizador Automático

- `POST /api/v1/optimizer/register-parameter` - Registrar parámetro
- `POST /api/v1/optimizer/record-performance` - Registar performance
- `GET /api/v1/optimizer/parameter/{parameter_name}` - Valor de parámetro
- `GET /api/v1/optimizer/optimizations` - Optimizaciones
- `GET /api/v1/optimizer/summary` - Resumen

### Bulk Operations (Operaciones Masivas)

- `POST /api/v1/bulk/sessions/create` - Crear múltiples sesiones
- `POST /api/v1/bulk/sessions/delete` - Eliminar múltiples sesiones
- `POST /api/v1/bulk/sessions/pause` - Pausar múltiples sesiones
- `POST /api/v1/bulk/sessions/resume` - Reanudar múltiples sesiones
- `POST /api/v1/bulk/messages/send` - Enviar mensaje a múltiples sesiones
- `POST /api/v1/bulk/export/sessions` - Exportar múltiples sesiones
- `GET /api/v1/bulk/export/status/{job_id}` - Estado de exportación
- `POST /api/v1/bulk/analytics/sessions` - Analizar múltiples sesiones
- `POST /api/v1/bulk/cleanup/sessions` - Limpiar sesiones antiguas
- `POST /api/v1/bulk/import/sessions` - Importar múltiples sesiones
- `GET /api/v1/bulk/import/status/{job_id}` - Estado de importación
- `POST /api/v1/bulk/notifications/send` - Enviar notificaciones masivas
- `POST /api/v1/bulk/search/execute` - Ejecutar búsqueda masiva
- `GET /api/v1/bulk/process/status/{job_id}` - Estado de procesamiento
- `POST /api/v1/bulk/process/cancel/{job_id}` - Cancelar job de procesamiento
- `POST /api/v1/bulk/backup/sessions` - Crear backup masivo de sesiones
- `GET /api/v1/bulk/backup/status/{job_id}` - Estado de backup
- `POST /api/v1/bulk/migration/start` - Iniciar migración masiva
- `GET /api/v1/bulk/migration/status/{job_id}` - Estado de migración
- `GET /api/v1/bulk/metrics/stats` - Estadísticas de operaciones bulk
- `GET /api/v1/bulk/metrics/history` - Historial de operaciones bulk
- `GET /api/v1/bulk/metrics/summary` - Resumen de operaciones bulk
- `POST /api/v1/bulk/scheduler/schedule` - Programar operación recurrente
- `GET /api/v1/bulk/scheduler/jobs` - Listar jobs programados
- `POST /api/v1/bulk/scheduler/{job_id}/enable` - Habilitar job programado
- `POST /api/v1/bulk/scheduler/{job_id}/disable` - Deshabilitar job programado
- `GET /api/v1/bulk/rate-limit/stats` - Estadísticas de rate limiting
- `POST /api/v1/bulk/rate-limit/check` - Verificar rate limit
- `POST /api/v1/bulk/auto-creation/start` - Iniciar auto-creación continua (nunca se detiene)
- `POST /api/v1/bulk/auto-creation/stop` - Detener auto-creación
- `GET /api/v1/bulk/auto-creation/stats` - Estadísticas de auto-creación
- `POST /api/v1/bulk/auto-expansion/start` - Iniciar auto-expansión
- `POST /api/v1/bulk/self-sustaining/start` - Iniciar sistema auto-sostenible completo
- `GET /api/v1/bulk/self-sustaining/stats` - Estadísticas del sistema auto-sostenible
- `POST /api/v1/bulk/self-sustaining/ensure-continuity` - Asegurar continuidad
- `POST /api/v1/bulk/infinite-generator/create` - Crear generador infinito

### Health & Performance

- `GET /health` - Health check básico
- `GET /health/detailed` - Health check detallado con recursos del sistema
- `GET /api/v1/performance/metrics` - Métricas de rendimiento
- `GET /api/v1/performance/slow-operations` - Operaciones lentas detectadas
- `GET /api/v1/tasks/queue` - Estado de cola de tareas

## ⚙️ Configuración

### Variables de Entorno

```env
# API Settings
CHAT_API_HOST=0.0.0.0
CHAT_API_PORT=8006

# LLM Settings
LLM_PROVIDER=openai
LLM_MODEL=gpt-4
OPENAI_API_KEY=tu-api-key-aqui
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=2000

# Chat Behavior
AUTO_CONTINUE=true
RESPONSE_INTERVAL=2.0
MAX_CONSECUTIVE_RESPONSES=100
MAX_MESSAGES_PER_SESSION=1000

# CORS
CORS_ORIGINS=*

# Logging
LOG_LEVEL=INFO

# Storage (Persistencia)
STORAGE_TYPE=json  # json o redis
STORAGE_PATH=sessions
REDIS_URL=redis://localhost:6379  # Solo si usas Redis
SESSION_TTL=86400  # TTL en segundos (24 horas)

# Auto-save
AUTO_SAVE=true
SAVE_INTERVAL=30.0  # Segundos entre guardados

# Rate Limiting
RATE_LIMIT_MAX_REQUESTS=60
RATE_LIMIT_WINDOW=60.0
RATE_LIMIT_MAX_CONCURRENT=10

# Cache
ENABLE_CACHE=true
CACHE_SIZE=1000
CACHE_TTL=3600  # 1 hora

# Plugins
ENABLE_PLUGINS=true

# Backups
ENABLE_BACKUPS=true
BACKUP_INTERVAL_HOURS=24
BACKUP_DIRECTORY=backups
```

## 🔄 Flujo de Trabajo

1. **Crear Sesión**: El usuario crea una nueva sesión de chat
2. **Iniciar Chat**: El chat comienza a generar respuestas automáticamente
3. **Generación Continua**: El chat genera respuestas cada `response_interval` segundos
4. **Control del Usuario**: El usuario puede pausar/reanudar en cualquier momento
5. **Detener**: El usuario puede detener completamente el chat

## 🎯 Casos de Uso

- **Brainstorming Continuo**: Generar ideas continuamente sin interrupciones
- **Análisis Progresivo**: Analizar un tema en profundidad con múltiples iteraciones
- **Generación de Contenido**: Crear contenido de forma continua y automática
- **Simulación de Conversación**: Simular conversaciones largas y naturales
- **Asistente Personal**: Asistente que genera respuestas proactivamente
- **Análisis de Datos**: Análisis continuo y generación de insights

## 🆕 Mejoras Recientes (v2.0.0)

Ver [IMPROVEMENTS.md](IMPROVEMENTS.md) y [ADVANCED_FEATURES.md](ADVANCED_FEATURES.md) para detalles completos:

### Mejoras Básicas
- ✅ **Persistencia de Sesiones**: Guardado automático en JSON/Redis
- ✅ **Sistema de Métricas**: Monitoreo completo de rendimiento
- ✅ **Rate Limiting**: Control de tasa de solicitudes
- ✅ **Retry Logic**: Reintentos automáticos con exponential backoff
- ✅ **Auto-guardado**: Guardado automático cada 30 segundos
- ✅ **Carga de Sesiones**: Recuperación automática de sesiones

### Características Avanzadas
- ✅ **WebSockets**: Streaming bidireccional en tiempo real
- ✅ **Cache de Respuestas**: Reduce hasta 80% las llamadas al LLM
- ✅ **Sistema de Plugins**: Extensible con plugins personalizados
- ✅ **Plugins Incluidos**: Sentiment Analyzer, Profanity Filter, Response Enhancer

### Características Enterprise
- ✅ **Análisis Avanzado**: Insights, temas, sentimientos, estadísticas
- ✅ **Exportación Multi-formato**: JSON, Markdown, CSV, HTML, TXT
- ✅ **Sistema de Plantillas**: Mensajes predefinidos con variables
- ✅ **Webhooks**: Integración con sistemas externos en tiempo real

### Características Finales
- ✅ **Autenticación JWT**: Sistema completo con roles y permisos
- ✅ **Backups Automáticos**: Backups programados cada 24h
- ✅ **Dashboard Web**: Interfaz interactiva para monitoreo en tiempo real

### Optimizaciones y Testing
- ✅ **Testing Framework**: Suite completa de tests con pytest
- ✅ **Optimizador de Rendimiento**: Métricas avanzadas y detección de problemas
- ✅ **Logging Estructurado**: Logs JSON con contexto completo
- ✅ **Monitor de Salud**: Health checks avanzados y monitoreo de recursos
- ✅ **Cola de Tareas**: Procesamiento asíncrono con workers

### Características Ultimate
- ✅ **API GraphQL**: Consultas flexibles solo con los campos necesarios
- ✅ **Sistema de Alertas**: Alertas multi-nivel con handlers personalizables
- ✅ **Dashboard Mejorado**: Gráficos en tiempo real con Chart.js

### Enterprise Plus
- ✅ **Clustering Distribuido**: Escalabilidad horizontal con consistent hashing
- ✅ **Feature Flags**: Activación/desactivación dinámica de características
- ✅ **Versionado de API**: Gestión completa con deprecación y migración
- ✅ **Analytics Avanzado**: Detección de patrones, análisis de comportamiento y ML

### Ultimate Plus
- ✅ **Recomendaciones ML**: Sistema de recomendaciones con collaborative filtering
- ✅ **A/B Testing**: Framework completo para experimentos controlados
- ✅ **Sistema de Eventos**: Pub/sub para eventos en tiempo real con historial
- ✅ **Seguridad Avanzada**: Audit logs, sanitización, validación y tokens seguros
- ✅ **Internacionalización**: Soporte multi-idioma con 9 idiomas incluidos

### Final Plus
- ✅ **Workflows**: Sistema completo de automatización con pasos, condiciones y retry
- ✅ **Notificaciones Push**: Sistema de notificaciones en tiempo real con suscripciones
- ✅ **Integraciones**: Sistema de integraciones con Webhooks, REST API y más
- ✅ **Benchmarking**: Sistema de performance testing con métricas P95/P99

### Ultimate Final
- ✅ **Documentación Automática**: Generación de OpenAPI 3.0 y Markdown
- ✅ **Monitoring Avanzado**: Sistema de métricas en tiempo real con alertas
- ✅ **Gestión de Secretos**: Almacenamiento seguro con encriptación
- ✅ **ML Optimizer**: Optimización de parámetros basada en datos históricos

### Production Ready
- ✅ **Deployment Automático**: Sistema completo de deployment con rollback automático
- ✅ **Reportes Automatizados**: Generación programada de reportes (diarios, semanales, mensuales)

### User & Search
- ✅ **Gestión de Usuarios Avanzada**: Sistema completo con roles (USER, ADMIN, MODERATOR, PREMIUM), permisos y estados
- ✅ **Búsqueda Avanzada**: Motor de búsqueda con índices, scoring, highlights y filtros

### Advanced Systems
- ✅ **Cola de Mensajes**: Sistema de cola con prioridades (LOW, MEDIUM, HIGH, URGENT), reintentos y procesamiento asíncrono
- ✅ **Validación Avanzada**: Sistema de validación con reglas personalizadas, validadores por defecto (email, URL, phone) y validadores custom
- ✅ **Throttling**: Sistema de throttling configurable con ventanas de tiempo y límites por identificador
- ✅ **Circuit Breaker**: Sistema de circuit breaker con estados (CLOSED, OPEN, HALF_OPEN) para proteger servicios externos

### AI & Learning Systems
- ✅ **Optimizador Inteligente**: Optimización automática con ML que analiza rendimiento y sugiere mejoras
- ✅ **Aprendizaje Adaptativo**: Sistema que aprende patrones de uso y predice mejores acciones automáticamente
- ✅ **Predicción de Demanda**: Sistema ML para predecir demanda futura de recursos con confianza y tendencias
- ✅ **Health Checks Inteligentes**: Sistema avanzado de health checks con auto-diagnóstico, métricas y auto-recovery
- ✅ **Auto-Scaling Predictivo**: Sistema que escala recursos proactivamente basado en predicciones de demanda
- ✅ **Optimizador de Costos**: Análisis y optimización de costos con sugerencias automáticas de ahorro
- ✅ **Alertas Inteligentes**: Detección proactiva de anomalías usando ML con sistema de handlers personalizables
- ✅ **Observabilidad Avanzada**: Sistema de tracing distribuido, métricas detalladas y logging estructurado
- ✅ **Balanceador de Carga Inteligente**: Balanceo adaptativo con múltiples algoritmos y detección de salud
- ✅ **Gestor de Recursos**: Sistema avanzado de gestión de recursos con límites, cuotas y asignación inteligente
- ✅ **Recuperación de Desastres**: Sistema avanzado con replicación, failover automático y restauración
- ✅ **Seguridad Avanzada**: Detección de amenazas, análisis de comportamiento y protección automática
- ✅ **Optimizador Automático**: Ajuste dinámico de parámetros basado en métricas de rendimiento
- ✅ **Aprendizaje Federado**: Entrenamiento distribuido de modelos sin compartir datos
- ✅ **Gestor de Conocimiento**: Sistema de gestión de conocimiento con indexación semántica y búsqueda vectorial
- ✅ **Generador Automático**: Generación automática de código, configuración y documentación
- ✅ **Recomendador de Arquitectura**: Recomendaciones de arquitecturas y patrones basados en requisitos
- ✅ **Gestor de MLOps**: Gestión completa de operaciones de ML con versionado, experimentos y monitoreo de drift
- ✅ **Gestor de Dependencias**: Gestión de dependencias con análisis de vulnerabilidades y actualizaciones
- ✅ **Gestor de CI/CD**: Pipelines de integración y despliegue continuo con stages y automatización
- ✅ **Análisis de Calidad de Código**: Métricas de código, detección de code smells y sugerencias de mejora
- ✅ **Métricas de Negocio**: KPIs, análisis de conversión, embudos y tendencias empresariales
- ✅ **Control de Versiones**: Sistema de versionado con commits, branches, diffs y merge
- ✅ **Analizador de Logs**: Análisis avanzado de logs con búsqueda, filtrado y detección de patrones
- ✅ **Rendimiento de API**: Métricas detalladas de API con profiling y detección de endpoints lentos
- ✅ **Gestión Avanzada de Secretos**: Rotación automática, encriptación y auditoría completa
- ✅ **Caché Inteligente**: Caché avanzado con invalidación inteligente, prefetching y análisis de patrones
- ✅ **Analizador de Sentimientos**: Análisis de sentimientos con detección de emociones y polaridad
- ✅ **Gestor de Tareas**: Sistema de gestión de tareas con prioridades, dependencias y seguimiento
- ✅ **Monitor de Recursos**: Monitoreo de recursos en tiempo real con alertas y análisis de tendencias
- ✅ **Notificaciones Push**: Sistema de notificaciones push en tiempo real con múltiples canales
- ✅ **Sincronización Distribuida**: Sistema de sincronización con resolución de conflictos
- ✅ **Analizador de Queries**: Análisis de performance de queries con detección de queries lentas
- ✅ **Gestor de Archivos**: Gestión avanzada de archivos con versionado, compresión y metadatos
- ✅ **Compresión de Datos**: Sistema de compresión con múltiples algoritmos y análisis de eficiencia
- ✅ **Backup Incremental**: Sistema de backup incremental con deduplicación y restauración
- ✅ **Analizador de Red**: Análisis de red con monitoreo de tráfico, latencia y detección de problemas
- ✅ **Gestor de Configuraciones**: Gestión avanzada de configuraciones con validación, versionado y hot-reload
- ✅ **Autenticación MFA**: Autenticación multi-factor con TOTP, SMS, Email y backup codes
- ✅ **Rate Limiter Avanzado**: Rate limiting con múltiples estrategias y análisis de patrones
- ✅ **Analizador de Comportamiento**: Análisis de comportamiento de usuarios con detección de anomalías
- ✅ **Flujo de Eventos**: Gestión de eventos en tiempo real con pub/sub y filtrado
- ✅ **Analizador de Seguridad**: Detección de amenazas, vulnerabilidades y patrones sospechosos
- ✅ **Gestor de Sesiones**: Gestión avanzada de sesiones con tracking, análisis y optimización
- ✅ **Métricas en Tiempo Real**: Sistema de métricas con agregación, alertas y visualización
- ✅ **Optimizador Automático**: Optimización automática con análisis de performance y ajuste dinámico
- ✅ **Análisis Predictivo**: Sistema de análisis predictivo con ML para predecir tendencias y comportamientos
- ✅ **Asignador de Recursos**: Sistema avanzado de asignación de recursos con cuotas y prioridades
- ✅ **Orquestador de Servicios**: Orquestación de servicios con gestión de dependencias y health checks
- ✅ **Perfilador de Rendimiento**: Sistema avanzado de profiling con análisis detallado y optimización

Ver [ENTERPRISE_FEATURES.md](ENTERPRISE_FEATURES.md), [FINAL_FEATURES.md](FINAL_FEATURES.md), [OPTIMIZATIONS.md](OPTIMIZATIONS.md), [ULTIMATE_FEATURES.md](ULTIMATE_FEATURES.md), [ENTERPRISE_PLUS.md](ENTERPRISE_PLUS.md), [ULTIMATE_PLUS.md](ULTIMATE_PLUS.md), [FINAL_PLUS.md](FINAL_PLUS.md), [ULTIMATE_FINAL.md](ULTIMATE_FINAL.md) y [BULK_OPERATIONS.md](BULK_OPERATIONS.md) para más detalles.

## 🔒 Seguridad

- Validación de sesiones
- Rate limiting (configurable)
- Sanitización de inputs
- Control de acceso por sesión

## 📊 Monitoreo

El sistema incluye:
- Health checks
- Métricas de sesiones activas
- Logging detallado
- Métricas de rendimiento (opcional con Prometheus)

## 🐛 Troubleshooting

### El chat no genera respuestas

1. Verificar que la sesión esté activa: `GET /api/v1/chat/sessions/{session_id}`
2. Verificar que `auto_continue` esté en `true`
3. Verificar logs para errores del LLM
4. Verificar que el API key del LLM sea válido

### El chat no se pausa

1. Verificar que el `session_id` sea correcto
2. Verificar el estado de la sesión
3. Revisar logs para errores

### Respuestas muy lentas

1. Ajustar `response_interval` para más tiempo entre respuestas
2. Verificar la latencia del proveedor de LLM
3. Considerar usar un modelo más rápido

## 🤝 Contribuir

Las contribuciones son bienvenidas. Por favor:

1. Fork el repositorio
2. Crea un branch para tu feature
3. Commit tus cambios
4. Push al branch
5. Abre un Pull Request

## 📝 Licencia

Este proyecto está bajo la licencia MIT.

## 🙏 Agradecimientos

- Blatam Academy
- Comunidad de desarrolladores

---

**Bulk Chat** - Chat continuo que no se detiene hasta que tú lo detengas 🚀


## 🚀 Descripción

**Bulk Chat** es un sistema de chat continuo y proactivo similar a ChatGPT, pero con la característica única de que **no se detiene automáticamente**. El chat genera respuestas de forma continua hasta que el usuario explícitamente lo pause.

### Características Principales

- ✅ **Chat Continuo**: Genera respuestas automáticamente sin detenerse
- ✅ **Control de Pausa**: El usuario puede pausar/reanudar el chat en cualquier momento
- ✅ **Streaming en Tiempo Real**: Soporte para Server-Sent Events (SSE)
- ✅ **Múltiples Sesiones**: Soporte para múltiples sesiones de chat simultáneas
- ✅ **Integración con LLMs**: Soporte para OpenAI, Anthropic y otros proveedores
- ✅ **API REST Completa**: Endpoints para control total del chat
- ✅ **Persistencia de Sesiones**: Guardado automático en JSON o Redis
- ✅ **Sistema de Métricas**: Monitoreo completo de rendimiento y uso
- ✅ **Rate Limiting**: Control de tasa de solicitudes para prevenir abuso
- ✅ **Retry Logic**: Reintentos automáticos con exponential backoff
- ✅ **Auto-guardado**: Guardado automático cada 30 segundos (configurable)
- ✅ **WebSockets**: Streaming bidireccional en tiempo real
- ✅ **Cache de Respuestas**: Reduce llamadas duplicadas al LLM
- ✅ **Sistema de Plugins**: Extensible con plugins personalizados
- ✅ **Análisis de Conversaciones**: Insights y estadísticas avanzadas
- ✅ **Exportación Multi-formato**: JSON, Markdown, CSV, HTML, TXT
- ✅ **Sistema de Plantillas**: Mensajes predefinidos con variables
- ✅ **Webhooks**: Notificaciones en tiempo real a sistemas externos
- ✅ **Autenticación JWT**: Sistema completo de auth con roles
- ✅ **Backups Automáticos**: Backups programados y restauración
- ✅ **Dashboard Web**: Interfaz web interactiva para monitoreo
- ✅ **Testing Framework**: Tests completos con pytest
- ✅ **Optimizador de Rendimiento**: Métricas P50/P95/P99 y detección de cuellos de botella
- ✅ **Logging Estructurado**: Logs JSON con contexto
- ✅ **Monitor de Salud**: Health checks avanzados del sistema
- ✅ **Cola de Tareas**: Procesamiento asíncrono en background
- ✅ **API GraphQL**: Consultas flexibles y eficientes
- ✅ **Sistema de Alertas**: Alertas y notificaciones avanzadas
- ✅ **Dashboard Mejorado**: Gráficos en tiempo real con Chart.js
- ✅ **Clustering Distribuido**: Escalabilidad horizontal con distribución automática
- ✅ **Feature Flags**: Control dinámico de características sin redesplegar
- ✅ **Versionado de API**: Gestión completa de versiones con deprecación
- ✅ **Analytics Avanzado**: Detección de patrones y análisis de comportamiento
- ✅ **Recomendaciones ML**: Sistema de recomendaciones con collaborative filtering
- ✅ **A/B Testing**: Framework completo para experimentos controlados
- ✅ **Sistema de Eventos**: Pub/sub para eventos en tiempo real
- ✅ **Seguridad Avanzada**: Audit logs, sanitización y validación
- ✅ **Internacionalización**: Soporte multi-idioma (i18n)
- ✅ **Workflows**: Sistema de automatización de tareas
- ✅ **Notificaciones Push**: Sistema de notificaciones en tiempo real
- ✅ **Integraciones**: Sistema de integraciones con servicios externos
- ✅ **Benchmarking**: Sistema de performance testing
- ✅ **Documentación Automática**: Generación de OpenAPI y Markdown
- ✅ **Monitoring Avanzado**: Sistema de métricas y alertas
- ✅ **Gestión de Secretos**: Sistema seguro de secretos y configuración
- ✅ **ML Optimizer**: Optimización basada en machine learning
- ✅ **Deployment Automático**: Sistema de deployment y rollback
- ✅ **Reportes Automatizados**: Generación automática de reportes
- ✅ **Gestión de Usuarios**: Sistema completo con roles y permisos
- ✅ **Búsqueda Avanzada**: Motor de búsqueda con índices y filtros
- ✅ **Cola de Mensajes**: Sistema de cola con prioridades y reintentos
- ✅ **Validación Avanzada**: Sistema de validación con reglas personalizadas

## 📋 Requisitos

- Python 3.8+
- FastAPI
- Un proveedor de LLM (OpenAI, Anthropic, etc.)

## 🔧 Instalación

### Opción 1: Instalación Automática (Recomendado)

```bash
# Instalación automática
python install.py

# Verificar instalación
python verify_setup.py
```

### Opción 2: Instalación Manual

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Verificar que todo esté listo
python verify_setup.py

# 3. Configurar variables de entorno (opcional)
# Copiar .env.example a .env y editar con tus API keys
```

**Nota**: Si no tienes una API key, puedes usar el modo `mock`:
```bash
python -m bulk_chat.main --llm-provider mock
```

### Scripts de Inicio Rápido

**Windows:**
```bash
start.bat
start.bat openai
```

**Linux/Mac:**
```bash
chmod +x start.sh
./start.sh
./start.sh openai
```

**Python (Multiplataforma):**
```bash
python run.py server
python run.py server --provider openai
```

Ver [COMMANDS.md](COMMANDS.md) para más comandos útiles.

## 🚀 Uso Rápido

### Iniciar el servidor

```bash
# Uso básico
python -m bulk_chat.main

# Con opciones personalizadas
python -m bulk_chat.main --host 0.0.0.0 --port 8006 --llm-provider openai --llm-model gpt-4
```

### Ejemplo de uso con cURL

```bash
# 1. Crear una sesión de chat
curl -X POST "http://localhost:8006/api/v1/chat/sessions" \
  -H "Content-Type: application/json" \
  -d '{
    "initial_message": "Hola, quiero que me expliques sobre inteligencia artificial",
    "auto_continue": true
  }'

# Respuesta:
# {
#   "session_id": "abc123...",
#   "state": "active",
#   "is_paused": false,
#   "message_count": 1,
#   "auto_continue": true
# }

# 2. El chat comenzará a generar respuestas automáticamente
# Puedes ver los mensajes en tiempo real:

curl "http://localhost:8006/api/v1/chat/sessions/{session_id}/messages"

# 3. Pausar el chat
curl -X POST "http://localhost:8006/api/v1/chat/sessions/{session_id}/pause" \
  -H "Content-Type: application/json" \
  -d '{"action": "pause", "reason": "Usuario pausó"}'

# 4. Reanudar el chat
curl -X POST "http://localhost:8006/api/v1/chat/sessions/{session_id}/resume"

# 5. Detener completamente el chat
curl -X POST "http://localhost:8006/api/v1/chat/sessions/{session_id}/stop"
```

### Ejemplo con Python

```python
import asyncio
from bulk_chat.core.chat_engine import ContinuousChatEngine
from bulk_chat.config.chat_config import ChatConfig

async def main():
    # Crear configuración
    config = ChatConfig()
    config.llm_provider = "openai"
    config.llm_model = "gpt-4"
    config.auto_continue = True
    config.response_interval = 2.0
    
    # Crear motor de chat
    engine = ContinuousChatEngine(
        llm_provider=config.get_llm_provider(),
        auto_continue=config.auto_continue,
        response_interval=config.response_interval,
    )
    
    # Crear sesión
    session = await engine.create_session(
        user_id="user123",
        initial_message="Hola, explícame sobre machine learning",
        auto_continue=True,
    )
    
    # Iniciar chat continuo
    await engine.start_continuous_chat(session.session_id)
    
    # El chat ahora generará respuestas automáticamente
    # Esperar un poco para ver las respuestas
    await asyncio.sleep(10)
    
    # Ver mensajes
    print(f"Mensajes generados: {len(session.messages)}")
    for msg in session.messages:
        print(f"{msg.role}: {msg.content[:100]}...")
    
    # Pausar el chat
    await engine.pause_session(session.session_id, "Pausado por usuario")
    
    # Reanudar
    await engine.resume_session(session.session_id)
    
    # Detener
    await engine.stop_session(session.session_id)

if __name__ == "__main__":
    asyncio.run(main())
```

## 📡 API Endpoints

### Sesiones

- `POST /api/v1/chat/sessions` - Crear nueva sesión
- `GET /api/v1/chat/sessions` - Listar todas las sesiones
- `GET /api/v1/chat/sessions/{session_id}` - Obtener información de sesión
- `DELETE /api/v1/chat/sessions/{session_id}` - Eliminar sesión

### Mensajes

- `POST /api/v1/chat/sessions/{session_id}/messages` - Enviar mensaje
- `GET /api/v1/chat/sessions/{session_id}/messages` - Obtener mensajes

### Control del Chat

- `POST /api/v1/chat/sessions/{session_id}/start` - Iniciar chat continuo
- `POST /api/v1/chat/sessions/{session_id}/pause` - Pausar chat
- `POST /api/v1/chat/sessions/{session_id}/resume` - Reanudar chat
- `POST /api/v1/chat/sessions/{session_id}/stop` - Detener chat

### Streaming

- `GET /api/v1/chat/sessions/{session_id}/stream` - Stream de respuestas (SSE)
- `WS /ws/chat/{session_id}` - WebSocket para streaming bidireccional

### Métricas

- `GET /api/v1/chat/sessions/{session_id}/metrics` - Métricas de sesión
- `GET /api/v1/chat/metrics` - Métricas globales del sistema
- `GET /api/v1/chat/rate-limit/{identifier}` - Estadísticas de rate limiting

### Cache

- `GET /api/v1/chat/cache/stats` - Estadísticas del cache
- `POST /api/v1/chat/cache/clear` - Limpiar cache

### Análisis

- `GET /api/v1/chat/sessions/{session_id}/analyze` - Analizar sesión
- `GET /api/v1/chat/sessions/{session_id}/summary` - Resumen de sesión

### Exportación

- `GET /api/v1/chat/sessions/{session_id}/export/{format}` - Exportar sesión (json, markdown, csv, html, txt)

### Plantillas

- `GET /api/v1/chat/templates` - Listar plantillas
- `POST /api/v1/chat/templates/{template_id}/render` - Renderizar plantilla

### Webhooks

- `POST /api/v1/chat/webhooks` - Registrar webhook
- `GET /api/v1/chat/webhooks` - Listar webhooks

### Autenticación

- `POST /api/v1/auth/register` - Registrar usuario
- `POST /api/v1/auth/login` - Iniciar sesión

### Backups

- `POST /api/v1/chat/backup/create` - Crear backup manual
- `GET /api/v1/chat/backup/list` - Listar backups
- `GET /api/v1/chat/backup/history` - Historial de backups

### Dashboard

- `GET /dashboard` - Dashboard web interactivo con gráficos

### GraphQL

- `POST /graphql` - Endpoint GraphQL para consultas flexibles

### Alertas

- `GET /api/v1/alerts` - Obtener alertas
- `POST /api/v1/alerts/{alert_id}/resolve` - Resolver alerta

### Feature Flags

- `GET /api/v1/feature-flags` - Listar feature flags
- `GET /api/v1/feature-flags/{flag_name}` - Obtener estado de flag
- `POST /api/v1/feature-flags/{flag_name}/enable` - Habilitar flag
- `POST /api/v1/feature-flags/{flag_name}/disable` - Deshabilitar flag

### Versionado de API

- `GET /api/versions` - Información de versiones disponibles

### Clustering

- `GET /api/v1/cluster/info` - Información del cluster

### Analytics Avanzado

- `GET /api/v1/analytics/patterns` - Patrones detectados
- `GET /api/v1/analytics/user/{user_id}/behavior` - Comportamiento de usuario
- `GET /api/v1/analytics/insights` - Insights generales

### Recomendaciones

- `GET /api/v1/recommendations/{user_id}` - Obtener recomendaciones
- `POST /api/v1/recommendations/interaction` - Registrar interacción

### A/B Testing

- `POST /api/v1/ab-testing/experiments` - Crear experimento
- `GET /api/v1/ab-testing/experiments/{experiment_id}/variant` - Obtener variante
- `GET /api/v1/ab-testing/experiments/{experiment_id}/stats` - Estadísticas

### Eventos

- `GET /api/v1/events/history` - Historial de eventos
- `GET /api/v1/events/subscribers` - Conteo de suscriptores

### Seguridad

- `GET /api/v1/security/audit-logs` - Logs de auditoría
- `GET /api/v1/security/stats` - Estadísticas de seguridad

### Internacionalización

- `GET /api/v1/i18n/translate` - Traducir clave
- `GET /api/v1/i18n/languages` - Idiomas soportados

### Workflows

- `POST /api/v1/workflows/execute` - Ejecutar workflow
- `GET /api/v1/workflows` - Listar workflows

### Notificaciones

- `POST /api/v1/notifications/send` - Enviar notificación
- `GET /api/v1/notifications/{user_id}` - Obtener notificaciones
- `POST /api/v1/notifications/{user_id}/read/{notification_id}` - Marcar como leída

### Integraciones

- `POST /api/v1/integrations/call` - Llamar integración
- `GET /api/v1/integrations` - Listar integraciones

### Benchmarking

- `POST /api/v1/benchmark/run` - Ejecutar benchmark
- `GET /api/v1/benchmark/results` - Obtener resultados

### Documentación

- `GET /api/v1/docs/openapi` - Especificación OpenAPI
- `GET /api/v1/docs/markdown` - Documentación Markdown
- `GET /api/v1/docs/endpoints` - Listar endpoints

### Monitoring

- `POST /api/v1/monitoring/metrics` - Registrar métrica
- `GET /api/v1/monitoring/metrics/{metric_name}/stats` - Estadísticas
- `GET /api/v1/monitoring/summary` - Resumen
- `GET /api/v1/monitoring/alerts` - Alertas

### Secretos

- `POST /api/v1/secrets/store` - Almacenar secreto
- `GET /api/v1/secrets/{secret_id}` - Obtener secreto
- `GET /api/v1/secrets` - Listar secretos

### ML Optimizer

- `POST /api/v1/ml-optimizer/record` - Registrar rendimiento
- `POST /api/v1/ml-optimizer/optimize` - Optimizar parámetro
- `POST /api/v1/ml-optimizer/predict` - Predecir rendimiento

### Deployment

- `POST /api/v1/deployment/deploy` - Ejecutar deployment
- `POST /api/v1/deployment/{deployment_id}/rollback` - Hacer rollback
- `GET /api/v1/deployment/current` - Versión actual
- `GET /api/v1/deployment` - Listar deployments

### Reportes

- `POST /api/v1/reports/generate` - Generar reporte
- `GET /api/v1/reports` - Listar reportes
- `GET /api/v1/reports/{report_id}` - Obtener reporte

### Gestión de Usuarios

- `POST /api/v1/users/register` - Registrar usuario
- `POST /api/v1/users/login` - Iniciar sesión
- `GET /api/v1/users` - Listar usuarios
- `GET /api/v1/users/{user_id}` - Obtener usuario

### Búsqueda

- `GET /api/v1/search` - Buscar items
- `POST /api/v1/search/index` - Indexar item
- `GET /api/v1/search/stats` - Estadísticas de búsqueda

### Cola de Mensajes

- `POST /api/v1/queue/enqueue` - Agregar mensaje a la cola
- `GET /api/v1/queue/stats` - Estadísticas de colas
- `GET /api/v1/queue/{queue_name}/size` - Tamaño de cola

### Validación

- `POST /api/v1/validation/validate` - Validar datos
- `POST /api/v1/validation/rules` - Registar regla de validación

### Throttling

- `POST /api/v1/throttle/configure` - Configurar throttling
- `GET /api/v1/throttle/status/{identifier}` - Estado de throttling

### Circuit Breaker

- `GET /api/v1/circuit-breaker/{identifier}/state` - Estado del circuit breaker
- `POST /api/v1/circuit-breaker/{identifier}/reset` - Resetear circuit breaker

### Optimizador Inteligente

- `POST /api/v1/optimizer/record-performance` - Registrar rendimiento
- `POST /api/v1/optimizer/analyze` - Analizar y obtener sugerencias
- `POST /api/v1/optimizer/apply/{suggestion_id}` - Aplicar optimización
- `GET /api/v1/optimizer/applied` - Optimizaciones aplicadas
- `GET /api/v1/optimizer/history` - Historial de optimizaciones

### Aprendizaje Adaptativo

- `POST /api/v1/learning/observe` - Observar resultado para aprendizaje
- `POST /api/v1/learning/predict` - Predecir mejor acción
- `GET /api/v1/learning/patterns` - Obtener patrones aprendidos

### Predicción de Demanda

- `POST /api/v1/demand/record` - Registrar demanda actual
- `POST /api/v1/demand/predict` - Predecir demanda futura
- `POST /api/v1/demand/predict-multiple` - Predecir demanda para múltiples recursos
- `GET /api/v1/demand/history/{resource_type}` - Historial de demanda
- `GET /api/v1/demand/forecasts` - Historial de pronósticos

### Health Checks Inteligentes

- `POST /api/v1/health/register-check` - Registar check de salud
- `POST /api/v1/health/register-metric` - Registrar métrica de salud
- `POST /api/v1/health/update-metric` - Actualizar valor de métrica
- `GET /api/v1/health/check` - Ejecutar todos los health checks
- `GET /api/v1/health/summary` - Resumen de salud

### Auto-Scaling Predictivo

- `POST /api/v1/scaling/evaluate` - Evaluar necesidad de scaling
- `POST /api/v1/scaling/apply` - Aplicar cambio de capacidad
- `GET /api/v1/scaling/history` - Historial de scaling
- `GET /api/v1/scaling/capacity/{resource_type}` - Capacidad actual

### Optimizador de Costos

- `POST /api/v1/costs/set-rate` - Establecer costo por unidad
- `POST /api/v1/costs/record` - Registrar costo
- `POST /api/v1/costs/analyze` - Analizar y obtener sugerencias
- `GET /api/v1/costs/summary` - Resumen de costos
- `GET /api/v1/costs/recent` - Costos recientes

### Alertas Inteligentes

- `POST /api/v1/alerts/record-metric` - Registrar métrica para alertas
- `POST /api/v1/alerts/register-handler` - Registrar handler de alertas
- `GET /api/v1/alerts/recent` - Alertas recientes
- `GET /api/v1/alerts/summary` - Resumen de alertas

### Observabilidad Avanzada

- `POST /api/v1/observability/start-trace` - Iniciar trace
- `POST /api/v1/observability/start-span` - Iniciar span
- `POST /api/v1/observability/end-span` - Finalizar span
- `POST /api/v1/observability/add-log` - Agregar log a span
- `POST /api/v1/observability/record-metric` - Registrar métrica
- `GET /api/v1/observability/trace/{trace_id}` - Obtener trace completo
- `GET /api/v1/observability/traces` - Obtener traces
- `GET /api/v1/observability/metrics` - Obtener métricas
- `GET /api/v1/observability/summary` - Resumen de observabilidad

### Balanceador de Carga Inteligente

- `POST /api/v1/load-balancer/add-node` - Agregar nodo
- `POST /api/v1/load-balancer/remove-node` - Remover nodo
- `POST /api/v1/load-balancer/select-node` - Seleccionar nodo
- `POST /api/v1/load-balancer/record-request` - Registrar petición
- `POST /api/v1/load-balancer/update-health` - Actualizar salud
- `GET /api/v1/load-balancer/stats` - Estadísticas de nodos
- `GET /api/v1/load-balancer/summary` - Resumen del balanceador

### Gestor de Recursos

- `POST /api/v1/resources/set-quota` - Establecer cuota
- `POST /api/v1/resources/allocate` - Asignar recurso
- `POST /api/v1/resources/release` - Liberar recurso
- `GET /api/v1/resources/quota` - Obtener cuota
- `GET /api/v1/resources/usage-history` - Historial de uso
- `POST /api/v1/resources/cleanup-expired` - Limpiar asignaciones expiradas

### Recuperación de Desastres

- `POST /api/v1/disaster-recovery/register-node` - Registrar nodo de replicación
- `POST /api/v1/disaster-recovery/create-point` - Crear punto de recuperación
- `POST /api/v1/disaster-recovery/verify-point` - Verificar punto de recuperación
- `POST /api/v1/disaster-recovery/restore` - Restaurar desde punto
- `POST /api/v1/disaster-recovery/failover` - Iniciar failover
- `GET /api/v1/disaster-recovery/points` - Obtener puntos de recuperación
- `GET /api/v1/disaster-recovery/failover-history` - Historial de failovers
- `GET /api/v1/disaster-recovery/status` - Estado de recuperación

### Seguridad Avanzada

- `POST /api/v1/security/add-rule` - Agregar regla de seguridad
- `POST /api/v1/security/record-event` - Registrar evento de seguridad
- `POST /api/v1/security/failed-auth` - Registrar autenticación fallida
- `GET /api/v1/security/check-blocked` - Verificar si está bloqueado
- `GET /api/v1/security/events` - Obtener eventos de seguridad
- `GET /api/v1/security/summary` - Resumen de seguridad
- `GET /api/v1/security/analyze/{source}` - Analizar comportamiento

### Optimizador Automático

- `POST /api/v1/optimizer/register-parameter` - Registrar parámetro
- `POST /api/v1/optimizer/record-performance` - Registrar rendimiento
- `POST /api/v1/optimizer/optimize` - Optimizar parámetro(s)
- `GET /api/v1/optimizer/parameters` - Estado de parámetros
- `GET /api/v1/optimizer/history` - Historial de optimizaciones
- `GET /api/v1/optimizer/summary` - Resumen de optimizaciones

### Controlador de Tasa Adaptativo

- `POST /api/v1/adaptive-rate/register` - Registrar identificador
- `POST /api/v1/adaptive-rate/record` - Registrar petición
- `POST /api/v1/adaptive-rate/check` - Verificar rate limit
- `GET /api/v1/adaptive-rate/{identifier}` - Obtener límite
- `GET /api/v1/adaptive-rate/{identifier}/history` - Historial de ajustes
- `GET /api/v1/adaptive-rate/summary` - Resumen del controlador

### Gestor Inteligente de Reintentos

- `POST /api/v1/retry/create` - Crear operación con reintentos
- `GET /api/v1/retry/{operation_id}` - Información de operación
- `GET /api/v1/retry/patterns/{operation_type}` - Patrones aprendidos
- `GET /api/v1/retry/summary` - Resumen del gestor

### Gestor de Locks Distribuidos

- `POST /api/v1/locks/acquire` - Adquirir lock
- `POST /api/v1/locks/{lock_id}/release` - Liberar lock
- `POST /api/v1/locks/{lock_id}/renew` - Renovar lock
- `GET /api/v1/locks/{lock_id}` - Información de lock
- `GET /api/v1/locks/resource/{resource_id}` - Lock de recurso
- `POST /api/v1/locks/cleanup` - Limpiar locks expirados
- `GET /api/v1/locks/summary` - Resumen del gestor

### Gestor de Pipelines de Datos

- `POST /api/v1/pipelines/create` - Crear pipeline
- `POST /api/v1/pipelines/{pipeline_id}/add-stage` - Agregar stage
- `POST /api/v1/pipelines/{pipeline_id}/execute` - Ejecutar pipeline
- `GET /api/v1/pipelines/{pipeline_id}` - Información de pipeline
- `GET /api/v1/pipelines/executions/{execution_id}` - Información de ejecución
- `POST /api/v1/pipelines/executions/{execution_id}/cancel` - Cancelar ejecución
- `GET /api/v1/pipelines/executions/history` - Historial de ejecuciones
- `GET /api/v1/pipelines/summary` - Resumen del gestor

### Programador de Eventos

- `POST /api/v1/scheduler/schedule` - Programar evento
- `POST /api/v1/scheduler/{event_id}/pause` - Pausar evento
- `POST /api/v1/scheduler/{event_id}/resume` - Reanudar evento
- `POST /api/v1/scheduler/{event_id}/cancel` - Cancelar evento
- `GET /api/v1/scheduler/{event_id}` - Información de evento
- `GET /api/v1/scheduler/history` - Historial de ejecuciones
- `GET /api/v1/scheduler/summary` - Resumen del scheduler

### Gestor de Degradación Gradual

- `POST /api/v1/degradation/register-service` - Registar servicio
- `POST /api/v1/degradation/register-fallback` - Registar fallback
- `POST /api/v1/degradation/add-rule` - Agregar regla de degradación
- `POST /api/v1/degradation/record-metric` - Registrar métrica
- `POST /api/v1/degradation/record-call` - Registrar llamada a servicio
- `GET /api/v1/degradation/service/{service_id}` - Salud de servicio
- `GET /api/v1/degradation/status` - Estado de degradación
- `GET /api/v1/degradation/history` - Historial de degradación
- `GET /api/v1/degradation/summary` - Resumen del gestor

### Sistema de Precalentamiento de Cache

- `POST /api/v1/cache-warmer/register-rule` - Registrar regla de precalentamiento
- `POST /api/v1/cache-warmer/record-access` - Registrar acceso a cache
- `POST /api/v1/cache-warmer/start` - Iniciar precalentamiento
- `POST /api/v1/cache-warmer/stop` - Detener precalentamiento
- `GET /api/v1/cache-warmer/patterns` - Patrones de acceso
- `GET /api/v1/cache-warmer/statistics` - Estadísticas de precalentamiento
- `GET /api/v1/cache-warmer/summary` - Resumen del warmer

### Sistema de Descarga de Carga

- `POST /api/v1/load-shedder/record-metric` - Registrar métrica de carga
- `POST /api/v1/load-shedder/add-rule` - Agregar regla de descarga
- `POST /api/v1/load-shedder/check-request` - Verificar si aceptar petición
- `GET /api/v1/load-shedder/statistics` - Estadísticas de carga
- `GET /api/v1/load-shedder/history` - Historial de descarga
- `GET /api/v1/load-shedder/summary` - Resumen del shedder

### Resolvedor de Conflictos

- `POST /api/v1/conflicts/register` - Registrar conflicto
- `POST /api/v1/conflicts/{conflict_id}/resolve` - Resolver conflicto
- `POST /api/v1/conflicts/register-rule` - Registrar regla de resolución
- `GET /api/v1/conflicts/{conflict_id}` - Información de conflicto
- `GET /api/v1/conflicts/pending` - Conflictos pendientes
- `GET /api/v1/conflicts/history` - Historial de resoluciones
- `GET /api/v1/conflicts/summary` - Resumen del resolvedor

### Máquina de Estados

- `POST /api/v1/state-machines/create` - Crear máquina de estados
- `POST /api/v1/state-machines/{machine_id}/add-transition` - Agregar transición
- `POST /api/v1/state-machines/{machine_id}/transition` - Realizar transición
- `GET /api/v1/state-machines/{machine_id}` - Información de máquina
- `GET /api/v1/state-machines/history` - Historial de estados
- `GET /api/v1/state-machines/summary` - Resumen del gestor

### Motor de Flujos de Trabajo V2

- `POST /api/v1/workflows-v2/create` - Crear workflow
- `POST /api/v1/workflows-v2/{workflow_id}/add-step` - Agregar step
- `POST /api/v1/workflows-v2/{workflow_id}/execute` - Ejecutar workflow
- `GET /api/v1/workflows-v2/{workflow_id}` - Información de workflow
- `GET /api/v1/workflows-v2/executions/{execution_id}` - Información de ejecución
- `POST /api/v1/workflows-v2/executions/{execution_id}/cancel` - Cancelar ejecución
- `GET /api/v1/workflows-v2/summary` - Resumen del motor

### Bus de Eventos

- `POST /api/v1/events/publish` - Publicar evento
- `POST /api/v1/events/subscribe` - Suscribirse a eventos
- `POST /api/v1/events/unsubscribe` - Desuscribirse de eventos
- `GET /api/v1/events/history` - Historial de eventos
- `GET /api/v1/events/subscriptions` - Suscripciones
- `GET /api/v1/events/summary` - Resumen del bus

### Gestor de Feature Toggles

- `POST /api/v1/feature-toggles/create` - Crear feature toggle
- `GET /api/v1/feature-toggles/{toggle_id}/check` - Verificar toggle
- `POST /api/v1/feature-toggles/{toggle_id}/update` - Actualizar toggle
- `GET /api/v1/feature-toggles/{toggle_id}` - Información de toggle
- `GET /api/v1/feature-toggles/{toggle_id}/statistics` - Estadísticas
- `GET /api/v1/feature-toggles/summary` - Resumen del gestor

### Limitador de Tasa V2

- `POST /api/v1/rate-limiter-v2/add-rule` - Agregar regla
- `POST /api/v1/rate-limiter-v2/check` - Verificar rate limit
- `GET /api/v1/rate-limiter-v2/status` - Estado de rate limiting
- `GET /api/v1/rate-limiter-v2/history` - Historial de bloqueos
- `GET /api/v1/rate-limiter-v2/summary` - Resumen del limitador

### Circuit Breaker V2

- `POST /api/v1/circuit-breakers-v2/create` - Crear circuit breaker
- `GET /api/v1/circuit-breakers-v2/{circuit_id}` - Información
- `POST /api/v1/circuit-breakers-v2/{circuit_id}/reset` - Resetear
- `GET /api/v1/circuit-breakers-v2/summary` - Resumen del gestor

### Optimizador Adaptativo

- `POST /api/v1/optimizer/register-parameter` - Registrar parámetro
- `POST /api/v1/optimizer/add-goal` - Agregar objetivo
- `POST /api/v1/optimizer/record-metric` - Registrar métrica
- `POST /api/v1/optimizer/start` - Iniciar optimización
- `POST /api/v1/optimizer/stop` - Detener optimización
- `GET /api/v1/optimizer/parameter/{parameter_id}` - Información de parámetro
- `GET /api/v1/optimizer/results` - Resultados de optimización
- `GET /api/v1/optimizer/summary` - Resumen del optimizador

### Verificador de Salud V2

- `POST /api/v1/health-v2/register-check` - Registrar health check
- `POST /api/v1/health-v2/{check_id}/run` - Ejecutar check manualmente
- `GET /api/v1/health-v2/overall` - Salud general
- `GET /api/v1/health-v2/{check_id}/history` - Historial de checks
- `GET /api/v1/health-v2/summary` - Resumen del verificador

### Auto Scaler

- `POST /api/v1/auto-scaler/add-rule` - Agregar regla de escalado
- `POST /api/v1/auto-scaler/record-metric` - Registrar métrica
- `POST /api/v1/auto-scaler/set-instances` - Establecer instancias manualmente
- `POST /api/v1/auto-scaler/start` - Iniciar auto-escalado
- `POST /api/v1/auto-scaler/stop` - Detener auto-escalado
- `GET /api/v1/auto-scaler/status` - Estado de escalado
- `GET /api/v1/auto-scaler/history` - Historial de escalado
- `GET /api/v1/auto-scaler/summary` - Resumen del escalador

### Procesador por Lotes

- `POST /api/v1/batch/add-item` - Agregar item a batch
- `POST /api/v1/batch/register-processor` - Registrar procesador
- `GET /api/v1/batch/queue-status` - Estado de cola(s)
- `GET /api/v1/batch/history` - Historial de batches
- `GET /api/v1/batch/summary` - Resumen del procesador

### Monitor de Rendimiento

- `POST /api/v1/performance/record-metric` - Registrar métrica
- `POST /api/v1/performance/record-latency` - Registrar latencia
- `POST /api/v1/performance/create-snapshot` - Crear snapshot
- `GET /api/v1/performance/summary` - Resumen de rendimiento
- `GET /api/v1/performance/metric/{metric_name}` - Historial de métrica
- `GET /api/v1/performance/monitor-summary` - Resumen del monitor

### Gestor de Colas

- `POST /api/v1/queues/create` - Crear cola
- `POST /api/v1/queues/{queue_name}/enqueue` - Encolar mensaje
- `POST /api/v1/queues/{queue_name}/dequeue` - Desencolar mensaje
- `POST /api/v1/queues/messages/{message_id}/ack` - Confirmar mensaje
- `POST /api/v1/queues/messages/{message_id}/nack` - Negar mensaje
- `GET /api/v1/queues/{queue_name}/status` - Estado de cola
- `GET /api/v1/queues/summary` - Resumen del gestor

### Gestor de Conexiones

- `POST /api/v1/connections/register` - Registrar conexión
- `POST /api/v1/connections/{connection_type}/acquire` - Adquirir conexión
- `POST /api/v1/connections/{connection_id}/release` - Liberar conexión
- `POST /api/v1/connections/{connection_id}/close` - Cerrar conexión
- `GET /api/v1/connections/{connection_id}` - Información de conexión
- `GET /api/v1/connections/type/{connection_type}` - Conexiones por tipo
- `GET /api/v1/connections/summary` - Resumen del gestor

### Gestor de Transacciones

- `POST /api/v1/transactions/begin` - Iniciar transacción
- `POST /api/v1/transactions/{transaction_id}/add-operation` - Agregar operación
- `POST /api/v1/transactions/{transaction_id}/commit` - Commit transacción
- `POST /api/v1/transactions/{transaction_id}/rollback` - Rollback transacción
- `GET /api/v1/transactions/{transaction_id}` - Información de transacción
- `GET /api/v1/transactions/history` - Historial de transacciones
- `GET /api/v1/transactions/summary` - Resumen del gestor

### Orquestador de Sagas

- `POST /api/v1/sagas/create` - Crear saga
- `POST /api/v1/sagas/{saga_id}/add-step` - Agregar step
- `POST /api/v1/sagas/{saga_id}/execute` - Ejecutar saga
- `GET /api/v1/sagas/{saga_id}` - Información de saga
- `GET /api/v1/sagas/history` - Historial de sagas
- `GET /api/v1/sagas/summary` - Resumen del orquestador

### Coordinador Distribuido

- `POST /api/v1/coordination/register-node` - Registrar nodo
- `POST /api/v1/coordination/propose` - Proponer valor para consenso
- `GET /api/v1/coordination/leader` - Información del líder
- `GET /api/v1/coordination/status` - Estado de coordinación
- `GET /api/v1/coordination/summary` - Resumen del coordinador

### Malla de Servicios

- `POST /api/v1/mesh/register-service` - Registrar servicio
- `POST /api/v1/mesh/register-instance` - Registrar instancia
- `GET /api/v1/mesh/service/{service_name}/instance` - Obtener instancia
- `POST /api/v1/mesh/instance/{instance_id}/status` - Actualizar estado
- `GET /api/v1/mesh/service/{service_name}/instances` - Instancias de servicio
- `GET /api/v1/mesh/summary` - Resumen de la malla

### Limitador Adaptativo

- `POST /api/v1/throttler/add-rule` - Agregar regla
- `POST /api/v1/throttler/record-metric` - Registrar métrica
- `POST /api/v1/throttler/check` - Verificar throttling
- `GET /api/v1/throttler/{rule_id}/status` - Estado de throttling
- `GET /api/v1/throttler/summary` - Resumen del limitador

### Gestor de Backpressure

- `POST /api/v1/backpressure/add-rule` - Agregar regla
- `POST /api/v1/backpressure/record-metric` - Registrar métrica
- `GET /api/v1/backpressure/{component_id}/level` - Nivel de backpressure
- `POST /api/v1/backpressure/{component_id}/check` - Verificar si aceptar
- `GET /api/v1/backpressure/status` - Estado de backpressure
- `GET /api/v1/backpressure/history` - Historial de backpressure
- `GET /api/v1/backpressure/summary` - Resumen del gestor

### Aprendizaje Federado

- `POST /api/v1/federated-learning/register-client` - Registrar cliente
- `POST /api/v1/federated-learning/start-round` - Iniciar ronda
- `POST /api/v1/federated-learning/submit-update` - Enviar actualización
- `GET /api/v1/federated-learning/global-model` - Obtener modelo global
- `GET /api/v1/federated-learning/round/{round_id}` - Estado de ronda
- `GET /api/v1/federated-learning/summary` - Resumen de aprendizaje

### Gestor de Conocimiento

- `POST /api/v1/knowledge/add` - Agregar entrada de conocimiento
- `POST /api/v1/knowledge/search` - Buscar conocimiento
- `POST /api/v1/knowledge/remove` - Remover entrada
- `POST /api/v1/knowledge/add-relationship` - Agregar relación
- `GET /api/v1/knowledge/related/{entry_id}` - Conocimiento relacionado
- `GET /api/v1/knowledge/stats` - Estadísticas de conocimiento

### Generador Automático

- `POST /api/v1/generator/register-template` - Registrar plantilla
- `POST /api/v1/generator/generate` - Generar artefacto
- `POST /api/v1/generator/generate-batch` - Generar múltiples artefactos
- `GET /api/v1/generator/templates` - Listar plantillas
- `GET /api/v1/generator/template/{template_id}` - Obtener plantilla
- `GET /api/v1/generator/history` - Historial de generaciones
- `GET /api/v1/generator/stats` - Estadísticas de generación

### Recomendador de Arquitectura

- `POST /api/v1/architecture/add-requirement` - Agregar requisito
- `POST /api/v1/architecture/recommend` - Recomendar arquitectura
- `GET /api/v1/architecture/recommendations` - Obtener recomendaciones
- `GET /api/v1/architecture/stats` - Estadísticas

### Gestor de MLOps

- `POST /api/v1/mlops/create-experiment` - Crear experimento
- `POST /api/v1/mlops/update-experiment` - Actualizar experimento
- `POST /api/v1/mlops/register-model` - Registar modelo
- `POST /api/v1/mlops/deploy-model` - Desplegar modelo
- `POST /api/v1/mlops/record-performance` - Registrar rendimiento
- `POST /api/v1/mlops/detect-drift` - Detectar drift
- `GET /api/v1/mlops/experiment/{experiment_id}` - Obtener experimento
- `GET /api/v1/mlops/model/{model_id}` - Obtener modelo
- `GET /api/v1/mlops/drift` - Detecciones de drift
- `GET /api/v1/mlops/summary` - Resumen de MLOps

### Gestor de Dependencias

- `POST /api/v1/dependencies/register` - Registrar dependencia
- `POST /api/v1/dependencies/check-updates` - Verificar actualizaciones
- `POST /api/v1/dependencies/scan-vulnerabilities` - Escanear vulnerabilidades
- `POST /api/v1/dependencies/update` - Actualizar dependencia
- `GET /api/v1/dependencies/tree/{dependency_name}` - Árbol de dependencias
- `GET /api/v1/dependencies/vulnerabilities` - Obtener vulnerabilidades
- `GET /api/v1/dependencies/summary` - Resumen de dependencias

### Gestor de CI/CD

- `POST /api/v1/cicd/register-template` - Registrar plantilla de pipeline
- `POST /api/v1/cicd/create-pipeline` - Crear pipeline
- `POST /api/v1/cicd/create-from-template` - Crear desde plantilla
- `POST /api/v1/cicd/run-pipeline` - Ejecutar pipeline
- `POST /api/v1/cicd/cancel-pipeline` - Cancelar pipeline
- `GET /api/v1/cicd/pipeline/{pipeline_id}` - Obtener pipeline
- `GET /api/v1/cicd/pipeline/{pipeline_id}/logs` - Logs de pipeline
- `GET /api/v1/cicd/summary` - Resumen de CI/CD

### Análisis de Calidad de Código

- `POST /api/v1/code-quality/analyze` - Analizar código
- `GET /api/v1/code-quality/report/{report_id}` - Obtener reporte
- `GET /api/v1/code-quality/summary` - Resumen de calidad

### Métricas de Negocio

- `POST /api/v1/business-metrics/record` - Registrar métrica
- `POST /api/v1/business-metrics/define-kpi` - Definir KPI
- `POST /api/v1/business-metrics/update-kpi` - Actualizar KPI
- `GET /api/v1/business-metrics/kpi/{kpi_id}` - Estado de KPI
- `POST /api/v1/business-metrics/create-funnel` - Crear embudo
- `POST /api/v1/business-metrics/record-funnel-stage` - Registrar stage
- `GET /api/v1/business-metrics/funnel/{funnel_id}` - Análisis de embudo
- `GET /api/v1/business-metrics/trend/{metric_name}` - Tendencias
- `GET /api/v1/business-metrics/summary` - Resumen de negocio

### Control de Versiones

- `POST /api/v1/version-control/create-branch` - Crear branch
- `POST /api/v1/version-control/switch-branch` - Cambiar de branch
- `POST /api/v1/version-control/commit` - Crear commit
- `POST /api/v1/version-control/merge` - Fusionar branch
- `GET /api/v1/version-control/commit/{commit_id}` - Obtener commit
- `GET /api/v1/version-control/branch/{branch_name}/history` - Historial de branch
- `GET /api/v1/version-control/file/{file_path}/history` - Historial de archivo
- `GET /api/v1/version-control/diff` - Diff entre commits
- `GET /api/v1/version-control/summary` - Resumen

### Analizador de Logs

- `POST /api/v1/logs/ingest` - Ingerir log
- `POST /api/v1/logs/search` - Buscar logs
- `GET /api/v1/logs/statistics` - Estadísticas de logs
- `GET /api/v1/logs/patterns` - Coincidencias de patrones
- `GET /api/v1/logs/summary` - Resumen de análisis

### Rendimiento de API

- `POST /api/v1/api-performance/record` - Registrar llamada
- `GET /api/v1/api-performance/endpoint` - Rendimiento de endpoint
- `GET /api/v1/api-performance/slow-endpoints` - Endpoints lentos
- `GET /api/v1/api-performance/error-endpoints` - Endpoints con errores
- `GET /api/v1/api-performance/summary` - Resumen de rendimiento

### Gestión Avanzada de Secretos

- `POST /api/v1/secrets/create` - Crear secreto
- `GET /api/v1/secrets/{secret_id}` - Obtener valor
- `GET /api/v1/secrets/{secret_id}/info` - Información de secreto
- `POST /api/v1/secrets/{secret_id}/rotate` - Rotar secreto
- `POST /api/v1/secrets/{secret_id}/revoke` - Revocar secreto
- `POST /api/v1/secrets/auto-rotate` - Rotación automática
- `GET /api/v1/secrets/access-log` - Log de accesos
- `GET /api/v1/secrets/summary` - Resumen de secretos

### Caché Inteligente

- `POST /api/v1/cache/get` - Obtener valor
- `POST /api/v1/cache/set` - Guardar valor
- `POST /api/v1/cache/invalidate` - Invalidar entrada
- `POST /api/v1/cache/invalidate-pattern` - Invalidar por patrón
- `POST /api/v1/cache/prefetch` - Pre-cargar entrada
- `GET /api/v1/cache/stats` - Estadísticas de caché
- `GET /api/v1/cache/patterns` - Patrones de acceso
- `POST /api/v1/cache/clear` - Limpiar caché

### Analizador de Sentimientos

- `POST /api/v1/sentiment/analyze` - Analizar sentimiento
- `POST /api/v1/sentiment/analyze-batch` - Analizar lote
- `POST /api/v1/sentiment/summary` - Resumen de sentimientos

### Gestor de Tareas

- `POST /api/v1/tasks/create` - Crear tarea
- `POST /api/v1/tasks/{task_id}/update-status` - Actualizar estado
- `POST /api/v1/tasks/{task_id}/update-progress` - Actualizar progreso
- `GET /api/v1/tasks/{task_id}` - Obtener tarea
- `GET /api/v1/tasks/status/{status}` - Tareas por estado
- `GET /api/v1/tasks/assignee/{assignee}` - Tareas por asignado
- `GET /api/v1/tasks/overdue` - Tareas vencidas
- `POST /api/v1/task-lists/create` - Crear lista
- `POST /api/v1/task-lists/{list_id}/add-task` - Agregar tarea
- `GET /api/v1/task-lists/{list_id}` - Obtener lista
- `GET /api/v1/tasks/summary` - Resumen

### Monitor de Recursos

- `GET /api/v1/resources/current` - Métricas actuales
- `GET /api/v1/resources/history/{resource_type}` - Historial
- `GET /api/v1/resources/statistics/{resource_type}` - Estadísticas
- `GET /api/v1/resources/alerts` - Alertas de recursos
- `POST /api/v1/resources/alerts/{alert_id}/resolve` - Resolver alerta
- `GET /api/v1/resources/summary` - Resumen del monitor

### Notificaciones Push

- `POST /api/v1/notifications/send` - Enviar notificación
- `POST /api/v1/notifications/subscribe` - Suscribir usuario
- `POST /api/v1/notifications/unsubscribe` - Desuscribir usuario
- `GET /api/v1/notifications/user/{user_id}` - Notificaciones de usuario
- `POST /api/v1/notifications/{notification_id}/read` - Marcar como leída
- `GET /api/v1/notifications/stats` - Estadísticas de notificaciones

### Sincronización Distribuida

- `POST /api/v1/sync/create-resource` - Crear recurso sincronizado
- `POST /api/v1/sync/update-resource` - Actualizar recurso
- `POST /api/v1/sync/sync-from-remote` - Sincronizar desde remoto
- `POST /api/v1/sync/resolve-conflict` - Resolver conflicto
- `GET /api/v1/sync/resource/{resource_id}` - Obtener recurso
- `GET /api/v1/sync/conflicts` - Obtener conflictos
- `GET /api/v1/sync/summary` - Resumen de sincronización

### Analizador de Queries

- `POST /api/v1/queries/record` - Registrar ejecución
- `GET /api/v1/queries/slow` - Queries lentas
- `GET /api/v1/queries/patterns` - Patrones de queries
- `GET /api/v1/queries/statistics` - Estadísticas de queries
- `GET /api/v1/queries/summary` - Resumen del analizador

### Gestor de Archivos

- `POST /api/v1/files/upload` - Subir archivo
- `GET /api/v1/files/{file_id}` - Descargar archivo
- `GET /api/v1/files/{file_id}/metadata` - Metadatos de archivo
- `GET /api/v1/files/{file_id}/versions` - Versiones de archivo
- `GET /api/v1/files/search` - Buscar archivos
- `DELETE /api/v1/files/{file_id}` - Eliminar archivo
- `POST /api/v1/files/{file_id}/restore` - Restaurar archivo
- `GET /api/v1/files/summary` - Resumen del gestor

### Compresión de Datos

- `POST /api/v1/compression/compress` - Comprimir datos
- `POST /api/v1/compression/decompress` - Descomprimir datos
- `POST /api/v1/compression/find-best` - Encontrar mejor algoritmo
- `GET /api/v1/compression/stats` - Estadísticas de compresión

### Backup Incremental

- `POST /api/v1/backup/create` - Crear backup
- `POST /api/v1/backup/restore` - Restaurar backup
- `GET /api/v1/backup/{backup_id}` - Información de backup
- `GET /api/v1/backup/list` - Listar backups
- `POST /api/v1/backup/create-set` - Crear conjunto de backups
- `GET /api/v1/backup/summary` - Resumen de backups

### Analizador de Red

- `POST /api/v1/network/record` - Registrar métrica
- `GET /api/v1/network/endpoint/{endpoint}` - Estadísticas de endpoint
- `GET /api/v1/network/slow-endpoints` - Endpoints lentos
- `GET /api/v1/network/events` - Eventos de red
- `GET /api/v1/network/summary` - Resumen de red

### Gestor de Configuraciones

- `POST /api/v1/config/register` - Registrar configuración
- `GET /api/v1/config/{config_id}` - Obtener configuración
- `GET /api/v1/config/{config_id}/history` - Historial de configuración
- `GET /api/v1/config/changes` - Cambios de configuración
- `POST /api/v1/config/{config_id}/rollback` - Revertir configuración
- `POST /api/v1/config/{config_id}/subscribe` - Suscribirse a cambios
- `GET /api/v1/config/summary` - Resumen del gestor

### Autenticación MFA

- `POST /api/v1/mfa/setup/totp` - Configurar TOTP
- `POST /api/v1/mfa/setup/sms` - Configurar SMS
- `POST /api/v1/mfa/setup/email` - Configurar Email
- `POST /api/v1/mfa/initiate` - Iniciar proceso MFA
- `POST /api/v1/mfa/verify` - Verificar código MFA
- `GET /api/v1/mfa/status/{user_id}` - Estado MFA del usuario

### Rate Limiter Avanzado

- `POST /api/v1/rate-limit/create-rule` - Crear regla
- `POST /api/v1/rate-limit/check` - Verificar rate limit
- `POST /api/v1/rate-limit/block` - Bloquear identificador
- `GET /api/v1/rate-limit/violations` - Violaciones
- `GET /api/v1/rate-limit/summary` - Resumen

### Analizador de Comportamiento

- `POST /api/v1/behavior/record` - Registar acción
- `GET /api/v1/behavior/profile/{user_id}` - Perfil de usuario
- `GET /api/v1/behavior/high-risk` - Usuarios de alto riesgo
- `GET /api/v1/behavior/anomalies` - Anomalías
- `GET /api/v1/behavior/summary` - Resumen

### Flujo de Eventos

- `POST /api/v1/events/publish` - Publicar evento
- `GET /api/v1/events` - Obtener eventos
- `POST /api/v1/events/subscribe` - Suscribirse
- `GET /api/v1/events/summary` - Resumen

### Analizador de Seguridad

- `POST /api/v1/security/analyze` - Analizar entrada
- `POST /api/v1/security/block` - Bloquear fuente
- `GET /api/v1/security/threats` - Obtener amenazas
- `POST /api/v1/security/threats/{threat_id}/resolve` - Resolver amenaza
- `GET /api/v1/security/summary` - Resumen

### Gestor de Sesiones

- `POST /api/v1/sessions/create` - Crear sesión
- `POST /api/v1/sessions/{session_id}/activity` - Actualizar actividad
- `POST /api/v1/sessions/{session_id}/status` - Actualizar estado
- `GET /api/v1/sessions/{session_id}` - Información de sesión
- `GET /api/v1/sessions/{session_id}/analytics` - Analíticas de sesión
- `GET /api/v1/sessions/active` - Sesiones activas
- `POST /api/v1/sessions/cleanup` - Limpiar sesiones
- `GET /api/v1/sessions/summary` - Resumen

### Métricas en Tiempo Real

- `POST /api/v1/metrics/record` - Registar métrica
- `GET /api/v1/metrics/aggregates/{metric_name}` - Agregados
- `GET /api/v1/metrics` - Obtener métricas
- `POST /api/v1/metrics/alerts/create` - Crear alerta
- `GET /api/v1/metrics/summary` - Resumen

### Optimizador Automático

- `POST /api/v1/optimizer/register-parameter` - Registrar parámetro
- `POST /api/v1/optimizer/record-performance` - Registar performance
- `GET /api/v1/optimizer/parameter/{parameter_name}` - Valor de parámetro
- `GET /api/v1/optimizer/optimizations` - Optimizaciones
- `GET /api/v1/optimizer/summary` - Resumen

### Bulk Operations (Operaciones Masivas)

- `POST /api/v1/bulk/sessions/create` - Crear múltiples sesiones
- `POST /api/v1/bulk/sessions/delete` - Eliminar múltiples sesiones
- `POST /api/v1/bulk/sessions/pause` - Pausar múltiples sesiones
- `POST /api/v1/bulk/sessions/resume` - Reanudar múltiples sesiones
- `POST /api/v1/bulk/messages/send` - Enviar mensaje a múltiples sesiones
- `POST /api/v1/bulk/export/sessions` - Exportar múltiples sesiones
- `GET /api/v1/bulk/export/status/{job_id}` - Estado de exportación
- `POST /api/v1/bulk/analytics/sessions` - Analizar múltiples sesiones
- `POST /api/v1/bulk/cleanup/sessions` - Limpiar sesiones antiguas
- `POST /api/v1/bulk/import/sessions` - Importar múltiples sesiones
- `GET /api/v1/bulk/import/status/{job_id}` - Estado de importación
- `POST /api/v1/bulk/notifications/send` - Enviar notificaciones masivas
- `POST /api/v1/bulk/search/execute` - Ejecutar búsqueda masiva
- `GET /api/v1/bulk/process/status/{job_id}` - Estado de procesamiento
- `POST /api/v1/bulk/process/cancel/{job_id}` - Cancelar job de procesamiento
- `POST /api/v1/bulk/backup/sessions` - Crear backup masivo de sesiones
- `GET /api/v1/bulk/backup/status/{job_id}` - Estado de backup
- `POST /api/v1/bulk/migration/start` - Iniciar migración masiva
- `GET /api/v1/bulk/migration/status/{job_id}` - Estado de migración
- `GET /api/v1/bulk/metrics/stats` - Estadísticas de operaciones bulk
- `GET /api/v1/bulk/metrics/history` - Historial de operaciones bulk
- `GET /api/v1/bulk/metrics/summary` - Resumen de operaciones bulk
- `POST /api/v1/bulk/scheduler/schedule` - Programar operación recurrente
- `GET /api/v1/bulk/scheduler/jobs` - Listar jobs programados
- `POST /api/v1/bulk/scheduler/{job_id}/enable` - Habilitar job programado
- `POST /api/v1/bulk/scheduler/{job_id}/disable` - Deshabilitar job programado
- `GET /api/v1/bulk/rate-limit/stats` - Estadísticas de rate limiting
- `POST /api/v1/bulk/rate-limit/check` - Verificar rate limit
- `POST /api/v1/bulk/auto-creation/start` - Iniciar auto-creación continua (nunca se detiene)
- `POST /api/v1/bulk/auto-creation/stop` - Detener auto-creación
- `GET /api/v1/bulk/auto-creation/stats` - Estadísticas de auto-creación
- `POST /api/v1/bulk/auto-expansion/start` - Iniciar auto-expansión
- `POST /api/v1/bulk/self-sustaining/start` - Iniciar sistema auto-sostenible completo
- `GET /api/v1/bulk/self-sustaining/stats` - Estadísticas del sistema auto-sostenible
- `POST /api/v1/bulk/self-sustaining/ensure-continuity` - Asegurar continuidad
- `POST /api/v1/bulk/infinite-generator/create` - Crear generador infinito

### Health & Performance

- `GET /health` - Health check básico
- `GET /health/detailed` - Health check detallado con recursos del sistema
- `GET /api/v1/performance/metrics` - Métricas de rendimiento
- `GET /api/v1/performance/slow-operations` - Operaciones lentas detectadas
- `GET /api/v1/tasks/queue` - Estado de cola de tareas

## ⚙️ Configuración

### Variables de Entorno

```env
# API Settings
CHAT_API_HOST=0.0.0.0
CHAT_API_PORT=8006

# LLM Settings
LLM_PROVIDER=openai
LLM_MODEL=gpt-4
OPENAI_API_KEY=tu-api-key-aqui
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=2000

# Chat Behavior
AUTO_CONTINUE=true
RESPONSE_INTERVAL=2.0
MAX_CONSECUTIVE_RESPONSES=100
MAX_MESSAGES_PER_SESSION=1000

# CORS
CORS_ORIGINS=*

# Logging
LOG_LEVEL=INFO

# Storage (Persistencia)
STORAGE_TYPE=json  # json o redis
STORAGE_PATH=sessions
REDIS_URL=redis://localhost:6379  # Solo si usas Redis
SESSION_TTL=86400  # TTL en segundos (24 horas)

# Auto-save
AUTO_SAVE=true
SAVE_INTERVAL=30.0  # Segundos entre guardados

# Rate Limiting
RATE_LIMIT_MAX_REQUESTS=60
RATE_LIMIT_WINDOW=60.0
RATE_LIMIT_MAX_CONCURRENT=10

# Cache
ENABLE_CACHE=true
CACHE_SIZE=1000
CACHE_TTL=3600  # 1 hora

# Plugins
ENABLE_PLUGINS=true

# Backups
ENABLE_BACKUPS=true
BACKUP_INTERVAL_HOURS=24
BACKUP_DIRECTORY=backups
```

## 🔄 Flujo de Trabajo

1. **Crear Sesión**: El usuario crea una nueva sesión de chat
2. **Iniciar Chat**: El chat comienza a generar respuestas automáticamente
3. **Generación Continua**: El chat genera respuestas cada `response_interval` segundos
4. **Control del Usuario**: El usuario puede pausar/reanudar en cualquier momento
5. **Detener**: El usuario puede detener completamente el chat

## 🎯 Casos de Uso

- **Brainstorming Continuo**: Generar ideas continuamente sin interrupciones
- **Análisis Progresivo**: Analizar un tema en profundidad con múltiples iteraciones
- **Generación de Contenido**: Crear contenido de forma continua y automática
- **Simulación de Conversación**: Simular conversaciones largas y naturales
- **Asistente Personal**: Asistente que genera respuestas proactivamente
- **Análisis de Datos**: Análisis continuo y generación de insights

## 🆕 Mejoras Recientes (v2.0.0)

Ver [IMPROVEMENTS.md](IMPROVEMENTS.md) y [ADVANCED_FEATURES.md](ADVANCED_FEATURES.md) para detalles completos:

### Mejoras Básicas
- ✅ **Persistencia de Sesiones**: Guardado automático en JSON/Redis
- ✅ **Sistema de Métricas**: Monitoreo completo de rendimiento
- ✅ **Rate Limiting**: Control de tasa de solicitudes
- ✅ **Retry Logic**: Reintentos automáticos con exponential backoff
- ✅ **Auto-guardado**: Guardado automático cada 30 segundos
- ✅ **Carga de Sesiones**: Recuperación automática de sesiones

### Características Avanzadas
- ✅ **WebSockets**: Streaming bidireccional en tiempo real
- ✅ **Cache de Respuestas**: Reduce hasta 80% las llamadas al LLM
- ✅ **Sistema de Plugins**: Extensible con plugins personalizados
- ✅ **Plugins Incluidos**: Sentiment Analyzer, Profanity Filter, Response Enhancer

### Características Enterprise
- ✅ **Análisis Avanzado**: Insights, temas, sentimientos, estadísticas
- ✅ **Exportación Multi-formato**: JSON, Markdown, CSV, HTML, TXT
- ✅ **Sistema de Plantillas**: Mensajes predefinidos con variables
- ✅ **Webhooks**: Integración con sistemas externos en tiempo real

### Características Finales
- ✅ **Autenticación JWT**: Sistema completo con roles y permisos
- ✅ **Backups Automáticos**: Backups programados cada 24h
- ✅ **Dashboard Web**: Interfaz interactiva para monitoreo en tiempo real

### Optimizaciones y Testing
- ✅ **Testing Framework**: Suite completa de tests con pytest
- ✅ **Optimizador de Rendimiento**: Métricas avanzadas y detección de problemas
- ✅ **Logging Estructurado**: Logs JSON con contexto completo
- ✅ **Monitor de Salud**: Health checks avanzados y monitoreo de recursos
- ✅ **Cola de Tareas**: Procesamiento asíncrono con workers

### Características Ultimate
- ✅ **API GraphQL**: Consultas flexibles solo con los campos necesarios
- ✅ **Sistema de Alertas**: Alertas multi-nivel con handlers personalizables
- ✅ **Dashboard Mejorado**: Gráficos en tiempo real con Chart.js

### Enterprise Plus
- ✅ **Clustering Distribuido**: Escalabilidad horizontal con consistent hashing
- ✅ **Feature Flags**: Activación/desactivación dinámica de características
- ✅ **Versionado de API**: Gestión completa con deprecación y migración
- ✅ **Analytics Avanzado**: Detección de patrones, análisis de comportamiento y ML

### Ultimate Plus
- ✅ **Recomendaciones ML**: Sistema de recomendaciones con collaborative filtering
- ✅ **A/B Testing**: Framework completo para experimentos controlados
- ✅ **Sistema de Eventos**: Pub/sub para eventos en tiempo real con historial
- ✅ **Seguridad Avanzada**: Audit logs, sanitización, validación y tokens seguros
- ✅ **Internacionalización**: Soporte multi-idioma con 9 idiomas incluidos

### Final Plus
- ✅ **Workflows**: Sistema completo de automatización con pasos, condiciones y retry
- ✅ **Notificaciones Push**: Sistema de notificaciones en tiempo real con suscripciones
- ✅ **Integraciones**: Sistema de integraciones con Webhooks, REST API y más
- ✅ **Benchmarking**: Sistema de performance testing con métricas P95/P99

### Ultimate Final
- ✅ **Documentación Automática**: Generación de OpenAPI 3.0 y Markdown
- ✅ **Monitoring Avanzado**: Sistema de métricas en tiempo real con alertas
- ✅ **Gestión de Secretos**: Almacenamiento seguro con encriptación
- ✅ **ML Optimizer**: Optimización de parámetros basada en datos históricos

### Production Ready
- ✅ **Deployment Automático**: Sistema completo de deployment con rollback automático
- ✅ **Reportes Automatizados**: Generación programada de reportes (diarios, semanales, mensuales)

### User & Search
- ✅ **Gestión de Usuarios Avanzada**: Sistema completo con roles (USER, ADMIN, MODERATOR, PREMIUM), permisos y estados
- ✅ **Búsqueda Avanzada**: Motor de búsqueda con índices, scoring, highlights y filtros

### Advanced Systems
- ✅ **Cola de Mensajes**: Sistema de cola con prioridades (LOW, MEDIUM, HIGH, URGENT), reintentos y procesamiento asíncrono
- ✅ **Validación Avanzada**: Sistema de validación con reglas personalizadas, validadores por defecto (email, URL, phone) y validadores custom
- ✅ **Throttling**: Sistema de throttling configurable con ventanas de tiempo y límites por identificador
- ✅ **Circuit Breaker**: Sistema de circuit breaker con estados (CLOSED, OPEN, HALF_OPEN) para proteger servicios externos

### AI & Learning Systems
- ✅ **Optimizador Inteligente**: Optimización automática con ML que analiza rendimiento y sugiere mejoras
- ✅ **Aprendizaje Adaptativo**: Sistema que aprende patrones de uso y predice mejores acciones automáticamente
- ✅ **Predicción de Demanda**: Sistema ML para predecir demanda futura de recursos con confianza y tendencias
- ✅ **Health Checks Inteligentes**: Sistema avanzado de health checks con auto-diagnóstico, métricas y auto-recovery
- ✅ **Auto-Scaling Predictivo**: Sistema que escala recursos proactivamente basado en predicciones de demanda
- ✅ **Optimizador de Costos**: Análisis y optimización de costos con sugerencias automáticas de ahorro
- ✅ **Alertas Inteligentes**: Detección proactiva de anomalías usando ML con sistema de handlers personalizables
- ✅ **Observabilidad Avanzada**: Sistema de tracing distribuido, métricas detalladas y logging estructurado
- ✅ **Balanceador de Carga Inteligente**: Balanceo adaptativo con múltiples algoritmos y detección de salud
- ✅ **Gestor de Recursos**: Sistema avanzado de gestión de recursos con límites, cuotas y asignación inteligente
- ✅ **Recuperación de Desastres**: Sistema avanzado con replicación, failover automático y restauración
- ✅ **Seguridad Avanzada**: Detección de amenazas, análisis de comportamiento y protección automática
- ✅ **Optimizador Automático**: Ajuste dinámico de parámetros basado en métricas de rendimiento
- ✅ **Aprendizaje Federado**: Entrenamiento distribuido de modelos sin compartir datos
- ✅ **Gestor de Conocimiento**: Sistema de gestión de conocimiento con indexación semántica y búsqueda vectorial
- ✅ **Generador Automático**: Generación automática de código, configuración y documentación
- ✅ **Recomendador de Arquitectura**: Recomendaciones de arquitecturas y patrones basados en requisitos
- ✅ **Gestor de MLOps**: Gestión completa de operaciones de ML con versionado, experimentos y monitoreo de drift
- ✅ **Gestor de Dependencias**: Gestión de dependencias con análisis de vulnerabilidades y actualizaciones
- ✅ **Gestor de CI/CD**: Pipelines de integración y despliegue continuo con stages y automatización
- ✅ **Análisis de Calidad de Código**: Métricas de código, detección de code smells y sugerencias de mejora
- ✅ **Métricas de Negocio**: KPIs, análisis de conversión, embudos y tendencias empresariales
- ✅ **Control de Versiones**: Sistema de versionado con commits, branches, diffs y merge
- ✅ **Analizador de Logs**: Análisis avanzado de logs con búsqueda, filtrado y detección de patrones
- ✅ **Rendimiento de API**: Métricas detalladas de API con profiling y detección de endpoints lentos
- ✅ **Gestión Avanzada de Secretos**: Rotación automática, encriptación y auditoría completa
- ✅ **Caché Inteligente**: Caché avanzado con invalidación inteligente, prefetching y análisis de patrones
- ✅ **Analizador de Sentimientos**: Análisis de sentimientos con detección de emociones y polaridad
- ✅ **Gestor de Tareas**: Sistema de gestión de tareas con prioridades, dependencias y seguimiento
- ✅ **Monitor de Recursos**: Monitoreo de recursos en tiempo real con alertas y análisis de tendencias
- ✅ **Notificaciones Push**: Sistema de notificaciones push en tiempo real con múltiples canales
- ✅ **Sincronización Distribuida**: Sistema de sincronización con resolución de conflictos
- ✅ **Analizador de Queries**: Análisis de performance de queries con detección de queries lentas
- ✅ **Gestor de Archivos**: Gestión avanzada de archivos con versionado, compresión y metadatos
- ✅ **Compresión de Datos**: Sistema de compresión con múltiples algoritmos y análisis de eficiencia
- ✅ **Backup Incremental**: Sistema de backup incremental con deduplicación y restauración
- ✅ **Analizador de Red**: Análisis de red con monitoreo de tráfico, latencia y detección de problemas
- ✅ **Gestor de Configuraciones**: Gestión avanzada de configuraciones con validación, versionado y hot-reload
- ✅ **Autenticación MFA**: Autenticación multi-factor con TOTP, SMS, Email y backup codes
- ✅ **Rate Limiter Avanzado**: Rate limiting con múltiples estrategias y análisis de patrones
- ✅ **Analizador de Comportamiento**: Análisis de comportamiento de usuarios con detección de anomalías
- ✅ **Flujo de Eventos**: Gestión de eventos en tiempo real con pub/sub y filtrado
- ✅ **Analizador de Seguridad**: Detección de amenazas, vulnerabilidades y patrones sospechosos
- ✅ **Gestor de Sesiones**: Gestión avanzada de sesiones con tracking, análisis y optimización
- ✅ **Métricas en Tiempo Real**: Sistema de métricas con agregación, alertas y visualización
- ✅ **Optimizador Automático**: Optimización automática con análisis de performance y ajuste dinámico
- ✅ **Análisis Predictivo**: Sistema de análisis predictivo con ML para predecir tendencias y resultados
- ✅ **Motor de Políticas**: Sistema avanzado de gestión de políticas con evaluación y aplicación
- ✅ **Sistema de Auditoría**: Sistema completo de auditoría con tracking de acciones y eventos
- ✅ **Orquestador de Tareas**: Sistema avanzado de orquestación con dependencias y ejecución paralela
- ✅ **Asignador de Recursos**: Sistema avanzado de asignación de recursos con cuotas, prioridades y balanceo
- ✅ **Orquestador de Servicios**: Sistema de orquestación de servicios con gestión de dependencias y health checks
- ✅ **Perfilador de Rendimiento**: Sistema avanzado de profiling con análisis detallado y optimización
- ✅ **Controlador de Tasa Adaptativo**: Control de tasa que ajusta dinámicamente límites basado en condiciones del sistema
- ✅ **Gestor Inteligente de Reintentos**: Sistema de reintentos con estrategias adaptativas y aprendizaje automático
- ✅ **Gestor de Locks Distribuidos**: Sistema de locks distribuidos con TTL, auto-renovación y detección de deadlocks
- ✅ **Gestor de Pipelines de Datos**: Sistema de pipelines con transformaciones, validaciones y procesamiento paralelo
- ✅ **Programador de Eventos**: Sistema avanzado de programación con soporte para cron, intervalos y eventos recurrentes
- ✅ **Gestor de Degradación Gradual**: Sistema de degradación con fallbacks, circuitos y modos de operación degradados
- ✅ **Sistema de Precalentamiento de Cache**: Precalentamiento inteligente basado en patrones de acceso y predicción
- ✅ **Sistema de Descarga de Carga**: Descarga inteligente de carga para prevenir sobrecarga del sistema
- ✅ **Resolvedor de Conflictos**: Sistema avanzado de resolución con múltiples estrategias y aprendizaje automático
- ✅ **Máquina de Estados**: Sistema de máquinas de estados con transiciones, validaciones y eventos
- ✅ **Motor de Flujos de Trabajo V2**: Sistema avanzado con soporte para bucles, paralelismo y compensación
- ✅ **Bus de Eventos**: Sistema de bus de eventos con pub/sub, filtrado y procesamiento asíncrono
- ✅ **Gestor de Feature Toggles**: Sistema avanzado con rollouts graduales, A/B testing y targeting
- ✅ **Limitador de Tasa V2**: Sistema avanzado con múltiples algoritmos, sliding windows y distributed support
- ✅ **Circuit Breaker V2**: Sistema avanzado con múltiples estrategias y auto-recovery
- ✅ **Optimizador Adaptativo**: Sistema de optimización que ajusta parámetros automáticamente basado en métricas
- ✅ **Verificador de Salud V2**: Sistema avanzado con dependencias, timeouts y auto-recovery
- ✅ **Auto Scaler**: Sistema de auto-escalado basado en métricas de carga y predicción de demanda
- ✅ **Procesador por Lotes**: Sistema de procesamiento por lotes con batching inteligente y ventanas de tiempo
- ✅ **Monitor de Rendimiento**: Sistema avanzado de monitoreo con análisis de latencias, throughput y recursos
- ✅ **Pool de Recursos**: Sistema de pool de recursos con reutilización, límites y estadísticas
- ✅ **Gestor de Colas**: Sistema avanzado de gestión de colas con prioridades, dead letter queues y procesamiento asíncrono
- ✅ **Gestor de Conexiones**: Sistema de gestión de conexiones con pooling, keep-alive y estadísticas
- ✅ **Gestor de Transacciones**: Sistema de gestión de transacciones con rollback y compensación
- ✅ **Orquestador de Sagas**: Sistema de orquestación de sagas para transacciones distribuidas con compensación
- ✅ **Coordinador Distribuido**: Sistema de coordinación distribuida con consenso, elección de líder y sincronización
- ✅ **Malla de Servicios**: Sistema de malla de servicios con discovery, load balancing y observabilidad
- ✅ **Limitador Adaptativo**: Sistema de throttling adaptativo que ajusta límites dinámicamente basado en condiciones
- ✅ **Gestor de Backpressure**: Sistema de gestión de backpressure para prevenir sobrecarga y mantener estabilidad

Ver [ENTERPRISE_FEATURES.md](ENTERPRISE_FEATURES.md), [FINAL_FEATURES.md](FINAL_FEATURES.md), [OPTIMIZATIONS.md](OPTIMIZATIONS.md), [ULTIMATE_FEATURES.md](ULTIMATE_FEATURES.md), [ENTERPRISE_PLUS.md](ENTERPRISE_PLUS.md), [ULTIMATE_PLUS.md](ULTIMATE_PLUS.md), [FINAL_PLUS.md](FINAL_PLUS.md), [ULTIMATE_FINAL.md](ULTIMATE_FINAL.md) y [BULK_OPERATIONS.md](BULK_OPERATIONS.md) para más detalles.

## 🔒 Seguridad

- Validación de sesiones
- Rate limiting (configurable)
- Sanitización de inputs
- Control de acceso por sesión

## 📊 Monitoreo

El sistema incluye:
- Health checks
- Métricas de sesiones activas
- Logging detallado
- Métricas de rendimiento (opcional con Prometheus)

## 🐛 Troubleshooting

### El chat no genera respuestas

1. Verificar que la sesión esté activa: `GET /api/v1/chat/sessions/{session_id}`
2. Verificar que `auto_continue` esté en `true`
3. Verificar logs para errores del LLM
4. Verificar que el API key del LLM sea válido

### El chat no se pausa

1. Verificar que el `session_id` sea correcto
2. Verificar el estado de la sesión
3. Revisar logs para errores

### Respuestas muy lentas

1. Ajustar `response_interval` para más tiempo entre respuestas
2. Verificar la latencia del proveedor de LLM
3. Considerar usar un modelo más rápido

## 🤝 Contribuir

Las contribuciones son bienvenidas. Por favor:

1. Fork el repositorio
2. Crea un branch para tu feature
3. Commit tus cambios
4. Push al branch
5. Abre un Pull Request

## 📝 Licencia

Este proyecto está bajo la licencia MIT.

## 🙏 Agradecimientos

- Blatam Academy
- Comunidad de desarrolladores

---

**Bulk Chat** - Chat continuo que no se detiene hasta que tú lo detengas 🚀

