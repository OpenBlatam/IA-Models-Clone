# Integración MCP con Cursor IDE

## Descripción

El Cursor Agent 24/7 ahora incluye soporte completo para el **Model Context Protocol (MCP)**, permitiendo que Cursor IDE se conecte directamente al agente y envíe comandos de forma nativa.

## Características

- ✅ **Servidor MCP**: Servidor MCP que expone el agente como recurso para Cursor IDE
- ✅ **Cliente MCP**: Cliente para conectarse con Cursor IDE y recibir comandos
- ✅ **WebSocket Support**: Comunicación en tiempo real vía WebSocket
- ✅ **HTTP API**: Endpoints REST para integración MCP
- ✅ **Compatibilidad**: Compatible con el protocolo MCP estándar
- ✅ **Reintentos Automáticos**: Reintentos con exponential backoff para operaciones fallidas
- ✅ **Reconexión Automática**: Reconexión automática si se pierde la conexión
- ✅ **Validación JSON-RPC**: Validación completa de mensajes JSON-RPC
- ✅ **Rate Limiting**: Protección contra abuso con rate limiting
- ✅ **Manejo de Errores Robusto**: Manejo de errores mejorado con logging detallado
- ✅ **Timeouts**: Timeouts configurables para prevenir operaciones colgadas
- ✅ **CORS**: Soporte CORS configurable para integración web
- ✅ **Circuit Breaker**: Circuit breaker para prevenir fallos en cascada
- ✅ **Métricas Avanzadas**: Sistema de métricas completo con estadísticas detalladas
- ✅ **Streaming de Resultados**: Server-Sent Events (SSE) para streaming de resultados
- ✅ **Monitoreo en Tiempo Real**: Métricas de rendimiento y uso en tiempo real
- ✅ **Autenticación Opcional**: Sistema de autenticación con API keys y roles
- ✅ **Caché de Resultados**: Caché inteligente de resultados de comandos
- ✅ **Operaciones en Lote**: Ejecutar múltiples comandos en una sola request
- ✅ **Webhooks**: Notificaciones automáticas de eventos
- ✅ **Configuración Flexible**: Sistema de configuración completo y personalizable
- ✅ **Logging Estructurado**: Request IDs y logging estructurado para trazabilidad
- ✅ **Compresión de Respuestas**: Compresión automática de respuestas grandes (gzip)
- ✅ **Headers de Seguridad**: Headers de seguridad HTTP automáticos
- ✅ **Rate Limiting por Usuario**: Rate limiting granular por usuario además de por IP
- ✅ **Sistema de Eventos**: Sistema pub/sub interno para eventos del servidor
- ✅ **WebSocket Heartbeat**: Heartbeat automático para mantener conexiones WebSocket vivas
- ✅ **Health Checks Avanzados**: Health checks detallados con métricas y estado de componentes
- ✅ **Documentación OpenAPI**: Documentación interactiva de API con Swagger/ReDoc
- ✅ **Gestión de Caché**: Endpoints para limpiar y obtener estadísticas del caché
- ✅ **Consulta de Eventos**: Endpoint para consultar eventos recientes del sistema
- ✅ **Códigos de Error Estandarizados**: Sistema de códigos de error específicos para mejor debugging
- ✅ **Validación de Comandos**: Validación y sanitización de comandos antes de ejecutarlos
- ✅ **Exception Handlers Globales**: Manejo centralizado de excepciones con respuestas consistentes
- ✅ **Protección contra Comandos Peligrosos**: Bloqueo de patrones peligrosos en comandos
- ✅ **Connection Pooling**: Pool de conexiones HTTP reutilizables para mejor rendimiento
- ✅ **Request Queue**: Cola de requests con prioridades para manejar picos de tráfico
- ✅ **Token Bucket Rate Limiting**: Algoritmo avanzado de rate limiting con recarga automática
- ✅ **Graceful Shutdown**: Cierre ordenado del servidor con limpieza de recursos
- ✅ **Request Deduplication**: Detección y prevención de requests duplicados
- ✅ **Advanced Metrics**: Métricas con percentiles (p50, p75, p90, p95, p99) y estadísticas por endpoint
- ✅ **WebSocket Connection Limits**: Límites configurables de conexiones WebSocket simultáneas
- ✅ **Prometheus Metrics Export**: Exportación de métricas en formato Prometheus para monitoreo
- ✅ **Adaptive Rate Limiting**: Rate limiting que se adapta automáticamente según la carga del sistema

## Configuración

### Configuración Básica

El servidor MCP puede configurarse de varias formas:

#### Opción 1: Usando parámetros de línea de comandos

```bash
python -m agents.backend.onyx.server.features.cursor_backend_clone.main \
    --enable-mcp \
    --mcp-port 8025
```

#### Opción 2: Usando archivo de configuración JSON

Crear un archivo `mcp_config.json`:

```json
{
  "host": "localhost",
  "port": 8025,
  "enable_cors": true,
  "enable_auth": false,
  "enable_cache": true,
  "enable_metrics": true,
  "enable_circuit_breaker": true,
  "rate_limit_max_requests": 100,
  "rate_limit_window_seconds": 60,
  "circuit_breaker_failure_threshold": 5,
  "circuit_breaker_recovery_timeout": 60.0,
  "cache_max_size": 1000,
  "cache_default_ttl": 300.0,
  "websocket_timeout": 300.0,
  "max_command_length": 10000,
  "allowed_origins": ["*"],
  "api_version": "v1",
  "enable_batch_operations": true,
  "enable_webhooks": false,
  "webhook_urls": []
}
```

Cargar configuración desde archivo:

```python
from agents.backend.onyx.server.features.cursor_backend_clone.core.mcp_config import MCPServerConfig

config = MCPServerConfig.from_file("mcp_config.json")
mcp_server = MCPServer(agent, config=config)
```

#### Opción 3: Configuración programática

```python
from agents.backend.onyx.server.features.cursor_backend_clone.core.mcp_config import MCPServerConfig

config = MCPServerConfig(
    host="0.0.0.0",
    port=8025,
    enable_auth=True,
    enable_cache=True,
    enable_webhooks=True,
    webhook_urls=["https://example.com/webhook"]
)

mcp_server = MCPServer(agent, config=config)
```

### Autenticación

Para habilitar autenticación:

1. Configurar `enable_auth: true` en la configuración
2. Usar el endpoint `/mcp/v1/auth/login` para obtener un API key
3. Incluir el API key en los headers de las requests:

```
X-API-Key: your_api_key_here
```

O usando Bearer token:

```
Authorization: Bearer your_api_key_here
```

**Roles disponibles:**
- `ADMIN`: Acceso completo, incluyendo métricas y configuración
- `USER`: Puede ejecutar comandos y leer estado
- `READONLY`: Solo lectura de estado
- `GUEST`: Sin permisos

### Webhooks

Para habilitar webhooks:

1. Configurar `enable_webhooks: true`
2. Agregar URLs de webhook en `webhook_urls`

Los webhooks se envían para los siguientes eventos:
- `command_executed`: Cuando se ejecuta un comando
- `command_cached`: Cuando se usa un resultado del caché
- `batch_commands_executed`: Cuando se ejecutan comandos en lote

Formato del payload:

```json
{
  "event": "command_executed",
  "timestamp": "2024-01-01T12:00:00",
  "data": {
    "task_id": "task_123",
    "command": "print('Hello')",
    "client_id": "127.0.0.1",
    "username": "admin"
  }
}
```

### Opciones Avanzadas de Configuración

#### Token Bucket Rate Limiting
```json
{
  "use_token_bucket_rate_limiting": true,
  "rate_limit_max_requests": 100,
  "rate_limit_window_seconds": 60
}
```

#### Request Queue
```json
{
  "enable_request_queue": true,
  "request_queue_max_size": 1000,
  "request_queue_max_workers": 10
}
```

#### Request Deduplication
```json
{
  "enable_request_deduplication": true,
  "deduplication_window_seconds": 60.0,
  "deduplication_max_cache_size": 10000
}
```

#### WebSocket Connection Limits
```json
{
  "max_websocket_connections": 100
}
```

#### Connection Pooling
```json
{
  "connection_pool_max_connections": 10
}
```

#### Adaptive Rate Limiting
```json
{
  "enable_adaptive_rate_limiting": true
}
```

**Nota:** El adaptive rate limiting ajusta automáticamente los límites según:
- Tiempo de respuesta promedio
- Tasa de errores
- Carga del sistema

## Instalación

### 1. Iniciar el agente con soporte MCP

```bash
python main.py --mode api --enable-mcp --mcp-port 8025
```

Esto iniciará:
- API REST en el puerto 8024 (por defecto)
- Servidor MCP en el puerto 8025 (configurable)

### 2. Configurar Cursor IDE

Para conectar Cursor IDE al agente, necesitas configurar Cursor para usar el servidor MCP.

#### Opción A: Configuración en Cursor Settings

1. Abre Cursor IDE
2. Ve a Settings → Extensions → MCP Servers
3. Agrega una nueva configuración:

```json
{
  "name": "cursor-agent-24-7",
  "url": "http://localhost:8025",
  "transport": "http"
}
```

#### Opción B: Configuración manual

Edita el archivo de configuración de Cursor (ubicación depende del sistema):

**Windows:**
```
%APPDATA%\Cursor\User\settings.json
```

**macOS:**
```
~/Library/Application Support/Cursor/User/settings.json
```

**Linux:**
```
~/.config/Cursor/User/settings.json
```

Agrega:

```json
{
  "mcp.servers": {
    "cursor-agent-24-7": {
      "url": "http://localhost:8025",
      "transport": "http"
    }
  }
}
```

## Uso

### Desde Cursor IDE

Una vez configurado, puedes usar el agente directamente desde Cursor:

1. Abre la paleta de comandos (Ctrl+Shift+P / Cmd+Shift+P)
2. Busca "MCP: Execute Command"
3. Selecciona "cursor-agent-24-7"
4. Escribe tu comando y presiona Enter

### Desde la API REST

```bash
# Enviar comando vía MCP endpoint
curl -X POST http://localhost:8025/mcp/v1/command \
  -H "Content-Type: application/json" \
  -d '{"command": "print(\"Hello from MCP!\")"}'
```

### Desde WebSocket

```javascript
const ws = new WebSocket('ws://localhost:8025/mcp/v1/ws');

ws.onopen = () => {
  ws.send(JSON.stringify({
    jsonrpc: "2.0",
    method: "mcp/command",
    params: {
      command: "print('Hello from WebSocket!')"
    },
    id: "1"
  }));
};

ws.onmessage = (event) => {
  const response = JSON.parse(event.data);
  console.log('Response:', response);
};
```

## Endpoints MCP

### `GET /mcp/v1/health`

Health check del servidor MCP.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "agent_status": "running"
}
```

### `GET /mcp/v1/resources`

Listar recursos disponibles.

**Response:**
```json
{
  "resources": [
    {
      "id": "cursor-agent",
      "type": "agent",
      "name": "Cursor Agent 24/7",
      "description": "Agente persistente que ejecuta comandos",
      "capabilities": ["execute_command", "get_status", "list_tasks"]
    }
  ]
}
```

### `POST /mcp/v1/command`

Ejecutar comando a través de MCP.

**Request:**
```json
{
  "command": "print('Hello World')"
}
```

**Response:**
```json
{
  "success": true,
  "task_id": "task_1234567890_0",
  "message": "Command queued for execution"
}
```

### `GET /mcp/v1/status`

Obtener estado del agente.

**Response:**
```json
{
  "status": "running",
  "running": true,
  "tasks_total": 10,
  "tasks_pending": 2,
  "tasks_running": 1,
  "tasks_completed": 6,
  "tasks_failed": 1
}
```

### `GET /mcp/v1/metrics`

Obtener métricas del servidor MCP.

### `GET /mcp/v1/metrics/prometheus`

Exportar métricas en formato Prometheus (compatible con Prometheus scraping).

**Response:** Texto plano en formato Prometheus

```
# HELP mcp_requests_total Total number of requests
# TYPE mcp_requests_total counter
mcp_requests_total 150
...
```

**Uso con Prometheus:**
```yaml
scrape_configs:
  - job_name: 'mcp-server'
    scrape_interval: 15s
    static_configs:
      - targets: ['localhost:8025']
    metrics_path: '/mcp/v1/metrics/prometheus'
```

**Response:**
```json
{
  "uptime_seconds": 3600,
  "total_requests": 150,
  "total_errors": 5,
  "commands_executed": 145,
  "commands_failed": 5,
  "websocket_connections": 3,
  "average_response_time_ms": 45.2,
  "response_time_percentiles": {
    "p50": 42.1,
    "p75": 48.5,
    "p90": 55.2,
    "p95": 62.3,
    "p99": 78.9,
    "min": 12.5,
    "max": 120.0
  },
  "requests_per_second": 0.04,
  "error_rate": 0.0333,
  "error_types": {
    "validation_error": 2,
    "internal_error": 3
  },
  "top_clients": {
    "127.0.0.1": 100,
    "192.168.1.1": 50
  },
  "endpoint_stats": {
    "/mcp/v1/command": {
      "request_count": 145,
      "avg_response_time_ms": 45.2,
      "percentiles": {
        "p50": 42.1,
        "p75": 48.5,
        "p90": 55.2,
        "p95": 62.3,
        "p99": 78.9
      }
    }
  }
}
```

### `GET /mcp/v1/tasks/{task_id}/stream`

Streaming de resultados de tarea usando Server-Sent Events (SSE).

**Response:** Stream de eventos SSE

```
data: {"status": "running", "task_id": "task_123"}

data: {"status": "completed", "result": "Task output here"}
```

### `GET /mcp/v1/tasks/{task_id}/result`

Obtener resultado de tarea (con caché).

**Response:**
```json
{
  "task_id": "task_1234567890_0",
  "status": "completed",
  "result": "Task output",
  "error": null,
  "timestamp": "2024-01-01T12:00:00"
}
```

### `POST /mcp/v1/commands/batch`

Ejecutar múltiples comandos en lote.

**Request:**
```json
{
  "commands": [
    "print('Hello')",
    "print('World')"
  ],
  "parallel": true,
  "metadata": {}
}
```

**Response:**
```json
{
  "success": true,
  "task_ids": ["task_1", "task_2"],
  "count": 2,
  "parallel": true,
  "timestamp": "2024-01-01T12:00:00",
  "response_time_ms": 45.2
}
```

### `POST /mcp/v1/auth/login`

Autenticar y obtener API key (requiere autenticación habilitada).

**Request:**
```json
{
  "username": "admin",
  "password": "admin"
}
```

**Response:**
```json
{
  "success": true,
  "session_id": "session_123",
  "api_key": "api_key_here",
  "role": "admin",
  "expires_at": "2024-01-02T12:00:00"
}
```

### `GET /mcp/v1/config`

Obtener configuración del servidor (requiere admin).

**Headers:**
```
X-API-Key: your_api_key
```

**Response:**
```json
{
  "host": "localhost",
  "port": 8025,
  "enable_cors": true,
  "enable_auth": false,
  "enable_cache": true,
  "rate_limit_max_requests": 100
}
```

### `GET /mcp/v1/events`

Obtener eventos recientes del sistema (requiere admin).

**Query Parameters:**
- `event_type` (opcional): Filtrar por tipo de evento
- `limit` (opcional, default: 100): Número máximo de eventos a retornar

**Headers:**
```
X-API-Key: your_api_key
```

**Response:**
```json
{
  "count": 10,
  "events": [
    {
      "event_type": "command_executed",
      "data": {
        "task_id": "task_123",
        "command": "print('Hello')",
        "client_id": "127.0.0.1",
        "username": "admin"
      },
      "timestamp": "2024-01-01T12:00:00",
      "source": "execute_command"
    }
  ]
}
```

### `DELETE /mcp/v1/cache`

Limpiar todo el caché del servidor (requiere admin).

**Headers:**
```
X-API-Key: your_api_key
```

**Response:**
```json
{
  "success": true,
  "message": "Cache cleared successfully",
  "timestamp": "2024-01-01T12:00:00"
}
```

### `GET /mcp/v1/cache/stats`

Obtener estadísticas del caché (requiere admin).

**Headers:**
```
X-API-Key: your_api_key
```

**Response:**
```json
{
  "size": 500,
  "max_size": 1000,
  "expired_entries": 10,
  "eviction_policy": "lru",
  "hit_rate": 0.85
}
```

### `WebSocket /mcp/v1/ws`

Conexión WebSocket para comunicación en tiempo real.

### `GET /mcp/v1/docs`

Documentación interactiva de la API (Swagger UI).

### `GET /mcp/v1/redoc`

Documentación alternativa de la API (ReDoc).

### `GET /mcp/v1/openapi.json`

Especificación OpenAPI en formato JSON.

**Mensajes soportados:**

1. **mcp/command**: Ejecutar comando
```json
{
  "jsonrpc": "2.0",
  "method": "mcp/command",
  "params": {
    "command": "print('Hello')"
  },
  "id": "1"
}
```

2. **mcp/ping**: Ping para mantener conexión
```json
{
  "jsonrpc": "2.0",
  "method": "mcp/ping",
  "id": "2"
}
```

## Arquitectura

```
┌─────────────┐
│ Cursor IDE  │
└──────┬──────┘
       │ MCP Protocol
       ▼
┌─────────────────┐
│   MCP Server    │
│  (Port 8025)    │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  Cursor Agent   │
│   (Port 8024)   │
└─────────────────┘
```

## Troubleshooting

### El servidor MCP no inicia

- Verifica que el puerto 8025 esté disponible
- Revisa los logs: `tail -f logs/agent.log`
- Asegúrate de usar `--enable-mcp` al iniciar

### Cursor IDE no se conecta

- Verifica que el servidor MCP esté corriendo: `curl http://localhost:8025/mcp/v1/health`
- Revisa la configuración de Cursor
- Asegúrate de que la URL en la configuración sea correcta

### Los comandos no se ejecutan

- Verifica que el agente esté en estado "running": `curl http://localhost:8024/api/status`
- Revisa los logs del agente
- Verifica que el formato del comando sea correcto

## Desarrollo

### Agregar nuevos endpoints MCP

Edita `core/mcp_server.py` y agrega nuevas rutas:

```python
@self.app.post("/mcp/v1/custom-endpoint")
async def custom_endpoint(request: dict):
    # Tu lógica aquí
    return {"result": "success"}
```

### Extender el cliente MCP

Edita `core/mcp_client.py` para agregar nuevas funcionalidades:

```python
async def custom_method(self, param: str) -> Dict[str, Any]:
    message = {
        "jsonrpc": "2.0",
        "method": "mcp/custom",
        "params": {"param": param},
        "id": f"custom_{datetime.now().timestamp()}"
    }
    return message
```

## Referencias

- [Model Context Protocol Specification](https://modelcontextprotocol.io)
- [Cursor IDE Documentation](https://cursor.sh/docs)

