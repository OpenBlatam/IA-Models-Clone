# Arquitectura Completa - Contabilidad Mexicana AI SAM3

## Tabla de Contenidos

1. [Visión General](#visión-general)
2. [Arquitectura de Alto Nivel](#arquitectura-de-alto-nivel)
3. [Componentes Principales](#componentes-principales)
4. [Flujos de Datos](#flujos-de-datos)
5. [Patrones de Diseño](#patrones-de-diseño)
6. [Integraciones](#integraciones)
7. [Gestión de Tareas](#gestión-de-tareas)
8. [Ejecución Paralela](#ejecución-paralela)
9. [Manejo de Errores](#manejo-de-errores)
10. [Seguridad](#seguridad)
11. [Performance](#performance)
12. [Escalabilidad](#escalabilidad)
13. [Extensibilidad](#extensibilidad)

---

## Visión General

**Contabilidad Mexicana AI SAM3** es un sistema autónomo de contabilidad fiscal mexicana que combina:

- **Arquitectura SAM3**: Procesamiento continuo 24/7 con ejecución paralela
- **OpenRouter**: Integración con modelos LLM de última generación
- **TruthGPT**: Optimización avanzada y analytics
- **Servicios Fiscales**: Cálculo de impuestos, asesoría, guías, trámites SAT

### Principios de Diseño

1. **Autonomía**: Sistema que opera continuamente sin intervención
2. **Paralelismo**: Procesamiento concurrente de múltiples tareas
3. **Resiliencia**: Recuperación automática de errores
4. **Extensibilidad**: Fácil agregar nuevos servicios
5. **Modularidad**: Componentes independientes y reutilizables

---

## Arquitectura de Alto Nivel

```
┌─────────────────────────────────────────────────────────────┐
│                    Contabilidad Mexicana AI SAM3            │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   API REST   │  │  Python API  │  │  Continuous  │    │
│  │   (FastAPI)  │  │   (Direct)   │  │     Mode     │    │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
│         │                  │                  │            │
│         └──────────────────┼──────────────────┘            │
│                            │                                 │
│                  ┌─────────▼─────────┐                      │
│                  │ ContadorSAM3Agent │                      │
│                  │   (Orchestrator)  │                      │
│                  └─────────┬─────────┘                      │
│                            │                                 │
│         ┌──────────────────┼──────────────────┐            │
│         │                  │                  │            │
│  ┌──────▼──────┐  ┌───────▼───────┐  ┌───────▼───────┐    │
│  │TaskManager  │  │ParallelExecutor│  │Service Handlers│   │
│  │(Queue/Prio) │  │  (Workers)    │  │  (Business)   │    │
│  └──────┬──────┘  └───────┬───────┘  └───────┬───────┘    │
│         │                  │                  │            │
│         └──────────────────┼──────────────────┘            │
│                            │                                 │
│         ┌──────────────────┼──────────────────┐            │
│         │                  │                  │            │
│  ┌──────▼──────┐  ┌───────▼───────┐  ┌───────▼───────┐    │
│  │OpenRouter   │  │  TruthGPT     │  │  Prompt       │    │
│  │  Client     │  │   Client      │  │  Builders     │    │
│  └─────────────┘  └───────────────┘  └───────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## Componentes Principales

### 1. ContadorSAM3Agent (Core Orchestrator)

**Ubicación**: `core/contador_sam3_agent.py`

**Responsabilidades**:
- Orquestación de todos los servicios
- Gestión del ciclo de vida del agente
- Enrutamiento de tareas a handlers específicos
- Coordinación entre componentes

**Estructura**:
```python
class ContadorSAM3Agent:
    - openrouter_client: OpenRouterClient
    - truthgpt_client: TruthGPTClient
    - task_manager: TaskManager
    - parallel_executor: ParallelExecutor
    - system_prompts: Dict[str, str]
    
    # Métodos principales
    - start() -> None
    - stop() -> None
    - _process_task(task) -> Dict
    - calcular_impuestos(...) -> str
    - asesoria_fiscal(...) -> str
    - guia_fiscal(...) -> str
    - tramite_sat(...) -> str
    - ayuda_declaracion(...) -> str
```

**Flujo de Operación**:
1. Inicialización de clientes y managers
2. Loop principal continuo (24/7)
3. Obtención de tareas pendientes
4. Envío a parallel executor
5. Procesamiento asíncrono
6. Almacenamiento de resultados

---

### 2. TaskManager (Gestión de Tareas)

**Ubicación**: `core/task_manager.py`

**Responsabilidades**:
- Creación y gestión de tareas
- Cola de prioridades
- Persistencia de estado
- Seguimiento de resultados

**Estructura de Datos**:
```python
@dataclass
class Task:
    id: str
    service_type: str
    parameters: Dict[str, Any]
    priority: int
    status: TaskStatus
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    result: Optional[Dict[str, Any]]
    error: Optional[str]
```

**Estados de Tarea**:
- `PENDING`: En cola esperando procesamiento
- `PROCESSING`: Actualmente siendo procesada
- `COMPLETED`: Completada exitosamente
- `FAILED`: Falló durante procesamiento
- `CANCELLED`: Cancelada manualmente

**Cola de Prioridades**:
- Tareas ordenadas por `priority` (mayor = más importante)
- FIFO dentro del mismo nivel de prioridad
- Persistencia en disco para recuperación

---

### 3. ParallelExecutor (Ejecución Paralela)

**Ubicación**: `core/parallel_executor.py`

**Responsabilidades**:
- Pool de workers asíncronos
- Distribución de tareas
- Balanceo de carga
- Estadísticas de ejecución

**Arquitectura**:
```
ParallelExecutor
├── Worker Pool (configurable, default: 10)
│   ├── Worker-0
│   ├── Worker-1
│   ├── ...
│   └── Worker-N
├── Task Queue (asyncio.Queue)
└── Statistics Tracker
```

**Características**:
- Workers independientes que procesan tareas concurrentemente
- Cola thread-safe con `asyncio.Queue`
- Timeout y cancelación de tareas
- Estadísticas en tiempo real

**Flujo de Worker**:
1. Worker espera en cola
2. Obtiene tarea cuando disponible
3. Ejecuta función asíncrona
4. Maneja errores y excepciones
5. Actualiza estadísticas
6. Vuelve a esperar

---

### 4. OpenRouterClient (Integración LLM)

**Ubicación**: `infrastructure/openrouter_client.py`

**Responsabilidades**:
- Comunicación con OpenRouter API
- Connection pooling
- Reintentos automáticos
- Manejo de timeouts

**Características**:
- **Connection Pooling**: Reutilización de conexiones HTTP
- **Retry Logic**: Backoff exponencial en errores
- **Timeout Management**: Límites configurables
- **HTTP/2 Support**: Mejor performance

**Flujo de Request**:
```
1. Preparar mensajes
2. Obtener cliente HTTP (pool)
3. Enviar request con retry
4. Parsear respuesta
5. Retornar resultado estructurado
```

**Manejo de Errores**:
- HTTP errors → Retry con backoff
- Timeouts → Retry con delay
- Invalid responses → Exception con detalles

---

### 5. TruthGPTClient (Optimización)

**Ubicación**: `infrastructure/truthgpt_client.py`

**Responsabilidades**:
- Integración con TruthGPT modules
- Optimización de queries
- Analytics y monitoreo
- Mejora de calidad

**Componentes TruthGPT**:
- `TruthGPTIntegrationManager`: Integración principal
- `TruthGPTAnalyticsManager`: Analytics y métricas
- `TruthGPTConfigManager`: Gestión de configuración

**Flujo de Optimización**:
```
1. Query original
2. TruthGPT processing
3. Query optimizada
4. Analytics tracking
5. Resultado mejorado
```

**Modo Degradado**:
- Si TruthGPT no disponible → Usa query original
- Logging de advertencias
- Sistema sigue funcionando

---

### 6. Prompt Builders

**Ubicación**: `core/prompt_builder.py`, `core/system_prompts_builder.py`

**Responsabilidades**:
- Construcción de prompts de usuario
- Gestión de system prompts
- Formateo de datos para LLM

**SystemPromptsBuilder**:
- Base prompt común
- Especializaciones por servicio
- Mantenimiento centralizado

**PromptBuilder**:
- Templates por tipo de servicio
- Formateo de datos
- Contexto adicional

**Estructura de Prompts**:
```
System Prompt (especializado)
├── Base: Conocimiento fiscal general
└── Especialización: Servicio específico

User Prompt
├── Datos estructurados
├── Contexto adicional
└── Instrucciones específicas
```

---

### 7. Helpers y Utilities

**Ubicación**: `core/helpers.py`, `utils/`

**Funcionalidades**:
- **Helpers**: Operaciones comunes (mensajes, JSON)
- **Formatters**: Formateo de moneda, porcentajes, períodos
- **Validators**: Validación de RFC, CURP, regímenes, datos

**Formatters**:
- `format_currency()`: Formato monetario mexicano
- `format_percentage()`: Porcentajes
- `format_fiscal_period()`: Períodos fiscales
- `format_tax_calculation_result()`: Resultados de cálculos
- `format_fiscal_advice()`: Asesorías formateadas

**Validators**:
- `validate_rfc()`: Validación RFC mexicano
- `validate_curp()`: Validación CURP
- `validate_regimen()`: Regímenes válidos
- `validate_calculation_data()`: Datos de cálculo
- `validate_declaration_data()`: Datos de declaración

---

## Flujos de Datos

### Flujo 1: Cálculo de Impuestos

```
Usuario
  │
  ├─> ContadorSAM3Agent.calcular_impuestos()
  │   │
  │   ├─> Validar datos (opcional)
  │   │
  │   ├─> TaskManager.create_task()
  │   │   ├─> Crear Task object
  │   │   ├─> Agregar a cola de prioridades
  │   │   └─> Persistir en disco
  │   │
  │   └─> Retornar task_id
  │
  │
Loop Principal (24/7)
  │
  ├─> TaskManager.get_pending_tasks()
  │   └─> Obtener tareas ordenadas por prioridad
  │
  ├─> ParallelExecutor.submit_task()
  │   └─> Agregar a cola de workers
  │
  │
Worker (asíncrono)
  │
  ├─> ContadorSAM3Agent._process_task()
  │   │
  │   ├─> TaskManager.update_task_status("processing")
  │   │
  │   ├─> ContadorSAM3Agent._handle_calcular_impuestos()
  │   │   │
  │   │   ├─> PromptBuilder.build_calculation_prompt()
  │   │   │
  │   │   ├─> TruthGPTClient.optimize_query()
  │   │   │   └─> Optimizar prompt (opcional)
  │   │   │
  │   │   ├─> Construir mensajes
  │   │   │   ├─> System prompt (especializado)
  │   │   │   └─> User prompt (optimizado)
  │   │   │
  │   │   └─> OpenRouterClient.chat_completion()
  │   │       ├─> Connection pooling
  │   │       ├─> Retry logic
  │   │       └─> Parsear respuesta
  │   │
  │   ├─> TaskManager.complete_task()
  │   │   ├─> Actualizar estado
  │   │   ├─> Guardar resultado
  │   │   └─> Persistir en disco
  │   │
  │   └─> Retornar resultado
  │
  │
Usuario
  │
  ├─> ContadorSAM3Agent.get_task_status(task_id)
  │   └─> Retornar estado actual
  │
  └─> ContadorSAM3Agent.get_task_result(task_id)
      └─> Retornar resultado si completado
```

### Flujo 2: Modo Continuo 24/7

```
Inicio
  │
  ├─> ContadorSAM3Agent.start()
  │   │
  │   ├─> ParallelExecutor.start()
  │   │   └─> Crear pool de workers
  │   │
  │   └─> Loop Principal (while running)
  │       │
  │       ├─> TaskManager.get_pending_tasks(limit=10)
  │       │   └─> Obtener tareas pendientes
  │       │
  │       ├─> Para cada tarea:
  │       │   └─> ParallelExecutor.submit_task()
  │       │
  │       ├─> Esperar 1 segundo
  │       │
  │       └─> Repetir
  │
  │
Workers (paralelos)
  │
  ├─> Worker-0: Procesa tarea A
  ├─> Worker-1: Procesa tarea B
  ├─> Worker-2: Procesa tarea C
  └─> ...
  │
  └─> Todos procesan concurrentemente
  │
  │
Shutdown
  │
  ├─> Signal handler (SIGINT/SIGTERM)
  │   └─> ContadorSAM3Agent.stop()
  │       │
  │       ├─> ParallelExecutor.stop()
  │       │   ├─> Esperar cola vacía
  │       │   └─> Cancelar workers
  │       │
  │       ├─> OpenRouterClient.close()
  │       │
  │       └─> TruthGPTClient.close()
```

---

## Patrones de Diseño

### 1. Arquitectura SAM3

**Características**:
- **Continuidad**: Operación 24/7 sin interrupciones
- **Paralelismo**: Múltiples tareas procesadas simultáneamente
- **Autonomía**: Sistema auto-gestionado
- **Resiliencia**: Recuperación automática

**Implementación**:
- Loop principal continuo
- Pool de workers asíncronos
- Cola de tareas con prioridades
- Persistencia para recuperación

### 2. Single Responsibility Principle

Cada clase tiene una responsabilidad única:

- `ContadorSAM3Agent`: Orquestación
- `TaskManager`: Gestión de tareas
- `ParallelExecutor`: Ejecución paralela
- `PromptBuilder`: Construcción de prompts
- `OpenRouterClient`: Comunicación LLM
- `TruthGPTClient`: Optimización

### 3. Dependency Injection

Configuración inyectada en lugar de hardcodeada:

```python
config = ContadorSAM3Config()
agent = ContadorSAM3Agent(config=config)
```

**Beneficios**:
- Testabilidad
- Flexibilidad
- Mantenibilidad

### 4. Strategy Pattern

Diferentes handlers para diferentes servicios:

```python
if service_type == "calcular_impuestos":
    result = await self._handle_calcular_impuestos(parameters)
elif service_type == "asesoria_fiscal":
    result = await self._handle_asesoria_fiscal(parameters)
```

### 5. Factory Pattern

Builders para creación de objetos complejos:

- `SystemPromptsBuilder`: Construye todos los system prompts
- `PromptBuilder`: Construye user prompts por servicio

### 6. Observer Pattern (implícito)

Task status tracking permite observación de estado:

```python
status = await agent.get_task_status(task_id)
```

---

## Integraciones

### OpenRouter Integration

**Propósito**: Acceso a modelos LLM de última generación

**Modelos Soportados**:
- Claude 3.5 Sonnet (default)
- GPT-4
- Gemini Pro
- Otros modelos disponibles en OpenRouter

**Características**:
- Connection pooling para eficiencia
- Retry automático con backoff exponencial
- Timeout configurables
- HTTP/2 para mejor performance

**Configuración**:
```python
OpenRouterConfig(
    api_key="...",
    model="anthropic/claude-3.5-sonnet",
    timeout=60.0,
    max_retries=3
)
```

### TruthGPT Integration

**Propósito**: Optimización y analytics

**Componentes**:
- `TruthGPTIntegrationManager`: Integración principal
- `TruthGPTAnalyticsManager`: Tracking y métricas
- `TruthGPTConfigManager`: Configuración

**Funcionalidades**:
- Optimización de queries antes de enviar a LLM
- Analytics de uso y performance
- Mejora de calidad de respuestas

**Modo Degradado**:
- Si TruthGPT no disponible → Funciona sin optimización
- Sistema sigue operando normalmente
- Logging de advertencias

---

## Gestión de Tareas

### Cola de Prioridades

**Implementación**:
- Lista ordenada por `priority` (descendente)
- FIFO dentro del mismo nivel
- Actualización en tiempo real

**Prioridades**:
- `10`: Crítico (urgente)
- `5`: Alto (importante)
- `0`: Normal (default)
- `-5`: Bajo (puede esperar)

### Persistencia

**Almacenamiento**:
- Archivos JSON por tarea
- Directorio: `task_storage/`
- Formato: `{task_id}.json`

**Recuperación**:
- Carga automática al iniciar
- Restauración de estado
- Continuidad de procesamiento

### Seguimiento de Estado

**Estados**:
1. `PENDING`: En cola
2. `PROCESSING`: En ejecución
3. `COMPLETED`: Finalizada exitosamente
4. `FAILED`: Error durante ejecución
5. `CANCELLED`: Cancelada manualmente

**Transiciones**:
```
PENDING → PROCESSING → COMPLETED
                    → FAILED
                    → CANCELLED
```

---

## Ejecución Paralela

### Worker Pool

**Configuración**:
- Número de workers: `max_parallel_tasks` (default: 10)
- Configurable por instancia
- Balanceo automático

**Distribución**:
- Workers independientes
- Cola compartida thread-safe
- Sin dependencias entre workers

### Concurrencia

**Modelo**:
- Asyncio para I/O bound operations
- Workers coroutines independientes
- No bloqueo entre tareas

**Ventajas**:
- Alto throughput
- Eficiente uso de recursos
- Escalabilidad horizontal

### Estadísticas

**Métricas**:
- Total de tareas
- Tareas completadas
- Tareas fallidas
- Workers activos
- Tamaño de cola

---

## Manejo de Errores

### Estrategias

1. **Retry con Backoff Exponencial**
   - OpenRouter requests
   - Timeouts y errores HTTP
   - Límite de reintentos

2. **Error Handling por Capa**
   - Infrastructure: Retry y logging
   - Core: Manejo de excepciones
   - API: Respuestas HTTP apropiadas

3. **Task Failure Tracking**
   - Estado `FAILED`
   - Mensaje de error guardado
   - Logging detallado

### Recuperación

**Automática**:
- Reintentos en requests HTTP
- Workers continúan después de errores
- Sistema sigue operando

**Manual**:
- Re-submit de tareas fallidas
- Análisis de logs
- Debugging con información guardada

---

## Seguridad

### API Keys

**Almacenamiento**:
- Variables de entorno
- No hardcodeadas
- Validación al inicio

**Validación**:
```python
if not self.api_key:
    raise ValueError("OpenRouter API key required")
```

### Validación de Entrada

**Validators**:
- RFC, CURP
- Regímenes fiscales
- Datos de cálculo
- Períodos fiscales

**Sanitización**:
- Validación antes de procesar
- Errores descriptivos
- Prevención de inyección

### Timeouts y Límites

**Protecciones**:
- Timeouts en requests HTTP
- Límite de workers
- Límite de tareas en cola
- Timeout en ejecución de tareas

---

## Performance

### Optimizaciones

1. **Connection Pooling**
   - Reutilización de conexiones HTTP
   - Menos overhead de conexión
   - Mejor throughput

2. **Paralelismo**
   - Múltiples tareas simultáneas
   - Workers independientes
   - Sin bloqueo

3. **Caching (futuro)**
   - Resultados frecuentes
   - Reducción de llamadas LLM
   - Mejor latencia

### Métricas

**Tracking**:
- Tiempo de respuesta
- Tokens utilizados
- Tareas por segundo
- Tasa de éxito

**Monitoreo**:
- Logs estructurados
- Estadísticas en tiempo real
- Analytics con TruthGPT

---

## Escalabilidad

### Horizontal

**Estrategias**:
- Múltiples instancias del agente
- Load balancing
- Shared task queue (futuro)

**Limitaciones Actuales**:
- Task storage local
- Sin coordinación entre instancias

### Vertical

**Mejoras**:
- Aumentar número de workers
- Más memoria para cache
- CPU para procesamiento

**Configuración**:
```python
agent = ContadorSAM3Agent(
    max_parallel_tasks=20  # Más workers
)
```

---

## Extensibilidad

### Agregar Nuevo Servicio

**Pasos**:

1. **Agregar Prompt Builder**
   ```python
   @staticmethod
   def build_nuevo_servicio_prompt(...):
       return "..."
   ```

2. **Agregar System Prompt**
   ```python
   def _get_nuevo_servicio_specialization():
       return "..."
   ```

3. **Agregar Handler**
   ```python
   async def _handle_nuevo_servicio(self, parameters):
       # Lógica de procesamiento
   ```

4. **Agregar Método Público**
   ```python
   async def nuevo_servicio(self, ...):
       task_id = await self.task_manager.create_task(...)
       return task_id
   ```

5. **Agregar API Endpoint** (opcional)
   ```python
   @app.post("/nuevo-servicio")
   async def nuevo_servicio(request):
       ...
   ```

### Personalización

**Configuración**:
- Variables de entorno
- `ContadorSAM3Config`
- Parámetros en inicialización

**Extensión**:
- Subclases de componentes
- Plugins (futuro)
- Hooks y callbacks

---

## Diagrama de Secuencia Completo

```
Usuario          Agent          TaskManager    ParallelExecutor  OpenRouter    TruthGPT
  │                │                 │              │               │            │
  │─calcular()────>│                 │              │               │            │
  │                │─create_task()──>│              │               │            │
  │                │<──task_id───────│              │               │            │
  │<──task_id──────│                 │              │               │            │
  │                │                 │              │               │            │
  │                │                 │              │               │            │
  │                │─get_pending()──>│              │               │            │
  │                │<──tasks─────────│              │               │            │
  │                │─submit_task()───┼─────────────>│               │            │
  │                │                 │              │               │            │
  │                │                 │              │─process()──────┼───────────>│
  │                │                 │              │                │            │
  │                │                 │              │─optimize()─────┼───────────>│
  │                │                 │              │<──optimized─────┼───────────<│
  │                │                 │              │                │            │
  │                │                 │              │─chat()─────────>│            │
  │                │                 │              │<──response─────│            │
  │                │                 │              │                │            │
  │                │                 │              │─complete()─────>│            │
  │                │                 │<──updated────│                │            │
  │                │                 │              │                │            │
  │─get_status()──>│                 │              │               │            │
  │                │─get_status()────>│              │               │            │
  │                │<──status────────│              │               │            │
  │<──status───────│                 │              │               │            │
  │                │                 │              │               │            │
  │─get_result()──>│                 │              │               │            │
  │                │─get_result()────>│              │               │            │
  │                │<──result────────│              │               │            │
  │<──result───────│                 │              │               │            │
```

---

## Conclusión

Esta arquitectura proporciona:

✅ **Autonomía**: Operación continua 24/7  
✅ **Escalabilidad**: Paralelismo y extensibilidad  
✅ **Resiliencia**: Manejo robusto de errores  
✅ **Modularidad**: Componentes independientes  
✅ **Performance**: Optimizaciones múltiples  
✅ **Extensibilidad**: Fácil agregar servicios  

El sistema está diseñado para ser robusto, escalable y fácil de mantener.
