# Arquitectura - Piel Mejorador AI SAM3

## 🏗️ Arquitectura del Sistema

### Estructura de Capas

```
┌─────────────────────────────────────┐
│         API Layer (FastAPI)          │
│  - Endpoints                          │
│  - Request/Response handling          │
│  - Authentication                    │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│         Core Layer                   │
│  - PielMejoradorAgent                │
│  - ServiceHandler                    │
│  - TaskManager                        │
│  - Managers (Cache, Webhook, etc.)   │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│      Infrastructure Layer            │
│  - OpenRouterClient                  │
│  - TruthGPTClient                    │
│  - BaseHTTPClient                    │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│         Utilities Layer              │
│  - Helpers                           │
│  - Validators                        │
│  - Error Handlers                    │
└─────────────────────────────────────┘
```

## 📦 Componentes Principales

### 1. Base Classes

#### BaseHTTPClient
- Funcionalidad común para clientes HTTP
- Connection pooling
- Retry logic
- Error handling

#### BaseService
- Interfaz común para servicios
- Request/response handling
- Validación
- Error handling

#### BaseManager
- Funcionalidad común para managers
- Statistics tracking
- Lifecycle management
- Resource management

### 2. Core Components

#### PielMejoradorAgent
- Orquestador principal
- Gestión de tareas
- Coordinación de servicios

#### ServiceHandler
- Manejo de servicios
- Integración con APIs externas
- Procesamiento de requests

#### TaskManager
- Gestión de tareas
- Estado y tracking
- Persistencia

### 3. Infrastructure

#### OpenRouterClient
- Cliente para OpenRouter API
- Soporte para vision models
- Retry y error handling

#### TruthGPTClient
- Cliente para TruthGPT
- Optimización de prompts
- Análisis avanzado

### 4. Managers

#### CacheManager
- Caché inteligente
- TTL management
- Estadísticas

#### WebhookManager
- Gestión de webhooks
- Notificaciones
- Retry logic

#### MemoryOptimizer
- Optimización de memoria
- Monitoreo
- Limpieza automática

## 🔄 Flujo de Datos

### Procesamiento de Imagen

```
1. API Request → FastAPI Endpoint
2. Endpoint → PielMejoradorAgent
3. Agent → TaskManager (crear tarea)
4. Agent → ServiceHandler
5. ServiceHandler → Validators
6. ServiceHandler → OpenRouterClient
7. OpenRouterClient → OpenRouter API
8. Response → ServiceHandler
9. ServiceHandler → TaskManager (completar)
10. TaskManager → WebhookManager (notificar)
```

## 🎯 Patrones de Diseño

### Factory Pattern
- ServiceFactory: Creación de servicios
- AgentBuilder: Construcción del agente

### Dependency Injection
- DIContainer: Inyección de dependencias
- Service resolution

### Strategy Pattern
- Diferentes estrategias de procesamiento
- Configuración flexible

### Observer Pattern
- EventBus: Pub/Sub
- Webhooks: Notificaciones

### Circuit Breaker
- Protección contra fallos
- Recuperación automática

## 📊 Organización de Código

### Estructura de Directorios

```
piel_mejorador_ai_sam3/
├── api/              # API endpoints
├── config/           # Configuración
├── core/             # Lógica principal
│   ├── common/       # Base classes
│   └── utils/        # Utilidades
├── infrastructure/   # Clientes externos
├── tests/            # Tests
└── examples/         # Ejemplos
```

## 🔧 Principios de Diseño

1. **Separation of Concerns**: Cada componente tiene una responsabilidad clara
2. **DRY (Don't Repeat Yourself)**: Base classes eliminan duplicación
3. **SOLID Principles**: Código siguiendo principios SOLID
4. **Dependency Inversion**: Dependencias inyectadas
5. **Open/Closed**: Extensible sin modificar código existente

## 🚀 Escalabilidad

- **Horizontal**: Múltiples instancias
- **Vertical**: Optimización de recursos
- **Caching**: Reducción de carga
- **Async**: Procesamiento no bloqueante
- **Queue**: Procesamiento distribuido

El sistema está diseñado para escalar eficientemente.




