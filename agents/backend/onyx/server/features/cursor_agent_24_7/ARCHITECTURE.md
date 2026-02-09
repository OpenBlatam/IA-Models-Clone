# 🏗️ Arquitectura - Cursor Agent 24/7

Arquitectura completa del sistema siguiendo principios de microservicios, serverless y cloud-native.

## 📐 Diagrama de Arquitectura

```
┌─────────────────────────────────────────────────────────────┐
│                    AWS API Gateway / ALB                     │
│              (Rate Limiting, Auth, Routing)                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Application                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Middleware │  │    Routes     │  │   WebSocket  │     │
│  │  - Logging   │  │  - Agent     │  │   Manager    │     │
│  │  - Auth      │  │  - Tasks     │  │              │     │
│  │  - Metrics   │  │  - Health    │  │              │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│   DynamoDB   │ │   ElastiCache│ │  CloudWatch  │
│   (State)    │ │   (Cache)    │ │  (Logs)      │
└──────────────┘ └──────────────┘ └──────────────┘
        │              │              │
        └──────────────┼──────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Celery Workers                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Task Queue  │  │ Heavy Tasks  │  │ Notifications│     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Message Brokers                           │
│  ┌──────────────┐  ┌──────────────┐                        │
│  │  RabbitMQ    │  │    Kafka     │                        │
│  └──────────────┘  └──────────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

## 🎯 Principios de Diseño

### 1. Stateless Design

- **Estado en DynamoDB**: Todo el estado persistente está en DynamoDB
- **Caché en Redis**: Caché distribuido usando ElastiCache
- **Sin estado local**: No hay archivos locales o estado en memoria que se pierda

### 2. Microservices Ready

- **Separación de concerns**: Cada módulo tiene responsabilidad única
- **API Gateway**: Integración con AWS API Gateway o Kong
- **Message Brokers**: Comunicación asíncrona entre servicios
- **Health Checks**: Endpoints de salud para orquestación

### 3. Serverless Optimized

- **Lambda compatible**: Funciona en AWS Lambda
- **Cold start optimization**: Lazy loading de dependencias
- **Minimal dependencies**: Solo lo esencial
- **Auto-scaling**: Escala automáticamente según carga

### 4. Observability First

- **OpenTelemetry**: Distributed tracing completo
- **Prometheus**: Métricas detalladas
- **Structured Logging**: Logs estructurados con contexto
- **CloudWatch**: Integración con AWS CloudWatch

## 🔧 Componentes Principales

### API Layer

**FastAPI Application**
- Middleware stack completo
- OAuth2 authentication
- Rate limiting
- Metrics collection
- Error handling

**Endpoints:**
- `/api/health` - Health check
- `/api/status` - Agent status
- `/api/tasks` - Task management
- `/api/auth/token` - Authentication
- `/metrics` - Prometheus metrics
- `/ws` - WebSocket

### Core Services

**Agent Core**
- Task processing
- State management
- Command execution
- Event handling

**Observability**
- OpenTelemetry tracing
- Prometheus metrics
- Structured logging

**Security**
- OAuth2/JWT
- Rate limiting
- Security headers

### Data Layer

**DynamoDB**
- Agent state
- Task history
- Configuration

**ElastiCache Redis**
- Cache
- Rate limiting counters
- Session storage

**CloudWatch**
- Logs
- Metrics
- Alarms

### Workers

**Celery Workers**
- Background task processing
- Heavy computations
- Notifications
- Report generation

**Queues:**
- `tasks` - Regular tasks
- `heavy` - Heavy computations
- `notifications` - Notifications

### Message Brokers

**RabbitMQ / Kafka**
- Event publishing
- Service communication
- Event-driven architecture

## 🔄 Flujos de Datos

### Request Flow

```
1. Client Request
   ↓
2. API Gateway (Rate Limiting, Auth)
   ↓
3. FastAPI Middleware (Logging, Tracing)
   ↓
4. Route Handler
   ↓
5. Agent Core / Celery Task
   ↓
6. DynamoDB / Redis
   ↓
7. Response
```

### Task Processing Flow

```
1. Task Created
   ↓
2. Enqueue to Celery
   ↓
3. Worker Picks Up
   ↓
4. Process Task
   ↓
5. Update DynamoDB
   ↓
6. Publish Event (Message Broker)
   ↓
7. Notification (if needed)
```

### Event Flow

```
1. Event Occurs
   ↓
2. Publish to Message Broker
   ↓
3. Subscribers Receive
   ↓
4. Process Event
   ↓
5. Update State / Trigger Actions
```

## 🚀 Deployment Options

### Option 1: ECS Fargate (Recommended)

```
ALB → ECS Fargate (2+ tasks) → DynamoDB
                          ↓
                    ElastiCache Redis
                          ↓
                    CloudWatch Logs
```

**Pros:**
- Auto-scaling
- High availability
- Managed infrastructure

### Option 2: AWS Lambda

```
API Gateway → Lambda → DynamoDB
                    ↓
              ElastiCache Redis
                    ↓
              CloudWatch Logs
```

**Pros:**
- Pay per use
- Auto-scaling
- No server management

### Option 3: EKS (Kubernetes)

```
Ingress → EKS Pods → DynamoDB
                  ↓
            ElastiCache Redis
                  ↓
            CloudWatch Logs
```

**Pros:**
- Full control
- Kubernetes ecosystem
- Advanced orchestration

## 📊 Observability Stack

### Metrics (Prometheus)

- HTTP metrics
- Agent metrics
- System metrics

### Tracing (OpenTelemetry)

- Request tracing
- Service dependencies
- Performance bottlenecks

### Logging (CloudWatch / ELK)

- Structured logs
- Centralized logging
- Log analysis

## 🔐 Security Architecture

### Authentication

- OAuth2 / JWT
- Token-based auth
- Role-based access control

### Rate Limiting

- Redis-based
- Per-user/IP limits
- API Gateway limits

### Network Security

- VPC isolation
- Security groups
- Encryption in transit

## 📈 Scalability

### Horizontal Scaling

- Stateless design
- Load balancing
- Auto-scaling groups

### Vertical Scaling

- Resource optimization
- Connection pooling
- Caching strategies

### Performance Optimization

- Async operations
- Connection pooling
- Caching layers
- Lazy loading

## 🔄 Resilience Patterns

### Circuit Breakers

- Protect against cascading failures
- Automatic recovery
- Fallback mechanisms

### Retries

- Exponential backoff
- Configurable retries
- Service-specific retries

### Health Checks

- Liveness probes
- Readiness probes
- Dependency checks

## 📚 Technology Stack

### Core
- **FastAPI**: Web framework
- **Python 3.11+**: Runtime
- **Uvicorn**: ASGI server

### Data
- **DynamoDB**: State storage
- **ElastiCache Redis**: Cache
- **S3**: File storage (optional)

### Workers
- **Celery**: Task queue
- **Redis**: Broker/Backend

### Message Brokers
- **RabbitMQ**: AMQP broker
- **Kafka**: Event streaming

### Observability
- **OpenTelemetry**: Tracing
- **Prometheus**: Metrics
- **CloudWatch**: Logs

### Security
- **OAuth2/JWT**: Authentication
- **bcrypt**: Password hashing
- **HTTPS**: Encryption

## 🎯 Best Practices

1. **Stateless Services**: No estado local
2. **Idempotency**: Operaciones idempotentes
3. **Error Handling**: Manejo robusto de errores
4. **Monitoring**: Observabilidad completa
5. **Security**: Defense in depth
6. **Documentation**: Documentación completa
7. **Testing**: Tests automatizados
8. **CI/CD**: Deployment automatizado

## 📖 Más Información

- [ADVANCED_FEATURES.md](ADVANCED_FEATURES.md) - Características avanzadas
- [AWS_DEPLOYMENT.md](aws/AWS_DEPLOYMENT.md) - Despliegue en AWS
- [IMPROVEMENTS_SUMMARY.md](IMPROVEMENTS_SUMMARY.md) - Resumen de mejoras




