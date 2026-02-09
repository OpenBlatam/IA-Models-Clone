# Architecture - Manuales Hogar AI

## 🏗️ Architecture Overview

This document describes the microservices, serverless, and cloud-native architecture of Manuales Hogar AI.

## 📐 Architecture Principles

### 1. Microservices Architecture

- **Stateless Services**: All services are stateless, leveraging external storage (PostgreSQL, Redis)
- **Service Separation**: Clear separation of concerns with dedicated services
- **API Gateway**: Ready for integration with Kong, AWS API Gateway, or NGINX
- **Event-Driven**: Support for message brokers (RabbitMQ, Kafka) for inter-service communication

### 2. Serverless & Cloud-Native

- **Containerized**: Optimized Docker images for serverless deployment
- **Auto-Scaling**: Built-in support for auto-scaling in ECS, Lambda, or Kubernetes
- **Cold Start Optimization**: Configuration flags for minimizing cold start times
- **Managed Services**: Designed to work with managed databases and caches

### 3. Resilience Patterns

- **Circuit Breaker**: Prevents cascading failures
- **Retry with Backoff**: Automatic retry with exponential backoff
- **Health Checks**: Comprehensive health checking
- **Graceful Degradation**: Fallback mechanisms for external dependencies

## 🔧 Component Architecture

### Application Layer

```
┌─────────────────────────────────────────────────┐
│              FastAPI Application                │
├─────────────────────────────────────────────────┤
│  Middleware Stack (in order):                   │
│  1. Logging Middleware                          │
│  2. Security Middleware                         │
│  3. Rate Limiting Middleware                    │
│  4. Tracing Middleware (OpenTelemetry)          │
│  5. Metrics Middleware (Prometheus)             │
│  6. CORS Middleware                             │
└─────────────────────────────────────────────────┘
```

### API Routes

```
/api/v1/
├── health          # Health checks
├── metrics         # Prometheus metrics
├── models          # AI models
├── categories      # Categories
├── generate-*      # Manual generation endpoints
├── manuals         # Manual management
├── history         # History and search
├── analytics       # Analytics
└── ...
```

### Infrastructure Layer

```
┌─────────────────────────────────────────────────┐
│           Infrastructure Components              │
├─────────────────────────────────────────────────┤
│  • Circuit Breaker                               │
│  • Retry Pattern                                 │
│  • Redis Cache                                   │
│  • Database Session Management                   │
│  • OpenRouter Client                             │
└─────────────────────────────────────────────────┘
```

## 🔄 Request Flow

```
Client Request
    ↓
[Security Middleware] → Headers, CORS
    ↓
[Rate Limiting] → Redis/Memory check
    ↓
[Logging Middleware] → Request ID, logging
    ↓
[Tracing Middleware] → OpenTelemetry span
    ↓
[Metrics Middleware] → Prometheus metrics
    ↓
[API Route Handler]
    ↓
[Circuit Breaker] → External service calls
    ↓
[Retry Logic] → Automatic retries
    ↓
[Cache Layer] → Redis cache check
    ↓
[Database/External API]
    ↓
Response
```

## 🗄️ Data Layer

### Primary Database (PostgreSQL)
- Manuals storage
- User history
- Analytics data
- Cache persistence

### Cache Layer (Redis)
- Rate limiting counters
- Frequently accessed data
- Session storage
- Temporary data

## 🔒 Security Architecture

### Security Layers

1. **Network Security**
   - VPC isolation (AWS)
   - Security groups
   - Private subnets for databases

2. **Application Security**
   - Security headers (CSP, HSTS, X-Frame-Options)
   - Rate limiting
   - Input validation
   - OAuth2 ready

3. **Data Security**
   - Secrets in AWS Secrets Manager
   - Encrypted connections (TLS)
   - Database encryption at rest

## 📊 Observability

### Logging
- Structured logging with request IDs
- JSON format for log aggregation
- CloudWatch/ELK integration ready

### Metrics
- Prometheus metrics endpoint
- Custom business metrics
- System metrics (CPU, memory, etc.)

### Tracing
- OpenTelemetry distributed tracing
- Request span tracking
- Error tracking

## 🚀 Deployment Architecture

### Development
```
Docker Compose
├── App Container (FastAPI)
├── PostgreSQL Container
└── Redis Container
```

### Production (AWS)
```
Application Load Balancer
    ↓
ECS Fargate Cluster
├── Service (Auto-scaling: 2-10 tasks)
│   └── Task Definition
│       └── Container (FastAPI)
    ↓
RDS PostgreSQL (Multi-AZ)
    ↓
ElastiCache Redis
    ↓
CloudWatch (Logs & Metrics)
    ↓
Secrets Manager
```

### Serverless (Lambda)
```
API Gateway
    ↓
Lambda Function (FastAPI)
    ↓
RDS Proxy / DynamoDB
    ↓
ElastiCache Redis
```

## 🔌 Integration Points

### External Services
- **OpenRouter API**: AI model inference
- **PostgreSQL**: Primary database
- **Redis**: Caching and rate limiting
- **CloudWatch**: Logging and monitoring
- **Secrets Manager**: Credential storage

### Future Integrations
- **Message Broker**: RabbitMQ/Kafka for events
- **API Gateway**: Kong/AWS API Gateway
- **Service Mesh**: Istio/Linkerd
- **CDN**: CloudFront for static assets

## 📈 Scalability

### Horizontal Scaling
- Stateless design enables easy horizontal scaling
- Load balancer distributes traffic
- Auto-scaling based on CPU/memory metrics

### Vertical Scaling
- Configurable resource limits
- Database connection pooling
- Redis connection pooling

### Caching Strategy
- Multi-level caching (memory + Redis)
- Cache warming strategies
- TTL-based expiration

## 🛡️ Resilience

### Failure Handling
- Circuit breaker for external services
- Automatic retries with backoff
- Graceful degradation
- Health check monitoring

### Data Consistency
- Database transactions
- Idempotent operations
- Eventual consistency where appropriate

## 🔐 Security Best Practices

1. **Authentication & Authorization**
   - OAuth2 ready
   - API key support
   - JWT token validation

2. **Input Validation**
   - Pydantic models
   - Content type validation
   - Size limits

3. **Output Sanitization**
   - XSS prevention
   - SQL injection prevention
   - CSRF protection

## 📝 Code Organization

```
manuales_hogar_ai/
├── api/              # API routes and handlers
├── core/             # Business logic
├── infrastructure/   # External integrations
├── middleware/       # Request/response middleware
├── services/         # Service layer
├── database/         # Database models and sessions
├── config/           # Configuration
└── utils/            # Utility functions
```

## 🎯 Design Patterns

- **Repository Pattern**: Database access abstraction
- **Service Layer**: Business logic separation
- **Dependency Injection**: Configuration and dependencies
- **Circuit Breaker**: Fault tolerance
- **Retry Pattern**: Resilience
- **Cache-Aside**: Caching strategy

## 📚 References

- [FastAPI Best Practices](https://fastapi.tiangolo.com/)
- [Microservices Patterns](https://microservices.io/patterns/)
- [12-Factor App](https://12factor.net/)
- [AWS Well-Architected Framework](https://aws.amazon.com/architecture/well-architected/)

---

**Last Updated**: 2024-01-XX
**Version**: 2.0.0




