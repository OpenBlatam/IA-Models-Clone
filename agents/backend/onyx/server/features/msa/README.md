# 🏗️ Microservices Architecture (MSA) Implementation

> Part of the [Blatam Academy Integrated Platform](../README.md)


## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Kubernetes (optional, for production)
- kubectl (optional)

### Local Development

```bash
# Start all services
docker-compose up -d

# Check service health
curl http://localhost:8001/health  # vLLM Service
curl http://localhost:8010/health  # KV Cache Service

# Stop all services
docker-compose down
```

### Service Endpoints

- **API Gateway**: http://localhost:8000
- **vLLM Service**: http://localhost:8001
- **KV Cache Service**: http://localhost:8010
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000

## 📚 Documentation

See [MSA_ARCHITECTURE.md](../MSA_ARCHITECTURE.md) for complete architecture documentation.

## 🔧 Development

### Adding a New Service

1. Create service directory in `services/`
2. Add Dockerfile
3. Add to `docker-compose.yml`
4. Update API Gateway configuration
5. Add service discovery registration

### Testing

```bash
# Run tests for a specific service
cd services/inference/vllm-service
pytest

# Run integration tests
pytest tests/integration/
```

## 📊 Monitoring

- **Prometheus**: Metrics collection
- **Grafana**: Dashboards and visualization
- **Jaeger**: Distributed tracing (coming soon)

## 🔐 Security

- TLS/SSL: Configured via API Gateway
- Authentication: JWT tokens
- Network: Isolated via Docker networks

---

*MSA Implementation v1.0.0*












