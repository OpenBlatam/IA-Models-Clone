# Docker Deployment Guide

Complete guide for deploying Music Analyzer AI using Docker.

## Table of Contents

- [Quick Start](#quick-start)
- [Docker Compose Configurations](#docker-compose-configurations)
- [Building Images](#building-images)
- [Running Containers](#running-containers)
- [Production Deployment](#production-deployment)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)

## Quick Start

### Development

```bash
# Start all services (development mode)
docker-compose -f deployment/docker-compose.dev.yml up -d

# View logs
docker-compose -f deployment/docker-compose.dev.yml logs -f

# Stop services
docker-compose -f deployment/docker-compose.dev.yml down
```

### Production

```bash
# Create .env file with your configuration
cp .env.example .env
# Edit .env with your values

# Start all services (production mode)
docker-compose -f deployment/docker-compose.prod.yml up -d

# View logs
docker-compose -f deployment/docker-compose.prod.yml logs -f
```

## Docker Compose Configurations

### docker-compose.yml (Default)
Basic setup with Redis cache. Good for quick testing.

**Services:**
- `music-analyzer-ai`: Main application
- `redis`: Caching layer

### docker-compose.dev.yml (Development)
Full development environment with hot-reload and monitoring.

**Services:**
- `music-analyzer-ai`: Application with hot-reload
- `redis`: Caching
- `postgres`: Database (optional)
- `nginx`: Reverse proxy
- `prometheus`: Metrics collection
- `grafana`: Metrics visualization

**Features:**
- Hot-reload enabled
- Volume mounts for live code changes
- Debug logging
- Development-friendly settings

### docker-compose.prod.yml (Production)
Production-ready setup with all optimizations.

**Services:**
- `music-analyzer-ai`: Application (multi-worker)
- `redis`: Caching with persistence
- `postgres`: Production database
- `nginx`: Reverse proxy with SSL
- `prometheus`: Metrics with retention
- `grafana`: Production dashboards

**Features:**
- Resource limits
- Health checks
- SSL/TLS support
- Production logging
- Monitoring stack

## Building Images

### Basic Build

```bash
docker build -f deployment/Dockerfile -t music-analyzer-ai:latest ..
```

### Using Build Script

```bash
# Development build
./deployment/scripts/docker_build.sh development latest

# Production build
./deployment/scripts/docker_build.sh production v2.21.0
```

### Build Arguments

```bash
docker build \
  -f deployment/Dockerfile \
  -t music-analyzer-ai:latest \
  --build-arg ENVIRONMENT=production \
  --build-arg BUILDKIT_INLINE_CACHE=1 \
  ..
```

### Multi-Architecture Build

```bash
docker buildx create --use
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -f deployment/Dockerfile \
  -t music-analyzer-ai:latest \
  --push \
  ..
```

## Running Containers

### Using Docker Compose (Recommended)

```bash
# Development
docker-compose -f deployment/docker-compose.dev.yml up -d

# Production
docker-compose -f deployment/docker-compose.prod.yml up -d

# With custom env file
docker-compose -f deployment/docker-compose.prod.yml --env-file .env.prod up -d
```

### Using Run Script

```bash
# Development
./deployment/scripts/docker_run.sh development

# Production
./deployment/scripts/docker_run.sh production
```

### Manual Docker Run

```bash
docker run -d \
  --name music-analyzer-ai \
  -p 8010:8010 \
  -e SPOTIFY_CLIENT_ID=your_id \
  -e SPOTIFY_CLIENT_SECRET=your_secret \
  -e ENVIRONMENT=production \
  -v $(pwd)/data:/app/data \
  music-analyzer-ai:latest
```

## Production Deployment

### Prerequisites

1. **Environment Variables**
   Create `.env` file:
   ```env
   ENVIRONMENT=production
   SPOTIFY_CLIENT_ID=your_client_id
   SPOTIFY_CLIENT_SECRET=your_client_secret
   POSTGRES_PASSWORD=strong_password
   REDIS_PASSWORD=strong_password
   GRAFANA_PASSWORD=admin_password
   DATABASE_URL=postgresql://music_analyzer:password@postgres:5432/music_analyzer_db
   ```

2. **SSL Certificates** (for HTTPS)
   ```bash
   mkdir -p deployment/nginx/ssl
   # Place your cert.pem and key.pem files here
   ```

### Deployment Steps

1. **Build Production Image**
   ```bash
   docker build -f deployment/Dockerfile -t music-analyzer-ai:latest ..
   ```

2. **Start Services**
   ```bash
   docker-compose -f deployment/docker-compose.prod.yml up -d
   ```

3. **Verify Health**
   ```bash
   curl http://localhost/health
   ```

4. **Check Logs**
   ```bash
   docker-compose -f deployment/docker-compose.prod.yml logs -f music-analyzer-ai
   ```

### Scaling

```bash
# Scale application containers
docker-compose -f deployment/docker-compose.prod.yml up -d --scale music-analyzer-ai=3

# Use with load balancer (nginx handles this automatically)
```

## Monitoring

### Prometheus

Access at: `http://localhost:9090`

**Metrics Endpoints:**
- Application: `http://music-analyzer-ai:8010/metrics`
- Redis: `redis:6379` (if exporter installed)
- PostgreSQL: `postgres:5432` (if exporter installed)

### Grafana

Access at: `http://localhost:3000`

**Default Credentials:**
- Username: `admin`
- Password: `admin` (change in production!)

**Pre-configured:**
- Prometheus datasource
- Sample dashboards

### Health Checks

```bash
# Basic health
curl http://localhost:8010/health

# Detailed health
curl http://localhost:8010/health/detailed

# Readiness
curl http://localhost:8010/health/ready

# Liveness
curl http://localhost:8010/health/live
```

## Docker Swarm (Optional)

### Initialize Swarm

```bash
docker swarm init
```

### Deploy Stack

```bash
docker stack deploy -c deployment/docker-compose.prod.yml music-analyzer
```

### Scale Services

```bash
docker service scale music-analyzer_music-analyzer-ai=3
```

## Kubernetes (Optional)

### Create Deployment

```bash
kubectl create -f deployment/kubernetes/
```

### Apply ConfigMap

```bash
kubectl create configmap music-analyzer-config --from-env-file=.env
```

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker logs music-analyzer-ai

# Check container status
docker ps -a

# Inspect container
docker inspect music-analyzer-ai
```

### Port Already in Use

```bash
# Find process using port
lsof -i :8010

# Change port in docker-compose.yml
ports:
  - "8011:8010"  # Use different host port
```

### Out of Memory

```bash
# Increase memory limits in docker-compose.prod.yml
deploy:
  resources:
    limits:
      memory: 8G  # Increase as needed
```

### Database Connection Issues

```bash
# Check postgres container
docker logs music-analyzer-postgres-prod

# Test connection
docker exec -it music-analyzer-postgres-prod psql -U music_analyzer -d music_analyzer_db
```

### Redis Connection Issues

```bash
# Check redis container
docker logs music-analyzer-redis-prod

# Test connection
docker exec -it music-analyzer-redis-prod redis-cli ping
```

### Build Failures

```bash
# Clean build (no cache)
docker build --no-cache -f deployment/Dockerfile -t music-analyzer-ai:latest ..

# Check build context
docker build --progress=plain -f deployment/Dockerfile -t music-analyzer-ai:latest ..
```

## Best Practices

1. **Use .dockerignore**: Reduces build context size
2. **Multi-stage builds**: Smaller final images
3. **Health checks**: Enable automatic recovery
4. **Resource limits**: Prevent resource exhaustion
5. **Secrets management**: Use Docker secrets or external vaults
6. **Logging**: Centralize logs with ELK or similar
7. **Backups**: Regular database and data backups
8. **Updates**: Keep base images updated
9. **Security**: Run as non-root user
10. **Monitoring**: Always enable monitoring in production

## Security Considerations

1. **Non-root user**: Container runs as `appuser`
2. **Secrets**: Never commit secrets, use environment variables or secrets management
3. **Network isolation**: Use Docker networks
4. **Image scanning**: Regularly scan images for vulnerabilities
5. **Updates**: Keep dependencies updated
6. **SSL/TLS**: Always use HTTPS in production
7. **Firewall**: Restrict access to necessary ports only

## Performance Optimization

1. **Layer caching**: Order Dockerfile commands for best cache hits
2. **Multi-stage builds**: Reduce final image size
3. **Resource limits**: Set appropriate CPU/memory limits
4. **Connection pooling**: Configure for your workload
5. **Caching**: Use Redis for frequently accessed data
6. **CDN**: Serve static assets via CDN
7. **Load balancing**: Distribute traffic across instances

## Support

For issues or questions:
- Check logs: `docker-compose logs -f`
- Review health endpoints
- Check monitoring dashboards
- Consult Docker documentation




