# Docker Deployment Guide

This guide covers Docker deployment for the 3D Prototype AI application, following DevOps best practices.

## 📋 Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Dockerfiles](#dockerfiles)
- [Docker Compose](#docker-compose)
- [Development](#development)
- [Production](#production)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

The project includes multiple Docker configurations optimized for different environments:

- **Dockerfile.production**: Multi-stage build for production
- **Dockerfile.development**: Development image with hot-reload
- **docker-compose.production.yml**: Full production stack
- **docker-compose.development.yml**: Development stack

## 🚀 Quick Start

### Development

```bash
# Build and start development environment
docker-compose -f docker-compose.development.yml up --build

# Or use the utility script
./cloud/scripts/docker_utils.sh build -e development
./cloud/scripts/docker_utils.sh up -e development
```

### Production

```bash
# Build production images
docker-compose -f docker-compose.production.yml build

# Start production stack
docker-compose -f docker-compose.production.yml up -d

# Or use the utility script
./cloud/scripts/docker_utils.sh build -e production
./cloud/scripts/docker_utils.sh up -e production
```

## 🐳 Dockerfiles

### Production Dockerfile

**Features:**
- Multi-stage build for smaller image size
- Non-root user for security
- Optimized layer caching
- Health checks
- Minimal runtime dependencies

**Build:**
```bash
docker build -f Dockerfile.production -t 3d-prototype-ai:latest .
```

### Development Dockerfile

**Features:**
- Hot-reload support
- Development tools included
- Debug mode enabled
- Volume mounting for live code updates

**Build:**
```bash
docker build -f Dockerfile.development -t 3d-prototype-ai:dev .
```

## 🎛️ Docker Compose

### Production Stack

Includes:
- API service (with multiple workers)
- Redis (with persistence)
- RabbitMQ (with management UI)
- Celery Worker
- Celery Beat
- Prometheus
- Grafana

**Start:**
```bash
docker-compose -f docker-compose.production.yml up -d
```

**Stop:**
```bash
docker-compose -f docker-compose.production.yml down
```

### Development Stack

Includes:
- API service (with hot-reload)
- Redis

**Start:**
```bash
docker-compose -f docker-compose.development.yml up
```

## 🛠️ Development

### Using Docker Compose

```bash
# Start services
docker-compose -f docker-compose.development.yml up

# View logs
docker-compose -f docker-compose.development.yml logs -f api

# Execute commands
docker-compose -f docker-compose.development.yml exec api python manage.py migrate

# Stop services
docker-compose -f docker-compose.development.yml down
```

### Using Utility Scripts

```bash
# Build
./cloud/scripts/docker_utils.sh build -e development

# Start
./cloud/scripts/docker_utils.sh up -e development

# View logs
./cloud/scripts/docker_utils.sh logs -s api

# Open shell
./cloud/scripts/docker_utils.sh shell -s api

# Check health
./cloud/scripts/docker_utils.sh health

# View statistics
./cloud/scripts/docker_utils.sh stats
```

## 🏭 Production

### Environment Variables

Create a `.env` file:

```env
# Application
APP_PORT=8030
DEBUG=false

# Redis
REDIS_PASSWORD=your_secure_password

# RabbitMQ
RABBITMQ_PASSWORD=your_secure_password

# Grafana
GRAFANA_PASSWORD=your_secure_password

# Version
VERSION=1.0.0
```

### Deployment Steps

1. **Build images:**
   ```bash
   docker-compose -f docker-compose.production.yml build
   ```

2. **Start services:**
   ```bash
   docker-compose -f docker-compose.production.yml up -d
   ```

3. **Verify health:**
   ```bash
   ./cloud/scripts/docker_utils.sh health
   ```

4. **View logs:**
   ```bash
   ./cloud/scripts/docker_utils.sh logs
   ```

### Resource Limits

Production compose file includes resource limits:
- API: 2 CPU, 2GB RAM
- Redis: 0.5 CPU, 512MB RAM
- RabbitMQ: 0.5 CPU, 512MB RAM
- Celery Worker: 1 CPU, 1GB RAM

Adjust in `docker-compose.production.yml` as needed.

## ✅ Best Practices

### Security

1. **Non-root user**: All containers run as non-root
2. **Secrets management**: Use environment variables or secrets
3. **Image scanning**: Regularly scan images for vulnerabilities
4. **Minimal base images**: Use slim/alpine variants
5. **No secrets in images**: Never commit secrets to images

### Performance

1. **Multi-stage builds**: Reduce final image size
2. **Layer caching**: Optimize Dockerfile layer order
3. **Resource limits**: Set appropriate CPU/memory limits
4. **Health checks**: Monitor container health
5. **Logging**: Configure log rotation

### Maintainability

1. **.dockerignore**: Exclude unnecessary files
2. **Labels**: Add metadata labels
3. **Documentation**: Document Dockerfile steps
4. **Versioning**: Tag images with versions
5. **CI/CD**: Automate builds and deployments

## 🔧 Troubleshooting

### Container won't start

```bash
# Check logs
docker-compose logs api

# Check container status
docker-compose ps

# Inspect container
docker inspect 3d-prototype-ai-api
```

### Health check failing

```bash
# Test health endpoint manually
docker-compose exec api curl http://localhost:8030/health

# Check application logs
docker-compose logs api
```

### Out of memory

```bash
# Check resource usage
docker stats

# Adjust resource limits in docker-compose.yml
# Or increase Docker memory limit
```

### Port conflicts

```bash
# Check what's using the port
lsof -i :8030

# Change port in docker-compose.yml
ports:
  - "8031:8030"  # Host:Container
```

### Volume permissions

```bash
# Fix volume permissions
docker-compose exec api chown -R appuser:appuser /app/storage
```

## 📊 Monitoring

### Prometheus

Access at: `http://localhost:9090`

### Grafana

Access at: `http://localhost:3000`
- Default username: `admin`
- Default password: `admin` (change in production)

### Container Stats

```bash
# Real-time stats
docker stats

# Or use utility script
./cloud/scripts/docker_utils.sh stats
```

## 🔄 Updates

### Update Application

```bash
# Pull latest code
git pull

# Rebuild and restart
docker-compose -f docker-compose.production.yml up -d --build
```

### Update Dependencies

```bash
# Update requirements.txt
pip freeze > requirements.txt

# Rebuild image
docker-compose -f docker-compose.production.yml build --no-cache api
```

## 📚 Additional Resources

- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Multi-stage Builds](https://docs.docker.com/build/building/multi-stage/)

---

**Last Updated**: 2024-01-XX
**Version**: 2.0.0

