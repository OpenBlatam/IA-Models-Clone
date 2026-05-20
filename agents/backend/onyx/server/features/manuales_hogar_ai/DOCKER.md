# Docker Guide - Manuales Hogar AI

This guide explains how to use Docker with the Manuales Hogar AI service.

## Quick Start

### Development Environment

Start all services (app, PostgreSQL, Redis):

```bash
docker-compose up -d
```

Access the API at: `http://localhost:8000`

### Production Environment

```bash
docker-compose -f docker-compose.prod.yml up -d
```

## Dockerfiles

### Dockerfile (Default)
Multi-stage build optimized for production. Use for AWS deployments.

### Dockerfile.dev
Development version with hot reload enabled.

### Dockerfile.prod
Production-optimized with multiple workers.

## Docker Compose Configurations

### docker-compose.yml (Development)
- **app**: FastAPI application with hot reload
- **postgres**: PostgreSQL 15 database
- **redis**: Redis cache
- **nginx**: Reverse proxy (optional, use `--profile production`)

### docker-compose.prod.yml (Production)
- **app**: Production FastAPI with multiple workers
- **redis**: Redis cache with persistence

## Usage

### Build Images

**Development:**
```bash
./scripts/docker-build.sh latest dev
```

**Production:**
```bash
./scripts/docker-build.sh latest prod
```

### Run Single Container

```bash
# Set environment variables
export OPENROUTER_API_KEY="your-key"
export DATABASE_URL="postgresql+asyncpg://user:pass@host:5432/db"

# Run
./scripts/docker-run.sh
```

### Run with Docker Compose

**Development:**
```bash
docker-compose up -d
```

**Production:**
```bash
docker-compose -f docker-compose.prod.yml up -d
```

**With Nginx:**
```bash
docker-compose --profile production up -d
```

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f app
docker-compose logs -f postgres
docker-compose logs -f redis
```

### Stop Services

```bash
# Stop but keep volumes
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

### Database Migrations

Run Alembic migrations inside the container:

```bash
# Development
docker-compose exec app alembic upgrade head

# Production
docker-compose -f docker-compose.prod.yml exec app alembic upgrade head
```

### Access Container Shell

```bash
docker-compose exec app bash
```

## Environment Variables

### Required

- `OPENROUTER_API_KEY`: OpenRouter API key

### Database (for docker-compose)

Automatically configured in docker-compose.yml:
- `DB_HOST=postgres`
- `DB_PORT=5432`
- `DB_USER=postgres`
- `DB_PASSWORD=postgres`
- `DB_NAME=manuales_hogar`

### Optional

- `ENVIRONMENT`: dev/staging/prod (default: dev)
- `WORKERS`: Number of Uvicorn workers (default: 4 for prod)
- `PORT`: Application port (default: 8000)
- `LOG_LEVEL`: Logging level (default: INFO)

## Volumes

### Development
- `./:/app`: Mounts current directory for hot reload
- `postgres_data`: PostgreSQL data persistence
- `redis_data`: Redis data persistence

### Production
- `./logs:/app/logs`: Application logs
- `redis_data_prod`: Redis data persistence

## Health Checks

All services include health checks:

```bash
# Check service health
docker-compose ps

# Manual health check
curl http://localhost:8000/api/v1/health
```

## Networking

Services communicate via Docker network `manuales-network`:

- **app**: Port 8000
- **postgres**: Port 5432
- **redis**: Port 6379
- **nginx**: Ports 80, 443

## Nginx Reverse Proxy

Nginx is included for production-like setup:

- Load balancing
- SSL termination (configure certificates)
- Rate limiting
- Request/response logging

Enable with:
```bash
docker-compose --profile production up -d
```

## Troubleshooting

### Container won't start

1. Check logs:
```bash
docker-compose logs app
```

2. Verify environment variables:
```bash
docker-compose config
```

3. Check port availability:
```bash
netstat -an | grep 8000
```

### Database connection errors

1. Wait for PostgreSQL to be healthy:
```bash
docker-compose ps postgres
```

2. Check database credentials
3. Verify network connectivity:
```bash
docker-compose exec app ping postgres
```

### Permission errors

If you see permission errors:

```bash
# Fix ownership
sudo chown -R $USER:$USER .
```

### Clean rebuild

```bash
# Remove containers, networks, and volumes
docker-compose down -v

# Rebuild images
docker-compose build --no-cache

# Start fresh
docker-compose up -d
```

## Production Deployment

### Build for Production

```bash
docker build -f Dockerfile.prod -t manuales-hogar-ai:latest .
```

### Push to Registry

```bash
# Tag for registry
docker tag manuales-hogar-ai:latest registry.example.com/manuales-hogar-ai:latest

# Push
docker push registry.example.com/manuales-hogar-ai:latest
```

### Run in Production

```bash
docker run -d \
  --name manuales-hogar-ai \
  -p 8000:8000 \
  -e OPENROUTER_API_KEY="${OPENROUTER_API_KEY}" \
  -e DATABASE_URL="${DATABASE_URL}" \
  -e ENVIRONMENT=prod \
  -e WORKERS=4 \
  manuales-hogar-ai:latest
```

## Best Practices

1. **Use .env file** for sensitive variables:
```bash
# .env
OPENROUTER_API_KEY=your-key
DB_PASSWORD=secure-password
```

2. **Use secrets** in production (Docker Swarm/Kubernetes)

3. **Monitor resources**:
```bash
docker stats
```

4. **Backup volumes**:
```bash
docker run --rm -v manuales_hogar_ai_postgres_data:/data -v $(pwd):/backup alpine tar czf /backup/postgres-backup.tar.gz /data
```

5. **Use health checks** for automatic restart

## Integration with AWS

The Docker setup integrates seamlessly with AWS:

1. Build image locally or in CI/CD
2. Push to ECR (see AWS deployment docs)
3. ECS uses the same Dockerfile.prod

See [AWS_DEPLOYMENT.md](AWS_DEPLOYMENT.md) for AWS-specific deployment.

## Scripts

Helper scripts in `scripts/`:

- `docker-build.sh`: Build Docker images
- `docker-run.sh`: Run single container
- `docker-compose-up.sh`: Start docker-compose services

## Support

For issues:
1. Check logs: `docker-compose logs`
2. Verify configuration: `docker-compose config`
3. Check health: `docker-compose ps`
4. Review [README.md](README.md) for application-specific issues




