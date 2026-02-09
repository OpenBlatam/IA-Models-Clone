# Docker Deployment Guide

## Quick Start

### Using Docker Compose (Recommended)

1. **Copy the example configuration:**
   ```bash
   cp config/config.yaml.example config/config.yaml
   ```

2. **Set your environment variables:**
   ```bash
   export OPENROUTER_API_KEY="your-api-key-here"
   ```

3. **Start the service:**
   ```bash
   docker-compose up -d
   ```

4. **View logs:**
   ```bash
   docker-compose logs -f
   ```

5. **Stop the service:**
   ```bash
   docker-compose down
   ```

### Using Docker directly

1. **Build the image:**
   ```bash
   docker build -t robot-maintenance-ai .
   ```

2. **Run the container:**
   ```bash
   docker run -d \
     --name robot-maintenance-ai \
     -p 8000:8000 \
     -e OPENROUTER_API_KEY="your-api-key-here" \
     -v $(pwd)/logs:/app/logs \
     -v $(pwd)/data:/app/data \
     robot-maintenance-ai
   ```

3. **View logs:**
   ```bash
   docker logs -f robot-maintenance-ai
   ```

4. **Stop the container:**
   ```bash
   docker stop robot-maintenance-ai
   docker rm robot-maintenance-ai
   ```

## Environment Variables

- `OPENROUTER_API_KEY` (required): Your OpenRouter API key
- `LOG_LEVEL` (optional): Logging level (default: INFO)
- `LOG_FILE` (optional): Path to log file (default: /app/logs/app.log)
- `HOST` (optional): Host to bind to (default: 0.0.0.0)
- `PORT` (optional): Port to listen on (default: 8000)

## Volumes

The Docker setup mounts the following volumes:

- `./logs:/app/logs` - Application logs
- `./data:/app/data` - Data directory for models and training data

## Health Check

The container includes a health check that verifies the service is running:

```bash
docker inspect --format='{{.State.Health.Status}}' robot-maintenance-ai
```

## Production Deployment

For production, consider:

1. **Using environment files:**
   ```bash
   docker-compose --env-file .env.production up -d
   ```

2. **Setting resource limits:**
   ```yaml
   deploy:
     resources:
       limits:
         cpus: '2'
         memory: 2G
   ```

3. **Using a reverse proxy** (nginx, traefik, etc.)

4. **Setting up SSL/TLS** certificates

5. **Configuring log rotation**

6. **Setting up monitoring** (Prometheus, Grafana, etc.)

## Troubleshooting

### Container won't start

Check logs:
```bash
docker-compose logs robot-maintenance-ai
```

### Port already in use

Change the port mapping in `docker-compose.yml`:
```yaml
ports:
  - "8001:8000"  # Use port 8001 instead
```

### Missing API key

Ensure `OPENROUTER_API_KEY` is set:
```bash
echo $OPENROUTER_API_KEY
```

### Permission errors

Ensure volumes have correct permissions:
```bash
chmod -R 755 logs data
```






