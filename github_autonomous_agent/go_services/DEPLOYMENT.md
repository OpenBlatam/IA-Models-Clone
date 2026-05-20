# 🚀 Deployment Guide - Go Services

## Local Development

### Prerequisites

- Go 1.22 or later
- Git

### Quick Start

```bash
# Clone and navigate
cd go_services

# Download dependencies
go mod download

# Build
go build ./cmd/agent

# Run
./agent-service --port 8080
```

### Using Make

```bash
make build    # Build binary
make run      # Run service
make test     # Run tests
make clean    # Clean artifacts
```

### Using Scripts

**Linux/Mac:**
```bash
chmod +x scripts/build.sh scripts/test.sh
./scripts/build.sh
./scripts/test.sh
```

**Windows:**
```powershell
.\build.ps1
```

## Docker Deployment

### Build Image

```bash
docker build -t github-autonomous-agent-go:latest .
```

### Run Container

```bash
docker run -p 8080:8080 \
  -e PORT=8080 \
  -e LOG_LEVEL=info \
  github-autonomous-agent-go:latest
```

### Using Docker Compose

Add to your `docker-compose.yml`:

```yaml
services:
  go-services:
    build:
      context: ./go_services
      dockerfile: Dockerfile
    ports:
      - "8080:8080"
    environment:
      - PORT=8080
      - LOG_LEVEL=info
      - CACHE_BADGER_PATH=/data/cache
      - SEARCH_INDEX_PATH=/data/search
    volumes:
      - go-cache:/data/cache
      - go-search:/data/search
    healthcheck:
      test: ["CMD", "wget", "--spider", "http://localhost:8080/health"]
      interval: 30s
      timeout: 3s
      retries: 3

volumes:
  go-cache:
  go-search:
```

## Production Deployment

### Environment Variables

```bash
PORT=8080
LOG_LEVEL=info
CACHE_MEMORY_SIZE=10000
CACHE_MEMORY_TTL=5m
CACHE_BADGER_PATH=/var/lib/agent/cache
CACHE_ENABLE_BADGER=true
CACHE_ENABLE_REDIS=false
REDIS_URL=redis://localhost:6379
SEARCH_INDEX_PATH=/var/lib/agent/search
```

### Systemd Service

Create `/etc/systemd/system/go-services.service`:

```ini
[Unit]
Description=GitHub Autonomous Agent Go Services
After=network.target

[Service]
Type=simple
User=agent
WorkingDirectory=/opt/agent/go_services
ExecStart=/opt/agent/go_services/agent-service --port 8080
Restart=always
RestartSec=5
Environment="LOG_LEVEL=info"
Environment="CACHE_BADGER_PATH=/var/lib/agent/cache"
Environment="SEARCH_INDEX_PATH=/var/lib/agent/search"

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable go-services
sudo systemctl start go-services
sudo systemctl status go-services
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: go-services
spec:
  replicas: 3
  selector:
    matchLabels:
      app: go-services
  template:
    metadata:
      labels:
        app: go-services
    spec:
      containers:
      - name: agent-service
        image: github-autonomous-agent-go:latest
        ports:
        - containerPort: 8080
        env:
        - name: PORT
          value: "8080"
        - name: LOG_LEVEL
          value: "info"
        resources:
          requests:
            memory: "128Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: go-services
spec:
  selector:
    app: go-services
  ports:
  - port: 8080
    targetPort: 8080
  type: LoadBalancer
```

## Monitoring

### Health Check

```bash
curl http://localhost:8080/health
```

Expected response:
```json
{"status":"healthy","service":"go-services"}
```

### Metrics (Future)

Prometheus metrics endpoint (to be implemented):
```bash
curl http://localhost:8080/metrics
```

## Troubleshooting

### Service won't start

```bash
# Check logs
journalctl -u go-services -f

# Check port
netstat -tulpn | grep 8080

# Test binary directly
./agent-service --port 8080 --log-level=debug
```

### High memory usage

- Reduce `CACHE_MEMORY_SIZE`
- Enable Redis for distributed caching
- Monitor BadgerDB disk usage

### Performance issues

- Enable Redis for distributed caching
- Increase worker pool size
- Monitor with profiling tools












