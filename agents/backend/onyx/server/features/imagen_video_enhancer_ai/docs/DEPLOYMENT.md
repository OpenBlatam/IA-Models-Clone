# Deployment Guide - Imagen Video Enhancer AI

## Production Deployment

### Requirements

- Python 3.8+
- 4GB+ RAM recommended
- 10GB+ disk space
- Network access to OpenRouter API

### Installation

```bash
# Clone or copy the project
cd imagen_video_enhancer_ai

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENROUTER_API_KEY="your-api-key"
export TRUTHGPT_API_KEY="your-truthgpt-key"  # Optional
```

### Configuration

Create `config.json`:

```json
{
  "openrouter": {
    "api_key": "${OPENROUTER_API_KEY}",
    "base_url": "https://openrouter.ai/api/v1",
    "timeout": 30.0
  },
  "truthgpt": {
    "enabled": false
  },
  "max_file_size_mb": 100,
  "max_parallel_tasks": 5,
  "output_dir": "/var/lib/enhancer/output"
}
```

### Running as Service

#### Systemd Service

Create `/etc/systemd/system/enhancer.service`:

```ini
[Unit]
Description=Imagen Video Enhancer AI
After=network.target

[Service]
Type=simple
User=enhancer
WorkingDirectory=/opt/enhancer
Environment="OPENROUTER_API_KEY=your-key"
ExecStart=/usr/bin/python3 -m imagen_video_enhancer_ai.main
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable enhancer
sudo systemctl start enhancer
```

#### Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "imagen_video_enhancer_ai.api.enhancer_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
docker build -t enhancer .
docker run -d -p 8000:8000 \
  -e OPENROUTER_API_KEY=your-key \
  -v /data/enhancer:/app/output \
  enhancer
```

### API Server

Start API server:

```bash
python -m imagen_video_enhancer_ai.run_api
```

Or with uvicorn:

```bash
uvicorn imagen_video_enhancer_ai.api.enhancer_api:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4
```

### Monitoring

- Use `/dashboard/health` endpoint for health checks
- Monitor `/stats` endpoint for metrics
- Set up alerts for `/dashboard/health` status
- Monitor disk space for output directory
- Monitor memory usage

### Backup

Regular backups:

```bash
# Daily backup
python scripts/backup_tasks.py --compress

# List backups
python scripts/backup_tasks.py --list

# Restore backup
python scripts/backup_tasks.py --restore backup_20240101_120000 --target-dir /restore
```

### Maintenance

Cleanup old files:

```bash
# Clean cache and old files
python scripts/cleanup.py --cache --logs --days 30

# Export statistics
python scripts/export_stats.py --format json
```

### Security

1. **API Keys**: Store in environment variables or secrets manager
2. **Authentication**: Enable API key authentication for production
3. **HTTPS**: Use reverse proxy (nginx) with SSL
4. **Rate Limiting**: Configure appropriate limits
5. **File Uploads**: Validate and sanitize all uploads

### Scaling

- Run multiple API workers with uvicorn
- Use load balancer for multiple instances
- Configure shared storage for output directory
- Use external cache (Redis) for distributed deployments

### Troubleshooting

- Check logs in `output/logs/`
- Monitor `/health` endpoint
- Check disk space
- Verify API keys are valid
- Check network connectivity to OpenRouter




