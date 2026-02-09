# Deployment Guide - Universal Model Benchmark AI

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# Build and run with Docker Compose
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Option 2: Manual Setup

```bash
# Run setup script
chmod +x scripts/setup.sh
./scripts/setup.sh

# Activate virtual environment
source venv/bin/activate

# Start API server
python -m api.rest_api
```

## 📦 Installation

### Prerequisites

- Python 3.10+
- Rust 1.70+
- Docker (optional)
- CUDA (for GPU support, optional)

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd universal_model_benchmark_ai
```

### Step 2: Setup Environment

```bash
# Run setup script
./scripts/setup.sh

# Or manually:
python3 -m venv venv
source venv/bin/activate
pip install -r python/requirements.txt
cd rust && cargo build --release && cd ..
```

### Step 3: Configure

Create `.env` file:

```env
DATABASE_URL=sqlite:///data/results.db
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
RUST_BACKTRACE=1
```

### Step 4: Start Services

```bash
# Start API server
python -m api.rest_api

# Or use CLI
python -m cli.main run --model llama2-7b --benchmark mmlu
```

## 🐳 Docker Deployment

### Build Image

```bash
docker build -t universal-model-benchmark:latest .
```

### Run Container

```bash
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/results:/app/results \
  universal-model-benchmark:latest
```

### Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

## ☁️ Cloud Deployment

### AWS

```bash
# Build and push to ECR
aws ecr get-login-password | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
docker build -t universal-model-benchmark .
docker tag universal-model-benchmark:latest <account>.dkr.ecr.<region>.amazonaws.com/universal-model-benchmark:latest
docker push <account>.dkr.ecr.<region>.amazonaws.com/universal-model-benchmark:latest
```

### Google Cloud

```bash
# Build and push to GCR
gcloud builds submit --tag gcr.io/<project>/universal-model-benchmark
```

### Azure

```bash
# Build and push to ACR
az acr build --registry <registry> --image universal-model-benchmark .
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | Database connection string | `sqlite:///data/results.db` |
| `API_HOST` | API host | `0.0.0.0` |
| `API_PORT` | API port | `8000` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `RUST_BACKTRACE` | Rust backtrace | `1` |

### API Configuration

Edit `python/api/rest_api.py` to customize:
- CORS settings
- Authentication
- Rate limiting
- Middleware

## 📊 Monitoring

### Health Checks

```bash
# Check API health
curl http://localhost:8000/health

# Get statistics
curl http://localhost:8000/api/v1/statistics
```

### Logs

```bash
# View logs
tail -f logs/app.log

# Docker logs
docker-compose logs -f
```

## 🔐 Security

### Production Checklist

- [ ] Change default admin credentials
- [ ] Enable HTTPS
- [ ] Configure CORS properly
- [ ] Use environment variables for secrets
- [ ] Enable authentication
- [ ] Set up firewall rules
- [ ] Regular security updates

### Authentication

```python
from core.auth import AuthManager

manager = AuthManager()
user = manager.create_user("admin", "admin@example.com", role=UserRole.ADMIN)
token = manager.generate_token(user)
```

## 📈 Scaling

### Horizontal Scaling

```yaml
# docker-compose.yml
services:
  api:
    deploy:
      replicas: 3
  worker:
    deploy:
      replicas: 5
```

### Load Balancing

Use nginx or similar:

```nginx
upstream api {
    server api1:8000;
    server api2:8000;
    server api3:8000;
}

server {
    listen 80;
    location / {
        proxy_pass http://api;
    }
}
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=core --cov-report=html

# Specific test
pytest tests/test_core.py::TestResultsManager -v
```

## 📝 CLI Usage

```bash
# Run benchmark
python -m cli.main run --model llama2-7b --benchmark mmlu

# List results
python -m cli.main results list

# Compare results
python -m cli.main results compare mmlu

# Export results
python -m cli.main results export results.json --format json

# Manage experiments
python -m cli.main experiments create --name test --model llama2-7b --benchmark mmlu

# View costs
python -m cli.main costs show

# System statistics
python -m cli.main stats
```

## 🐛 Troubleshooting

### Common Issues

1. **Rust build fails**
   ```bash
   rustup update
   cargo clean
   cargo build --release
   ```

2. **Python imports fail**
   ```bash
   export PYTHONPATH=/app/python
   ```

3. **Database errors**
   ```bash
   mkdir -p data
   chmod 755 data
   ```

4. **Port already in use**
   ```bash
   # Change port in .env
   API_PORT=8001
   ```

## 📚 Additional Resources

- [API Documentation](http://localhost:8000/docs)
- [CLI Help](python -m cli.main --help)
- [README](README.md)












