# Security Toolkit - Comprehensive Cybersecurity Tooling

A modular, secure, and high-performance Python toolkit for cybersecurity operations implementing OWASP, NIST, and FastAPI best practices.

## Features

### 🔒 Security Features
- **Authentication & Authorization**: JWT-based token validation
- **Rate Limiting**: Configurable per-client rate limiting
- **Input Validation**: Pydantic models with strict validation
- **Secure Headers**: CORS, CSP, HSTS, and other security headers
- **Structured Logging**: JSON logging for SIEM integration
- **Secret Management**: Environment variable and cloud secret store support

### 🌐 Network Operations
- **Port Scanning**: TCP/UDP port scanning with nmap integration
- **SSH Operations**: Secure SSH command execution
- **HTTP Operations**: Async HTTP requests with SSL verification
- **Network Layer Abstraction**: Protocol-independent network operations

### ⚡ Performance Features
- **Async I/O**: Full async support for high concurrency
- **Connection Pooling**: HTTP and SSH connection reuse
- **Caching**: DNS resolution and vulnerability database caching
- **Batch Processing**: Efficient processing of large target lists
- **Rate Limiting**: Configurable back-off and retry mechanisms

### 📊 Monitoring & Metrics
- **Structured Logging**: JSON format for easy SIEM ingestion
- **Performance Metrics**: Scan completion time, success rates
- **Health Checks**: API health and security status endpoints
- **Request Tracking**: Request ID tracking for debugging

## Installation

```bash
pip install -r requirements_security.txt
```

## Quick Start

### Basic Port Scanning

```python
from security_toolkit import scan_ports_basic

result = scan_ports_basic({
    target:192.16811,
    ports": [80443221,
   scan_type": "tcp,
   timeout": 5,
    max_workers": 10})

print(result)
```

### SSH Command Execution

```python
from security_toolkit import run_ssh_command

result = await run_ssh_command([object Object]
    host:192.1680.1,
    username": "admin,password:password",
    command: ls -la,
    timeout": 30})

print(result)
```

### HTTP Request

```python
from security_toolkit import make_http_request

result = await make_http_request({
  url": "https://api.example.com,   method": "GET",headers:[object Object]Authorization":Bearer token},
    timeout: 30
    verify_ssl": True
})

print(result)
```

## API Usage

### Start the API Server

```bash
python security_api.py
```

### API Endpoints

#### Health Check
```bash
curl -X GET http://localhost:8000/health"
```

#### Port Scan
```bash
curl -X POST http://localhost:800/scan/ports" \
  -H "Authorization: Bearer valid_token" \
  -H "Content-Type: application/json" \
  -d {    target:192.16811,
    ports:[80432,
   scan_type": "tcp,
   timeout": 5,
    max_workers":10  }'
```

#### SSH Command
```bash
curl -X POST http://localhost:8000ssh/execute" \
  -H "Authorization: Bearer valid_token" \
  -H "Content-Type: application/json" \
  -d '[object Object]
    host:192.1680.1,
    username": "admin,password:password",
    command: ls -la,
   timeout":30 }'
```

#### HTTP Request
```bash
curl -X POST http://localhost:8000/http/request" \
  -H "Authorization: Bearer valid_token" \
  -H "Content-Type: application/json" \
  -d '{
  url": "https://api.example.com,   method": "GET",
   headers": {"User-Agent": SecurityToolkit/10},
    timeout: 30
    verify_ssl: true
  }
```

## Advanced Features

### Rate Limiting

```python
from security_toolkit import AsyncRateLimiter

limiter = AsyncRateLimiter(max_calls_per_second=10)

async def rate_limited_operation():
    await limiter.acquire()
    # Your operation here
    returnresult"
```

### Retry with Back-off

```python
from security_toolkit import retry_with_backoff

async def unreliable_operation():
    # Operation that might fail
    if random.random() < 0.5   raise Exception(Simulated failure")
    returnsuccess"

result = await retry_with_backoff(unreliable_operation, max_retries=3)
```

### Batch Processing

```python
from security_toolkit import process_batch_async

targets = ["192168.11, "192168.1.2, "192.168.1
async def process_target(target):
    return await scan_ports_basic({"target": target, ports": [80443)

results = await process_batch_async(targets, process_target, batch_size=10, max_concurrent=5)
```

### Network Layer Abstraction

```python
from security_toolkit import NetworkLayerFactory

# HTTP layer
http_layer = NetworkLayerFactory.create_layer(http)
await http_layer.connect({"timeout:30verify_ssl": True})

result = await http_layer.send({
   method:GET",
  url": "https://api.example.com",headers:[object Object]Authorization": "Bearer token"}
})

await http_layer.close()
```

## Configuration

### Environment Variables

```bash
# API Configuration
API_KEY=your_api_key
SECRET_KEY=your_secret_key
ALLOWED_HOSTS=yourdomain.com

# Database Configuration
DATABASE_URL=postgresql://user:pass@localhost/db

# Redis Configuration
REDIS_URL=redis://localhost:6379

# Logging Configuration
LOG_LEVEL=INFO
LOG_FORMAT=json
```

### Security Configuration

```python
# CORS settings
ALLOWED_ORIGINS = ["https://yourdomain.com]
ALLOWED_METHODS = [GET, "POST]
ALLOWED_HEADERS = ["*"]

# Rate limiting
RATE_LIMIT_PER_MINUTE = 100ATE_LIMIT_PER_HOUR = 1000

# SSL/TLS settings
SSL_VERIFY = True
SSL_CERT_FILE = cert.pem
SSL_KEY_FILE =key.pem"
```

## Best Practices

### Security
1. **Always validate input** using Pydantic models
2. **Use HTTPS** for all API communications
3. **Implement proper authentication** and authorization4**Rate limit** all endpoints to prevent abuse5og security events** for monitoring and auditing
6. **Never log secrets** or sensitive data
7. **Use secure defaults** for all configurations

### Performance
1. **Use async operations** for I/O-bound tasks
2. **Implement connection pooling** for network operations
3. **Cache expensive operations** like DNS lookups
4 **Batch process** large datasets
5. **Set appropriate timeouts** for all operations
6. **Monitor and log** performance metrics

### Code Quality
1. **Use type hints** for all functions
2. **Follow RORO pattern** (Receive Object, Return Object)
3. **Implement proper error handling** with custom exceptions
4. **Write comprehensive tests** for all modules
5. **Use structured logging** for debugging and monitoring6ocument all public APIs** with docstrings

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=security_toolkit

# Run specific test file
pytest test_security_toolkit.py

# Run async tests
pytest test_async_operations.py
```

## Deployment

### Docker

```dockerfile
FROM python:311slim

WORKDIR /app
COPY requirements_security.txt .
RUN pip install -r requirements_security.txt

COPY . .
EXPOSE 800
CMD ["uvicorn",security_api:app, --host", 0000,--port", "8000"]
```

### Production Checklist

- [ ] Configure HTTPS with valid certificates
-per authentication and authorization
- gure rate limiting and monitoring
- [ ] Set up structured logging and SIEM integration
- [ ] Configure backup and disaster recovery
- [ ] Set up monitoring and alerting
- [ ] Perform security audit and penetration testing
- [ ] Document deployment procedures

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Create an issue in the repository
- Check the documentation
- Review the demo scripts for examples 