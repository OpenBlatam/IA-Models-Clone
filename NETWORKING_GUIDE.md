# 🌐 Guía de Networking - Blatam Academy Features

## 🔌 Configuración de Red

### Docker Network Setup

```yaml
# docker-compose.yml
version: '3.8'

services:
  bul:
    image: blatam-academy:latest
    networks:
      - internal
      - external
    ports:
      - "8002:8002"

  postgres:
    image: postgres:14
    networks:
      - internal

  redis:
    image: redis:7
    networks:
      - internal

networks:
  internal:
    driver: bridge
    internal: true  # Sin acceso externo
  external:
    driver: bridge  # Con acceso externo
```

### Network Policies

```yaml
# network-policy.yml (Kubernetes)
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: bul-network-policy
spec:
  podSelector:
    matchLabels:
      app: bul
  policyTypes:
    - Ingress
    - Egress
  ingress:
    - from:
      - podSelector:
          matchLabels:
            app: api-gateway
      ports:
        - protocol: TCP
          port: 8002
  egress:
    - to:
      - podSelector:
          matchLabels:
            app: postgres
      ports:
        - protocol: TCP
          port: 5432
    - to:
      - podSelector:
          matchLabels:
            app: redis
      ports:
        - protocol: TCP
          port: 6379
```

## 📡 Comunicación entre Servicios

### REST API Client

```python
import httpx
from typing import Optional

class ServiceClient:
    """Cliente para comunicación REST."""
    
    def __init__(self, base_url: str, timeout: int = 30):
        self.client = httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout,
            limits=httpx.Limits(max_keepalive_connections=10)
        )
    
    async def request(self, method: str, endpoint: str, **kwargs):
        """Realizar request con retry."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await self.client.request(method, endpoint, **kwargs)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    async def close(self):
        """Cerrar cliente."""
        await self.client.aclose()

# Uso
bul_client = ServiceClient('http://bul:8002')
result = await bul_client.request('POST', '/api/query', json=data)
```

### gRPC Client

```python
import grpc
from proto import bul_pb2, bul_pb2_grpc

class GRPCClient:
    """Cliente gRPC."""
    
    def __init__(self, server_address: str):
        self.channel = grpc.aio.insecure_channel(server_address)
        self.stub = bul_pb2_grpc.BulServiceStub(self.channel)
    
    async def query(self, request_data):
        """Enviar query."""
        request = bul_pb2.QueryRequest(data=request_data)
        response = await self.stub.ProcessQuery(request)
        return response.result
    
    async def close(self):
        """Cerrar canal."""
        await self.channel.close()

# Uso
client = GRPCClient('bul:50051')
result = await client.query(data)
```

### WebSocket Client

```python
import websockets
import json

class WebSocketClient:
    """Cliente WebSocket."""
    
    def __init__(self, uri: str):
        self.uri = uri
        self.websocket = None
    
    async def connect(self):
        """Conectar."""
        self.websocket = await websockets.connect(self.uri)
    
    async def send(self, data):
        """Enviar datos."""
        await self.websocket.send(json.dumps(data))
        response = await self.websocket.recv()
        return json.loads(response)
    
    async def close(self):
        """Cerrar conexión."""
        if self.websocket:
            await self.websocket.close()

# Uso
client = WebSocketClient('ws://bul:8002/ws')
await client.connect()
result = await client.send({'query': 'test'})
```

## 🔒 Security en Networking

### TLS/SSL Configuration

```python
import ssl
import httpx

# Configurar SSL
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_REQUIRED
ssl_context.load_verify_locations('ca-cert.pem')

client = httpx.AsyncClient(verify=ssl_context)
```

### Authentication Headers

```python
class AuthenticatedClient:
    """Cliente con autenticación."""
    
    def __init__(self, base_url: str, api_key: str):
        self.client = httpx.AsyncClient(
            base_url=base_url,
            headers={
                'Authorization': f'Bearer {api_key}',
                'X-API-Key': api_key
            }
        )
    
    async def request(self, method: str, endpoint: str, **kwargs):
        """Request autenticado."""
        response = await self.client.request(method, endpoint, **kwargs)
        response.raise_for_status()
        return response.json()
```

## 📊 Load Balancing

### Nginx Load Balancer

```nginx
# nginx.conf
upstream bul_backend {
    least_conn;  # Load balancing method
    server bul1:8002 weight=3;
    server bul2:8002 weight=2;
    server bul3:8002 weight=1 backup;
    
    keepalive 32;
}

server {
    listen 80;
    
    location / {
        proxy_pass http://bul_backend;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}
```

### Client-side Load Balancing

```python
import random
from typing import List

class LoadBalancer:
    """Load balancer client-side."""
    
    def __init__(self, servers: List[str]):
        self.servers = servers
    
    def get_server(self, strategy: str = 'round_robin'):
        """Obtener servidor según estrategia."""
        if strategy == 'round_robin':
            return self._round_robin()
        elif strategy == 'random':
            return self._random()
        elif strategy == 'least_connections':
            return self._least_connections()
    
    def _round_robin(self):
        """Round robin."""
        server = self.servers[0]
        self.servers = self.servers[1:] + [self.servers[0]]
        return server
    
    def _random(self):
        """Random selection."""
        return random.choice(self.servers)
```

## 🔍 Network Monitoring

### Health Check Endpoint

```python
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    # Verificar conexiones
    redis_healthy = await check_redis()
    postgres_healthy = await check_postgres()
    
    if redis_healthy and postgres_healthy:
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"status": "healthy"}
        )
    else:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unhealthy"}
        )

async def check_redis():
    """Verificar Redis."""
    try:
        await redis_client.ping()
        return True
    except:
        return False

async def check_postgres():
    """Verificar PostgreSQL."""
    try:
        await db.execute("SELECT 1")
        return True
    except:
        return False
```

### Network Latency Monitoring

```python
import time
import statistics

class LatencyMonitor:
    """Monitor de latencia de red."""
    
    def __init__(self):
        self.latencies = []
    
    async def measure_latency(self, url: str):
        """Medir latencia a URL."""
        start = time.perf_counter()
        try:
            async with httpx.AsyncClient() as client:
                await client.get(url, timeout=5.0)
            latency = (time.perf_counter() - start) * 1000  # ms
            self.latencies.append(latency)
            return latency
        except:
            return None
    
    def get_stats(self):
        """Obtener estadísticas."""
        if not self.latencies:
            return None
        
        return {
            'mean': statistics.mean(self.latencies),
            'median': statistics.median(self.latencies),
            'p95': statistics.quantiles(self.latencies, n=20)[18],
            'p99': statistics.quantiles(self.latencies, n=100)[98]
        }

# Uso
monitor = LatencyMonitor()
latency = await monitor.measure_latency('http://bul:8002/health')
stats = monitor.get_stats()
```

## ✅ Checklist de Networking

### Configuración Inicial
- [ ] Redes Docker configuradas
- [ ] Network policies definidas
- [ ] Load balancing configurado
- [ ] TLS/SSL configurado

### Monitoreo
- [ ] Health checks implementados
- [ ] Latency monitoring activo
- [ ] Connection pooling configurado
- [ ] Timeouts apropiados

### Seguridad
- [ ] Autenticación configurada
- [ ] Rate limiting activo
- [ ] Firewall rules configuradas
- [ ] Network isolation implementado

---

**Más información:**
- [Configuration Recipes](CONFIGURATION_RECIPES.md)
- [Security Guide](SECURITY_GUIDE.md)
- [Production Ready](bulk/PRODUCTION_READY.md)

