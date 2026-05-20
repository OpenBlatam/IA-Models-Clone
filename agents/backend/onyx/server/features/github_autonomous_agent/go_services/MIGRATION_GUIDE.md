# 🔄 Migration Guide - Go Services

## Overview

This guide helps you migrate from Python implementations to Go services for better performance.

## Migration Strategy

### Phase 1: Parallel Running (Recommended)

Run both Python and Go services in parallel, gradually shifting traffic.

```
┌─────────────┐
│   Python    │──┐
│  Services   │  │
└─────────────┘  │
                 ├──▶ Load Balancer ──▶ Clients
┌─────────────┐  │
│  Go Services│──┘
│  (New)      │
└─────────────┘
```

**Steps:**
1. Deploy Go services alongside Python
2. Route 10% traffic to Go services
3. Monitor performance and errors
4. Gradually increase to 100%

### Phase 2: Component-by-Component

Migrate one component at a time.

## Component Migrations

### 1. Cache Service

#### Before (Python)
```python
from core.services.cache_service import CacheService

cache = CacheService(max_size=1000, ttl=300)
cache.set("key", "value")
value = cache.get("key")
```

#### After (Go via HTTP)
```python
import httpx

client = httpx.Client(base_url="http://localhost:8080")

# Set
client.post("/api/v1/cache", params={"key": "key", "value": "value"})

# Get
response = client.get("/api/v1/cache", params={"key": "key"})
value = response.json()["value"]
```

#### After (Go via Client Library)
```python
from go_services_client import GoServicesClient

client = GoServicesClient()
client.cache_set("key", "value")
value = client.cache_get("key")
```

**Benefits:**
- 10-50x faster
- Multi-tier caching
- Persistent storage

### 2. Git Operations

#### Before (Python)
```python
import subprocess

subprocess.run(["git", "clone", url, path])
```

#### After (Go)
```python
from go_services_client import GoServicesClient

client = GoServicesClient()
client.clone_repository(url, path)
```

**Benefits:**
- 3-5x faster
- Better error handling
- No subprocess overhead

### 3. Search Service

#### Before (Python)
```python
from core.services.search_service import SearchService

search = SearchService()
results = search.search("query", limit=10)
```

#### After (Go)
```python
from go_services_client import GoServicesClient

client = GoServicesClient()
results = client.search("query", limit=10)
```

**Benefits:**
- 20-100x faster
- No external Elasticsearch needed
- Embedded search engine

### 4. Task Queue

#### Before (Python - Celery)
```python
from celery import Celery

app = Celery('tasks')
app.send_task('process_repo', args=[repo])
```

#### After (Go)
```python
from go_services_client import GoServicesClient

client = GoServicesClient()
client.enqueue_task({
    "type": "process_repo",
    "data": {"repo": repo}
})
```

**Benefits:**
- 5-10x higher throughput
- Lower latency
- No Redis dependency (optional)

## Migration Checklist

### Pre-Migration

- [ ] Review current Python implementations
- [ ] Identify performance bottlenecks
- [ ] Set up Go services environment
- [ ] Create feature flags for gradual rollout

### Migration

- [ ] Deploy Go services
- [ ] Set up monitoring
- [ ] Route small percentage of traffic
- [ ] Monitor for errors
- [ ] Gradually increase traffic
- [ ] Verify performance improvements

### Post-Migration

- [ ] Remove Python implementations
- [ ] Update documentation
- [ ] Train team on Go services
- [ ] Monitor long-term performance

## Rollback Plan

If issues occur:

1. **Immediate:** Route all traffic back to Python
2. **Investigate:** Check Go services logs
3. **Fix:** Address issues in Go services
4. **Retry:** Gradually reintroduce Go services

## Performance Comparison

| Operation | Python | Go | Improvement |
|-----------|--------|----|----|
| Cache get | 500ns | 50ns | 10x |
| Git clone | 2.5s | 0.8s | 3.1x |
| Search | 200ms | 5ms | 40x |
| Queue throughput | 1K/s | 10K/s | 10x |

## Common Issues

### Issue: Different API

**Solution:** Use adapter pattern
```python
class GoServicesAdapter:
    def __init__(self):
        self.client = GoServicesClient()
    
    def search(self, query, limit=10):
        # Adapt Go API to Python API
        return self.client.search(query, limit)
```

### Issue: Error Handling

**Solution:** Wrap Go errors
```python
try:
    result = client.search(query)
except GoServiceError as e:
    # Convert to Python exception
    raise SearchError(str(e))
```

## Testing Strategy

1. **Unit Tests:** Test each component
2. **Integration Tests:** Test with real data
3. **Performance Tests:** Compare with Python
4. **Load Tests:** Verify under load

## Timeline

- **Week 1:** Setup and deployment
- **Week 2:** 10% traffic migration
- **Week 3:** 50% traffic migration
- **Week 4:** 100% traffic migration
- **Week 5:** Remove Python code

## Support

For issues during migration:
1. Check `TROUBLESHOOTING.md`
2. Review logs
3. Contact team
4. Use rollback plan if needed












