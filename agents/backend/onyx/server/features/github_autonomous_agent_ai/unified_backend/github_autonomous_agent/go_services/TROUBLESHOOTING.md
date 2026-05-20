# 🔧 Troubleshooting Guide - Go Services

## Common Issues and Solutions

### Build Issues

#### Error: `go: cannot find module`
```bash
# Solution: Download dependencies
go mod download
go mod tidy
```

#### Error: `package not found`
```bash
# Solution: Update dependencies
go get -u ./...
go mod tidy
```

#### Error: `CGO_ENABLED=0` issues
```bash
# Solution: Enable CGO if needed, or build without it
CGO_ENABLED=0 go build ./cmd/agent
```

### Runtime Issues

#### Service won't start

**Check port availability:**
```bash
# Linux/Mac
lsof -i :8080
netstat -tulpn | grep 8080

# Windows
netstat -ano | findstr :8080
```

**Check logs:**
```bash
./agent-service --port 8080 --log-level=debug
```

#### High memory usage

**Solutions:**
1. Reduce cache size:
   ```bash
   export CACHE_MEMORY_SIZE=5000
   ```

2. Enable Redis for distributed caching:
   ```bash
   export CACHE_ENABLE_REDIS=true
   export REDIS_URL=redis://localhost:6379
   ```

3. Monitor BadgerDB size:
   ```bash
   du -sh /tmp/agent_cache
   ```

#### Slow performance

**Check:**
1. Worker pool size - increase if needed
2. Cache hit rate - should be >80%
3. Search index size - may need optimization

**Profile:**
```bash
go tool pprof http://localhost:8080/debug/pprof/profile
```

### Cache Issues

#### Cache not working

**Check:**
```bash
# Verify cache is enabled
curl http://localhost:8080/api/v1/cache?key=test

# Check BadgerDB
ls -la /tmp/agent_cache
```

#### Cache corruption

**Solution:**
```bash
# Clear cache
rm -rf /tmp/agent_cache
# Restart service
```

### Search Issues

#### Search index corrupted

**Solution:**
```bash
# Rebuild index
rm -rf /tmp/agent_search
# Restart and re-index
```

#### Slow search queries

**Solutions:**
1. Optimize index:
   ```go
   // Use batch indexing
   index.IndexBatch(docs)
   ```

2. Limit results:
   ```go
   search.Search(query, SearchOptions{Limit: 10})
   ```

### Git Operations Issues

#### Git clone fails

**Check:**
1. Network connectivity
2. Repository URL validity
3. Permissions for target directory

**Debug:**
```bash
# Test manually
git clone https://github.com/user/repo.git /tmp/test
```

#### Repository not found

**Solution:**
```go
// Check if repository exists
repo, err := git.OpenRepository(path)
if err != nil {
    // Handle error
}
```

### Docker Issues

#### Container won't start

**Check logs:**
```bash
docker logs agent-service
```

**Check health:**
```bash
docker exec agent-service wget -qO- http://localhost:8080/health
```

#### Permission issues

**Solution:**
```bash
# Run with proper user
docker run -u $(id -u):$(id -g) ...
```

### Network Issues

#### Can't connect from Python

**Check:**
1. Service is running: `curl http://localhost:8080/health`
2. Firewall rules
3. CORS settings (if applicable)

**Test:**
```python
import httpx
try:
    response = httpx.get("http://localhost:8080/health", timeout=5.0)
    print("Connected!")
except Exception as e:
    print(f"Error: {e}")
```

## Performance Tuning

### Increase throughput

**Queue settings:**
```go
queue := queue.NewTaskQueue(queue.Config{
    MaxWorkers: 200,  // Increase workers
    QueueSize:  20000, // Increase queue size
})
```

### Reduce latency

**Cache settings:**
```go
cache := cache.NewMultiTierCache(cache.Config{
    MemorySize: 20000,  // Larger memory cache
    MemoryTTL:  10 * time.Minute,
})
```

### Optimize memory

**BadgerDB settings:**
```go
// Use compression
opts := badger.DefaultOptions(path)
opts.Compression = options.ZSTD
```

## Debugging

### Enable debug logging

```bash
./agent-service --log-level=debug
```

### Profiling

**CPU profiling:**
```bash
go tool pprof http://localhost:8080/debug/pprof/profile
```

**Memory profiling:**
```bash
go tool pprof http://localhost:8080/debug/pprof/heap
```

### Trace

```bash
go tool trace trace.out
```

## Getting Help

1. Check logs: `./agent-service --log-level=debug`
2. Review documentation: `README.md`, `INTEGRATION_GUIDE.md`
3. Check GitHub issues
4. Enable verbose logging

## Common Error Messages

### `bind: address already in use`
**Solution:** Change port or stop existing service

### `no such file or directory`
**Solution:** Check file paths and permissions

### `connection refused`
**Solution:** Verify service is running and accessible

### `timeout`
**Solution:** Increase timeout or check network connectivity












