# 🚀 Quick Start - Go Services

## Instalación Rápida

```bash
cd go_services
go mod download
go build ./cmd/agent
./agent --port 8080
```

## Uso Básico

### Desde Go

```go
import "github.com/blatam-academy/github-autonomous-agent/go_services/internal/git"

repo, err := git.OpenRepository("/path/to/repo")
commits, err := repo.GetCommits("main", 10)
```

### Desde Python

```python
import httpx

client = httpx.Client(base_url="http://localhost:8080")

# Git clone (3-5x más rápido)
response = client.post(
    "/api/v1/git/clone",
    params={"url": "https://github.com/user/repo.git", "path": "/tmp/repo"}
)

# Cache (10-50x más rápido)
client.post("/api/v1/cache", params={"key": "test", "value": "data"})
value = client.get("/api/v1/cache", params={"key": "test"}).json()

# Search (20-100x más rápido)
results = client.get("/api/v1/search", params={"q": "query"}).json()
```

## Endpoints Disponibles

- `GET /health` - Health check
- `POST /api/v1/git/clone` - Clone repository
- `GET /api/v1/search?q=query` - Full-text search
- `GET /api/v1/cache?key=key` - Get from cache
- `POST /api/v1/cache?key=key&value=value` - Set cache

## Próximos Pasos

Ver `INTEGRATION_GUIDE.md` para integración completa.












