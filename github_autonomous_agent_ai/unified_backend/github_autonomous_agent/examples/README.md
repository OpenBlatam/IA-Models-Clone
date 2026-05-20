# Ejemplos de Uso - GitHub Autonomous Agent

Colección de ejemplos prácticos para usar el GitHub Autonomous Agent.

---

## 📋 Ejemplos Disponibles

### `example_usage.py`
Ejemplos completos de uso de todas las funcionalidades.

**Incluye:**
- Cliente de GitHub básico
- Procesamiento de tareas
- Worker Manager
- API REST
- Flujo completo
- Manejo de errores

**Ejecutar:**
```bash
python examples/example_usage.py
```

---

## 🚀 Ejemplos Rápidos

### 1. Cliente de GitHub

```python
from core.github_client import GitHubClient

client = GitHubClient(token="your_token")
repo = await client.get_repository("owner", "repo-name")
```

### 2. Crear Tarea

```python
from api.schemas import TaskCreate

task = TaskCreate(
    repository="owner/repo",
    instruction="Crear archivo README.md"
)
```

### 3. API REST

```python
import httpx

async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8030/api/v1/tasks",
        json={"repository": "owner/repo", "instruction": "..."}
    )
```

---

## 📚 Más Información

- Ver [DEVELOPMENT.md](../DEVELOPMENT.md) para guía completa
- Ver [README.md](../README.md) para documentación principal
- Ver templates en [templates/](../templates/) para crear nuevas funcionalidades

---

**Última actualización:** 2024




