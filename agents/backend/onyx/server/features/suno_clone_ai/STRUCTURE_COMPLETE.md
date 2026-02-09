# Estructura Completa - Arquitectura Modular IA

## 📁 Estructura de Archivos por Módulo

Cada módulo sigue esta estructura estándar:

```
module_name/
├── __init__.py          # Exports públicos con documentación de rol y reglas
├── main.py              # Funciones base y entry points
├── base.py              # Clases base abstractas
├── service.py           # Servicio principal
├── repository.py        # Repositorio (si aplica)
├── models.py            # Modelos de datos (si aplica)
└── [archivos específicos]
```

## ✅ Módulos Completados

### Nivel 0 - Base (Sin Dependencias)
- ✅ `configs/` - Configuraciones del sistema
- ✅ `utils/` - Utilidades generales

### Nivel 1 - Infraestructura
- ✅ `db/` - Base de datos y ORM
- ✅ `redis/` - Cliente Redis
- ✅ `tracing/` - Trazabilidad y observabilidad
- ✅ `httpx/` - Cliente HTTP asíncrono

### Nivel 2 - Servicios Base
- ✅ `auth/` - Autenticación y autorización
- ✅ `prompts/` - Gestión de prompts
- ✅ `llm/` - Modelos de lenguaje

### Nivel 3 - Servicios Especializados
- ✅ `access/` - Control de acceso
- ✅ `tools/` - Herramientas
- ✅ `context/` - Gestión de contexto
- ✅ `chat/` - Sistema de chat
- ✅ `agents/` - Agentes de IA

## 📋 Módulos Restantes (Estructura Similar)

Los siguientes módulos siguen el mismo patrón. Ejemplo de estructura:

### `server/`
```python
# server/main.py
def get_server_service() -> ServerService
def initialize_server() -> ServerService

# server/__init__.py
# Incluye exports y documentación de rol
```

### `background/`
```python
# background/main.py
def get_background_service() -> BackgroundService
async def enqueue_task(task, *args, **kwargs) -> str

# background/__init__.py
# Incluye exports y documentación de rol
```

### `file_store/`
```python
# file_store/main.py
def get_file_store_service() -> FileStoreService
async def save_file(file_path: str, content: bytes) -> str

# file_store/__init__.py
# Incluye exports y documentación de rol
```

### `document_index/`
```python
# document_index/main.py
def get_document_index_service() -> DocumentIndexService
async def index_document(document: dict) -> str

# document_index/__init__.py
# Incluye exports y documentación de rol
```

## 🔄 Patrón de Importación Seguro

### Ejemplo: `chat/main.py`
```python
# ✅ CORRECTO - Importar desde main.py de otros módulos
from llm.main import get_llm_service
from db.main import get_db_service
from context.main import get_context_service

# ❌ INCORRECTO - Importar directamente desde service.py
# from llm.service import LLMService  # Evitar esto
```

### Ejemplo: `llm/service.py`
```python
# ✅ CORRECTO - Usar TYPE_CHECKING para tipos
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from chat.service import ChatService  # Solo para type hints

# ✅ CORRECTO - Inyección de dependencias
class LLMService:
    def __init__(self, chat_service=None):  # Inyectado desde fuera
        self.chat_service = chat_service
```

## 📝 Reglas de Importación por Nivel

### Nivel 0 (Base)
- **Puede importar**: Solo librerías externas
- **NO puede importar**: Otros módulos del proyecto

### Nivel 1 (Infraestructura)
- **Puede importar**: Nivel 0 (configs, utils)
- **NO puede importar**: Niveles superiores

### Nivel 2 (Servicios Base)
- **Puede importar**: Niveles 0 y 1
- **NO puede importar**: Niveles 3+ (evitar ciclos)

### Nivel 3 (Servicios Especializados)
- **Puede importar**: Niveles 0, 1, 2
- **NO puede importar**: Otros del mismo nivel si crean ciclos

### Nivel 4 (Lógica de Negocio)
- **Puede importar**: Todos los niveles anteriores
- **NO puede importar**: Otros del mismo nivel si crean ciclos

### Nivel 5 (Interfaz)
- **Puede importar**: Todos los niveles anteriores
- Es el punto de entrada del sistema

## 🎯 Funciones Principales en `main.py`

Cada `main.py` debe incluir:

1. **get_[module]_service()** - Obtiene instancia singleton del servicio
2. **initialize_[module]()** - Inicializa el módulo
3. **Funciones de acceso rápido** - Wrappers para operaciones comunes

## 📚 Documentación en `__init__.py`

Cada `__init__.py` debe incluir:

1. **Descripción del módulo**
2. **Rol en el Ecosistema IA** - Cómo contribuye al sistema
3. **Reglas de Importación** - Qué puede y no puede importar
4. **Exports organizados** - Clases y funciones públicas

## 🔍 Detección de Ciclos

Para detectar ciclos de importación:

```bash
# Usar herramientas como:
- pylint --disable=all --enable=import-error
- mypy --follow-imports=silent
- python -m py_compile [archivo]
```

## 🚀 Inicialización del Sistema

Orden recomendado de inicialización:

```python
# 1. Base
from configs import initialize_config
from utils import get_util_service

# 2. Infraestructura
from db import initialize_db
from redis import initialize_redis
from tracing import initialize_tracing

# 3. Servicios Base
from auth import initialize_auth
from prompts import initialize_prompts
from llm import initialize_llm

# 4. Servicios Especializados
from tools import initialize_tools
from context import initialize_context
from chat import initialize_chat
from agents import initialize_agents

# 5. Interfaz
from server import initialize_server
```

## ✅ Checklist de Implementación

Para cada módulo nuevo:

- [ ] Crear `__init__.py` con documentación completa
- [ ] Crear `main.py` con funciones base
- [ ] Crear `base.py` con clases abstractas (si aplica)
- [ ] Crear `service.py` con servicio principal
- [ ] Crear `repository.py` si maneja datos (si aplica)
- [ ] Crear `models.py` si tiene modelos de datos (si aplica)
- [ ] Documentar rol en el ecosistema IA
- [ ] Documentar reglas de importación
- [ ] Verificar que no hay ciclos de importación
- [ ] Agregar funciones de inicialización

