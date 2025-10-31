# PDF Variantes API - Arquitectura Organizada

La API ha sido reorganizada para mejorar la mantenibilidad, escalabilidad y claridad del código.

## 📁 Estructura de Archivos

```
api/
├── main.py          # Punto de entrada principal - App FastAPI
├── lifecycle.py     # Gestión de startup/shutdown y servicios
├── dependencies.py # Inyección de dependencias (DI)
├── middleware.py    # Middleware de requests/responses
├── config.py        # Configuración de la aplicación
├── routers.py       # Registro de routers
├── routes.py        # Definición de endpoints
└── models.py        # Modelos de datos (si aplica)
```

## 🏗️ Organización de Responsabilidades

### `main.py`
- Punto de entrada principal de la aplicación
- Creación de la instancia FastAPI
- Configuración de OpenAPI
- Manejo de excepciones globales
- Endpoints raíz (`/`, `/health`)

### `lifecycle.py`
- Gestión del ciclo de vida de la aplicación
- Inicialización de servicios durante startup
- Cleanup de servicios durante shutdown
- Registro de servicios en el sistema de dependencias

### `dependencies.py`
- Sistema de inyección de dependencias (FastAPI Dependency Injection)
- Funciones helper para obtener servicios individuales
- Registro centralizado de servicios

### `middleware.py`
- Configuración de CORS
- Rate limiting
- Logging de requests
- Monitoring de performance
- Middleware de seguridad

### `config.py`
- Configuración de la aplicación
- Setup de middleware base
- Configuración de OpenAPI schema
- Funciones de configuración centralizadas

### `routers.py`
- Registro centralizado de todos los routers
- Organización de prefijos y tags
- Facilita agregar nuevos routers

### `routes.py`
- Definición de todos los endpoints
- Lógica de negocio de cada endpoint
- Validación y manejo de errores a nivel de endpoint

## 🔄 Flujo de la Aplicación

```
1. main.py
   ├── Crea app FastAPI
   ├── Importa configuración
   └── Registra componentes

2. lifecycle.py
   ├── Inicializa servicios (startup)
   ├── Registra servicios en DI
   └── Cleanup servicios (shutdown)

3. config.py
   ├── Setup middleware base
   └── Configura OpenAPI

4. routers.py
   └── Registra todos los routers

5. middleware.py
   ├── Intercepta requests
   ├── Aplica CORS, rate limiting, etc.
   └── Pasa al siguiente middleware

6. routes.py
   ├── Recibe request
   ├── Inyecta dependencias (servicios)
   ├── Ejecuta lógica de negocio
   └── Retorna respuesta
```

## 🔌 Inyección de Dependencias

### Obtener todos los servicios
```python
from api.dependencies import get_services

@app.get("/endpoint")
async def my_endpoint(services: dict = Depends(get_services)):
    pdf_service = services["pdf_service"]
    # ...
```

### Obtener un servicio específico
```python
from api.dependencies import get_pdf_service

@app.get("/endpoint")
async def my_endpoint(pdf_service: PDFVariantesService = Depends(get_pdf_service)):
    # Usar pdf_service directamente
    # ...
```

## 📝 Agregar Nuevos Routers

1. **Definir router en `routes.py`:**
```python
my_router = APIRouter()

@my_router.get("/items")
async def get_items():
    return {"items": []}
```

2. **Registrar en `routers.py`:**
```python
from .routes import my_router

def register_routers(app: FastAPI) -> None:
    # ... otros routers
    app.include_router(
        my_router,
        prefix="/api/v1/my-feature",
        tags=["My Feature"]
    )
```

3. **Listo!** El router estará disponible en `/api/v1/my-feature/items`

## 🛠️ Agregar Nuevos Servicios

1. **Crear el servicio** en `services/` (fuera de `api/`)

2. **Importar en `lifecycle.py`:**
```python
from ..services.my_service import MyService

async def initialize_services() -> Dict[str, Any]:
    services = {}
    # ...
    services["my_service"] = MyService(settings)
    # ...
    return services
```

3. **Crear dependencia en `dependencies.py`:**
```python
def get_my_service(services: Dict[str, Any] = Depends(get_services)) -> MyService:
    service = services.get("my_service")
    if not service:
        raise RuntimeError("My service not available")
    return service
```

4. **Usar en endpoints:**
```python
from api.dependencies import get_my_service

@router.get("/endpoint")
async def my_endpoint(my_service: MyService = Depends(get_my_service)):
    result = await my_service.do_something()
    return result
```

## 🎯 Ventajas de esta Organización

✅ **Separación de Responsabilidades**: Cada módulo tiene un propósito claro
✅ **Mantenibilidad**: Fácil encontrar y modificar código
✅ **Testabilidad**: Servicios y dependencias fácilmente mockeables
✅ **Escalabilidad**: Fácil agregar nuevos routers, servicios, middleware
✅ **Claridad**: Código más limpio y fácil de entender
✅ **Reutilización**: Dependencias reutilizables en múltiples endpoints

## 🔍 Debugging

Para debuggear el flujo:

1. **Ver servicios inicializados:**
```python
# En cualquier endpoint
services = Depends(get_services)
print(services.keys())  # Ver todos los servicios disponibles
```

2. **Ver middleware aplicado:**
```python
# Los logs muestran qué middleware se ejecutó
```

3. **Ver routers registrados:**
```python
# En FastAPI, visita /docs para ver todos los endpoints
```

## 📚 Referencias

- [FastAPI Dependency Injection](https://fastapi.tiangolo.com/tutorial/dependencies/)
- [FastAPI Application Lifetime](https://fastapi.tiangolo.com/advanced/events/)
- [FastAPI Middleware](https://fastapi.tiangolo.com/advanced/middleware/)






