# Arquitectura Modular

Este documento describe la arquitectura modular del proyecto Plastic Surgery Visualization AI.

## Estructura Modular

El código ha sido refactorizado para seguir principios de diseño modular, separando responsabilidades en módulos pequeños y enfocados.

### Módulos Core

#### `core/app_factory.py`
- **Responsabilidad**: Factory principal para crear la aplicación FastAPI
- **Funciones**:
  - `create_app()`: Crea y configura la aplicación FastAPI completa

#### `core/lifespan.py`
- **Responsabilidad**: Gestión del ciclo de vida de la aplicación
- **Funciones**:
  - `lifespan()`: Context manager para startup/shutdown
  - `startup()`: Tareas de inicialización
  - `shutdown()`: Tareas de limpieza
  - `ensure_directories()`: Creación de directorios necesarios
  - `check_dependencies()`: Verificación de dependencias

#### `core/middleware_config.py`
- **Responsabilidad**: Configuración de middleware
- **Funciones**:
  - `setup_middleware()`: Configura todos los middlewares

#### `core/exceptions_config.py`
- **Responsabilidad**: Configuración de manejadores de excepciones
- **Funciones**:
  - `setup_exception_handlers()`: Configura todos los exception handlers

#### `core/routes_config.py`
- **Responsabilidad**: Configuración de rutas
- **Funciones**:
  - `setup_routes()`: Registra todas las rutas
  - `setup_root_endpoint()`: Configura el endpoint raíz

#### `core/factories.py`
- **Responsabilidad**: Factory functions para crear instancias de servicios
- **Funciones**:
  - `create_cache()`: Crea instancia de caché (singleton)
  - `create_image_processor()`: Crea procesador de imágenes (singleton)
  - `create_ai_processor()`: Crea procesador AI (singleton)
  - `create_visualization_service()`: Crea servicio de visualización (singleton)

#### `core/dependencies.py`
- **Responsabilidad**: Dependencias de FastAPI (para inyección)
- **Funciones**:
  - `get_cache()`: Dependency para obtener caché
  - `get_image_processor()`: Dependency para procesador de imágenes
  - `get_ai_processor()`: Dependency para procesador AI
  - `get_visualization_service()`: Dependency para servicio de visualización

#### `core/app_config.py`
- **Responsabilidad**: Configuración de aplicación (mantenido para compatibilidad)
- **Nota**: Este módulo está deprecado, usar `app_factory` en su lugar

## Flujo de Inicialización

```
main.py
  └─> setup_dev_environment()
  └─> setup_logging()
  └─> create_app() [app_factory.py]
        ├─> setup_middleware() [middleware_config.py]
        ├─> setup_exception_handlers() [exceptions_config.py]
        └─> setup_routes() [routes_config.py]
              └─> lifespan() [lifespan.py]
                    ├─> startup()
                    │     ├─> ensure_directories()
                    │     └─> check_dependencies()
                    └─> shutdown()
```

## Principios de Diseño

### 1. Separación de Responsabilidades
Cada módulo tiene una responsabilidad única y bien definida.

### 2. Single Responsibility Principle
Cada función hace una sola cosa y la hace bien.

### 3. Dependency Injection
Los servicios se crean mediante factories y se inyectan como dependencias.

### 4. Singleton Pattern
Los servicios pesados (cache, processors) se crean una sola vez usando `@lru_cache`.

### 5. Factory Pattern
Las factories centralizan la creación de instancias.

## Beneficios de la Arquitectura Modular

1. **Mantenibilidad**: Código más fácil de entender y modificar
2. **Testabilidad**: Cada módulo puede probarse independientemente
3. **Reutilización**: Los módulos pueden reutilizarse en otros contextos
4. **Escalabilidad**: Fácil agregar nuevas funcionalidades sin afectar existentes
5. **Debugging**: Más fácil identificar y solucionar problemas

## Migración

Si tienes código que usa los módulos antiguos:

```python
# Antes
from core.app_config import setup_middleware, setup_exception_handlers

# Ahora
from core.middleware_config import setup_middleware
from core.exceptions_config import setup_exception_handlers
```

O simplemente usa el factory:

```python
# Antes
from core.app_config import create_app_config
app = FastAPI(**create_app_config())

# Ahora
from core.app_factory import create_app
app = create_app()
```

## Extensibilidad

Para agregar nuevas funcionalidades:

1. **Nuevo middleware**: Agregar en `core/middleware_config.py`
2. **Nueva ruta**: Agregar en `core/routes_config.py`
3. **Nuevo servicio**: Crear factory en `core/factories.py`
4. **Nueva excepción**: Agregar handler en `core/exceptions_config.py`

