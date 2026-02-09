# Sistema Modular - PDF Variantes API

## Arquitectura Modular Completa

Sistema completamente modular donde cada funcionalidad es un módulo independiente, auto-contenido y reutilizable.

## Estructura de Módulos

```
modules/
├── document/           # Módulo de documentos
│   ├── domain.py       # Entidades y lógica
│   ├── application.py  # Casos de uso
│   ├── infrastructure.py # Repositorios
│   └── presentation.py # Controllers
│
├── variant/            # Módulo de variantes
│   └── ... (misma estructura)
│
├── topic/              # Módulo de topics
│   └── ... (misma estructura)
│
├── shared/             # Componentes compartidos
│   ├── interfaces.py   # Interfaces comunes
│   ├── events.py       # Sistema de eventos
│   └── utilities.py    # Utilidades
│
├── module_registry.py  # Registry de módulos
├── module_factory.py   # Factory pattern
├── module_loader.py    # Carga dinámica
├── module_manager.py   # Gestión completa
├── module_router.py    # Auto-generación routers
└── integration.py      # Integración con FastAPI
```

## Uso Rápido

### Setup Automático

```python
from api.modules.integration import setup_modules

# En main.py
app = FastAPI()
module_manager = setup_modules(app)
```

### Uso Manual

```python
from api.modules.module_manager import ModuleManager
from api.modules.config import ModulesConfig

# Crear manager
manager = ModuleManager()

# Descubrir módulos
manager.discover_and_register_modules(Path("api/modules"))

# Inicializar
manager.initialize_all_modules()

# Registrar routers
manager.register_routers(app)
```

### Usar Módulo Específico

```python
from api.modules.document import DocumentController

# Usar directamente
controller = DocumentController(...)
result = await controller.upload(request, command)
```

## Características

### ✅ Independencia Total
- Cada módulo es independiente
- Sin dependencias directas entre módulos
- Removible sin afectar otros

### ✅ Auto-Discovery
- Descubrimiento automático de módulos
- Registro automático
- Inicialización automática

### ✅ Auto-Generación de Routers
- Routers generados automáticamente
- Endpoints creados desde controllers
- Sin código repetitivo

### ✅ Configuración por Módulo
- Configuración individual
- Dependencias declaradas
- Habilitar/deshabilitar módulos

### ✅ Shared Components
- Interfaces comunes
- Eventos compartidos
- Utilidades reutilizables

## Ejemplo Completo

```python
# 1. Auto-setup (recomendado)
from api.modules.integration import setup_modules
module_manager = setup_modules(app)

# 2. Los módulos están listos
# Routers registrados automáticamente
# Controllers disponibles
# Todo funcionando

# 3. Acceder a módulos
controller = module_manager.get_module_controller("document")
result = await controller.upload(request, command)
```

## Beneficios

- **Desarrollo Paralelo**: Equipos trabajan en módulos separados
- **Testing Independiente**: Cada módulo testeable solo
- **Deployment Flexible**: Deploy módulos por separado
- **Reutilización**: Módulos usables en otros proyectos
- **Escalabilidad**: Agregar módulos sin tocar existentes

¡Sistema modular completo y listo para usar!






