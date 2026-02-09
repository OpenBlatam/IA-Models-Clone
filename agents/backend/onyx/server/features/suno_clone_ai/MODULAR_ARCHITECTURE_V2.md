# Arquitectura Modular V2

## Visión General

La aplicación ha sido reorganizada en una arquitectura modular siguiendo principios de microservicios y separación de concerns.

## Estructura de Directorios

```
suno_clone_ai/
├── modules/              # Módulos independientes (pueden ser microservicios)
│   ├── base.py          # Interfaz base para módulos
│   ├── registry.py      # Registro y gestión de módulos
│   ├── music_generation/ # Módulo de generación de música
│   ├── audio_processing/ # Módulo de procesamiento de audio
│   └── search/          # Módulo de búsqueda
├── infrastructure/      # Capa de infraestructura compartida
│   ├── database.py      # Abstracción de base de datos
│   ├── cache.py         # Abstracción de caché
│   ├── messaging.py     # Abstracción de message broker
│   └── storage.py       # Abstracción de almacenamiento
├── application/         # Capa de aplicación
│   └── use_cases/       # Casos de uso
├── api/                 # Capa de presentación (adapters)
├── core/                # Lógica de dominio core
├── services/            # Servicios de dominio
├── middleware/          # Middleware compartido
├── aws/                 # Integraciones AWS
└── bootstrap.py         # Inicialización modular
```

## Principios de Diseño

### 1. Separación de Concerns

- **Modules**: Módulos independientes con responsabilidades específicas
- **Infrastructure**: Abstracciones para servicios externos
- **Application**: Casos de uso que orquestan la lógica
- **API**: Adaptadores HTTP que exponen funcionalidad

### 2. Independencia de Módulos

Cada módulo:
- Puede funcionar independientemente
- Tiene su propia inicialización y shutdown
- Puede ser desplegado como microservicio
- Define sus dependencias explícitamente

### 3. Abstracciones de Infraestructura

Las abstracciones permiten:
- Cambiar backends sin cambiar código de negocio
- Testing más fácil (mocks/stubs)
- Soporte para múltiples proveedores (AWS, Azure, GCP, local)

## Módulos Disponibles

### Music Generation Module

```python
from modules.music_generation import MusicGenerationModule
from modules.base import ModuleConfig

module = MusicGenerationModule(
    ModuleConfig(
        name="music_generation",
        version="1.0.0",
        enabled=True
    )
)

await module.initialize()
result = await module.generate("A happy song", duration=30)
```

### Audio Processing Module

```python
from modules.audio_processing import AudioProcessingModule

module = AudioProcessingModule(config)
await module.initialize()
result = await module.process_audio("audio.mp3")
```

### Search Module

```python
from modules.search import SearchModule

module = SearchModule(config)
await module.initialize()
results = await module.search("jazz", limit=10)
```

## Registry de Módulos

El registry gestiona:
- Registro de módulos
- Inicialización en orden de dependencias
- Health checks
- Shutdown ordenado

```python
from modules.registry import get_module_registry

registry = get_module_registry()
module = registry.get_module("music_generation")
health = registry.get_health_report()
```

## Infraestructura

### Database

```python
from infrastructure.database import create_database_manager, DatabaseType

db = create_database_manager(
    DatabaseType.DYNAMODB,
    connection_string="table-name",
    region="us-east-1"
)
await db.connect()
```

### Cache

```python
from infrastructure.cache import create_cache_manager, CacheType

cache = create_cache_manager(CacheType.REDIS, redis_url="redis://...")
await cache.set("key", "value", ttl=3600)
```

### Message Broker

```python
from infrastructure.messaging import create_message_broker, MessageBrokerType

broker = create_message_broker(MessageBrokerType.SQS, queue_url="...")
await broker.publish("topic", {"message": "data"})
```

### Storage

```python
from infrastructure.storage import create_storage_manager, StorageType

storage = create_storage_manager(StorageType.S3, bucket_name="...")
await storage.upload("key", file_obj)
```

## Casos de Uso

Los casos de uso orquestan la lógica de negocio:

```python
from application.use_cases.music_generation import MusicGenerationUseCase
from modules.music_generation import MusicGenerationModule

module = MusicGenerationModule(config)
await module.initialize()

use_case = MusicGenerationUseCase(module)
result = await use_case.generate_music(
    prompt="A happy song",
    duration=30,
    user_id="user-123"
)
```

## Bootstrap

El bootstrap inicializa toda la aplicación:

```python
from bootstrap import bootstrap_application

result = bootstrap_application(app)
registry = result["registry"]
```

## Despliegue como Microservicios

Cada módulo puede desplegarse independientemente:

### Lambda Function para Music Generation

```python
# lambda_music_generation.py
from modules.music_generation import MusicGenerationModule
from modules.base import ModuleConfig

module = MusicGenerationModule(ModuleConfig(...))

async def handler(event, context):
    await module.initialize()
    result = await module.generate(event['prompt'])
    return result
```

### API Gateway Routing

- `/music/generate` → Music Generation Module
- `/audio/process` → Audio Processing Module
- `/search` → Search Module

## Ventajas

1. **Modularidad**: Cada módulo es independiente
2. **Escalabilidad**: Escalar módulos individualmente
3. **Mantenibilidad**: Código organizado y separado
4. **Testabilidad**: Fácil de testear módulos aislados
5. **Flexibilidad**: Agregar/quitar módulos sin afectar otros
6. **Cloud-Native**: Optimizado para serverless y microservicios

## Próximos Pasos

1. Agregar más módulos (analytics, recommendations, etc.)
2. Implementar service mesh para comunicación entre módulos
3. Agregar API Gateway routing por módulo
4. Implementar circuit breakers entre módulos
5. Agregar monitoring por módulo















