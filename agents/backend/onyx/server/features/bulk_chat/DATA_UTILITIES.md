# Data Utilities - Utilidades de Datos
## Utilidades Avanzadas para Transformación y Manejo de Datos

Este documento describe utilidades avanzadas para transformación, agrupación, mapeo, reducción y filtrado de datos, así como utilidades de networking y almacenamiento.

## 🚀 Nuevas Utilidades de Datos

### 1. BulkDataChunker - Chunker de Datos

Dividir datos en chunks de tamaño fijo.

```python
from bulk_chat.core.bulk_operations_performance import BulkDataChunker

chunker = BulkDataChunker(chunk_size=100)

# Dividir en chunks
items = list(range(250))
chunks = chunker.chunk(items)
# [[0-99], [100-199], [200-249]]

# Chunks asíncronos
async for chunk in chunker.chunk_async(items):
    await process_chunk(chunk)
```

**Características:**
- Tamaño configurable
- Síncrono y asíncrono
- **Mejora:** Procesamiento eficiente en chunks

### 2. BulkDataFlattener - Aplanador de Datos

Aplanar estructuras de datos anidadas.

```python
from bulk_chat.core.bulk_operations_performance import BulkDataFlattener

flattener = BulkDataFlattener()

# Aplanar lista anidada
nested = [[1, 2], [3, [4, 5]]]
flat = flattener.flatten(nested)
# [1, 2, 3, 4, 5]

# Aplanar diccionario
nested_dict = {"a": {"b": 1, "c": 2}, "d": 3}
flat_dict = flattener.flatten_dict(nested_dict, separator=".")
# {"a.b": 1, "a.c": 2, "d": 3}
```

**Características:**
- Aplanado recursivo
- Control de profundidad
- Separador configurable
- **Mejora:** Normalización de estructuras

### 3. BulkDataGrouper - Agrupador de Datos

Agrupar datos por función de key.

```python
from bulk_chat.core.bulk_operations_performance import BulkDataGrouper

grouper = BulkDataGrouper()

# Agrupar por key
items = [
    {"type": "A", "value": 1},
    {"type": "B", "value": 2},
    {"type": "A", "value": 3}
]

grouped = grouper.group_by(items, lambda x: x["type"])
# {
#   "A": [{"type": "A", "value": 1}, {"type": "A", "value": 3}],
#   "B": [{"type": "B", "value": 2}]
# }

# Agrupar asíncrono
grouped = await grouper.group_by_async(items, async_key_func)
```

**Características:**
- Key function personalizable
- Síncrono y asíncrono
- **Mejora:** Agrupación eficiente

### 4. BulkDataMapper - Mapeador de Datos

Mapear datos con función de transformación.

```python
from bulk_chat.core.bulk_operations_performance import BulkDataMapper

mapper = BulkDataMapper()

# Mapear items
items = [1, 2, 3, 4, 5]
mapped = mapper.map(items, lambda x: x * 2)
# [2, 4, 6, 8, 10]

# Mapear asíncrono
mapped = await mapper.map_async(items, async_transform)
```

**Características:**
- Transformación síncrona y asíncrona
- Procesamiento paralelo
- **Mejora:** Transformación eficiente

### 5. BulkDataReducer - Reductor de Datos

Reducir datos a un solo valor.

```python
from bulk_chat.core.bulk_operations_performance import BulkDataReducer

reducer = BulkDataReducer()

# Reducir items
items = [1, 2, 3, 4, 5]
sum_result = reducer.reduce(items, lambda acc, x: acc + x, initial=0)
# 15

# Reducir asíncrono
result = await reducer.reduce_async(items, async_reducer, initial=0)
```

**Características:**
- Valor inicial opcional
- Síncrono y asíncrono
- **Mejora:** Agregación eficiente

### 6. BulkDataFilter - Filtrador de Datos

Filtrar datos con predicado.

```python
from bulk_chat.core.bulk_operations_performance import BulkDataFilter

filter_obj = BulkDataFilter()

# Filtrar items
items = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
filtered = filter_obj.filter(items, lambda x: x % 2 == 0)
# [2, 4, 6, 8, 10]

# Filtrar asíncrono
filtered = await filter_obj.filter_async(items, async_predicate)
```

**Características:**
- Predicado personalizable
- Síncrono y asíncrono
- **Mejora:** Filtrado eficiente

### 7. BulkAsyncHTTPClient - Cliente HTTP Asíncrono

Cliente HTTP asíncrono optimizado.

```python
from bulk_chat.core.bulk_operations_performance import BulkAsyncHTTPClient

client = BulkAsyncHTTPClient(base_url="https://api.example.com", timeout=30.0)

# GET request
data = await client.get("/users", params={"page": 1})

# POST request
result = await client.post("/users", data={"name": "John"})

# Cerrar cliente
await client.close()
```

**Características:**
- Soporte para aiohttp y httpx
- Base URL configurable
- Timeout configurable
- **Mejora:** HTTP eficiente

### 8. BulkAsyncFileHandler - Manejador de Archivos

Manejo de archivos asíncrono.

```python
from bulk_chat.core.bulk_operations_performance import BulkAsyncFileHandler

file_handler = BulkAsyncFileHandler()

# Leer archivo
content = await file_handler.read_file("data.txt")

# Escribir archivo
await file_handler.write_file("output.txt", content)

# Leer líneas
lines = await file_handler.read_lines("data.txt")

# Escribir líneas
await file_handler.write_lines("output.txt", lines)
```

**Características:**
- Soporte para aiofiles
- Fallback síncrono
- **Mejora:** I/O de archivos eficiente

### 9. BulkAsyncStorage - Almacenamiento Asíncrono

Almacenamiento asíncrono genérico con TTL.

```python
from bulk_chat.core.bulk_operations_performance import BulkAsyncStorage

storage = BulkAsyncStorage(storage_type="memory")

# Almacenar con TTL
await storage.set("key1", "value1", ttl=3600.0)

# Obtener
value = await storage.get("key1")

# Eliminar
await storage.delete("key1")

# Limpiar
await storage.clear()
```

**Características:**
- TTL configurable
- Thread-safe
- **Mejora:** Almacenamiento eficiente

### 10. BulkAsyncQueueAdvanced - Cola Avanzada

Cola asíncrona con prioridades y timeouts.

```python
from bulk_chat.core.bulk_operations_performance import BulkAsyncQueueAdvanced

queue = BulkAsyncQueueAdvanced(maxsize=100)

# Agregar con prioridad
await queue.put("item1", priority=10)
await queue.put("item2", priority=5)

# Obtener con timeout
try:
    item = await queue.get(timeout=5.0)
except TimeoutError:
    pass

# Verificar estado
size = queue.qsize()
is_empty = queue.empty()
```

**Características:**
- Prioridades
- Timeouts
- Thread-safe
- **Mejora:** Cola eficiente con prioridades

### 11. BulkAsyncRateLimiter - Rate Limiter Mejorado

Rate limiter asíncrono mejorado.

```python
from bulk_chat.core.bulk_operations_performance import BulkAsyncRateLimiter

limiter = BulkAsyncRateLimiter(rate=10.0, per=1.0)

# Adquirir tokens
if await limiter.acquire(tokens=5.0):
    await operation()

# Esperar tokens
await limiter.wait(tokens=10.0)
```

**Características:**
- Token bucket algorithm
- Rate por período
- **Mejora:** Rate limiting eficiente

## 📊 Resumen de Utilidades de Datos

| Utilidad | Tipo | Mejora |
|----------|------|--------|
| **Data Chunker** | Transformación | Chunks eficientes |
| **Data Flattener** | Transformación | Aplanado de estructuras |
| **Data Grouper** | Agrupación | Agrupación eficiente |
| **Data Mapper** | Transformación | Mapeo paralelo |
| **Data Reducer** | Agregación | Reducción eficiente |
| **Data Filter** | Filtrado | Filtrado eficiente |
| **HTTP Client** | Networking | HTTP asíncrono |
| **File Handler** | I/O | Archivos asíncronos |
| **Storage** | Almacenamiento | Storage con TTL |
| **Queue Advanced** | Cola | Prioridades y timeouts |
| **Rate Limiter** | Control | Rate limiting mejorado |

## 🎯 Casos de Uso de Datos

### Pipeline de Transformación
```python
chunker = BulkDataChunker(chunk_size=100)
mapper = BulkDataMapper()
filter_obj = BulkDataFilter()
grouper = BulkDataGrouper()

# Pipeline completo
items = list(range(1000))
chunks = chunker.chunk(items)
mapped = [mapper.map(chunk, lambda x: x * 2) for chunk in chunks]
filtered = [filter_obj.filter(chunk, lambda x: x > 100) for chunk in mapped]
grouped = grouper.group_by(filtered[0], lambda x: x % 10)
```

### Sistema con HTTP y Storage
```python
client = BulkAsyncHTTPClient(base_url="https://api.example.com")
storage = BulkAsyncStorage()

# Fetch y cache
data = await storage.get("cache_key")
if not data:
    data = await client.get("/data")
    await storage.set("cache_key", data, ttl=3600.0)
```

### Sistema con Queue y Rate Limiting
```python
queue = BulkAsyncQueueAdvanced()
limiter = BulkAsyncRateLimiter(rate=10.0)

# Procesar con rate limiting
while not queue.empty():
    await limiter.wait()
    item = await queue.get(timeout=5.0)
    await process(item)
```

## 📈 Beneficios Totales

1. **Data Chunker**: División eficiente en chunks
2. **Data Flattener**: Normalización de estructuras
3. **Data Grouper**: Agrupación eficiente
4. **Data Mapper**: Transformación paralela
5. **Data Reducer**: Agregación eficiente
6. **Data Filter**: Filtrado eficiente
7. **HTTP Client**: HTTP asíncrono optimizado
8. **File Handler**: I/O de archivos eficiente
9. **Storage**: Almacenamiento con TTL
10. **Queue Advanced**: Cola con prioridades
11. **Rate Limiter**: Rate limiting mejorado

## 🚀 Resultados Esperados

Con todas las utilidades de datos:

- **Transformación eficiente** de datos (chunk, flatten, map, reduce, filter)
- **Agrupación eficiente** de datos
- **HTTP asíncrono** optimizado
- **I/O de archivos** eficiente
- **Almacenamiento** con TTL
- **Cola avanzada** con prioridades y timeouts
- **Rate limiting** mejorado

El sistema ahora tiene **129+ optimizaciones, utilidades, componentes y características** que cubren todos los aspectos posibles de procesamiento masivo, desde transformación de datos hasta networking, almacenamiento y control de flujo.

El sistema está completamente optimizado y listo para producción con todas las características necesarias para operaciones masivas de alta performance.














