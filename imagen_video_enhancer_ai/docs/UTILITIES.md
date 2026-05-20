# Utilities Guide - Imagen Video Enhancer AI

## Overview

This guide covers the utility modules available in the project.

## Serialization Utilities

### JSONSerializer

```python
from imagen_video_enhancer_ai.utils.serialization import JSONSerializer

# Serialize
data = {"key": "value"}
json_bytes = JSONSerializer.serialize(data)

# Deserialize
data = JSONSerializer.deserialize(json_bytes)
```

### Base64Serializer

```python
from imagen_video_enhancer_ai.utils.serialization import Base64Serializer

# Encode
encoded = Base64Serializer.encode_json({"key": "value"})

# Decode
data = Base64Serializer.decode_json(encoded)
```

### File Serialization

```python
from imagen_video_enhancer_ai.utils.serialization import save_serialized, load_serialized

# Save
save_serialized(data, "output.json", format="json")

# Load
data = load_serialized("output.json", format="json")
```

## Response Builder

### Success Response

```python
from imagen_video_enhancer_ai.utils.response_builder import ResponseBuilder

response = ResponseBuilder.success(
    data={"result": "enhanced"},
    message="Enhancement completed",
    metadata={"task_id": "123"}
)
```

### Error Response

```python
response = ResponseBuilder.error(
    error="File not found",
    code="FILE_NOT_FOUND",
    details={"file_path": "/path/to/file"}
)
```

### Paginated Response

```python
response = ResponseBuilder.paginated(
    items=[...],
    page=1,
    page_size=10,
    total=100
)
```

## Data Transformers

### Dictionary Transformers

```python
from imagen_video_enhancer_ai.utils.transformers import DictTransformer

# Flatten
nested = {"a": {"b": {"c": 1}}}
flat = DictTransformer.flatten(nested)
# {"a.b.c": 1}

# Unflatten
nested = DictTransformer.unflatten(flat)

# Filter keys
filtered = DictTransformer.filter_keys(data, ["key1", "key2"], include=True)

# Map values
mapped = DictTransformer.map_values(data, lambda x: x * 2)
```

### List Transformers

```python
from imagen_video_enhancer_ai.utils.transformers import ListTransformer

# Chunk
chunks = ListTransformer.chunk(items, chunk_size=10)

# Group by
grouped = ListTransformer.group_by(items, key="category")

# Unique
unique_items = ListTransformer.unique(items, key=lambda x: x.id)
```

### DateTime Transformers

```python
from imagen_video_enhancer_ai.utils.transformers import DateTimeTransformer

# To ISO string
iso_string = DateTimeTransformer.to_iso_string(datetime.now())

# From ISO string
dt = DateTimeTransformer.from_iso_string(iso_string)

# To/from timestamp
timestamp = DateTimeTransformer.to_timestamp(dt)
dt = DateTimeTransformer.from_timestamp(timestamp)
```

## Error Context

### Context Manager

```python
from imagen_video_enhancer_ai.utils.error_context import error_context

with error_context(user_id="123", task_id="456") as ctx:
    ctx.add("step", "processing")
    # Code that might raise
```

### Decorator

```python
from imagen_video_enhancer_ai.utils.error_context import wrap_with_context

@wrap_with_context({"service": "enhancement"})
async def my_function():
    ...
```

## Configuration Loader

```python
from imagen_video_enhancer_ai.utils.config_loader import ConfigLoader

# Load from multiple sources
config = ConfigLoader.create_config(
    file_path="config.json",
    env_prefix="ENHANCER_",
    use_env=True
)
```

## Type Checker

```python
from imagen_video_enhancer_ai.utils.type_checker import TypeChecker

# Check type
is_valid = TypeChecker.check_type(value, List[str])

# Validate function args
is_valid, error = TypeChecker.validate_function_args(func, *args, **kwargs)
```

## Best Practices

1. **Use ResponseBuilder** for consistent API responses
2. **Use error_context** for better error tracking
3. **Use transformers** for data manipulation
4. **Use serialization** for data persistence
5. **Use type checker** for runtime validation

## Examples

### Complete API Endpoint

```python
from fastapi import APIRouter
from imagen_video_enhancer_ai.utils.response_builder import ResponseBuilder
from imagen_video_enhancer_ai.utils.error_context import error_context

router = APIRouter()

@router.get("/items")
async def get_items():
    try:
        with error_context(endpoint="get_items") as ctx:
            items = await fetch_items()
            ctx.add("count", len(items))
            
            return ResponseBuilder.list_response(
                items=items,
                count=len(items)
            )
    except Exception as e:
        return ResponseBuilder.error(
            error=e,
            code="FETCH_ERROR"
        )
```




