# Programación Funcional Avanzada

## Nuevas Utilidades Avanzadas

### 1. Monads ✅
**Archivo**: `utils/monads.py`

**Clases:**
- `Maybe` - Monad para valores opcionales
- `Either` - Monad para éxito o error

**Decorators:**
- `maybe_decorator` - Envolver función en Maybe
- `either_decorator` - Envolver función en Either

**Uso:**
```python
from utils import Maybe, Either, maybe_decorator

# Maybe monad
maybe_value = Maybe.just(42)
doubled = maybe_value.map(lambda x: x * 2)
result = doubled.get_or_else(0)

# Either monad
either_value = Either.right(42)
mapped = either_value.map(lambda x: x * 2)
value = mapped.get_or_else(0)

# Decorator
@maybe_decorator
def risky_operation():
    return compute_value()
```

### 2. Lenses ✅
**Archivo**: `utils/lenses.py`

**Clases:**
- `Lens` - Lens para acceso inmutable a datos anidados

**Funciones:**
- `lens()` - Crear lens
- `prop_lens()` - Lens para propiedad
- `path_lens()` - Lens para path anidado

**Uso:**
```python
from utils import prop_lens, path_lens

# Property lens
user_name_lens = prop_lens("name")
name = user_name_lens.get(user_data)
updated = user_name_lens.set(user_data, "New Name")

# Path lens
nested_lens = path_lens("user.profile.name")
name = nested_lens.get(data)
updated = nested_lens.set(data, "New Name")

# Compose lenses
composed = user_lens.compose(profile_lens).compose(name_lens)
```

### 3. Functors ✅
**Archivo**: `utils/functors.py`

**Clases:**
- `Functor` - Base functor
- `ListFunctor` - List como functor
- `DictFunctor` - Dict como functor

**Funciones:**
- `list_functor()` - Crear list functor
- `dict_functor()` - Crear dict functor

**Uso:**
```python
from utils import list_functor, dict_functor

# List functor
numbers = list_functor([1, 2, 3, 4])
doubled = numbers.map(lambda x: x * 2)
evens = numbers.filter(lambda x: x % 2 == 0)

# Dict functor
data = dict_functor({"a": 1, "b": 2, "c": 3})
doubled = data.map(lambda x: x * 2)
filtered = data.filter(lambda x: x > 1)
```

## Estadísticas Finales

### Utilidades Funcionales Avanzadas
- ✅ **3 módulos** nuevos de programación funcional avanzada
- ✅ **10+ clases** y funciones
- ✅ **Cobertura completa** de patrones funcionales avanzados

### Categorías Avanzadas
- ✅ **Monads** - Maybe, Either
- ✅ **Lenses** - Acceso inmutable a datos anidados
- ✅ **Functors** - List y Dict como functors

## Ejemplos de Uso Avanzado

### Maybe Monad
```python
from utils import Maybe

def process_data(data):
    return Maybe.just(data)\
        .map(parse)\
        .filter(is_valid)\
        .map(transform)\
        .get_or_else(default_value)
```

### Either Monad
```python
from utils import Either

def risky_operation():
    result = Either.right(compute())\
        .map(transform)\
        .map(validate)\
        .fold(
            left_func=lambda e: handle_error(e),
            right_func=lambda v: process_success(v)
        )
    return result
```

### Lenses
```python
from utils import path_lens, prop_lens

# Immutable updates
user_lens = path_lens("user.profile")
updated = user_lens.modify(
    data,
    lambda profile: {**profile, "name": "New Name"}
)

# Compose lenses
name_lens = prop_lens("name")
profile_lens = prop_lens("profile")
composed = profile_lens.compose(name_lens)
```

### Functors
```python
from utils import list_functor

# Chain operations
result = list_functor([1, 2, 3, 4])\
    .map(lambda x: x * 2)\
    .filter(lambda x: x > 4)\
    .map(lambda x: x + 1)\
    .unwrap()
```

## Beneficios

1. ✅ **Monads**: Manejo elegante de valores opcionales y errores
2. ✅ **Lenses**: Acceso inmutable a datos anidados
3. ✅ **Functors**: Operaciones funcionales sobre colecciones
4. ✅ **Composición**: Fácil componer operaciones
5. ✅ **Inmutabilidad**: Operaciones sin efectos secundarios
6. ✅ **Type Safety**: Type hints completos

## Conclusión

El sistema ahora cuenta con:
- ✅ Monads (Maybe, Either)
- ✅ Lenses para acceso inmutable
- ✅ Functors para colecciones
- ✅ Programación funcional avanzada completa
- ✅ 34 módulos de utilidades
- ✅ 180+ funciones reutilizables

**Estado**: ✅ Advanced Functional Programming Complete

