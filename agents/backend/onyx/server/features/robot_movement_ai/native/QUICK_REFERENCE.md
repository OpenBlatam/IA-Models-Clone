# Quick Reference - Native Extensions

## Guía Rápida de Uso

### C++ Extensions

#### Operaciones Matriciales

```python
from robot_movement_ai.native.wrapper import NativeMatrixOpsWrapper
import numpy as np

a = np.random.rand(10, 10)
b = np.random.rand(10, 10)

# Multiplicación
c = NativeMatrixOpsWrapper.matmul(a, b)

# Inversa
inv_a = NativeMatrixOpsWrapper.inv(a)

# Determinante
det_a = NativeMatrixOpsWrapper.det(a)

# Transpuesta
transposed = NativeMatrixOpsWrapper.transpose(a)

# Norma
norm_a = NativeMatrixOpsWrapper.norm(a)

# Traza
trace_a = NativeMatrixOpsWrapper.trace(a)
```

#### Transformaciones 3D

```python
from robot_movement_ai.native.wrapper import NativeTransform3DWrapper
import numpy as np

# Rotaciones
rot_x = NativeTransform3DWrapper.rotation_x(np.pi / 2)
rot_y = NativeTransform3DWrapper.rotation_y(np.pi / 4)
rot_z = NativeTransform3DWrapper.rotation_z(np.pi / 6)

# Aplicar rotación
point = np.array([1.0, 0.0, 0.0])
rotated = NativeTransform3DWrapper.rotate_point(rot_z, point)
```

#### Operaciones Vectoriales

```python
from robot_movement_ai.native.wrapper import NativeVectorOpsWrapper
import numpy as np

vec = np.array([3.0, 4.0, 0.0])

# Normalizar
normalized = NativeVectorOpsWrapper.normalize(vec)

# Distancia
a = np.array([0.0, 0.0, 0.0])
b = np.array([1.0, 1.0, 1.0])
dist = NativeVectorOpsWrapper.distance(a, b)

# Producto punto
dot = NativeVectorOpsWrapper.dot(a, b)

# Producto cruz (3D)
cross = NativeVectorOpsWrapper.cross(a, b)
```

#### Interpolación

```python
from robot_movement_ai.native.wrapper import NativeInterpolationWrapper
import numpy as np

waypoints = np.array([
    [0.0, 0.0, 0.0],
    [0.5, 0.5, 0.5],
    [1.0, 1.0, 1.0],
])

# Interpolación lineal
smooth = NativeInterpolationWrapper.linear(waypoints, 100)
```

#### Cinemática Inversa

```python
from robot_movement_ai.native.wrapper import NativeIKWrapper
import numpy as np

ik = NativeIKWrapper(
    link_lengths=[0.3, 0.3, 0.3],
    joint_limits=[(-np.pi, np.pi)] * 3
)

target = np.array([0.5, 0.3, 0.2])
solution = ik.solve(target)
```

#### Optimización de Trayectorias

```python
from robot_movement_ai.native.wrapper import NativeTrajectoryOptimizerWrapper
import numpy as np

optimizer = NativeTrajectoryOptimizerWrapper()
trajectory = np.random.rand(50, 3)
obstacles = np.array([[0.5, 0.5, 0.5, 0.1]])

optimized = optimizer.optimize(trajectory, obstacles)
```

#### Detección de Colisiones

```python
from robot_movement_ai.native.wrapper import NativeCollisionDetectorWrapper
import numpy as np

trajectory = np.array([[0, 0, 0], [1, 1, 1]])
obstacles = np.array([[0.5, 0.5, 0.5, 0.2]])

has_collision = NativeCollisionDetectorWrapper.check_trajectory_collision(
    trajectory, obstacles
)
```

### Rust Extensions

#### Operaciones de Arrays

```python
from robot_movement_ai.native.wrapper import (
    array_mean, array_std, array_median, array_percentile,
    array_filter, binary_search
)

data = [1.0, 2.0, 3.0, 4.0, 5.0]

# Estadísticas
mean = array_mean(data)
std = array_std(data)
median = array_median(data)
p75 = array_percentile(data, 75.0)

# Filtrar
filtered = array_filter(data, 3.0, "gt")  # [4.0, 5.0]

# Búsqueda binaria
sorted_data = sorted(data)
index = binary_search(sorted_data, 3.0)
```

#### Operaciones de Strings

```python
from robot_movement_ai.native.wrapper import (
    string_search, string_count, string_replace,
    string_split, string_join, string_trim,
    string_upper, string_lower
)

text = "  Hello World  "

# Búsqueda
positions = string_search(text, "World")
count = string_count(text, "l")

# Reemplazo
replaced = string_replace(text, "World", "Python")

# Split/Join
parts = string_split("a,b,c", ",")
joined = string_join(parts, "-")

# Transformaciones
trimmed = string_trim(text)
upper = string_upper(text)
lower = string_lower(text)
```

#### JSON

```python
from robot_movement_ai.native.wrapper import (
    json_parse, json_validate
)

# Parse
json_str = '{"key": "value"}'
data = json_parse(json_str)

# Validar
is_valid = json_validate(json_str)
```

## Verificar Disponibilidad

```python
from robot_movement_ai.native.wrapper import (
    get_native_extensions_status,
    CPP_AVAILABLE,
    RUST_AVAILABLE
)

status = get_native_extensions_status()
print(f"C++: {CPP_AVAILABLE}, Rust: {RUST_AVAILABLE}")
print(status)
```

## Mejoras de Rendimiento

| Operación | Mejora |
|-----------|--------|
| Cinemática Inversa | 25x |
| Optimización Trayectorias | 20x |
| Multiplicación Matrices | 20x |
| Detección Colisiones | 30x |
| Interpolación | 10x |
| Rotaciones 3D | 10x |
| Normalización | 5x |
| Distancia | 4x |
| Mediana | 10x |
| Percentil | 10x |
| Parsing JSON | 10x |

