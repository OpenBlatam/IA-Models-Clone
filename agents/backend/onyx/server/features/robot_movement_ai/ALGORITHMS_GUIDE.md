# Guía de Algoritmos - Robot Movement AI

## 🧠 Algoritmos de Optimización Disponibles

### 1. Proximal Policy Optimization (PPO)

**Descripción**: Algoritmo de política que optimiza directamente la política de movimiento.

**Características:**
- ✅ Clipping de políticas para estabilidad
- ✅ Actualización conservadora
- ✅ Buen rendimiento en espacios continuos

**Uso:**
```python
optimizer.rl_algorithm = "ppo"
trajectory = optimizer.optimize_trajectory(start, goal, obstacles)
```

**Cuándo usar:**
- Cuando se necesita aprendizaje continuo
- Para políticas que requieren estabilidad
- Espacios de acción continuos

### 2. Deep Q-Network (DQN)

**Descripción**: Aprende función de valor Q que estima calidad de acciones.

**Características:**
- ✅ Estimación de Q-values
- ✅ Selección de mejor acción
- ✅ Bueno para espacios discretos

**Uso:**
```python
optimizer.rl_algorithm = "dqn"
trajectory = optimizer.optimize_trajectory(start, goal, obstacles)
```

**Cuándo usar:**
- Espacios de acción discretos
- Cuando se necesita estimación de valor
- Para comparar múltiples acciones

### 3. A* (A-Star)

**Descripción**: Algoritmo de búsqueda en grafos que encuentra el camino más corto.

**Características:**
- ✅ Garantiza camino óptimo
- ✅ Considera heurística (distancia al objetivo)
- ✅ Eficiente en espacios discretizados

**Uso:**
```python
trajectory = optimizer.optimize_with_astar(
    start, goal, obstacles,
    grid_resolution=0.05  # 5cm
)
```

**Cuándo usar:**
- Espacios con muchos obstáculos
- Cuando se necesita camino garantizado
- Para planificación inicial rápida

**Parámetros:**
- `grid_resolution`: Resolución del grid (más pequeño = más preciso pero más lento)

### 4. RRT (Rapidly-exploring Random Tree)

**Descripción**: Algoritmo probabilístico que explora el espacio eficientemente.

**Características:**
- ✅ Bueno para espacios complejos
- ✅ No requiere discretización completa
- ✅ Encuentra caminos en espacios con muchos obstáculos

**Uso:**
```python
trajectory = optimizer.optimize_with_rrt(
    start, goal, obstacles,
    max_iterations=1000,
    step_size=0.1  # 10cm
)
```

**Cuándo usar:**
- Espacios muy complejos
- Cuando A* es demasiado lento
- Para exploración inicial del espacio

**Parámetros:**
- `max_iterations`: Máximo de iteraciones (más = mejor pero más lento)
- `step_size`: Tamaño del paso (más pequeño = más preciso)

### 5. Optimización Multi-objetivo (NSGA-II simplificado)

**Descripción**: Optimiza múltiples objetivos simultáneamente.

**Objetivos:**
- **Tiempo**: Minimizar duración
- **Energía**: Minimizar consumo
- **Suavidad**: Maximizar suavidad
- **Seguridad**: Maximizar distancia a obstáculos

**Uso:**
```python
trajectory = optimizer.optimize_multi_objective(
    start, goal, obstacles,
    objectives={
        "time": 0.3,      # Priorizar tiempo
        "energy": 0.2,    # Menor prioridad a energía
        "smoothness": 0.3, # Priorizar suavidad
        "safety": 0.2     # Menor prioridad a seguridad
    }
)
```

**Cuándo usar:**
- Cuando se necesitan balancear múltiples objetivos
- Para optimización general
- Cuando no hay un objetivo dominante

### 6. Optimización Heurística (Fallback)

**Descripción**: Métodos heurísticos cuando no hay modelo RL.

**Características:**
- ✅ Suavizado de trayectorias
- ✅ Evasión de obstáculos
- ✅ Optimización de energía
- ✅ Compensación de vibraciones

**Uso:**
```python
optimizer.rl_algorithm = "heuristic"
trajectory = optimizer.optimize_trajectory(start, goal, obstacles)
```

**Cuándo usar:**
- Cuando no hay modelo RL disponible
- Para casos simples
- Como fallback

## 📊 Comparación de Algoritmos

| Algoritmo | Velocidad | Precisión | Complejidad | Mejor Para |
|-----------|-----------|-----------|-------------|------------|
| PPO | Media | Alta | Alta | Aprendizaje continuo |
| DQN | Media | Alta | Alta | Espacios discretos |
| A* | Rápida | Muy Alta | Media | Obstáculos simples |
| RRT | Lenta | Media | Media | Espacios complejos |
| Multi-objetivo | Media | Alta | Media | Balance de objetivos |
| Heurístico | Muy Rápida | Media | Baja | Casos simples |

## 🎯 Selección de Algoritmo

### Por Tipo de Tarea

**Soldadura Robótica:**
- Precisión crítica → **A*** o **Multi-objetivo**
- Obstáculos complejos → **RRT**

**Pick and Place:**
- Velocidad importante → **Heurístico** o **A***
- Obstáculos dinámicos → **PPO** o **DQN**

**Ensamblaje:**
- Múltiples objetivos → **Multi-objetivo**
- Aprendizaje continuo → **PPO**

### Por Condiciones

**Muchos Obstáculos:**
- → **RRT** o **A***

**Espacio Simple:**
- → **Heurístico** o **A***

**Aprendizaje Requerido:**
- → **PPO** o **DQN**

**Balance de Objetivos:**
- → **Multi-objetivo**

## 🔧 Configuración Avanzada

### Combinar Algoritmos

```python
# Usar A* para planificación inicial
initial_trajectory = optimizer.optimize_with_astar(start, goal, obstacles)

# Refinar con PPO
optimizer.rl_algorithm = "ppo"
refined_trajectory = optimizer.optimize_trajectory(
    initial_trajectory[0],
    initial_trajectory[-1],
    obstacles
)
```

### Análisis de Trayectorias

```python
# Analizar trayectoria
analysis = optimizer.analyze_trajectory(trajectory)
print(f"Distance: {analysis['total_distance']:.2f}m")
print(f"Duration: {analysis['duration']:.2f}s")
print(f"Max velocity: {analysis['max_velocity']:.2f}m/s")
print(f"Smoothness: {analysis['smoothness']:.2f}")
```

### Exportar/Importar Trayectorias

```python
# Exportar
optimizer.export_trajectory(trajectory, "path.json", format="json")
optimizer.export_trajectory(trajectory, "path.csv", format="csv")

# Importar
trajectory = optimizer.import_trajectory("path.json")
```

## 📈 Mejores Prácticas

1. **Empezar Simple**: Usar heurístico o A* para casos básicos
2. **Escalar Según Necesidad**: Si hay problemas, probar RRT o RL
3. **Analizar Resultados**: Usar `analyze_trajectory()` para evaluar
4. **Ajustar Parámetros**: Experimentar con diferentes configuraciones
5. **Combinar Algoritmos**: Usar múltiples algoritmos en secuencia

## 🚀 Próximas Mejoras

- [ ] Implementación completa de NSGA-II
- [ ] Algoritmo RRT* (versión optimizada)
- [ ] Algoritmo PRM (Probabilistic Roadmap)
- [ ] Integración de algoritmos de aprendizaje por demostración
- [ ] Optimización basada en gradientes






