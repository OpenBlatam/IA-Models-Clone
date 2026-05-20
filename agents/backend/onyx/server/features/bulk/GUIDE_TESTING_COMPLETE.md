# 🧪 Guía Completa de Testing - API BUL

## 📚 Scripts de Prueba Disponibles

### 1. **test_api_responses.py** - Pruebas Básicas
Script mejorado con validaciones, colores y reportes detallados.

**Características:**
- ✅ Validación de todos los endpoints
- ✅ Colores para mejor legibilidad
- ✅ Barra de progreso visual
- ✅ Validaciones automáticas
- ✅ Resumen de resultados

**Uso:**
```bash
python test_api_responses.py
```

### 2. **test_api_advanced.py** - Pruebas Avanzadas
Script para pruebas más complejas incluyendo carga, WebSocket y más.

**Características:**
- ✅ Pruebas de carga
- ✅ Pruebas de requests concurrentes
- ✅ Pruebas de rate limiting
- ✅ Pruebas de WebSocket
- ✅ Exportación a JSON y CSV
- ✅ Métricas detalladas

**Uso:**
```bash
python test_api_advanced.py
```

## 🎯 Tipos de Pruebas

### Pruebas Básicas

1. **Root Endpoint** - Verificar información del sistema
2. **Health Check** - Estado de salud
3. **Stats** - Estadísticas del sistema
4. **Generar Documento** - Proceso completo
5. **Estado de Tarea** - Polling con progreso
6. **Obtener Documento** - Documento completo
7. **Listar Tareas** - Lista de tareas
8. **Listar Documentos** - Lista de documentos
9. **Validaciones** - Prueba de validación de campos

### Pruebas Avanzadas

1. **Prueba de Carga**
   - Múltiples requests por segundo
   - Duración configurable
   - Métricas de rendimiento

2. **Requests Concurrentes**
   - Múltiples requests simultáneos
   - Verificación de estabilidad

3. **Rate Limiting**
   - Verificar límite de 10 req/min
   - Detección de bloqueos

4. **WebSocket**
   - Conexión y mensajes
   - Actualizaciones en tiempo real

## 📊 Exportación de Resultados

### Formato JSON
```json
{
  "summary": {
    "total": 10,
    "passed": 9,
    "failed": 1,
    "success_rate": 90.0,
    "duration": 45.23,
    "timestamp": "2024-01-15T10:30:00"
  },
  "tests": [...],
  "errors": [...],
  "metrics": {...}
}
```

### Formato CSV
```csv
Test,Status,Duration (s),Timestamp,Details
Root Endpoint,PASS,0.12,2024-01-15T10:30:00,{}
Health Check,PASS,0.08,2024-01-15T10:30:01,{}
```

## 🚀 Ejecución Completa

### Opción 1: Pruebas Básicas
```bash
# Iniciar servidor
python api_frontend_ready.py

# En otra terminal
python test_api_responses.py
```

### Opción 2: Pruebas Avanzadas
```bash
# Iniciar servidor
python api_frontend_ready.py

# En otra terminal
python test_api_advanced.py
```

### Opción 3: Ambas
```bash
# Ejecutar básicas primero
python test_api_responses.py

# Luego avanzadas
python test_api_advanced.py
```

## 📈 Métricas Generadas

### Pruebas Básicas
- Total de pruebas
- Exitosas vs Fallidas
- Tasa de éxito
- Tiempo total
- Errores detallados

### Pruebas Avanzadas
- **Carga:**
  - Total de requests
  - Requests exitosos/fallidos
  - Tiempo promedio de respuesta
  - RPS (Requests Per Second)
  - Tiempo mínimo/máximo

- **Concurrentes:**
  - Número de requests simultáneos
  - Tasa de éxito
  - Tiempo total

- **Rate Limiting:**
  - Requests enviados
  - Detección de límite
  - Status 429 recibido

## 🔍 Interpretación de Resultados

### Tasa de Éxito
- **>95%**: Excelente ✅
- **90-95%**: Bueno ⚠️
- **<90%**: Revisar errores ❌

### Tiempo de Respuesta
- **<100ms**: Excelente ✅
- **100-500ms**: Bueno ⚠️
- **>500ms**: Revisar optimización ❌

### Pruebas de Carga
- **RPS estable**: Sistema funcionando bien ✅
- **Caída de RPS**: Posible cuello de botella ⚠️
- **Muchos errores**: Sistema sobrecargado ❌

## 🛠️ Troubleshooting

### Error: "No se puede conectar al servidor"
- Verificar que el servidor esté corriendo
- Verificar URL y puerto
- Verificar firewall

### Error: "Timeout"
- Aumentar TIMEOUT en el script
- Verificar rendimiento del servidor
- Reducir carga de pruebas

### Error: "Rate limiting no funciona"
- Verificar configuración de slowapi
- Verificar que el límite esté activo
- Revisar logs del servidor

## 📝 Personalización

### Modificar Timeout
```python
TIMEOUT = 60  # 60 segundos
```

### Modificar Pruebas de Carga
```python
test_load(
    duration_seconds=30,  # 30 segundos
    requests_per_second=5  # 5 RPS
)
```

### Modificar Requests Concurrentes
```python
test_concurrent_requests(count=10)  # 10 requests
```

## 🎯 Mejores Prácticas

1. **Ejecutar pruebas básicas primero**
   - Verificar que todo funcione
   - Identificar problemas básicos

2. **Luego pruebas avanzadas**
   - Validar rendimiento
   - Probar límites

3. **Revisar resultados exportados**
   - Analizar métricas
   - Identificar patrones

4. **Ejecutar regularmente**
   - En CI/CD
   - Antes de releases
   - Después de cambios

## 📦 Dependencias

### Básicas
- `requests` - Peticiones HTTP
- `colorama` (opcional) - Colores

### Avanzadas
- `requests` - Peticiones HTTP
- `websockets` - Pruebas WebSocket
- `colorama` (opcional) - Colores

Instalar:
```bash
pip install requests websockets colorama
```

## 🚀 Integración CI/CD

### GitHub Actions
```yaml
- name: Test API
  run: |
    python api_frontend_ready.py &
    sleep 5
    python test_api_responses.py
```

### GitLab CI
```yaml
test:
  script:
    - python api_frontend_ready.py &
    - sleep 5
    - python test_api_responses.py
```

---

**Estado**: ✅ **Guía completa de testing lista**
































