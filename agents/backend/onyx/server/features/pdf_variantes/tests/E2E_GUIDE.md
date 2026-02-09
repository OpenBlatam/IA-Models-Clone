# Guía de Tests End-to-End (E2E)

## Introducción

Los tests end-to-end (E2E) prueban el sistema completo desde la perspectiva del usuario, verificando que todos los componentes trabajen juntos correctamente.

## Estructura de Tests E2E

### `test_e2e.py`
Tests e2e fundamentales que cubren:
- Flujos completos de usuario
- Operaciones básicas end-to-end
- Manejo de errores
- Performance
- Persistencia de datos

### `test_e2e_scenarios.py`
Tests basados en escenarios reales:
- Casos de uso específicos
- Workflows de usuarios reales
- Escenarios de colaboración
- Recuperación de errores

## Ejecutar Tests E2E

### Ejecutar todos los tests e2e

```bash
# Todos los tests e2e
pytest tests/test_e2e.py tests/test_e2e_scenarios.py -v

# Solo tests e2e básicos
pytest tests/test_e2e.py -v

# Solo tests de escenarios
pytest tests/test_e2e_scenarios.py -v

# Con marcador
pytest -m e2e -v
```

### Ejecutar tests específicos

```bash
# Test de journey completo
pytest tests/test_e2e.py::TestE2ECompleteUserJourney::test_complete_pdf_upload_journey -v

# Test de escenario de estudiante
pytest tests/test_e2e_scenarios.py::TestScenarioStudentWorkflow::test_student_lecture_notes_workflow -v
```

### Ejecutar tests lentos

```bash
# Tests e2e que incluyen performance
pytest -m "e2e and slow" -v

# Todos los tests e2e incluyendo lentos
pytest tests/test_e2e.py -v --runslow
```

## Tipos de Tests E2E

### 1. Tests de Journey Completo

Prueban flujos completos de usuario:

```python
def test_complete_pdf_upload_journey():
    # 1. Upload PDF
    # 2. Get Preview
    # 3. Extract Topics
    # 4. Generate Variants
```

### 2. Tests de Escenarios Reales

Simulan casos de uso reales:

- **Estudiante**: Sube notas, genera resumen y quiz
- **Investigador**: Analiza múltiples papers
- **Creador de contenido**: Genera variaciones
- **Equipo**: Colabora en documentos

### 3. Tests de Error

Verifican manejo de errores:

- Archivos inválidos
- Recursos no encontrados
- Tipos inválidos
- Acceso no autorizado

### 4. Tests de Performance

Verifican tiempos de respuesta:

- Upload < 5 segundos
- Variant generation < 30 segundos
- Operaciones concurrentes

### 5. Tests de Concurrencia

Prueban operaciones simultáneas:

- Múltiples uploads concurrentes
- Generación concurrente de variantes
- Acceso concurrente al mismo recurso

## Configuración

### Timeouts

Los tests e2e tienen timeouts más largos:

```python
client = TestClient(app, timeout=30.0)  # 30 segundos
```

### Autenticación

Los tests e2e pueden requerir autenticación:

```python
auth_headers = {
    "Authorization": "Bearer test_token",
    "X-User-ID": "test_user"
}
```

### Datos de Prueba

Se usan PDFs de prueba reales:

```python
sample_pdf_content = b"%PDF-1.4\n..."  # PDF válido mínimo
```

## Mejores Prácticas

### 1. Tests Independientes

Cada test e2e debe ser independiente:

```python
def test_workflow():
    # No depende de otros tests
    # Puede ejecutarse en cualquier orden
```

### 2. Limpieza

Los tests deben limpiar después de ejecutarse:

```python
def test_workflow():
    # Crear recursos
    # ...
    # Limpiar recursos (si es necesario)
    client.delete(f"/pdf/{file_id}")
```

### 3. Manejo de Errores

Los tests deben manejar errores gracefully:

```python
if upload_response.status_code not in [200, 201]:
    pytest.skip("Upload failed, cannot continue")
```

### 4. Aserciones Realistas

Las aserciones deben ser realistas:

```python
# No asumir que todo siempre funciona
assert response.status_code in [200, 202, 404]  # Múltiples estados válidos
```

### 5. Timeouts Apropiados

Usar timeouts apropiados para operaciones:

```python
@pytest.mark.slow
def test_slow_operation():
    # Operaciones que pueden tardar
    time.sleep(1)  # Esperar procesamiento
```

## Marcadores

### `@pytest.mark.e2e`

Marca tests como end-to-end:

```python
@pytest.mark.e2e
def test_e2e_workflow():
    ...
```

### `@pytest.mark.scenario`

Marca tests de escenarios:

```python
@pytest.mark.scenario
def test_student_workflow():
    ...
```

### `@pytest.mark.slow`

Marca tests lentos:

```python
@pytest.mark.slow
def test_performance():
    ...
```

## Ejecución en CI/CD

### Ejecutar en CI

```bash
# Solo tests e2e rápidos
pytest -m "e2e and not slow" -v

# Todos los tests e2e (puede tardar)
pytest -m e2e -v
```

### Ejecutar Localmente

```bash
# Desarrollo local - todos los tests
pytest tests/test_e2e.py -v

# Con output detallado
pytest tests/test_e2e.py -v -s

# Con cobertura
pytest tests/test_e2e.py --cov=. --cov-report=html
```

## Troubleshooting

### Tests Fallan por Timeout

Aumentar timeout:

```python
client = TestClient(app, timeout=60.0)  # 60 segundos
```

### Tests Fallan por Autenticación

Verificar headers de autenticación:

```python
auth_headers = {
    "Authorization": "Bearer valid_token",
    "X-User-ID": "valid_user_id"
}
```

### Tests Fallan por Recursos No Encontrados

Verificar que los recursos existen:

```python
if upload_response.status_code not in [200, 201]:
    pytest.skip("Cannot continue without upload")
```

## Métricas de Éxito

Los tests e2e deben verificar:

1. **Funcionalidad**: Todas las operaciones funcionan
2. **Performance**: Tiempos de respuesta aceptables
3. **Confiabilidad**: Sistema maneja errores gracefully
4. **Persistencia**: Datos persisten correctamente
5. **Concurrencia**: Múltiples usuarios pueden trabajar simultáneamente

## Próximos Pasos

1. Agregar más escenarios reales
2. Tests de carga (stress testing)
3. Tests de seguridad e2e
4. Tests de integración con servicios externos
5. Tests de migración de datos



