# 🧪 Cómo Ejecutar los Tests de la API

## ✅ Tests Disponibles

### 1. Test Rápido (Verificación Básica)
```bash
cd truthgpt_api
python test_api_quick.py
```

### 2. Tests Unitarios (NO requieren servidor)
```bash
cd truthgpt_api
pytest tests/test_unit_models.py -v
pytest tests/test_unit_utils.py -v
pytest tests/test_unit_api_helpers.py -v

# Todos los unitarios
pytest tests/test_unit_*.py -v
```

### 3. Tests de Endpoints (Requieren servidor)
```bash
# Terminal 1: Iniciar servidor
cd truthgpt_api
python start_server.py

# Terminal 2: Ejecutar tests
cd truthgpt_api
pytest tests/test_api_endpoints.py -v
```

### 4. Tests de Layers
```bash
pytest tests/test_layers.py -v
```

### 5. Tests de Optimizers
```bash
pytest tests/test_optimizers.py -v
```

### 6. Tests de Losses
```bash
pytest tests/test_losses.py -v
```

### 7. Tests de Integración
```bash
pytest tests/test_integration.py -v
```

### 8. Tests de Validación
```bash
pytest tests/test_validation.py -v
```

### 9. Tests de Rendimiento
```bash
pytest tests/test_performance.py -v -s
```

### 10. Tests E2E (End-to-End)
```bash
python run_e2e_tests.py
# O directamente:
pytest tests/test_e2e.py -v -s
```

### 11. TODOS los Tests
```bash
# Script interactivo que ejecuta todo
python run_all_tests.py

# O directamente con pytest
pytest tests/ -v
```

## 📋 Checklist Rápido

### Antes de Ejecutar Tests

1. ✅ Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

2. ✅ Para tests que requieren servidor:
   ```bash
   python start_server.py
   ```

3. ✅ Verificar que el servidor responde:
   ```bash
   curl http://localhost:8000/health
   # O en navegador: http://localhost:8000/health
   ```

### Ejecutar Tests Específicos

```bash
# Solo un test específico
pytest tests/test_api_endpoints.py::TestHealthCheck::test_health_endpoint -v

# Solo una clase de tests
pytest tests/test_api_endpoints.py::TestModelCreation -v

# Con coverage
pytest tests/ --cov=. --cov-report=html
```

## 🚀 Ejecución Rápida

### Opción 1: Test Rápido (Recomendado para empezar)
```bash
cd truthgpt_api
python test_api_quick.py
```

### Opción 2: Test Rápido desde Navegador
1. Inicia el servidor: `python start_server.py`
2. Abre: http://localhost:8000/docs
3. Prueba los endpoints directamente desde la interfaz

### Opción 3: Todos los Tests
```bash
cd truthgpt_api
python run_all_tests.py
```

## 📊 Ver Resultados

### Reporte HTML de Coverage
```bash
pytest tests/ --cov=. --cov-report=html
# Luego abre: htmlcov/index.html
```

### Reporte de Playwright (Frontend)
```bash
cd truthgpt-model-builder
npm run test:e2e
# Ver reporte: npx playwright show-report
```

## ⚠️ Solución de Problemas

### Error: "Module not found"
```bash
pip install -r requirements.txt
```

### Error: "Server not running"
```bash
python start_server.py
```

### Error: "Port already in use"
```bash
python start_server.py --port 8001
```

## ✅ Verificación Rápida

Si todo está bien, deberías ver:
- ✅ Health check responde
- ✅ Puedes crear modelos
- ✅ Puedes listar modelos
- ✅ Tests unitarios pasan sin servidor
- ✅ Tests de integración pasan con servidor











