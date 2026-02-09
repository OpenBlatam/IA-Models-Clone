# Quick Start Guide - TruthGPT Tests

## 🚀 Inicio Rápido

### 1. Verificar Entorno
```bash
# Verificar Python
python --version  # Debe ser 3.7+

# Verificar PyTorch
python -c "import torch; print(torch.__version__)"
```

### 2. Instalar Dependencias
```bash
pip install torch numpy psutil
```

### 3. Ejecutar Tests
```bash
# Todos los tests
python run_unified_tests.py

# Tests específicos
python run_unified_tests.py core
python run_unified_tests.py performance
```

## 📋 Comandos Útiles

### Básicos
```bash
# Ver ayuda
python run_unified_tests.py help

# Listar categorías (con runner mejorado)
python run_tests_improved.py --list

# Test con verbose
python run_tests_improved.py core --verbose
```

### Avanzados
```bash
# Guardar reporte
python run_tests_improved.py all --save-report

# Failfast (parar en primer error)
python run_tests_improved.py integration --failfast

# Modo quiet
python run_tests_improved.py performance --quiet
```

## 🎯 Categorías Disponibles

| Categoría | Comando | Tests |
|-----------|---------|-------|
| Core | `core` | 13 |
| Optimization | `optimization` | 24 |
| Models | `models` | 18 |
| Training | `training` | 23 |
| Inference | `inference` | 26 |
| Monitoring | `monitoring` | 24 |
| Integration | `integration` | 9 |
| Edge Cases | `edge` | 18 |
| Performance | `performance` | 10 |
| Security | `security` | 10 |
| Compatibility | `compatibility` | 12 |

## ⚡ Ejemplos Rápidos

### Test Rápido
```bash
python run_unified_tests.py core
```

### Test Completo con Reporte
```bash
python run_tests_improved.py all --save-report
```

### Test de Rendimiento
```bash
python run_unified_tests.py performance
```

### Test de Seguridad
```bash
python run_unified_tests.py security
```

## 🔧 Troubleshooting

### Error: Python not found
- Instala Python desde python.org
- Asegúrate de agregar Python al PATH

### Error: Module not found
- Ejecuta desde el directorio `TruthGPT-main`
- Verifica que `core/` y `tests/` existan

### Error: CUDA tests skipped
- Normal si no tienes GPU
- Los tests se saltan automáticamente

## 📊 Resultados Esperados

### Éxito
```
🎉 All tests passed!
```

### Con Fallos
```
❌ Some tests failed!
```
(Ver reporte detallado para más información)

## 📚 Más Información

- Ver `tests/README.md` para detalles de tests
- Ver `TEST_SUMMARY.md` para resumen completo
- Ver `READY_TO_TEST.md` para checklist

---

**¡Listo para empezar!** 🚀








