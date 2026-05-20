# 🚀 Guía de Inicio Rápido - TruthGPT Test Suite

## Instalación Rápida

```bash
# 1. Instalar dependencias
pip install -r requirements-test.txt

# 2. (Opcional) Instalar pre-commit hooks
pip install pre-commit
pre-commit install

# 3. Validar estructura
python validate_structure.py
```

## Ejecutar Tests

### Opción 1: Usando el script (Recomendado)
```bash
# Todos los tests
python run_tests.py all

# Tests unitarios
python run_tests.py unit

# Analizador específico
python run_tests.py performance
```

### Opción 2: Usando Makefile
```bash
# Ver todos los comandos disponibles
make help

# Ejecutar tests
make test

# Tests unitarios
make test-unit

# Con cobertura
make test-coverage
```

### Opción 3: Directamente con pytest
```bash
# Todos los tests
pytest core/ -v

# Tests específicos
pytest core/unit/ -v
pytest analyzers/performance/ -v
```

## Comandos Útiles

### Validación
```bash
# Validar estructura
python validate_structure.py
make validate
```

### Limpieza
```bash
# Limpiar archivos temporales
make clean
```

### Estadísticas
```bash
# Ver estadísticas
make stats

# Listar tests
make list

# Listar analizadores
make list-analyzers
```

### Formateo y Linting
```bash
# Formatear código
make format

# Verificar formato
make format-check

# Ejecutar linter
make lint
```

## Crear Nuevos Tests

### 1. Usar Template
```bash
# Copiar template
cp templates/test_template.py core/unit/test_mi_nuevo_test.py

# Editar y personalizar
```

### 2. Estructura Básica
```python
import unittest
from core.fixtures.test_utils import create_test_model
from core.fixtures.test_helpers import performance_test

class TestMiNuevoTest(unittest.TestCase):
    def test_ejemplo(self):
        # Tu test aquí
        self.assertTrue(True)
```

## Crear Nuevos Analizadores

### 1. Usar Template
```bash
# Copiar template
cp templates/analyzer_template.py analyzers/mi_categoria/mi_analizador.py

# Editar y personalizar
```

### 2. Estructura Básica
```python
from pathlib import Path
from typing import Dict, List

class MiAnalizador:
    def __init__(self, project_root: Path):
        self.project_root = project_root
    
    def analyze(self, lookback_days: int = 30) -> Dict:
        # Tu análisis aquí
        return {'result': 'data'}
```

## Estructura de Carpetas

```
tests/
├── core/
│   ├── unit/          # Tests unitarios
│   ├── integration/   # Tests de integración
│   └── fixtures/      # Utilidades compartidas
├── analyzers/         # Analizadores por categoría
├── systems/           # Sistemas y servicios
├── reporters/         # Módulos de reportes
├── exporters/         # Utilidades de exportación
└── utilities/         # Utilidades varias
```

## Importar Utilidades

```python
# Desde core
from core import create_test_model, assert_model_valid
from core import retry_on_failure, performance_test

# Desde analyzers
from analyzers.performance import PerformanceAnalyzer
from analyzers.quality import QualityAnalyzer

# Desde systems
from systems import PredictionSystem
```

## Troubleshooting

### Error: Module not found
```bash
# Asegúrate de estar en el directorio correcto
cd tests
python run_tests.py all
```

### Error: Dependencies missing
```bash
# Instalar dependencias
pip install -r requirements-test.txt
```

### Tests muy lentos
```bash
# Ejecutar solo tests rápidos
make test-fast

# O excluir tests marcados como slow
pytest core/ -v -m "not slow"
```

## Recursos Adicionales

- **README.md**: Documentación completa
- **INDEX.md**: Índice de todos los archivos
- **Makefile**: Comandos útiles
- **templates/**: Templates para nuevos tests y analizadores

## Siguiente Paso

Lee el [README.md](README.md) para documentación completa y mejores prácticas.

