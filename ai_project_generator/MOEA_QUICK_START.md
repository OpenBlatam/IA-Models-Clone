# 🚀 MOEA Project - Quick Start Guide

Guía rápida para generar y usar el proyecto MOEA (Multi-Objective Evolutionary Algorithm).

## ⚡ Generación Rápida

### Opción 1: Script Automático (Windows)
```bash
generate_moea.bat
```

### Opción 2: Script Automático (Linux/Mac)
```bash
chmod +x generate_moea.sh
./generate_moea.sh
```

### Opción 3: Python Directo
```bash
python generate_moea_direct.py
```

### Opción 4: API Server
```bash
# 1. Iniciar servidor
python main.py

# 2. En otra terminal, hacer request
python generate_moea.py
```

## 📋 Verificación

Después de generar, verifica que todo esté correcto:

```bash
python verify_moea_project.py
```

## 🎯 Características del Proyecto MOEA

El proyecto generado incluye:

### Backend (FastAPI)
- ✅ API REST para operaciones MOEA
- ✅ Implementaciones de algoritmos:
  - NSGA-II (Non-dominated Sorting Genetic Algorithm II)
  - NSGA-III (Non-dominated Sorting Genetic Algorithm III)
  - MOEA/D (Multi-Objective Evolutionary Algorithm based on Decomposition)
  - SPEA2 (Strength Pareto Evolutionary Algorithm 2)
- ✅ Cálculo de métricas de performance:
  - Hypervolume (HV)
  - Inverted Generational Distance (IGD)
  - Generational Distance (GD)
- ✅ Procesamiento en batch
- ✅ Optimización en tiempo real
- ✅ Exportación de resultados

### Frontend (React + TypeScript)
- ✅ Interfaz de ajuste de parámetros interactiva
- ✅ Visualización de frentes de Pareto
- ✅ Dashboard de métricas de performance
- ✅ Herramientas de comparación de algoritmos
- ✅ Monitoreo de optimización en tiempo real
- ✅ Interfaz de exportación de resultados

## 📦 Instalación

### Backend
```bash
cd generated_projects/moea_optimization_system/backend
pip install -r requirements.txt
```

### Frontend
```bash
cd generated_projects/moea_optimization_system/frontend
npm install
```

## 🏃 Ejecución

### Backend
```bash
cd generated_projects/moea_optimization_system/backend
uvicorn app.main:app --reload
```

El backend estará disponible en: `http://localhost:8000`

### Frontend
```bash
cd generated_projects/moea_optimization_system/frontend
npm run dev
```

El frontend estará disponible en: `http://localhost:5173`

## 📚 Documentación de la API

Una vez que el backend esté corriendo, visita:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 🔧 Configuración

### Variables de Entorno

Crea un archivo `.env` en el directorio `backend/`:

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True

# MOEA Configuration
MOEA_DEFAULT_POPULATION_SIZE=100
MOEA_DEFAULT_GENERATIONS=100
MOEA_DEFAULT_MUTATION_RATE=0.1
MOEA_DEFAULT_CROSSOVER_RATE=0.9
```

## 🧪 Testing

### Backend Tests
```bash
cd generated_projects/moea_optimization_system/backend
pytest
```

### Frontend Tests
```bash
cd generated_projects/moea_optimization_system/frontend
npm test
```

## 🐳 Docker

Si el proyecto incluye Docker:

```bash
cd generated_projects/moea_optimization_system
docker-compose up -d
```

## 📊 Uso Básico

### Ejemplo: Ejecutar NSGA-II

```python
import requests

# Configurar problema de optimización
problem_config = {
    "objectives": ["minimize_cost", "maximize_quality"],
    "variables": {
        "x1": {"min": 0, "max": 10},
        "x2": {"min": 0, "max": 10}
    },
    "constraints": []
}

# Configurar algoritmo
algorithm_config = {
    "algorithm": "nsga2",
    "population_size": 100,
    "generations": 50,
    "mutation_rate": 0.1,
    "crossover_rate": 0.9
}

# Ejecutar optimización
response = requests.post(
    "http://localhost:8000/api/v1/moea/optimize",
    json={
        "problem": problem_config,
        "algorithm": algorithm_config
    }
)

result = response.json()
print(f"Pareto Front: {result['pareto_front']}")
print(f"Hypervolume: {result['metrics']['hypervolume']}")
```

## 🎨 Visualización

El frontend incluye visualizaciones interactivas de:
- Frentes de Pareto 2D y 3D
- Evolución de métricas a lo largo de las generaciones
- Comparación de múltiples algoritmos
- Distribución de soluciones

## 🔍 Troubleshooting

### Error: Python no encontrado
- Instala Python 3.8+ desde [python.org](https://www.python.org/)
- Asegúrate de agregarlo al PATH

### Error: Módulos no encontrados
```bash
pip install -r requirements.txt
```

### Error: Puerto ocupado
- Cambia el puerto en `backend/.env` o `frontend/vite.config.ts`

### Error: npm no encontrado
- Instala Node.js desde [nodejs.org](https://nodejs.org/)

## 📞 Soporte

Para más información, consulta:
- `README.md` - Documentación completa del generador
- `GENERATE_MOEA.md` - Guía detallada de generación
- `example_usage.py` - Ejemplos de uso del generador

## 🎯 Próximos Pasos

1. ✅ Generar el proyecto
2. ✅ Verificar la estructura
3. ✅ Instalar dependencias
4. ✅ Ejecutar servidores
5. ✅ Probar la API
6. ✅ Personalizar algoritmos
7. ✅ Agregar nuevos problemas de optimización
8. ✅ Mejorar visualizaciones

---

**¡Listo para optimizar! 🚀**

