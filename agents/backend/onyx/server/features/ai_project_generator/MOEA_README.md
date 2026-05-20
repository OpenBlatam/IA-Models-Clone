# 🚀 MOEA Project - README Completo

## ⚡ Generación Ultra-Rápida

```bash
# Método más simple (recomendado)
python quick_moea.py

# O usar el script automático
generate_moea.bat  # Windows
./generate_moea.sh  # Linux/Mac
```

## 📖 ¿Qué es MOEA?

**MOEA** = **Multi-Objective Evolutionary Algorithm**

Sistema completo para resolver problemas de optimización con múltiples objetivos conflictivos usando algoritmos evolutivos.

## 🎯 Características

### Algoritmos Incluidos
- ✅ **NSGA-II** - Non-dominated Sorting Genetic Algorithm II
- ✅ **NSGA-III** - Non-dominated Sorting Genetic Algorithm III  
- ✅ **MOEA/D** - Multi-Objective Evolutionary Algorithm based on Decomposition
- ✅ **SPEA2** - Strength Pareto Evolutionary Algorithm 2

### Métricas de Performance
- ✅ **Hypervolume (HV)** - Medida de calidad del frente de Pareto
- ✅ **IGD** - Inverted Generational Distance
- ✅ **GD** - Generational Distance

### Funcionalidades
- ✅ Optimización en tiempo real
- ✅ Procesamiento en batch
- ✅ Visualización de Pareto fronts (2D y 3D)
- ✅ Comparación de algoritmos
- ✅ Exportación de resultados (JSON, CSV, Excel)
- ✅ API REST completa
- ✅ WebSocket para actualizaciones en tiempo real
- ✅ Interfaz web interactiva

## 🏗️ Estructura del Proyecto

```
moea_optimization_system/
├── backend/          # FastAPI
│   ├── app/
│   │   ├── api/     # Endpoints REST
│   │   ├── core/    # Algoritmos y métricas
│   │   ├── models/  # Modelos de datos
│   │   └── services/# Lógica de negocio
│   └── main.py
│
├── frontend/        # React + TypeScript
│   ├── src/
│   │   ├── components/  # Componentes React
│   │   ├── pages/       # Páginas
│   │   └── services/    # API client
│   └── package.json
│
└── docker-compose.yml
```

## 🚀 Inicio Rápido

### 1. Generar Proyecto
```bash
python quick_moea.py
```

### 2. Instalar Dependencias

**Backend:**
```bash
cd generated_projects/moea_optimization_system/backend
pip install -r requirements.txt
```

**Frontend:**
```bash
cd generated_projects/moea_optimization_system/frontend
npm install
```

### 3. Ejecutar

**Backend (Terminal 1):**
```bash
cd backend
uvicorn app.main:app --reload
# http://localhost:8000
```

**Frontend (Terminal 2):**
```bash
cd frontend
npm run dev
# http://localhost:5173
```

### 4. Usar la API

```bash
# Documentación interactiva
http://localhost:8000/docs

# Health check
curl http://localhost:8000/health
```

## 💻 Ejemplo de Uso

```python
import requests

# Configurar problema
problem = {
    "name": "ZDT1",
    "objectives": ["minimize_f1", "minimize_f2"],
    "variables": [
        {"name": "x1", "min": 0, "max": 1},
        {"name": "x2", "min": 0, "max": 1}
    ]
}

# Configurar algoritmo
algorithm = {
    "algorithm": "nsga2",
    "population_size": 100,
    "generations": 50
}

# Ejecutar optimización
response = requests.post(
    "http://localhost:8000/api/v1/moea/optimize",
    json={"problem": problem, "algorithm": algorithm}
)

result = response.json()
print(f"Hypervolume: {result['metrics']['hypervolume']}")
print(f"Pareto solutions: {len(result['pareto_front'])}")
```

Ver `moea_example_usage.py` para más ejemplos.

## 📚 Documentación

- **[MOEA_QUICK_START.md](MOEA_QUICK_START.md)** - Guía rápida
- **[MOEA_INDEX.md](MOEA_INDEX.md)** - Índice completo
- **[MOEA_SUMMARY.md](MOEA_SUMMARY.md)** - Resumen
- **[GENERATE_MOEA.md](GENERATE_MOEA.md)** - Guía de generación

## 🧪 Testing

```bash
# Backend
cd backend && pytest

# Frontend
cd frontend && npm test
```

## 🐳 Docker

```bash
cd generated_projects/moea_optimization_system
docker-compose up -d
```

## 📊 API Endpoints

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/api/v1/moea/optimize` | POST | Ejecutar optimización |
| `/api/v1/moea/batch` | POST | Optimización en batch |
| `/api/v1/moea/pareto/{id}` | GET | Obtener Pareto front |
| `/api/v1/moea/metrics/{id}` | GET | Obtener métricas |
| `/api/v1/moea/export/{id}` | GET | Exportar resultados |
| `/ws/moea/{id}` | WS | WebSocket tiempo real |

## 🔧 Troubleshooting

### Python no encontrado
```bash
# Instalar Python 3.8+
# Agregar al PATH
python --version
```

### Módulos faltantes
```bash
pip install -r requirements.txt
```

### Puerto ocupado
```bash
# Cambiar en .env o configuración
API_PORT=8001
```

### npm no encontrado
```bash
# Instalar Node.js 16+
npm --version
```

## ✨ Características Avanzadas

- **Problemas de Prueba**: ZDT1-6, DTLZ1-7
- **Visualizaciones 3D**: Frentes de Pareto interactivos
- **Comparación**: Múltiples algoritmos lado a lado
- **Exportación**: JSON, CSV, Excel, PNG/SVG
- **WebSocket**: Actualizaciones en tiempo real
- **Batch Processing**: Múltiples problemas simultáneos

## 🎓 Conceptos Clave

### Pareto Front
Conjunto de soluciones no dominadas donde mejorar un objetivo empeora otro.

### Hypervolume
Medida de calidad que calcula el volumen del espacio dominado por el frente de Pareto.

### IGD (Inverted Generational Distance)
Distancia promedio desde el frente de referencia al frente obtenido.

### GD (Generational Distance)
Distancia promedio desde el frente obtenido al frente de referencia.

## 📞 Soporte

Para más información:
- Ver `MOEA_INDEX.md` para índice completo
- Ver `moea_example_usage.py` para ejemplos
- Consultar documentación del generador en `README.md`

---

**¡Listo para optimizar! 🚀**

```bash
python quick_moea.py
```

