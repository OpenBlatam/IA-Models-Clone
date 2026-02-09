# 📑 MOEA Project - Complete Index

Índice completo de todos los archivos y recursos relacionados con el proyecto MOEA.

## 🚀 Inicio Rápido

**¿Quieres generar el proyecto MOEA ahora mismo?**

### Windows:
```bash
generate_moea.bat
```

### Linux/Mac:
```bash
chmod +x generate_moea.sh && ./generate_moea.sh
```

### Python Directo:
```bash
python generate_moea_direct.py
```

---

## 📚 Documentación

### Guías Principales
1. **[MOEA_QUICK_START.md](MOEA_QUICK_START.md)** ⚡
   - Guía rápida de inicio
   - Instalación y ejecución
   - Troubleshooting básico

2. **[GENERATE_MOEA.md](GENERATE_MOEA.md)** 📖
   - Guía detallada de generación
   - Múltiples métodos de generación
   - Configuración avanzada

3. **[MOEA_SUMMARY.md](MOEA_SUMMARY.md)** 📋
   - Resumen completo del proyecto
   - Estructura de archivos
   - Características principales

4. **[MOEA_INDEX.md](MOEA_INDEX.md)** 📑
   - Este archivo (índice completo)

---

## 🛠️ Scripts de Generación

### Scripts Automáticos
- **`generate_moea.bat`** - Script Windows (todo-en-uno)
- **`generate_moea.sh`** - Script Linux/Mac (todo-en-uno)

### Scripts Python
- **`generate_moea.py`** - Generación vía API (mejorado)
  - Espera automática del servidor
  - Monitoreo de estado
  - Manejo de errores mejorado

- **`generate_moea_direct.py`** - Generación directa
  - No requiere servidor API
  - Uso directo del generador

### Scripts de Utilidad
- **`verify_moea_project.py`** - Verificación de proyecto
  - Valida estructura de directorios
  - Verifica archivos requeridos
  - Muestra información del proyecto

---

## 💻 Ejemplos de Código

### **`moea_example_usage.py`** - Ejemplos Completos
Incluye 6 ejemplos prácticos:

1. **NSGA-II Optimization**
   - Optimización básica con NSGA-II
   - Configuración de problema ZDT1
   - Obtención de métricas

2. **Compare Algorithms**
   - Comparación de NSGA-II, NSGA-III, MOEA/D, SPEA2
   - Tabla comparativa de resultados
   - Análisis de performance

3. **Batch Optimization**
   - Optimización de múltiples problemas
   - Procesamiento en lote
   - Resultados agregados

4. **Get Pareto Front**
   - Obtención de frente de Pareto
   - Visualización de soluciones
   - Exportación de datos

5. **Export Results**
   - Exportación a JSON
   - Guardado de archivos
   - Múltiples formatos

6. **Real-time Optimization**
   - Uso de WebSocket
   - Actualizaciones en tiempo real
   - Monitoreo de progreso

---

## ⚙️ Configuración

### **`project_queue.json`** - Cola de Proyectos
- Proyecto MOEA pre-configurado
- Se procesa automáticamente al iniciar servidor
- Prioridad: 5 (alta)

```json
{
  "queue": [
    {
      "id": "moea_project_20250101_000000",
      "description": "A Multi-Objective Evolutionary Algorithm...",
      "project_name": "moea_optimization_system",
      "author": "Blatam Academy",
      "priority": 5
    }
  ]
}
```

---

## 📁 Estructura del Proyecto Generado

```
generated_projects/moea_optimization_system/
│
├── backend/                    # Backend FastAPI
│   ├── app/
│   │   ├── api/v1/            # Endpoints REST
│   │   ├── core/
│   │   │   ├── algorithms/    # NSGA-II, NSGA-III, MOEA/D, SPEA2
│   │   │   ├── metrics/       # Hypervolume, IGD, GD
│   │   │   └── problems/      # ZDT, DTLZ
│   │   ├── models/            # Modelos de datos
│   │   ├── services/          # Servicios de negocio
│   │   └── utils/             # Utilidades
│   ├── main.py
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/                   # Frontend React + TypeScript
│   ├── src/
│   │   ├── components/        # Componentes React
│   │   │   ├── ParetoFront.tsx
│   │   │   ├── MetricsChart.tsx
│   │   │   ├── AlgorithmComparison.tsx
│   │   │   └── ParameterTuning.tsx
│   │   ├── pages/             # Páginas
│   │   ├── services/          # API client
│   │   └── utils/             # Utilidades
│   ├── package.json
│   ├── vite.config.ts
│   └── Dockerfile
│
├── docker-compose.yml         # Orquestación Docker
├── README.md                   # Documentación del proyecto
└── project_info.json          # Metadata del proyecto
```

---

## 🎯 Características del Proyecto

### Algoritmos MOEA
- ✅ NSGA-II
- ✅ NSGA-III
- ✅ MOEA/D
- ✅ SPEA2

### Métricas de Performance
- ✅ Hypervolume (HV)
- ✅ Inverted Generational Distance (IGD)
- ✅ Generational Distance (GD)

### Problemas de Prueba
- ✅ ZDT1, ZDT2, ZDT3, ZDT4, ZDT6
- ✅ DTLZ1-7

### Funcionalidades
- ✅ Optimización en tiempo real
- ✅ Procesamiento en batch
- ✅ Visualización de Pareto fronts
- ✅ Comparación de algoritmos
- ✅ Exportación de resultados
- ✅ WebSocket para actualizaciones
- ✅ API REST completa

---

## 🔄 Flujo de Trabajo Recomendado

### 1. Generación
```bash
# Opción A: Script automático
generate_moea.bat  # o .sh en Linux/Mac

# Opción B: Python directo
python generate_moea_direct.py

# Opción C: API Server
python main.py  # Terminal 1
python generate_moea.py  # Terminal 2
```

### 2. Verificación
```bash
python verify_moea_project.py
```

### 3. Instalación
```bash
# Backend
cd generated_projects/moea_optimization_system/backend
pip install -r requirements.txt

# Frontend
cd ../frontend
npm install
```

### 4. Ejecución
```bash
# Backend (Terminal 1)
cd backend
uvicorn app.main:app --reload

# Frontend (Terminal 2)
cd frontend
npm run dev
```

### 5. Pruebas
```bash
# Ejecutar ejemplos
python moea_example_usage.py

# O usar la API directamente
curl http://localhost:8000/docs
```

---

## 📊 API Endpoints

### Optimización
- `POST /api/v1/moea/optimize` - Ejecutar optimización
- `POST /api/v1/moea/batch` - Optimización en batch

### Resultados
- `GET /api/v1/moea/pareto/{project_id}` - Obtener Pareto front
- `GET /api/v1/moea/metrics/{project_id}` - Obtener métricas
- `GET /api/v1/moea/export/{project_id}` - Exportar resultados

### WebSocket
- `WS /ws/moea/{project_id}` - Actualizaciones en tiempo real

### Documentación
- `GET /docs` - Swagger UI
- `GET /redoc` - ReDoc

---

## 🧪 Testing

```bash
# Backend
cd backend
pytest

# Frontend
cd frontend
npm test
```

---

## 🐳 Docker

```bash
cd generated_projects/moea_optimization_system
docker-compose up -d
```

---

## 🔍 Troubleshooting

### Problemas Comunes

| Problema | Solución |
|----------|----------|
| Python no encontrado | Instalar Python 3.8+ y agregar al PATH |
| Módulos faltantes | `pip install -r requirements.txt` |
| Puerto ocupado | Cambiar puerto en `.env` |
| npm no encontrado | Instalar Node.js 16+ |
| Servidor no responde | Verificar que esté corriendo en puerto 8020 |

### Verificación Rápida

```bash
# Verificar Python
python --version

# Verificar servidor
curl http://localhost:8020/health

# Verificar proyecto generado
python verify_moea_project.py
```

---

## 📞 Recursos Adicionales

- **[README.md](README.md)** - Documentación completa del generador
- **[example_usage.py](example_usage.py)** - Ejemplos del generador
- **[COMPLETE_SYSTEM.md](COMPLETE_SYSTEM.md)** - Sistema completo

---

## ✨ Checklist de Generación

- [ ] Leer `MOEA_QUICK_START.md`
- [ ] Ejecutar script de generación
- [ ] Verificar con `verify_moea_project.py`
- [ ] Instalar dependencias (backend y frontend)
- [ ] Ejecutar servidores
- [ ] Probar API en `/docs`
- [ ] Ejecutar ejemplos en `moea_example_usage.py`
- [ ] Personalizar según necesidades

---

## 🎓 Aprendizaje

### Conceptos MOEA
- **Pareto Front**: Conjunto de soluciones no dominadas
- **Hypervolume**: Medida de calidad del frente de Pareto
- **IGD**: Distancia generacional invertida
- **GD**: Distancia generacional

### Algoritmos
- **NSGA-II**: Algoritmo de clasificación no dominada
- **NSGA-III**: Extensión para muchos objetivos
- **MOEA/D**: Descomposición de objetivos
- **SPEA2**: Algoritmo de fuerza de Pareto

---

**¡Todo listo para generar y usar el proyecto MOEA! 🚀**

Para empezar, ejecuta: `generate_moea.bat` (Windows) o `./generate_moea.sh` (Linux/Mac)

