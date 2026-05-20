# 🛠️ MOEA Tools - Guía Completa de Herramientas

Guía completa de todas las herramientas disponibles para el proyecto MOEA.

## 📋 Índice de Herramientas

1. **[Generación](#generación)**
2. **[Configuración](#configuración)**
3. **[Testing](#testing)**
4. **[Benchmarking](#benchmarking)**
5. **[Visualización](#visualización)**
6. **[Verificación](#verificación)**

---

## 🚀 Generación

### `quick_moea.py` ⚡ (Recomendado)
**Generación rápida y simple**

```bash
python quick_moea.py
```

**Características:**
- ✅ Generación directa sin servidor API
- ✅ Mensajes claros y progreso visible
- ✅ Instrucciones de próximos pasos automáticas

### `generate_moea.py`
**Generación vía API**

```bash
# Terminal 1: Iniciar servidor
python main.py

# Terminal 2: Generar
python generate_moea.py
```

**Características:**
- ✅ Espera automática del servidor
- ✅ Monitoreo de estado
- ✅ Manejo de errores mejorado

### `generate_moea_direct.py`
**Generación directa**

```bash
python generate_moea_direct.py
```

**Características:**
- ✅ No requiere servidor API
- ✅ Uso directo del generador

### Scripts Automáticos
- **`generate_moea.bat`** (Windows)
- **`generate_moea.sh`** (Linux/Mac)

```bash
# Windows
generate_moea.bat

# Linux/Mac
chmod +x generate_moea.sh
./generate_moea.sh
```

---

## ⚙️ Configuración

### `moea_setup.py` 🔧
**Setup automático del proyecto**

```bash
python moea_setup.py
```

**Características:**
- ✅ Verifica herramientas (Python, Node.js, npm)
- ✅ Instala dependencias automáticamente
- ✅ Crea archivo .env
- ✅ Muestra información del proyecto

**Opciones:**
```bash
# Solo backend
python moea_setup.py --no-frontend

# Solo frontend
python moea_setup.py --no-backend

# Directorio personalizado
python moea_setup.py --project-dir ruta/personalizada
```

**Salida:**
```
✅ Python: 3.10.0
✅ Node.js: v18.0.0
✅ npm: 9.0.0
📦 Instalando dependencias...
✅ Dependencias del backend instaladas
✅ Dependencias del frontend instaladas
✅ Archivo .env creado
```

---

## 🧪 Testing

### `moea_test_api.py` 🧪
**Suite completa de tests de API**

```bash
python moea_test_api.py
```

**Tests incluidos:**
1. ✅ Optimización NSGA-II
2. ✅ Obtención de métricas
3. ✅ Optimización en batch
4. ✅ Exportación de resultados
5. ✅ Documentación de API

**Ejemplo de salida:**
```
🧪 Test 1: NSGA-II Optimization
   ✅ Éxito!
   Pareto solutions: 50
   Hypervolume: 0.123456

🧪 Test 2: Get Metrics
   ✅ Métricas obtenidas:
      hypervolume: 0.123456
      igd: 0.001234
      gd: 0.000567
```

**URL personalizada:**
```bash
python moea_test_api.py http://localhost:8001
```

---

## 📊 Benchmarking

### `moea_benchmark.py` 📈
**Benchmark de algoritmos MOEA**

```bash
# Benchmark básico
python moea_benchmark.py

# Benchmark personalizado
python moea_benchmark.py \
  --algorithms nsga2 nsga3 moead \
  --problems ZDT1 ZDT2 DTLZ2 \
  --population 100 \
  --generations 50 \
  --runs 5
```

**Características:**
- ✅ Compara múltiples algoritmos
- ✅ Múltiples problemas de prueba
- ✅ Múltiples runs para estadísticas
- ✅ Guarda resultados en JSON
- ✅ Resumen comparativo

**Ejemplo de salida:**
```
🔬 Benchmarking NSGA2 on ZDT1
   Run 1/3... ✅ (12.34s, HV: 0.123456)
   Run 2/3... ✅ (11.89s, HV: 0.123789)
   Run 3/3... ✅ (12.01s, HV: 0.123567)
   ⏱️  Tiempo promedio: 12.08s
   📊 Hypervolume promedio: 0.123604

📊 Benchmark Summary
   NSGA2: 12.08s, HV: 0.123604
   NSGA3: 13.45s, HV: 0.124123
   MOEA/D: 11.23s, HV: 0.122987
```

**Resultados guardados en:**
- `moea_benchmark_YYYYMMDD_HHMMSS.json`

---

## 📊 Visualización

### `moea_visualize.py` 🎨
**Generar visualizaciones de resultados**

```bash
# Visualizar proyecto
python moea_visualize.py PROJECT_ID

# Directorio personalizado
python moea_visualize.py PROJECT_ID --output mi_directorio
```

**Características:**
- ✅ Gráficos 2D de Pareto fronts
- ✅ Gráficos 3D de Pareto fronts
- ✅ Comparación de algoritmos
- ✅ Exportación a PNG (alta resolución)

**Requisitos:**
```bash
pip install matplotlib numpy
```

**Salida:**
```
📊 Visualizando proyecto: project_12345
✅ Gráfico guardado: moea_visualizations/project_12345_pareto_2d.png
✅ Gráfico 3D guardado: moea_visualizations/project_12345_pareto_3d.png
```

**Tipos de visualización:**
- **2D**: Para problemas con 2 objetivos
- **3D**: Para problemas con 3+ objetivos
- **Comparación**: Gráficos de barras comparativos

---

## ✅ Verificación

### `verify_moea_project.py` ✔️
**Verificar estructura del proyecto**

```bash
python verify_moea_project.py
```

**Verifica:**
- ✅ Directorios requeridos
- ✅ Archivos esenciales
- ✅ Información del proyecto
- ✅ Estructura completa

**Ejemplo de salida:**
```
✅ Backend directory: .../backend
✅ Frontend directory: .../frontend
✅ Backend main.py: .../backend/main.py
✅ Frontend package.json: .../frontend/package.json

📄 Project info found:
   Project Name: moea_optimization_system
   Author: Blatam Academy
   Version: 1.0.0
   AI Type: analytics

✅ Project structure is valid!
```

---

## 🔄 Flujo de Trabajo Completo

### 1. Generar Proyecto
```bash
python quick_moea.py
```

### 2. Configurar
```bash
python moea_setup.py
```

### 3. Verificar
```bash
python verify_moea_project.py
```

### 4. Iniciar Servidores
```bash
# Terminal 1: Backend
cd generated_projects/moea_optimization_system/backend
uvicorn app.main:app --reload

# Terminal 2: Frontend
cd generated_projects/moea_optimization_system/frontend
npm run dev
```

### 5. Probar API
```bash
python moea_test_api.py
```

### 6. Benchmark
```bash
python moea_benchmark.py --algorithms nsga2 nsga3 --problems ZDT1 ZDT2
```

### 7. Visualizar
```bash
# Obtener project_id de los tests
python moea_visualize.py project_12345
```

---

## 📦 Instalación de Dependencias

### Para todas las herramientas:
```bash
pip install requests matplotlib numpy
```

### Solo para visualización:
```bash
pip install matplotlib numpy
```

### Solo para testing/benchmark:
```bash
pip install requests
```

---

## 🎯 Casos de Uso

### Caso 1: Desarrollo Rápido
```bash
python quick_moea.py
python moea_setup.py
python verify_moea_project.py
```

### Caso 2: Testing Completo
```bash
python moea_test_api.py
python moea_benchmark.py
```

### Caso 3: Análisis de Resultados
```bash
python moea_visualize.py project_id
```

### Caso 4: Comparación de Algoritmos
```bash
python moea_benchmark.py \
  --algorithms nsga2 nsga3 moead spea2 \
  --problems ZDT1 ZDT2 ZDT3 \
  --runs 10
```

---

## 🔧 Troubleshooting

### Error: "Module not found"
```bash
pip install requests matplotlib numpy
```

### Error: "Server not available"
```bash
# Verificar que el backend esté corriendo
curl http://localhost:8000/health
```

### Error: "Project not found"
```bash
# Generar primero
python quick_moea.py
```

### Error: "matplotlib 3D not available"
```bash
pip install matplotlib[all]
```

---

## 📚 Documentación Relacionada

- **[MOEA_README.md](MOEA_README.md)** - README principal
- **[MOEA_QUICK_START.md](MOEA_QUICK_START.md)** - Inicio rápido
- **[MOEA_INDEX.md](MOEA_INDEX.md)** - Índice completo
- **[moea_example_usage.py](moea_example_usage.py)** - Ejemplos de código

---

## ✨ Tips y Mejores Prácticas

1. **Siempre verifica** después de generar:
   ```bash
   python verify_moea_project.py
   ```

2. **Usa setup automático** para ahorrar tiempo:
   ```bash
   python moea_setup.py
   ```

3. **Haz benchmarks** antes de optimizar:
   ```bash
   python moea_benchmark.py --runs 5
   ```

4. **Visualiza resultados** para entender mejor:
   ```bash
   python moea_visualize.py project_id
   ```

5. **Ejecuta tests** después de cambios:
   ```bash
   python moea_test_api.py
   ```

---

**¡Todas las herramientas listas para usar! 🚀**

