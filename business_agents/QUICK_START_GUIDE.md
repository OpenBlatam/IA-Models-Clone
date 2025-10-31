# Guía de Inicio Rápido - ML NLP Benchmark System

## 🚀 Sistemas Disponibles

### 1. Sistema Básico ML NLP Benchmark
- **Archivo**: `ml_nlp_benchmark.py`
- **App**: `ml_nlp_benchmark_app.py`
- **Puerto**: 8000
- **Características**: NLP, ML, Benchmark, Análisis Comprehensivo

### 2. Sistema Avanzado ML NLP Benchmark
- **Archivo**: `advanced_ml_nlp_benchmark.py`
- **App**: `advanced_ml_nlp_benchmark_app.py`
- **Puerto**: 8000
- **Características**: 15 niveles de análisis (Advanced, Enhanced, Super, Hyper, Ultimate, Extreme, Maximum, Peak, Supreme, Perfect, Flawless, Infallible, Ultimate Perfection, Ultimate Mastery)

### 3. Sistema Ultimate ML NLP Benchmark
- **Archivo**: `ultimate_ml_nlp_benchmark.py`
- **Características**: 20 niveles de análisis con capacidades máximas

## 📦 Instalación Rápida

```bash
# Instalar dependencias
pip install fastapi uvicorn nltk spacy scikit-learn numpy pandas torch transformers sentence-transformers faiss-cpu redis memcached psutil GPUtil

# Descargar modelos
python -c "import nltk; nltk.download('punkt'); nltk.download('vader_lexicon'); nltk.download('stopwords')"
python -m spacy download en_core_web_sm
```

## 🎯 Inicio Rápido

### Opción 1: Sistema Básico
```bash
python ml_nlp_benchmark_app.py
```

### Opción 2: Sistema Avanzado
```bash
python advanced_ml_nlp_benchmark_app.py
```

### Opción 3: Con Uvicorn
```bash
uvicorn ml_nlp_benchmark_app:app --host 0.0.0.0 --port 8000 --reload
```

## 🔥 Ejemplos de Uso

### 1. Análisis Básico
```bash
curl -X POST "http://localhost:8000/api/v1/ml-nlp-benchmark/analyze" \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text here", "analysis_type": "comprehensive", "method": "benchmark"}'
```

### 2. Análisis Avanzado
```bash
curl -X POST "http://localhost:8000/api/v1/advanced-ml-nlp-benchmark/advanced-analyze" \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text here"}'
```

### 3. Análisis por Lotes
```bash
curl -X POST "http://localhost:8000/api/v1/ml-nlp-benchmark/analyze-batch" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Text 1", "Text 2", "Text 3"], "analysis_type": "comprehensive"}'
```

## 📊 Monitoreo

### Ver Estadísticas
```bash
curl "http://localhost:8000/stats"
```

### Ver Estado de Salud
```bash
curl "http://localhost:8000/health"
```

### Ver Comparación de Sistemas
```bash
curl "http://localhost:8000/compare"
```

## 🌐 Endpoints Disponibles

### Sistema Básico
- `/api/v1/ml-nlp-benchmark/analyze` - Análisis básico
- `/api/v1/ml-nlp-benchmark/nlp-analyze` - Análisis NLP
- `/api/v1/ml-nlp-benchmark/ml-analyze` - Análisis ML
- `/api/v1/ml-nlp-benchmark/benchmark-analyze` - Análisis Benchmark
- `/api/v1/ml-nlp-benchmark/analyze-batch` - Procesamiento por lotes
- `/api/v1/ml-nlp-benchmark/stats` - Estadísticas
- `/api/v1/ml-nlp-benchmark/health` - Estado de salud
- `/api/v1/ml-nlp-benchmark/models` - Modelos disponibles
- `/api/v1/ml-nlp-benchmark/performance` - Métricas de rendimiento

### Sistema Avanzado (15 niveles adicionales)
- `/api/v1/advanced-ml-nlp-benchmark/analyze` - Análisis avanzado
- `/api/v1/advanced-ml-nlp-benchmark/advanced-analyze` - Advanced
- `/api/v1/advanced-ml-nlp-benchmark/enhanced-analyze` - Enhanced
- `/api/v1/advanced-ml-nlp-benchmark/super-analyze` - Super
- `/api/v1/advanced-ml-nlp-benchmark/hyper-analyze` - Hyper
- `/api/v1/advanced-ml-nlp-benchmark/ultimate-analyze` - Ultimate
- `/api/v1/advanced-ml-nlp-benchmark/extreme-analyze` - Extreme
- `/api/v1/advanced-ml-nlp-benchmark/maximum-analyze` - Maximum
- `/api/v1/advanced-ml-nlp-benchmark/peak-analyze` - Peak
- `/api/v1/advanced-ml-nlp-benchmark/supreme-analyze` - Supreme
- `/api/v1/advanced-ml-nlp-benchmark/perfect-analyze` - Perfect
- `/api/v1/advanced-ml-nlp-benchmark/flawless-analyze` - Flawless
- `/api/v1/advanced-ml-nlp-benchmark/infallible-analyze` - Infallible
- `/api/v1/advanced-ml-nlp-benchmark/ultimate-perfection-analyze` - Ultimate Perfection
- `/api/v1/advanced-ml-nlp-benchmark/ultimate-mastery-analyze` - Ultimate Mastery

## 📚 Documentación Completa

- `README_ML_NLP_BENCHMARK.md` - Documentación del sistema básico
- `README_ADVANCED_ML_NLP_BENCHMARK.md` - Documentación del sistema avanzado

## 🎨 Swagger UI

Visita: `http://localhost:8000/docs` para ver la documentación interactiva de la API.

## 🔧 Configuración

### Variables de Entorno
```bash
HOST=0.0.0.0
PORT=8000
DEBUG=True
MAX_WORKERS=8
BATCH_SIZE=1000
CACHE_SIZE=10000
```

## 💡 Tips

1. **Para desarrollo**: Usa `python ml_nlp_benchmark_app.py` con reload automático
2. **Para producción**: Usa `uvicorn` con múltiples workers
3. **Para máximo rendimiento**: Usa el sistema avanzado con GPU habilitado
4. **Para análisis rápido**: Usa el sistema básico
5. **Para análisis completo**: Usa el sistema avanzado con análisis "comprehensive"

## 🚨 Solución de Problemas

### Error de memoria
```bash
# Reducir batch_size o cache_size
export BATCH_SIZE=500
export CACHE_SIZE=5000
```

### Error de GPU
```bash
# Verificar CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### Error de Redis
```bash
# Iniciar Redis
redis-server
```

## 📞 Soporte

Para más información, consulta la documentación completa en los archivos README.












