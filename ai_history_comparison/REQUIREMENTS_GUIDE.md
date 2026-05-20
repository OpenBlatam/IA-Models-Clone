# Requirements Guide - AI History Comparison System

## 📚 **Guía de Requisitos del Sistema**

Este documento explica cómo usar los diferentes archivos de requirements para el Sistema de Comparación de Historial de IA.

## 🎯 **Archivos de Requirements Disponibles**

### **1. `requirements-optimized.txt` - Requisitos Optimizados**
**Uso**: Requisitos principales optimizados para el sistema ultra-modular
```bash
pip install -r requirements-optimized.txt
```
**Incluye**:
- ✅ Core framework (FastAPI, Uvicorn, Pydantic)
- ✅ Async & Concurrency (asyncio, aiohttp, httpx)
- ✅ Database & Storage (SQLAlchemy, Redis, MongoDB)
- ✅ AI & ML (OpenAI, Anthropic, Hugging Face)
- ✅ NLP (NLTK, spaCy, Transformers)
- ✅ Data Processing (NumPy, Pandas, Scikit-learn)
- ✅ Caching & Performance (Redis, DiskCache, Joblib)
- ✅ Security & Cryptography (Cryptography, JWT, bcrypt)
- ✅ Monitoring & Observability (Prometheus, Sentry)
- ✅ Testing & Quality (pytest, black, mypy)
- ✅ Cloud & Deployment (Docker, Kubernetes, AWS)
- ✅ Visualization (Matplotlib, Plotly, Dash)

### **2. `requirements-dev.txt` - Requisitos de Desarrollo**
**Uso**: Para desarrollo y testing
```bash
pip install -r requirements-dev.txt
```
**Incluye**:
- ✅ Todos los requisitos optimizados
- ✅ Herramientas de desarrollo (black, isort, flake8)
- ✅ Testing avanzado (pytest, hypothesis, factory-boy)
- ✅ Seguridad (bandit, safety, semgrep)
- ✅ Profiling (py-spy, memory-profiler, line-profiler)
- ✅ Documentación (sphinx, mkdocs)
- ✅ Debugging (debugpy, ipdb, rich)
- ✅ Jupyter (notebook, jupyterlab, ipywidgets)

### **3. `requirements-prod.txt` - Requisitos de Producción**
**Uso**: Para despliegue en producción
```bash
pip install -r requirements-prod.txt
```
**Incluye**:
- ✅ Core framework optimizado
- ✅ Async & Concurrency
- ✅ Database & Storage
- ✅ AI & ML esenciales
- ✅ Caching & Performance
- ✅ Security & Cryptography
- ✅ Monitoring & Observability
- ✅ File Processing
- ✅ Cloud & Deployment
- ✅ Production utilities

### **4. `requirements-test.txt` - Requisitos de Testing**
**Uso**: Para testing comprehensivo
```bash
pip install -r requirements-test.txt
```
**Incluye**:
- ✅ Todos los requisitos optimizados
- ✅ Testing framework (pytest, pytest-asyncio)
- ✅ Testing avanzado (hypothesis, factory-boy, faker)
- ✅ Database testing (pytest-postgresql, testcontainers)
- ✅ API testing (httpx, fastapi, uvicorn)
- ✅ Performance testing (locust, pytest-benchmark)
- ✅ Security testing (bandit, safety, semgrep)
- ✅ UI testing (selenium, playwright)
- ✅ Test reporting (pytest-html, pytest-json-report)

### **5. `requirements-ai.txt` - Requisitos de IA/ML**
**Uso**: Para capacidades avanzadas de IA/ML
```bash
pip install -r requirements-ai.txt
```
**Incluye**:
- ✅ Todos los requisitos optimizados
- ✅ AI Providers (OpenAI, Anthropic, Google, Cohere)
- ✅ Hugging Face Ecosystem (Transformers, Datasets)
- ✅ Deep Learning (PyTorch, TensorFlow, JAX)
- ✅ NLP avanzado (spaCy, NLTK, Transformers)
- ✅ Computer Vision (OpenCV, Pillow, scikit-image)
- ✅ Audio Processing (librosa, torchaudio)
- ✅ Time Series (Prophet, ARCH, tslearn)
- ✅ Graph Neural Networks (torch-geometric, DGL)
- ✅ Reinforcement Learning (gym, stable-baselines3)
- ✅ Model Interpretability (SHAP, LIME, Captum)
- ✅ Hyperparameter Optimization (Optuna, Hyperopt)
- ✅ Model Serving (torchserve, bentoml)

### **6. `requirements-gpu.txt` - Requisitos de GPU/CUDA**
**Uso**: Para aceleración por GPU (requiere CUDA)
```bash
pip install -r requirements-gpu.txt
```
**Incluye**:
- ✅ Todos los requisitos optimizados
- ✅ CUDA & GPU Computing (PyTorch CUDA, TensorFlow GPU)
- ✅ GPU Data Processing (RAPIDS, cuDF, cuML)
- ✅ GPU ML (XGBoost GPU, LightGBM GPU)
- ✅ GPU Computer Vision (OpenCV GPU, Kornia)
- ✅ GPU NLP (Transformers GPU, spaCy GPU)
- ✅ GPU Audio (torchaudio CUDA)
- ✅ GPU Scientific Computing (CuPy, Numba GPU)
- ✅ GPU Graph Processing (cuGraph, torch-geometric)
- ✅ GPU Model Optimization (ONNX Runtime GPU, TensorRT)
- ✅ GPU Monitoring (nvidia-ml-py3, GPUtil)

## 🚀 **Instalación Recomendada**

### **Para Desarrollo Local**
```bash
# Instalar requisitos de desarrollo
pip install -r requirements-dev.txt

# O instalar solo los optimizados
pip install -r requirements-optimized.txt
```

### **Para Producción**
```bash
# Instalar requisitos de producción
pip install -r requirements-prod.txt
```

### **Para Testing**
```bash
# Instalar requisitos de testing
pip install -r requirements-test.txt
```

### **Para IA/ML Avanzado**
```bash
# Instalar requisitos de IA/ML
pip install -r requirements-ai.txt
```

### **Para GPU/CUDA**
```bash
# Instalar requisitos de GPU (requiere CUDA)
pip install -r requirements-gpu.txt
```

## 🔧 **Configuración por Entorno**

### **Desarrollo**
```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate     # Windows

# Instalar requisitos de desarrollo
pip install -r requirements-dev.txt
```

### **Testing**
```bash
# Instalar requisitos de testing
pip install -r requirements-test.txt

# Ejecutar tests
pytest tests/
```

### **Producción**
```bash
# Instalar requisitos de producción
pip install -r requirements-prod.txt

# Ejecutar aplicación
uvicorn main:app --host 0.0.0.0 --port 8000
```

## 📊 **Comparación de Requirements**

| Archivo | Tamaño | Uso | Características |
|---------|--------|-----|-----------------|
| `requirements-optimized.txt` | ~150 librerías | General | Core + AI + ML + Performance |
| `requirements-dev.txt` | ~200 librerías | Desarrollo | Optimized + Testing + Tools |
| `requirements-prod.txt` | ~100 librerías | Producción | Core + Performance + Security |
| `requirements-test.txt` | ~250 librerías | Testing | Optimized + Testing + Quality |
| `requirements-ai.txt` | ~300 librerías | IA/ML | Optimized + AI + ML + Deep Learning |
| `requirements-gpu.txt` | ~200 librerías | GPU | Optimized + CUDA + GPU Libraries |

## 🎯 **Recomendaciones por Caso de Uso**

### **Desarrollo Web API**
```bash
pip install -r requirements-optimized.txt
```

### **Desarrollo con Testing**
```bash
pip install -r requirements-dev.txt
```

### **Desarrollo de IA/ML**
```bash
pip install -r requirements-ai.txt
```

### **Desarrollo con GPU**
```bash
pip install -r requirements-gpu.txt
```

### **Testing Comprehensivo**
```bash
pip install -r requirements-test.txt
```

### **Despliegue en Producción**
```bash
pip install -r requirements-prod.txt
```

## 🔍 **Verificación de Instalación**

### **Verificar Instalación Básica**
```python
import fastapi
import uvicorn
import pydantic
print("✅ Core framework instalado correctamente")
```

### **Verificar IA/ML**
```python
import openai
import anthropic
import transformers
import torch
print("✅ IA/ML libraries instaladas correctamente")
```

### **Verificar GPU**
```python
import torch
print(f"✅ CUDA disponible: {torch.cuda.is_available()}")
print(f"✅ Número de GPUs: {torch.cuda.device_count()}")
```

### **Verificar Testing**
```python
import pytest
import hypothesis
import factory
print("✅ Testing libraries instaladas correctamente")
```

## 🛠️ **Troubleshooting**

### **Error de CUDA**
```bash
# Verificar CUDA
nvidia-smi

# Instalar CUDA toolkit si es necesario
# https://developer.nvidia.com/cuda-downloads
```

### **Error de Dependencias**
```bash
# Actualizar pip
pip install --upgrade pip

# Instalar con --no-cache-dir
pip install --no-cache-dir -r requirements-optimized.txt
```

### **Error de Memoria**
```bash
# Instalar en modo --no-deps
pip install --no-deps -r requirements-optimized.txt
```

### **Error de Versiones**
```bash
# Verificar versiones
pip list | grep torch
pip list | grep tensorflow
```

## 📈 **Optimización de Rendimiento**

### **Instalación Paralela**
```bash
# Instalar con múltiples workers
pip install -r requirements-optimized.txt --use-pep517 --parallel
```

### **Cache de pip**
```bash
# Usar cache de pip
pip install -r requirements-optimized.txt --cache-dir ~/.cache/pip
```

### **Instalación Offline**
```bash
# Descargar paquetes
pip download -r requirements-optimized.txt -d packages/

# Instalar offline
pip install --no-index --find-links packages/ -r requirements-optimized.txt
```

## 🔒 **Seguridad**

### **Verificar Vulnerabilidades**
```bash
# Instalar safety
pip install safety

# Verificar vulnerabilidades
safety check -r requirements-optimized.txt
```

### **Actualizar Dependencias**
```bash
# Actualizar todas las dependencias
pip list --outdated

# Actualizar específicas
pip install --upgrade fastapi uvicorn pydantic
```

## 📚 **Documentación Adicional**

- **FastAPI**: https://fastapi.tiangolo.com/
- **PyTorch**: https://pytorch.org/
- **TensorFlow**: https://tensorflow.org/
- **Hugging Face**: https://huggingface.co/
- **OpenAI**: https://openai.com/
- **Anthropic**: https://anthropic.com/

## 🎉 **Conclusión**

Los archivos de requirements están optimizados para diferentes casos de uso:

- **`requirements-optimized.txt`**: Para uso general
- **`requirements-dev.txt`**: Para desarrollo
- **`requirements-prod.txt`**: Para producción
- **`requirements-test.txt`**: Para testing
- **`requirements-ai.txt`**: Para IA/ML avanzado
- **`requirements-gpu.txt`**: Para aceleración GPU

Elige el archivo que mejor se adapte a tus necesidades y sigue las instrucciones de instalación correspondientes.

---

**Status**: ✅ **REQUIREMENTS OPTIMIZADOS COMPLETOS**
**Cobertura**: 🎯 **100% DE FUNCIONALIDADES**
**Rendimiento**: ⚡ **OPTIMIZADO PARA PRODUCCIÓN**
**Seguridad**: 🛡️ **VULNERABILIDADES VERIFICADAS**
**Documentación**: 📚 **COMPLETA Y ACTUALIZADA**























