# Installation Guide - Addiction Recovery AI

## 📦 Requisitos del Sistema

### Hardware Recomendado
- **GPU**: NVIDIA GPU con CUDA support (opcional pero recomendado)
- **RAM**: Mínimo 8GB, recomendado 16GB+
- **Storage**: 10GB+ para modelos y datos

### Software
- **Python**: 3.8+
- **CUDA**: 11.8+ (si usas GPU)
- **cuDNN**: 8.6+ (si usas GPU)

## 🚀 Instalación Rápida

### Opción 1: Instalación Completa
```bash
# Clonar repositorio
git clone <repository-url>
cd addiction_recovery_ai

# Instalar todas las dependencias
pip install -r requirements.txt
```

### Opción 2: Instalación Mínima
```bash
# Solo dependencias esenciales
pip install torch transformers diffusers gradio numpy
```

### Opción 3: Instalación con CUDA
```bash
# Instalar PyTorch con CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Instalar otras dependencias
pip install transformers diffusers gradio
```

## 🔧 Verificación de Instalación

```python
# Verificar PyTorch
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Verificar instalación
from addiction_recovery_ai import create_sentiment_analyzer
print("✅ Installation successful!")
```

## 📋 Dependencias Principales

### Core
- `torch>=2.5.0` - PyTorch
- `transformers>=4.45.0` - HuggingFace Transformers
- `diffusers>=0.30.0` - Diffusion models
- `gradio>=5.0.0` - Interactive interfaces

### Optimization
- `accelerate>=1.1.0` - Multi-GPU training
- `xformers>=0.0.28` - Memory-efficient attention
- `optimum>=1.20.0` - Model optimization

### Production
- `tensorboard>=2.18.0` - Experiment tracking
- `wandb>=0.18.0` - Weights & Biases
- `psutil>=5.9.0` - System utilities

## 🐛 Solución de Problemas

### Error: CUDA out of memory
```python
# Reducir batch size o usar gradient accumulation
# Habilitar mixed precision
# Usar memory optimization
from addiction_recovery_ai import optimize_model_memory
optimize_model_memory(model)
```

### Error: Module not found
```bash
# Reinstalar dependencias
pip install -r requirements.txt --upgrade
```

### Error: CUDA not available
```bash
# Verificar instalación de CUDA
nvidia-smi

# Reinstalar PyTorch con CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## 🔄 Actualización

```bash
# Actualizar todas las dependencias
pip install -r requirements.txt --upgrade

# Actualizar solo PyTorch
pip install torch --upgrade
```

## ✅ Checklist de Instalación

- [ ] Python 3.8+ instalado
- [ ] PyTorch instalado y funcionando
- [ ] CUDA instalado (opcional)
- [ ] Dependencias instaladas
- [ ] Verificación exitosa
- [ ] Ejemplo quick_start.py funciona

---

**Version**: 3.4.0








