# ⚡ INSTALACIÓN MÁS RÁPIDA - MÁXIMA VELOCIDAD

## 🚀 Instalación Ultra Rápida

### 1️⃣ **Una Línea - Todo Automático**
```bash
pip install -r requirements_fastest.txt
```

### 2️⃣ **Instalación por Partes**

#### 🔴 Crítico (Mínimo para funcionar)
```bash
pip install torch>=2.1.0 transformers>=4.35.0 accelerate>=0.25.0
```

#### 🟠 GPU (Máxima velocidad)
```bash
pip install flash-attn xformers triton
```

#### 🟡 Optimización
```bash
pip install peft>=0.6.0 deepspeed>=0.12.0 bitsandbytes>=0.41.0
```

## ⚡ Speedup Esperado

| Optimización | Speedup | Descripción |
|--------------|---------|-------------|
| Flash Attention | **3x** | Atención GPU optimizada |
| XFormers | **2x** | Operaciones eficientes |
| Mixed Precision | **2x** | FP16 entrenamiento |
| DeepSpeed | **2-4x** | Distributed training |
| **COMBINADO** | **10-20x** ⚡ | Todo junto |

## 📦 Librerías Mínimas

Si necesitas solo lo esencial:
```bash
pip install torch transformers accelerate flash-attn wandb gradio
```

## 🔧 Optimización GPU

### Verificar GPU
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Instalar Flash Attention
```bash
pip install flash-attn --no-build-isolation
```

### Optimizar XFormers
```bash
pip install xformers -U
```

## ✅ Verificación Rápida

```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPUs: {torch.cuda.device_count()}")

import transformers
print(f"Transformers: {transformers.__version__}")

try:
    import flash_attn
    print("✅ Flash Attention instalado")
except:
    print("❌ Flash Attention no instalado")
```

## 🎯 Para Desarrollo TruthGPT

### Solo lo esencial:
```bash
pip install torch transformers accelerate gradio tqdm
```

### Con optimizaciones:
```bash
pip install -r requirements_fastest.txt
```

## 📊 Tiempo de Instalación

- **Mínimo**: 2-3 minutos
- **Recomendado**: 5-7 minutos
- **Completo**: 10-15 minutos

## 🚨 Troubleshooting

### Flash Attention falla
```bash
pip install flash-attn==2.3.6 --no-build-isolation
```

### CUDA version mismatch
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Instalación limpia
```bash
pip uninstall -y torch transformers
pip install -r requirements_fastest.txt
```

## ✅ Ready!

Una vez instalado, verifica con:
```bash
python -c "import torch; from flash_attn import flash_attn_func; print('✅ Todo listo!')"
```

**¡Ahora eres 10-20x más rápido!** ⚡🚀

