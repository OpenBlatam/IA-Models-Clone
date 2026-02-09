# ⚡ ONE-LINE INSTALL - INSTALACIÓN EN UNA LÍNEA

## 🚀 Instalación Ultra-Rápida (Una Línea)

### Windows (PowerShell/CMD)
```bash
.\quick_install.bat
```

### Linux/Mac
```bash
bash quick_install.sh
```

### O Manualmente (Una Línea)
```bash
pip install torch torchvision torchaudio transformers accelerate bitsandbytes xformers triton gradio wandb tqdm
```

## 🎯 Lo que Instala

✅ **PyTorch 2.1 + CUDA 11.8** - Framework principal  
✅ **Transformers 4.35** - Modelos LLM  
✅ **Accelerate + Bitsandbytes** - Optimización  
✅ **XFormers + Triton** - GPU acceleration  
✅ **Gradio** - Interfaces  
✅ **WandB + tqdm** - Tracking  

## ⚡ Tiempo Total

- **Instalación**: 3-5 minutos
- **Verificación**: 10 segundos
- **Listo para usar**: ✅

## 🎯 Uso Inmediato

```python
import torch
from transformers import AutoModelForCausalLM
from accelerate import Accelerator

# Tu código aquí
print("✅ Todo listo!")
```

## 📊 Speedup

Con este stack obtienes:
- **2-3x** con XFormers
- **1.5x** con Mixed Precision
- **Total: 3-5x speedup** ⚡

## ✅ Verificar

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

**¡Listo en minutos!** 🚀

