# 🚀 README - TRUTHGPT INSTALLATION

## ⚡ Instalación Rápida

### Opción 1: Script Automático (Recomendado)

**Windows:**
```bash
quick_install.bat
```

**Linux/Mac:**
```bash
bash quick_install.sh
```

### Opción 2: Una Línea
```bash
pip install torch torchvision torchaudio transformers accelerate bitsandbytes xformers triton gradio wandb tqdm
```

### Opción 3: Requirements File
```bash
pip install -r requirements.txt
```

## 📦 Qué Incluye

### Core Framework
- **PyTorch 2.1+** - Deep learning framework
- **Transformers 4.35+** - LLM models
- **Accelerate** - Training optimization

### GPU Optimization
- **Flash Attention** - 3x faster attention
- **XFormers** - 2x faster operations
- **Triton** - JIT compilation
- **CuPy** - NumPy for GPU

### Training
- **PEFT** - Efficient fine-tuning (LoRA)
- **DeepSpeed** - Distributed training
- **Bitsandbytes** - 8-bit optimization

### UI & Tracking
- **Gradio** - Interactive interfaces
- **WandB** - Experiment tracking
- **TensorBoard** - Visualization

## ✅ Verificar Instalación

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

## 🎯 Speedup Esperado

Con estas librerías obtienes:
- **3-5x** speedup general
- **2-3x** en atención
- **50-75%** menos memoria
- **2x** más rápido en entrenamiento

## 📚 Documentación

- `OPTIMAL_GUIDE.md` - Guía de optimización
- `FASTEST_INSTALL.md` - Instalación rápida
- `BEST_LIBRARIES_GUIDE.md` - Mejores librerías
- `ONE_LINE_INSTALL.md` - Una línea

## 🎓 Uso Básico

```python
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2")
# Tu código aquí
```

---

**¡Listo para empezar!** 🚀✨

