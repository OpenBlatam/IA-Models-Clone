# 🚀 HeyGen AI - Setup Guide
# ============================

## 🎯 Overview
This guide will help you set up the optimized HeyGen AI system with cutting-edge performance optimizations, clean architecture, and the latest libraries for deep learning, transformers, diffusion models, and LLMs.

## 🧠 System Requirements

### Hardware Requirements
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3080, RTX 4080, RTX 4090, or A100)
- **RAM**: 16GB+ system RAM
- **Storage**: 50GB+ free space for models and datasets
- **CPU**: 8+ cores recommended

### Software Requirements
- **OS**: Ubuntu 20.04+, Windows 10+, or macOS 12+
- **Python**: 3.9+ (3.11+ recommended)
- **CUDA**: 11.8+ or 12.1+ (for GPU acceleration)

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone <repository-url>
cd blatam-academy/agents/backend/onyx/server/features/heygen_ai
```

### 2. Create Virtual Environment
```bash
# Using conda (recommended)
conda create -n heygen-ai python=3.11
conda activate heygen-ai

# Or using venv
python -m venv heygen-ai
source heygen-ai/bin/activate  # Linux/Mac
# or
heygen-ai\Scripts\activate     # Windows
```

### 3. Install Dependencies
```bash
# Install PyTorch first (with CUDA support)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 🔧 Advanced Installation

### GPU Optimization Setup
```bash
# Install Flash Attention 2.0 for maximum performance
pip install flash-attn --no-build-isolation

# Install xFormers for memory-efficient attention
pip install xformers

# Install Triton for optimized kernels
pip install triton
```

### Quantization Support
```bash
# Install ONNX and Optimum for model optimization
pip install onnx onnxruntime optimum[onnxruntime-gpu]
```

## 🏗️ Architecture Overview

### Core Components
- **Enhanced Diffusion Models**: Advanced diffusion model implementations
- **Enhanced Transformer Models**: Optimized transformer architectures
- **Training Manager**: Refactored training pipeline with ultra-performance optimizations
- **Configuration System**: Centralized YAML-based configuration
- **Performance Monitoring**: Real-time GPU and memory monitoring

### Performance Features
- **Flash Attention 2.0**: 2-4x faster attention computation
- **xFormers**: Memory-efficient attention mechanisms
- **Triton Kernels**: Custom CUDA kernels for optimal performance
- **Mixed Precision Training**: Automatic mixed precision for speed and memory efficiency
- **Gradient Accumulation**: Large batch training on limited GPU memory

## 🚀 Running the System

### 1. Basic Demo
```bash
python run_refactored_demo.py
```

### 2. Training Models
```bash
python core/training_manager_refactored.py --config configs/training_config.yaml
```

### 3. Interactive Gradio Interface
```bash
python gradio_app.py
```

## 📊 Performance Benchmarks

### Expected Speedups
- **Training**: 3-5x faster with Flash Attention 2.0
- **Inference**: 2-3x faster with optimized kernels
- **Memory Usage**: 30-50% reduction with xFormers
- **Batch Size**: 2-4x larger batches with mixed precision

### GPU Memory Optimization
- **Flash Attention**: Reduces memory usage by 30-50%
- **xFormers**: Efficient memory layout for attention
- **Gradient Checkpointing**: Trade compute for memory
- **Dynamic Batching**: Adaptive batch sizes based on available memory

## 🔍 Troubleshooting

### Common Issues

#### CUDA Out of Memory
```bash
# Reduce batch size
export BATCH_SIZE=4

# Enable gradient checkpointing
export GRADIENT_CHECKPOINTING=1

# Use mixed precision
export MIXED_PRECISION=1
```

#### Flash Attention Installation Issues
```bash
# Install build dependencies
sudo apt-get install build-essential

# Install with specific CUDA version
pip install flash-attn --no-build-isolation --extra-index-url https://download.pytorch.org/whl/cu121
```

#### Performance Issues
```bash
# Check GPU utilization
nvidia-smi

# Monitor memory usage
watch -n 1 nvidia-smi

# Profile with PyTorch profiler
python -m torch.utils.bottleneck your_script.py
```

## 📚 Additional Resources

### Documentation
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)
- [Diffusers Documentation](https://huggingface.co/docs/diffusers/)
- [Flash Attention Paper](https://arxiv.org/abs/2205.14135)

### Performance Tuning
- [PyTorch Performance Tuning](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [Memory Management](https://pytorch.org/docs/stable/notes/cuda.html#memory-management)
- [Distributed Training](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)

## 🎉 Next Steps

1. **Explore the Codebase**: Review the enhanced models and training pipeline
2. **Run Benchmarks**: Test performance improvements on your hardware
3. **Customize Models**: Adapt the architecture for your specific use case
4. **Scale Up**: Use distributed training for larger models and datasets

## 🤝 Support

For issues and questions:
- Check the troubleshooting section above
- Review the code comments and documentation
- Open an issue in the repository

---

**Happy AI Development! 🚀✨**
