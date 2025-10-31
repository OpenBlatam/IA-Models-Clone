# Advanced AI Models - Quick Start Guide

## 🚀 Getting Started in 5 Minutes

This guide will help you get up and running with the Advanced AI Models module quickly.

## 📋 Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **Memory**: 8GB+ RAM, 4GB+ GPU memory
- **Storage**: 10GB+ free space

### CUDA Setup
```bash
# Check CUDA availability
nvidia-smi

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## ⚡ Quick Installation

### 1. Install Dependencies
```bash
# Navigate to the advanced_ai_models directory
cd agents/backend/onyx/server/features/advanced_ai_models

# Install all dependencies
pip install -r requirements_advanced.txt
```

### 2. Verify Installation
```python
import torch
import transformers
import diffusers
import gradio

print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
```

## 🎯 Quick Examples

### 1. Text Generation with LLM
```python
from models.llm_models import AdvancedLLMModel

# Initialize model
llm = AdvancedLLMModel(
    model_name="microsoft/DialoGPT-small",
    use_8bit=True
)

# Generate text
prompts = ["Hello, how are you?", "What is AI?"]
generated_texts = llm.generate(prompts, max_length=50)
print(generated_texts)
```

### 2. Image Classification
```python
from models.vision_models import ImageClassificationModel
from training.trainer import ImageProcessor

# Initialize model
model = ImageClassificationModel(
    model_name="resnet50",
    num_classes=1000,
    pretrained=True
)

# Process image
processor = ImageProcessor(image_size=224)
image = processor.preprocess("path/to/image.jpg")
predictions = model(image.unsqueeze(0))
results = processor.postprocess(predictions)
print(results)
```

### 3. Text-to-Image Generation
```python
from models.diffusion_models import StableDiffusionPipeline

# Initialize pipeline
pipeline = StableDiffusionPipeline(
    model_id="runwayml/stable-diffusion-v1-5",
    use_fp16=True
)

# Generate image
prompt = "A beautiful sunset over mountains"
images = pipeline.generate(prompt, height=512, width=512)
images[0].save("generated_image.png")
```

## 🔧 Basic Configuration

### Model Configuration
```python
config = {
    # Training settings
    "batch_size": 32,
    "num_epochs": 100,
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    
    # Optimization
    "use_mixed_precision": True,
    "gradient_clip_val": 1.0,
    "accumulate_grad_batches": 1,
    
    # Logging
    "log_every_n_steps": 100,
    "save_every_n_epochs": 5,
    "output_dir": "./outputs",
    
    # Advanced features
    "use_wandb": True,
    "wandb_project": "my-ai-project"
}
```

### Training Setup
```python
from training.trainer import AdvancedTrainer

# Initialize trainer
trainer = AdvancedTrainer(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    config=config
)

# Start training
trainer.train()
```

## 📊 Performance Monitoring

### Real-time Metrics
```python
# Monitor GPU usage
import torch
gpu_utilization = torch.cuda.utilization()
memory_usage = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
print(f"GPU Usage: {gpu_utilization}%")
print(f"Memory Usage: {memory_usage:.2%}")

# Monitor training progress
from training.trainer import TrainingUtils
lr = TrainingUtils.get_learning_rate(trainer.optimizer)
params = TrainingUtils.count_parameters(trainer.model)
print(f"Learning Rate: {lr}")
print(f"Parameters: {params}")
```

### TensorBoard Integration
```bash
# Start TensorBoard
tensorboard --logdir ./outputs/logs

# Open in browser: http://localhost:6006
```

## 🎨 Gradio Interface

### Quick Web Interface
```python
import gradio as gr
from models.llm_models import AdvancedLLMModel

# Initialize model
llm = AdvancedLLMModel("microsoft/DialoGPT-small")

# Create interface
def generate_text(prompt, max_length=50):
    return llm.generate([prompt], max_length=max_length)[0]

interface = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(label="Input Prompt"),
        gr.Slider(minimum=10, maximum=200, value=50, label="Max Length")
    ],
    outputs=gr.Textbox(label="Generated Text"),
    title="AI Text Generator"
)

interface.launch()
```

## 🔍 Debugging Tips

### Common Issues

#### 1. CUDA Out of Memory
```python
# Reduce batch size
config["batch_size"] = 16

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Use mixed precision
config["use_mixed_precision"] = True
```

#### 2. Slow Training
```python
# Increase batch size
config["batch_size"] = 64

# Use multiple GPUs
config["use_distributed"] = True

# Optimize data loading
config["num_workers"] = 8
```

#### 3. Model Not Loading
```python
# Check model path
print(f"Model path: {model_path}")

# Verify dependencies
import transformers
print(f"Transformers version: {transformers.__version__}")

# Clear cache
torch.cuda.empty_cache()
```

### Performance Optimization
```python
# Enable optimizations
model.enable_xformers_memory_efficient_attention()
model.enable_attention_slicing()

# Use quantization
from transformers import BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
```

## 📈 Advanced Features

### 1. Custom Training Loop
```python
from training.trainer import AdvancedTrainer, CustomLossFunctions

# Custom loss function
def custom_loss(predictions, targets):
    ce_loss = F.cross_entropy(predictions, targets)
    focal_loss = CustomLossFunctions.focal_loss(predictions, targets)
    return 0.7 * ce_loss + 0.3 * focal_loss

# Override training step
class CustomTrainer(AdvancedTrainer):
    def _training_step(self, batch):
        outputs = self.model(**batch)
        loss = custom_loss(outputs.logits, batch['labels'])
        return loss
```

### 2. Multi-Modal Processing
```python
from models.transformer_models import MultiModalTransformer

# Initialize multi-modal model
multimodal = MultiModalTransformer(
    text_config={"vocab_size": 32000, "d_model": 768},
    image_config={"image_size": 224, "patch_size": 16},
    audio_config={"input_dim": 128, "d_model": 768},
    fusion_config={"d_model": 768, "n_layers": 6}
)

# Process multiple modalities
outputs = multimodal(
    text_input=text_tensor,
    image_input=image_tensor,
    audio_input=audio_tensor
)
```

### 3. Distributed Training
```bash
# Single node, multiple GPUs
torchrun --nproc_per_node=4 train_script.py

# Multi-node training
torchrun --nnodes=2 --nproc_per_node=4 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=MASTER_IP:29400 train_script.py
```

## 🚀 Production Deployment

### 1. Model Serving
```python
from inference.inference_engine import ModelInferenceEngine

# Initialize inference engine
engine = ModelInferenceEngine(
    model=model,
    use_cache=True,
    max_cache_size=1000
)

# Serve predictions
results = engine.generate_batch(prompts)
```

### 2. Docker Deployment
```dockerfile
FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime

WORKDIR /app
COPY requirements_advanced.txt .
RUN pip install -r requirements_advanced.txt

COPY . .
EXPOSE 8000

CMD ["python", "app.py"]
```

### 3. API Endpoint
```python
from fastapi import FastAPI
from models.llm_models import AdvancedLLMModel

app = FastAPI()
model = AdvancedLLMModel("microsoft/DialoGPT-small")

@app.post("/generate")
async def generate_text(prompt: str, max_length: int = 50):
    result = model.generate([prompt], max_length=max_length)[0]
    return {"generated_text": result}
```

## 📚 Next Steps

### 1. Explore Examples
- Check the `examples/` directory for more detailed examples
- Run the demo script: `python demo_advanced_models.py`
- Experiment with different model configurations

### 2. Customize Models
- Modify model architectures in `models/`
- Add custom loss functions in `training/trainer.py`
- Create new data loaders in `data/`

### 3. Scale Up
- Implement distributed training
- Add model serving capabilities
- Integrate with monitoring systems

### 4. Join Community
- Report issues on GitHub
- Share your use cases
- Contribute improvements

## 🆘 Getting Help

### Documentation
- [API Reference](docs/api.md)
- [Model Architectures](docs/models.md)
- [Training Guide](docs/training.md)

### Support Channels
- **GitHub Issues**: Report bugs and request features
- **Discussions**: Community Q&A and discussions
- **Examples**: Code examples and tutorials

### Common Commands
```bash
# Run tests
python -m pytest tests/

# Check model performance
python benchmark_models.py

# Generate documentation
python generate_docs.py

# Clean up cache
python cleanup_cache.py
```

---

**🎉 Congratulations!** You're now ready to use the Advanced AI Models module. Start with the examples above and gradually explore more advanced features as you become comfortable with the basics.

**💡 Pro Tip**: Always start with smaller models and datasets to verify your setup before scaling up to larger models and full datasets. 