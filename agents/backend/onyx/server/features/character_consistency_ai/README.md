# Character Consistency AI - Flux2-based Model

> Part of the [Blatam Academy Integrated Platform](../README.md)

Flux2-based model for generating character consistency safe tensors. It accepts one or multiple images and generates embeddings that maintain character consistency across different generations.

## 🎯 Features

- **Flux2 Model**: Based on Black Forest Labs' Flux2 architecture
- **Multiple Images**: Processes one or multiple images to extract consistent features
- **Safe Tensors**: Generates workflow-ready safe tensor files
- **REST API**: FastAPI endpoints for easy integration
- **Optimizations**: Support for memory and speed optimizations

## 📋 Requirements

```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### 1. Basic Python Usage

```python
from character_consistency_ai.core.character_consistency_service import CharacterConsistencyService
from character_consistency_ai.config.character_consistency_config import CharacterConsistencyConfig

# Create configuration
config = CharacterConsistencyConfig.from_env()

# Create service
service = CharacterConsistencyService(config=config)

# Initialize model
service.initialize_model()

# Generate embedding from images
result = service.generate_character_embedding(
    images=["path/to/image1.jpg", "path/to/image2.jpg"],
    character_name="MyCharacter",
    save_tensor=True,
)

print(f"Embedding saved at: {result['saved_path']}")
```

### 2. Create Workflow Tensor

```python
# Create optimized tensor for workflow
workflow_result = service.create_workflow_tensor(
    embedding_path="character_embeddings/my_character_20240101_120000.safetensors",
    prompt_template="A photo of {character} in a {setting}",
    negative_prompt="blurry, low quality",
    num_inference_steps=50,
    guidance_scale=7.5,
)
```

### 3. Use the API

```bash
# Start server
python -m character_consistency_ai.main

# Or use uvicorn directly
uvicorn character_consistency_ai.api.character_consistency_api:app --host 0.0.0.0 --port 8001
```

#### Available Endpoints

- `POST /api/v1/generate` - Generate embedding from images
- `POST /api/v1/workflow` - Create workflow tensor
- `GET /api/v1/embeddings` - List all embeddings
- `GET /api/v1/embedding/{id}` - Download specific embedding
- `GET /api/v1/model/info` - Model information
- `POST /api/v1/initialize` - Initialize model
- `GET /api/v1/health` - Health check

#### cURL Usage Example

```bash
# Generate embedding
curl -X POST "http://localhost:8001/api/v1/generate" \
  -F "images=@image1.jpg" \
  -F "images=@image2.jpg" \
  -F "character_name=MyCharacter" \
  -F "save_tensor=true"

# Create workflow tensor
curl -X POST "http://localhost:8001/api/v1/workflow" \
  -F "embedding_path=character_embeddings/my_character.safetensors" \
  -F "prompt_template=A photo of {character}" \
  -F "negative_prompt=blurry" \
  -F "num_inference_steps=50" \
  -F "guidance_scale=7.5"
```

## 📁 Project Structure

```
character_consistency_ai/
├── models/
│   ├── flux2_character_model.py      # Main Flux2 model
│   └── safe_tensor_generator.py     # Safe tensor generator
├── core/
│   └── character_consistency_service.py  # Main service
├── api/
│   └── character_consistency_api.py      # FastAPI API
├── config/
│   └── character_consistency_config.py   # Configuration
├── main.py                               # Entry point
└── requirements.txt                      # Dependencies
```

## ⚙️ Configuration

### Environment Variables

```bash
# Model
CHARACTER_CONSISTENCY_MODEL_ID=black-forest-labs/flux2-dev
CHARACTER_CONSISTENCY_DEVICE=cuda  # or cpu, auto
CHARACTER_CONSISTENCY_DTYPE=float16  # or float32

# Optimizations
CHARACTER_CONSISTENCY_OPTIMIZATIONS=true
CHARACTER_CONSISTENCY_EMBEDDING_DIM=768

# Output
CHARACTER_CONSISTENCY_OUTPUT_DIR=./character_embeddings

# API
CHARACTER_CONSISTENCY_API_HOST=0.0.0.0
CHARACTER_CONSISTENCY_API_PORT=8001

# Performance
CHARACTER_CONSISTENCY_MAX_BATCH_SIZE=4
CHARACTER_CONSISTENCY_MAX_IMAGE_SIZE=1024

# Workflow defaults
CHARACTER_CONSISTENCY_INFERENCE_STEPS=50
CHARACTER_CONSISTENCY_GUIDANCE_SCALE=7.5
```

### Programmatic Configuration

```python
from character_consistency_ai.config.character_consistency_config import CharacterConsistencyConfig

config = CharacterConsistencyConfig(
    model_id="black-forest-labs/flux2-dev",
    device="cuda",
    enable_optimizations=True,
    embedding_dim=768,
    output_dir="./my_embeddings",
)
```

## 🔧 Workflow Usage

Generated safe tensors can be used in image generation workflows:

```python
from safetensors.torch import load_file
import torch

# Load embedding
data = load_file("character_embeddings/my_character.safetensors")
character_embedding = data["character_embedding"]

# Use in your generation pipeline
# (diffusers example)
from diffusers import StableDiffusionPipeline

pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

# Inject embedding into pipeline
# (specific implementation depends on your workflow)
```

## 📊 Safe Tensors Format

Generated safe tensors contain:

- `character_embedding`: Tensor with character embedding [embedding_dim]
- Associated JSON Metadata with information about:
  - Character name
  - Number of images used
  - Creation date
  - Model configuration

## 🎨 Usage Examples

### Processing Multiple Images

```python
# Process variable number of images for same character
images = [
    "character_front.jpg",
    "character_side.jpg",
    "character_back.jpg",
]

result = service.generate_character_embedding(
    images=images,
    character_name="Hero",
    metadata={"source": "game_assets", "version": "1.0"},
)
```

### Batch Processing

```python
# Process multiple characters
image_groups = [
    ["hero1_front.jpg", "hero1_side.jpg"],
    ["hero2_front.jpg", "hero2_side.jpg"],
    ["villain_front.jpg"],
]

character_names = ["Hero1", "Hero2", "Villain"]

results = service.batch_generate(
    image_groups=image_groups,
    character_names=character_names,
)
```

## 🔍 Model Information

```python
# Get model info
info = service.get_model_info()
print(info)
```

## 📝 Notes

- Model requires GPU for best performance (CUDA)
- Embeddings have configurable dimension (default: 768)
- Safe tensors are compatible with `safetensors` library
- Model uses lazy loading

## 🐛 Troubleshooting

### Error: "Diffusers not available"
```bash
pip install diffusers transformers
```

### Error: "CUDA out of memory"
- Reduce `max_batch_size` in configuration
- Use `enable_optimizations=True`
- Consider using CPU if GPU lacks memory

### Error: "Model not found"
- Verify model ID is correct
- Ensure access to HuggingFace (token if needed)

## 📚 References

- [Flux2 Model](https://github.com/black-forest-labs/flux2)
- [Safe Tensors](https://huggingface.co/docs/safetensors/)
- [Diffusers Library](https://huggingface.co/docs/diffusers/)

## 📄 License

This project follows the licenses of the base models used.

---

[← Back to Main README](../README.md)
