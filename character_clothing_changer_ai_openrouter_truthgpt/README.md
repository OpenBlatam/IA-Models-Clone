# Character Clothing Changer AI - OpenRouter & TruthGPT

> Part of the [Blatam Academy Integrated Platform](../README.md)

Advanced AI-powered character clothing changer that integrates:
- **ComfyUI** workflow for Flux Fill inpainting
- **OpenRouter** for intelligent prompt processing and optimization
- **TruthGPT** for advanced optimization and enhancement

## Features

- 🎨 **Intelligent Prompt Optimization**: Uses OpenRouter to optimize prompts for better results
- 🚀 **TruthGPT Enhancement**: Leverages TruthGPT for advanced processing and analytics
- 🖼️ **ComfyUI Integration**: Executes Flux Fill workflows for high-quality inpainting
- 👤 **Face Swap**: Swap faces in images that are being processed with inpainting
- 📊 **Analytics**: Tracks usage and performance across all services
- ⚡ **Async Processing**: Fully asynchronous for optimal performance

## Architecture

```
┌─────────────────┐
│   FastAPI App   │
└────────┬─────────┘
         │
    ┌────┴────┐
    │         │
┌───▼───┐ ┌──▼──────┐
│OpenRouter│ │ TruthGPT │
└───┬───┘ └──┬───────┘
    │        │
    └───┬────┘
        │
┌───────▼────────┐
│ ClothingService│
└───────┬────────┘
        │
┌───────▼───────┐
│ ComfyUI Service│
└───────────────┘
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set environment variables:
```bash
export OPENROUTER_API_KEY="your-openrouter-api-key"
export COMFYUI_API_URL="http://localhost:8188"
export TRUTHGPT_ENABLED="true"
export OPENROUTER_ENABLED="true"
```

## Configuration

Create a `.env` file or set environment variables:

```env
# Server
HOST=0.0.0.0
PORT=8000
DEBUG=false

# OpenRouter
OPENROUTER_API_KEY=your-api-key
OPENROUTER_ENABLED=true
OPENROUTER_MODEL=openai/gpt-4
OPENROUTER_TEMPERATURE=0.7
OPENROUTER_MAX_TOKENS=2000

# TruthGPT
TRUTHGPT_ENABLED=true
TRUTHGPT_ENDPOINT=
TRUTHGPT_TIMEOUT=120.0

# ComfyUI
COMFYUI_API_URL=http://localhost:8188
COMFYUI_WORKFLOW_PATH=workflows/flux_fill_clothing_changer.json

# Image Processing
MAX_IMAGE_SIZE=10485760
OUTPUT_DIR=outputs
SAVE_TENSORS=true
```

## Usage

### Start the server:

```bash
python main.py
```

Or with uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### API Endpoints

#### Change Clothing

```bash
POST /api/v1/clothing/change
```

Request body:
```json
{
  "image_url": "https://example.com/image.png",
  "clothing_description": "a red elegant dress",
  "mask_url": "https://example.com/mask.png",
  "character_name": "MyCharacter",
  "negative_prompt": "blurry, low quality",
  "guidance_scale": 50.0,
  "num_steps": 12,
  "seed": 12345,
  "optimize_prompt": true
}
```

#### Get Status

```bash
GET /api/v1/clothing/status/{prompt_id}
```

#### Face Swap

```bash
POST /api/v1/face-swap
```

Request body:
```json
{
  "image_url": "https://example.com/image_in_painting.png",
  "face_url": "https://example.com/new_face.png",
  "mask_url": "https://example.com/mask.png",
  "character_name": "MyCharacter",
  "prompt": "best quality, face swap, high quality portrait",
  "negative_prompt": "blurry, low quality",
  "guidance_scale": 50.0,
  "num_steps": 12,
  "seed": 12345,
  "optimize_prompt": true
}
```

#### Get Analytics

```bash
GET /api/v1/clothing/analytics
```

#### Health Check

```bash
GET /api/v1/health
GET /api/v1/health/detailed
```

## Workflow

The service follows this comprehensive workflow:

### 1. Request Validation
- Validates all input parameters
- Checks parameter ranges (guidance_scale: 1.0-100.0, num_steps: 1-100)
- Verifies required fields (image_url, clothing_description)

### 2. Prompt Optimization (Optional)
- **If OpenRouter enabled**:
  - Uses configured LLM model (default: GPT-4) to optimize prompt
  - Considers context: character name, task type, clothing description
  - Returns optimized prompt or falls back to original on error

### 3. TruthGPT Enhancement (Optional)
- **If TruthGPT enabled**:
  - Enhances the prompt using TruthGPT optimization modules
  - Tracks analytics and performance metrics
  - Applies advanced query processing
  - Falls back gracefully if enhancement fails

### 4. ComfyUI Execution
- Loads workflow template from configured path
- Updates workflow nodes with:
  - Image URL and mask (if provided)
  - Optimized prompt and negative prompt
  - Generation parameters (guidance_scale, num_steps, seed)
- Submits workflow to ComfyUI queue
- Returns prompt_id for status tracking

### 5. Response Building
- Collects execution metadata
- Includes service usage flags (openrouter_used, truthgpt_used)
- Returns success/error response with all relevant information

## ComfyUI Workflow

The workflow is based on Flux Fill for inpainting:
- Uses Flux Fill FP8 model
- Supports portrait LoRA
- Includes Turbo LoRA for faster generation
- Implements crop and stitch for better quality

## Requirements

- Python 3.8+
- ComfyUI server running (for image generation)
- OpenRouter API key (for prompt optimization)
- TruthGPT modules (optional, for enhancement)

## License

MIT

---

[← Back to Main README](../README.md)
