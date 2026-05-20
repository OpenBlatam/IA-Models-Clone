# 📡 API Documentation

## Base URL

```
http://localhost:8002/api/v1
```

## Endpoints

### Health Check

#### GET `/health`

Check server and model status.

**Response:**
```json
{
  "status": "healthy",
  "model_initialized": true,
  "using_deepseek_fallback": false,
  "model_type": "Flux2"
}
```

### Model Information

#### GET `/model/info`

Get model information.

**Response:**
```json
{
  "status": "initialized",
  "primary_model": "Flux2",
  "fallback_mode": false,
  "device": "cuda",
  "dtype": "float16"
}
```

#### POST `/initialize`

Initialize the model.

**Response:**
```json
{
  "status": "initialized",
  "message": "Flux2 model initialized successfully",
  "using_deepseek_fallback": false,
  "model_type": "Flux2"
}
```

### Clothing Change

#### POST `/change-clothing`

Change clothing in a character image.

**Request:**
- `image` (file, required): Character image (PNG, JPEG, JPG, max 10MB)
- `clothing_description` (string, required): Description of new clothing (3-500 characters)
- `character_name` (string, optional): Character name (1-100 characters)
- `prompt` (string, optional): Full prompt (max 1000 characters)
- `negative_prompt` (string, optional): Negative prompt
- `num_inference_steps` (integer, optional): Number of inference steps (1-100, default: 50)
- `guidance_scale` (float, optional): Guidance scale (1.0-20.0, default: 7.5)
- `strength` (float, optional): Inpainting strength (0.0-1.0, default: 0.8)
- `save_tensor` (boolean, optional): Save as ComfyUI safe tensor (default: true)

**Response:**
```json
{
  "clothing_description": "a red elegant dress",
  "character_name": "MyCharacter",
  "changed": true,
  "image_base64": "data:image/png;base64,...",
  "image_url": "/api/v1/image/result.png",
  "saved_path": "./comfyui_tensors/result.safetensors",
  "saved": true,
  "prompt_used": "...",
  "negative_prompt_used": "...",
  "quality_metrics": {
    "similarity": 0.95,
    "quality": 0.92
  }
}
```

**Error Responses:**

Validation Error (400):
```json
{
  "error": "VALIDATION_ERROR",
  "message": "Validation failed",
  "details": {
    "errors": [
      "Image: Image file size exceeds maximum of 10MB",
      "Clothing description: Description must be at least 3 characters"
    ]
  }
}
```

Server Error (500):
```json
{
  "error": "Internal server error",
  "type": "unexpected_error"
}
```

### Tensors

#### GET `/tensors`

List all saved tensors.

**Response:**
```json
[
  {
    "filename": "result.safetensors",
    "path": "./comfyui_tensors/result.safetensors",
    "size": 1024000,
    "created_at": "2024-01-01T00:00:00"
  }
]
```

#### GET `/tensor/{tensor_id}`

Download a tensor file.

**Response:** Binary file (application/octet-stream)

### Images

#### GET `/image/{image_id}`

Get a processed image.

**Response:** Image file (image/png)

## Error Codes

- `VALIDATION_ERROR` - Validation failed
- `IMAGE_VALIDATION_ERROR` - Image validation failed
- `TEXT_VALIDATION_ERROR` - Text validation failed
- `PARAMETER_VALIDATION_ERROR` - Parameter validation failed
- `MODEL_ERROR` - Model-related error
- `MODEL_NOT_INITIALIZED` - Model not initialized
- `MODEL_LOAD_ERROR` - Model failed to load
- `PROCESSING_ERROR` - Image processing error
- `TENSOR_GENERATION_ERROR` - Tensor generation error
- `API_ERROR` - API-related error
- `CONFIGURATION_ERROR` - Configuration error

## Rate Limiting

Currently no rate limiting is implemented. Consider adding rate limiting for production use.

## Authentication

Currently no authentication is required. Consider adding authentication for production use.

