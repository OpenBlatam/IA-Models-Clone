# Features Overview

## Core Features

### 1. Clothing Change
- Change character clothing using AI-powered inpainting
- OpenRouter prompt optimization
- TruthGPT enhancement
- ComfyUI workflow execution

### 2. Face Swap ⭐ NEW
- Swap faces in images that are being processed with inpainting
- Uses "Load New Face" node from ComfyUI workflow
- Integrated with OpenRouter and TruthGPT
- Supports mask-based inpainting

### 3. Workflow Management
- Workflow template loading and caching
- Node parameter updates
- Workflow structure validation
- Workflow information retrieval

### 4. Queue Management
- Queue status checking
- Prompt cancellation
- History retrieval
- Status tracking

### 5. Image Retrieval
- Get output images for completed prompts
- Wait for completion with timeout
- Image URL generation

## API Endpoints

### Clothing Change
- `POST /api/v1/clothing/change` - Change character clothing
- `GET /api/v1/clothing/status/{prompt_id}` - Get workflow status
- `GET /api/v1/clothing/analytics` - Get service analytics

### Face Swap ⭐ NEW
- `POST /api/v1/face-swap` - Swap face in inpainting image

### Workflow Management ⭐ NEW
- `POST /api/v1/clothing/cancel/{prompt_id}` - Cancel workflow
- `GET /api/v1/clothing/images/{prompt_id}` - Get output images
- `GET /api/v1/clothing/workflow/info` - Get workflow information

## Integration Features

### OpenRouter
- Prompt optimization
- Context-aware suggestions
- Multiple model support
- Connection pooling

### TruthGPT
- Advanced optimization
- Analytics tracking
- Performance monitoring
- Graceful fallback

### ComfyUI
- Flux Fill workflow execution
- Face swap support
- Inpainting with masks
- Crop and stitch

## Utilities

### Validators
- Image URL validation
- Prompt validation
- Parameter validation
- Seed validation

### Helpers
- Prompt ID generation
- Workflow summary formatting
- Filename sanitization
- Timestamp formatting
- Image info extraction

## Error Handling

- Retry logic with exponential backoff
- Comprehensive error messages
- Graceful degradation
- Detailed logging
- Validation at multiple levels

## Performance

- Connection pooling
- Workflow template caching
- Async/await throughout
- Efficient node updates
- Optimized HTTP client

