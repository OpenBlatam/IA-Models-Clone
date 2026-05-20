# Architecture Overview

## System Components

### 1. API Layer (`api/`)
FastAPI routers that handle HTTP requests and responses.

- **health_router.py**: Health check endpoints
  - `/api/v1/health` - Basic health check
  - `/api/v1/health/detailed` - Detailed service status
  
- **clothing_router.py**: Main clothing change endpoints
  - `POST /api/v1/clothing/change` - Execute clothing change
  - `GET /api/v1/clothing/status/{prompt_id}` - Get workflow status
  - `GET /api/v1/clothing/analytics` - Get service analytics

### 2. Services Layer (`services/`)
Business logic and orchestration services.

- **clothing_service.py**: Main orchestration service
  - Coordinates the complete workflow
  - Manages prompt optimization and enhancement
  - Handles error recovery and fallbacks
  - Provides analytics and status tracking
  
- **comfyui_service.py**: ComfyUI workflow execution
  - Loads and manages workflow templates
  - Executes ComfyUI API calls
  - Monitors queue status
  - Handles workflow node updates

### 3. Infrastructure Layer (`infrastructure/`)
External service clients and integrations.

- **openrouter_client.py**: OpenRouter API client
  - Prompt optimization via LLM models
  - Connection pooling and async operations
  - Error handling and retries
  
- **truthgpt_client.py**: TruthGPT integration client
  - Advanced query enhancement
  - Analytics and monitoring
  - Integration with TruthGPT modules
  
- **truthgpt_status.py**: TruthGPT availability checks
  - Module availability detection
  - Fallback response generation
  - Error response handling
  
- **truthgpt_helpers.py**: Helper functions
  - Safe TruthGPT operation calls
  - Ready state checks
  - Error recovery utilities

### 4. Configuration (`config/`)
Application settings and configuration management.

- **settings.py**: Application settings
  - Environment variable management
  - Pydantic-based validation
  - Default value handling
  - Service enable/disable flags

### 5. Workflows (`workflows/`)
ComfyUI workflow templates.

- **flux_fill_clothing_changer.json**: Main workflow template
  - Flux Fill FP8 model configuration
  - Inpainting pipeline setup
  - Node connections and parameters

## Data Flow

### Request Processing Flow

```
┌─────────────────┐
│  User Request   │
│  (FastAPI)      │
└────────┬─────────┘
         │
         ▼
┌─────────────────────────┐
│  ClothingChangeService  │
│  (Orchestration)        │
└────────┬────────────────┘
         │
         ├─────────────────┐
         │                 │
         ▼                 ▼
┌──────────────────┐  ┌──────────────────┐
│ OpenRouterClient │  │ TruthGPTClient   │
│ (Optimization)   │  │ (Enhancement)    │
└────────┬─────────┘  └────────┬─────────┘
         │                     │
         └──────────┬──────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │  ComfyUIService      │
         │  (Workflow Exec)     │
         └──────────┬────────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │    ComfyUI API       │
         │  (Image Generation)  │
         └──────────┬────────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │    Result Response    │
         └──────────────────────┘
```

### Detailed Workflow Steps

1. **Request Validation**
   - Validate input parameters (image_url, clothing_description, etc.)
   - Check parameter ranges (guidance_scale, num_steps)
   - Verify required fields are present

2. **Prompt Optimization (Optional)**
   - If OpenRouter enabled: Optimize prompt with LLM
   - Context-aware prompt enhancement
   - Fallback to original if optimization fails

3. **TruthGPT Enhancement (Optional)**
   - If TruthGPT enabled: Further enhance prompt
   - Advanced query processing
   - Analytics tracking

4. **Workflow Execution**
   - Load ComfyUI workflow template
   - Update workflow nodes with parameters
   - Submit to ComfyUI queue
   - Monitor execution status

5. **Response Building**
   - Collect execution metadata
   - Build success/error response
   - Include service usage flags

## Integration Points

### OpenRouter Integration
- **Purpose**: Optimize prompts for better image generation
- **Model**: Configurable via `OPENROUTER_MODEL` (default: `openai/gpt-4`)
- **Features**:
  - Prompt optimization with context awareness
  - Task-specific prompt enhancement
  - Token usage tracking
  - Temperature and max_tokens configuration
- **Error Handling**: Graceful fallback to original prompt
- **Configuration**:
  - `OPENROUTER_API_KEY`: API key for authentication
  - `OPENROUTER_ENABLED`: Enable/disable service
  - `OPENROUTER_TEMPERATURE`: LLM temperature (default: 0.7)
  - `OPENROUTER_MAX_TOKENS`: Max tokens per request (default: 2000)

### TruthGPT Integration
- **Purpose**: Advanced optimization and analytics
- **Features**:
  - Query enhancement and processing
  - Analytics tracking and monitoring
  - Performance optimization
  - Integration with TruthGPT modules
- **Error Handling**: Safe fallback with error logging
- **Configuration**:
  - `TRUTHGPT_ENABLED`: Enable/disable service
  - `TRUTHGPT_ENDPOINT`: TruthGPT endpoint URL
  - `TRUTHGPT_TIMEOUT`: Request timeout in seconds (default: 120.0)
- **Availability**: Checks for TruthGPT modules at runtime

### ComfyUI Integration
- **Purpose**: Execute Flux Fill inpainting workflows
- **Workflow**: Based on Flux Fill FP8 model
- **Features**:
  - Inpainting with mask support
  - Crop & stitch operations
  - Mask processing and generation
  - Queue management
  - Status monitoring
- **Configuration**:
  - `COMFYUI_API_URL`: ComfyUI API endpoint (default: `http://localhost:8188`)
  - `COMFYUI_WORKFLOW_PATH`: Path to workflow template
- **Workflow Template**: JSON-based node configuration

## Configuration

### Environment Variables

All configuration is managed through environment variables and the `Settings` class in `config/settings.py`.

#### Server Settings
- `HOST`: Server host (default: `0.0.0.0`)
- `PORT`: Server port (default: `8000`)
- `DEBUG`: Debug mode (default: `false`)

#### OpenRouter Settings
- `OPENROUTER_API_KEY`: API key for OpenRouter
- `OPENROUTER_ENABLED`: Enable OpenRouter (default: `true`)
- `OPENROUTER_MODEL`: Model to use (default: `openai/gpt-4`)
- `OPENROUTER_TEMPERATURE`: Temperature setting (default: `0.7`)
- `OPENROUTER_MAX_TOKENS`: Max tokens (default: `2000`)

#### TruthGPT Settings
- `TRUTHGPT_ENABLED`: Enable TruthGPT (default: `true`)
- `TRUTHGPT_ENDPOINT`: TruthGPT endpoint URL
- `TRUTHGPT_TIMEOUT`: Timeout in seconds (default: `120.0`)

#### ComfyUI Settings
- `COMFYUI_API_URL`: ComfyUI API URL (default: `http://localhost:8188`)
- `COMFYUI_WORKFLOW_PATH`: Workflow template path

#### Image Processing Settings
- `MAX_IMAGE_SIZE`: Maximum image size in bytes (default: `10485760` = 10MB)
- `ALLOWED_IMAGE_TYPES`: Allowed image MIME types

#### Output Settings
- `OUTPUT_DIR`: Output directory for results
- `SAVE_TENSORS`: Whether to save tensors (default: `true`)

## Error Handling

### Strategy
- **Graceful Degradation**: Services continue to work if optional services fail
- **Fallback Responses**: Original prompts used if optimization fails
- **Comprehensive Logging**: All errors logged with context
- **Error Types**:
  - `ValueError`: Invalid input parameters
  - `Exception`: General service errors
  - HTTP errors from external APIs

### Error Recovery
1. **OpenRouter Failure**: Falls back to original prompt
2. **TruthGPT Failure**: Uses prompt from previous step
3. **ComfyUI Failure**: Returns error response with details
4. **Validation Errors**: Returns immediately with error message

### Error Response Format
```json
{
  "success": false,
  "error": "Error message",
  "original_prompt": "User's original prompt"
}
```

## Performance

### Optimizations
- **Async/Await**: All I/O operations are asynchronous
- **Connection Pooling**: HTTP clients use connection pooling
- **Workflow Caching**: Workflow templates cached in memory
- **Lazy Initialization**: Services initialized only when needed

### HTTP Client Configuration
- **Max Connections**: 100 concurrent connections
- **Keep-Alive**: 20 connections kept alive
- **Keep-Alive Expiry**: 30 seconds
- **HTTP/2**: Enabled for better performance
- **Timeouts**: Configurable per service

### Resource Management
- **Client Lifecycle**: Proper cleanup on shutdown
- **Memory Management**: Efficient data structures
- **Error Recovery**: Minimal overhead on failures

## Security

### API Security
- **CORS**: Configurable CORS middleware
- **Input Validation**: All inputs validated before processing
- **Error Messages**: Sanitized error messages (no sensitive data)

### Configuration Security
- **Environment Variables**: Sensitive data via environment
- **API Keys**: Never logged or exposed
- **Settings Validation**: Pydantic-based validation

## Testing

### Test Structure
- Unit tests for individual services
- Integration tests for workflow
- Mock external services for testing

### Test Coverage
- Service initialization
- Error handling paths
- Validation logic
- Response building

## Monitoring & Analytics

### Analytics Endpoints
- `/api/v1/clothing/analytics`: Service analytics
- Service status and configuration
- Usage statistics (if available)

### Logging
- Structured logging throughout
- Log levels: INFO, WARNING, ERROR
- Exception tracebacks for debugging
- Service initialization logging

## Deployment

### Requirements
- Python 3.8+
- FastAPI
- ComfyUI server running
- (Optional) OpenRouter API key
- (Optional) TruthGPT modules

### Startup Sequence
1. Load configuration from environment
2. Initialize service clients
3. Verify ComfyUI connectivity
4. Start FastAPI server

### Shutdown Sequence
1. Close HTTP clients
2. Cleanup resources
3. Log shutdown event

