# Changelog

## [1.0.0] - 2025-12-11

### Added
- **Face Swap Functionality**: Complete face swap integration for images in inpainting
  - New endpoint `/api/v1/face-swap` for face swap operations
  - Integration with "Load New Face" node (ID 240) in ComfyUI workflow
  - Face swap with OpenRouter and TruthGPT optimization

- **Enhanced ComfyUI Service**:
  - Workflow structure validation
  - Better error handling with retry logic and exponential backoff
  - Connection pooling for HTTP client
  - Workflow info endpoint
  - Cancel prompt functionality
  - Get output images functionality
  - Wait for completion with timeout
  - Helper methods for node updates

- **New API Endpoints**:
  - `POST /api/v1/face-swap` - Swap face in inpainting image
  - `POST /api/v1/clothing/cancel/{prompt_id}` - Cancel workflow
  - `GET /api/v1/clothing/images/{prompt_id}` - Get output images
  - `GET /api/v1/clothing/workflow/info` - Get workflow information

- **Utilities**:
  - Validation utilities (`utils/validators.py`)
  - Helper utilities (`utils/helpers.py`)
  - Image URL validation
  - Prompt validation
  - Parameter validation

- **Improved Error Handling**:
  - Better error messages
  - Retry logic with exponential backoff
  - Comprehensive logging
  - Graceful degradation

### Improved
- Better workflow node management
- Enhanced validation
- Improved documentation
- Better code organization
- More comprehensive error handling

### Fixed
- Workflow template loading
- Node ID references
- HTTP client management

