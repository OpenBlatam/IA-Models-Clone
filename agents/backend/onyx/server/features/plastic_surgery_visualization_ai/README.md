# Plastic Surgery Visualization AI

AI system that visualizes how you'll look after plastic surgery procedures.

> **Quick Start**: See [QUICK_START.md](QUICK_START.md) for a quick setup guide.
> **Features**: See [FEATURES.md](FEATURES.md) for complete feature list.

## Description

This AI-powered service allows users to upload their photos and see how they would look after various plastic surgery procedures. The system uses advanced AI models to generate realistic previews of surgical outcomes.

## Features

- **Multiple Surgery Types**: Support for various plastic surgery procedures including:
  - Rhinoplasty (nose reshaping)
  - Facelift (facial rejuvenation)
  - Blepharoplasty (eyelid surgery)
  - Liposuction (fat removal)
  - Breast Augmentation
  - Chin Augmentation

- **Adjustable Intensity**: Control the intensity of the surgical effect (0.0 to 1.0)

- **Image Processing**: Support for multiple image formats (JPG, PNG, WebP) with automatic orientation correction

- **RESTful API**: Clean API design with FastAPI

- **Rate Limiting**: Built-in rate limiting to prevent abuse (60 requests/minute)

- **Caching**: Intelligent caching system to reduce processing time for repeated requests

- **Multiple AI Models**: Support for different AI providers (OpenAI, Anthropic, Local models)

- **Security**: Security headers and input validation

- **Metrics & Monitoring**: Built-in metrics collection and monitoring endpoints

- **Error Handling**: Comprehensive error handling with custom exceptions

## Installation

### Production (Recommended)
```bash
pip install -r requirements.txt
```

### Development
```bash
pip install -r requirements-dev.txt
```

### Minimal (Basic functionality only)
```bash
pip install -r requirements-minimal.txt
```

### Optional (Advanced features)
```bash
pip install -r requirements.txt
pip install -r requirements-optional.txt
```

### Environment Setup
```bash
# Set up environment variables
cp .env.example .env
# Edit .env with your configuration
```

See [LIBRARIES_GUIDE.md](LIBRARIES_GUIDE.md) for detailed information about all libraries.

## Configuration

Create a `.env` file with the following variables:

```env
# Server settings
HOST=0.0.0.0
PORT=8025
LOG_LEVEL=INFO

# AI Model settings
MODEL_PROVIDER=openai
MODEL_NAME=gpt-4-vision-preview
API_KEY=your_api_key_here

# Image processing
MAX_IMAGE_SIZE_MB=10
OUTPUT_FORMAT=png

# Storage
UPLOAD_DIR=./storage/uploads
OUTPUT_DIR=./storage/outputs
```

## Usage

### Quick Start

```bash
# Setup storage directories
python scripts/setup_storage.py

# Start the server
python main.py
```

Or with uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 8025 --reload
```

### Using Makefile

```bash
# Install dependencies
make install-dev

# Setup storage
make setup

# Run tests
make test

# Format code
make format

# Run server
make run
```

### API Endpoints

#### Health Check
```
GET /health/
GET /health/ready
```

#### Create Visualization
```
POST /api/v1/visualize
```

Request body:
```json
{
  "surgery_type": "rhinoplasty",
  "intensity": 0.7,
  "image_url": "https://example.com/image.jpg",
  "target_areas": ["nose"],
  "additional_notes": "Make nose slightly smaller"
}
```

#### Upload Image and Visualize
```
POST /api/v1/visualize/upload
```

Form data:
- `file`: Image file
- `surgery_type`: Type of surgery (rhinoplasty, facelift, etc.)
- `intensity`: Optional, 0.0 to 1.0 (default: 0.5)

#### Get Visualization
```
GET /api/v1/visualize/{visualization_id}
```

#### Get Available Surgery Types
```
GET /api/v1/surgery-types
```

#### Get Metrics
```
GET /api/v1/metrics/
GET /api/v1/metrics/counters
GET /api/v1/metrics/timings
```

#### Create Comparison
```
POST /api/v1/compare
```

Request body:
```json
{
  "visualization_id": "uuid",
  "include_original": true,
  "layout": "side_by_side"
}
```

#### Batch Processing
```
POST /api/v1/batch
```

Request body:
```json
{
  "requests": [
    {"surgery_type": "rhinoplasty", "intensity": 0.7, "image_url": "..."}
  ],
  "max_concurrent": 3
}
```

#### Get Service Info
```
GET /api/v1/info
GET /api/v1/version
```

## Architecture

The project follows a **modular architecture** with clear separation of concerns. See [MODULAR_ARCHITECTURE.md](MODULAR_ARCHITECTURE.md) for detailed documentation.

### Project Structure

```
plastic_surgery_visualization_ai/
├── api/                    # Presentation layer (HTTP)
│   ├── routes/            # API endpoints
│   └── schemas/           # Pydantic models
├── config/                # Configuration
│   └── settings.py        # Application settings
├── core/                  # Core application modules
│   ├── app_factory.py    # Application factory
│   ├── lifespan.py       # Lifecycle management
│   ├── middleware_config.py  # Middleware setup
│   ├── exceptions_config.py   # Exception handlers
│   ├── routes_config.py   # Routes registration
│   ├── factories.py      # Service factories
│   ├── dependencies.py   # FastAPI dependencies
│   ├── interfaces.py      # Protocol interfaces
│   └── services/         # Core services
├── domain/                # Domain layer (business logic)
│   └── use_cases/        # Use cases
├── infrastructure/        # Infrastructure layer
│   ├── repositories/     # Data repositories
│   └── adapters/         # Service adapters
├── services/              # Application services (facade)
│   └── visualization_service.py
├── utils/                 # Utilities
├── tests/                 # Tests
├── examples/             # Example code
├── main.py               # Application entry point
└── requirements.txt      # Dependencies
```

### Core Modules

- **`core/app_factory.py`**: Application factory for creating FastAPI app
- **`core/lifespan.py`**: Application lifecycle management (startup/shutdown)
- **`core/middleware_config.py`**: Middleware configuration
- **`core/exceptions_config.py`**: Exception handlers configuration
- **`core/routes_config.py`**: Routes registration
- **`core/factories.py`**: Service factory functions
- **`core/dependencies.py`**: FastAPI dependency injection

### Architecture Layers

The project follows **Clean Architecture** principles with clear layer separation:

1. **Domain Layer** (`domain/`): Pure business logic, use cases
2. **Infrastructure Layer** (`infrastructure/`): Implementations (repositories, adapters)
3. **Application Layer** (`services/`): Service facades, orchestration
4. **Presentation Layer** (`api/`): HTTP endpoints, validation
5. **Core Layer** (`core/`): Interfaces, exceptions, constants

See [ARCHITECTURE_IMPROVEMENTS.md](ARCHITECTURE_IMPROVEMENTS.md) for detailed architecture documentation.

### Key Benefits

- **Modularity**: Each module has a single, well-defined responsibility
- **Testability**: Easy to test individual components with interfaces
- **Maintainability**: Clear structure makes code easier to understand and modify
- **Scalability**: Easy to add new features without affecting existing code
- **Flexibility**: Easy to swap implementations (e.g., S3 storage, Redis cache)
- **SOLID Principles**: Follows SOLID design principles throughout

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

This project follows PEP 8 style guidelines.

## Architecture Improvements

### Middleware
- **Rate Limiting**: Prevents API abuse with configurable limits
- **Security Headers**: Adds security headers to all responses
- **CORS**: Configurable CORS support

### Error Handling
- Custom exception hierarchy for better error messages
- Proper HTTP status codes
- Detailed error logging

### Caching
- File-based caching system
- Configurable TTL (default: 24 hours)
- Automatic cache key generation

### Metrics
- Request counters
- Processing time tracking
- Per-surgery-type metrics
- Daily metrics export

## Future Enhancements

- Integration with actual AI vision models (GPT-4 Vision, Claude Vision)
- Face detection and landmark detection using dlib/MediaPipe
- 3D visualization support
- Before/after comparison views
- Multiple surgery combination previews
- Real-time processing with WebSockets
- Batch processing for multiple images
- User authentication and history

## License

Copyright © Blatam Academy

## Support

For issues and questions, please contact the development team.

