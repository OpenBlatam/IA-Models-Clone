# Logistics AI Platform

A comprehensive freight forwarding and logistics management system - A Nowports clone built with FastAPI.

## 🚀 Description

Logistics AI Platform is a complete digital freight forwarding solution that simplifies international trade in Latin America. It provides a centralized platform for managing quotes, bookings, shipments, containers, tracking, invoices, documents, alerts, and insurance.

## ✨ Features

### Core Functionality

- **Quotes Management**: Generate and manage freight quotes for air, maritime, and ground transportation
- **Bookings**: Create bookings from quotes with automatic shipment generation
- **Shipments**: Comprehensive shipment management with real-time status tracking
- **Containers**: Container lifecycle management with GPS tracking
- **Real-time Tracking**: Track shipments and containers with detailed event history
- **Invoices**: Generate and manage invoices for shipments
- **Documents**: Upload, manage, and organize shipping documents
- **Alerts**: Create and manage alerts for shipments and containers
- **Insurance**: Manage cargo and container insurance policies
- **Reports**: Dashboard statistics and shipment reports

### Architecture Highlights

- **Modular Design**: Separated concerns with dedicated modules for routes, business logic, and data access
- **Repository Pattern**: Abstracted data access for easy database switching
- **Pure Functions**: Business logic extracted into testable, pure functions
- **Dependency Injection**: FastAPI's DI system for clean, testable code
- **Validation Layer**: Dedicated validators for input validation
- **Caching**: Redis support with in-memory fallback, optimized with OrJSON
- **Error Handling**: Custom exceptions with structured error responses
- **Performance**: Batch processing, parallel execution, lazy loading
- **Logging**: Loguru for structured, rotating logs
- **Monitoring**: Performance middleware with request timing
- **Geospatial**: Real distance calculations with Geopy

### Transportation Modes

- **Air Freight**: Fast delivery with express service
- **Maritime Freight**: Cost-effective ocean freight with container tracking
- **Ground Transport**: Door-to-door ground transportation
- **Multimodal**: Combined transportation solutions

### Key Capabilities

- Real-time GPS tracking for containers
- Comprehensive tracking event history
- Document management and organization
- Automated alert system
- Insurance policy management
- Dashboard with key metrics
- RESTful API with OpenAPI documentation

## 📁 Project Structure

```
logistics_ai_platform/
├── api/                    # API Routes (HTTP layer)
│   ├── quotes/            # Quote routes module
│   ├── bookings/          # Booking routes module
│   ├── shipments/         # Shipment routes module
│   ├── containers/        # Container routes module
│   ├── forwarding_routes.py
│   ├── tracking_routes.py
│   ├── invoice_routes.py
│   ├── document_routes.py
│   ├── alert_routes.py
│   ├── insurance_routes.py
│   └── report_routes.py
├── handlers/               # Request Handlers (orchestration)
│   ├── quote_handlers.py
│   ├── booking_handlers.py
│   └── shipment_handlers.py
├── domain/                 # Domain Logic (pure functions)
│   ├── quotes.py
│   ├── bookings.py
│   └── shipments.py
├── factories/              # Object Factories (creation)
│   ├── quote_factory.py
│   ├── booking_factory.py
│   └── shipment_factory.py
├── repositories/           # Data Access (abstraction)
│   ├── quote_repository.py
│   ├── booking_repository.py
│   ├── shipment_repository.py
│   └── container_repository.py
├── business_logic/         # Business Rules (pure functions)
│   ├── quote_logic.py
│   ├── booking_logic.py
│   └── shipment_logic.py
├── validators/             # Validation (pure functions)
│   ├── quote_validators.py
│   ├── booking_validators.py
│   └── shipment_validators.py
├── core/                   # Service layer (legacy)
│   ├── quote_service.py
│   ├── booking_service.py
│   ├── shipment_service.py
│   ├── container_service.py
│   └── tracking_service.py
├── services/               # Additional services
│   ├── invoice_service.py
│   ├── document_service.py
│   ├── alert_service.py
│   └── insurance_service.py
├── models/                 # Data models (Pydantic schemas)
│   └── schemas.py
├── config/                 # Configuration
│   └── settings.py
├── middleware/             # Middleware
│   ├── rate_limiter.py
│   └── performance.py
├── utils/                   # Utilities
│   ├── cache.py
│   ├── dependencies.py
│   ├── exceptions.py
│   ├── error_handler.py
│   ├── decorators.py
│   ├── performance.py
│   ├── async_helpers.py
│   ├── response.py
│   ├── geospatial.py
│   ├── json_serializer.py
│   └── logger.py
├── tests/                   # Tests
├── main.py                  # Main application
├── requirements.txt
├── requirements-dev.txt
├── requirements-prod.txt
├── README.md
├── ARCHITECTURE.md
├── IMPROVEMENTS.md
└── LIBRARIES.md
```

### Architecture Layers

1. **API Layer** (`api/`): HTTP endpoints, declarative route definitions
2. **Handler Layer** (`handlers/`): Request orchestration, cache, side effects
3. **Domain Layer** (`domain/`): Pure business logic functions
4. **Factory Layer** (`factories/`): Object creation functions
5. **Business Logic Layer** (`business_logic/`): Pure functions for calculations
6. **Repository Layer** (`repositories/`): Data access abstraction
7. **Validation Layer** (`validators/`): Input validation functions
8. **Models Layer** (`models/`): Data schemas and types

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed architecture documentation.

## 🔧 Installation

### Prerrequisites

- Python 3.8+
- pip

### Setup

1. Navigate to the project directory:
```bash
cd logistics_ai_platform
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file (optional):
```env
# Server
HOST=0.0.0.0
PORT=8030
LOG_LEVEL=INFO

# Database (optional)
DATABASE_URL=sqlite+aiosqlite:///./logistics.db

# Redis (optional)
REDIS_URL=redis://localhost:6379/0

# External APIs (optional)
GOOGLE_MAPS_API_KEY=your_google_maps_api_key
WEATHER_API_KEY=your_weather_api_key

# AI Services (optional)
OPENAI_API_KEY=your_openai_api_key

# Security
SECRET_KEY=your-secret-key-change-in-production
```

## 🚀 Usage

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
python main.py
```

The server will be available at `http://localhost:8030`

### Docker Deployment

```bash
# Using docker-compose (includes Redis)
docker-compose up -d

# Or build and run manually
docker build -t logistics-ai-platform .
docker run -p 8030:8030 logistics-ai-platform
```

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment instructions.

### API Documentation

Once the server is running, access:
- **Swagger UI**: `http://localhost:8030/docs`
- **ReDoc**: `http://localhost:8030/redoc`

## 📖 API Endpoints

### Forwarding

#### Quotes
- `POST /forwarding/quotes` - Create a new quote
- `GET /forwarding/quotes/{quote_id}` - Get quote by ID

#### Bookings
- `POST /forwarding/bookings` - Create a new booking
- `GET /forwarding/bookings/{booking_id}` - Get booking by ID

#### Shipments
- `POST /forwarding/shipments` - Create a new shipment
- `GET /forwarding/shipments` - Get shipments (with optional filtering)
- `GET /forwarding/shipments/{shipment_id}` - Get shipment by ID
- `PATCH /forwarding/shipments/{shipment_id}/status` - Update shipment status

#### Containers
- `POST /forwarding/containers` - Create a new container
- `GET /forwarding/containers/{container_id}` - Get container by ID
- `GET /forwarding/containers/shipment/{shipment_id}` - Get containers for shipment
- `PATCH /forwarding/containers/{container_id}/status` - Update container status

### Tracking

- `GET /tracking/shipment/{shipment_id}` - Get tracking info for shipment
- `GET /tracking/container/{container_id}` - Get tracking info for container
- `GET /tracking/shipment/{shipment_id}/history` - Get tracking history
- `POST /tracking/shipment/{shipment_id}/update` - Add tracking update
- `GET /tracking/summary` - Get tracking summary (departing, arriving, in transit)

### Invoices

- `POST /invoices` - Create a new invoice
- `GET /invoices` - Get all invoices
- `GET /invoices/{invoice_id}` - Get invoice by ID
- `GET /invoices/shipment/{shipment_id}` - Get invoices for shipment

### Documents

- `POST /documents` - Upload a document
- `GET /documents/{document_id}` - Get document by ID
- `GET /documents/shipment/{shipment_id}` - Get documents for shipment
- `DELETE /documents/{document_id}` - Delete a document

### Alerts

- `POST /alerts` - Create a new alert
- `GET /alerts` - Get alerts (with optional filtering)
- `GET /alerts/{alert_id}` - Get alert by ID
- `PATCH /alerts/{alert_id}/read` - Mark alert as read
- `DELETE /alerts/{alert_id}` - Delete an alert

### Insurance

- `POST /insurance` - Create insurance policy
- `GET /insurance/{insurance_id}` - Get insurance by ID
- `GET /insurance/shipment/{shipment_id}` - Get insurance for shipment

### Reports

- `GET /reports/dashboard` - Get dashboard statistics
- `GET /reports/shipments` - Get shipment report

### Metrics & Monitoring

- `GET /metrics` - Prometheus metrics endpoint
- `GET /metrics/info` - Metrics information
- `GET /health` - Health check endpoint
- `GET /ready` - Readiness check endpoint

## 📝 Example Usage

### Create a Quote

```python
import requests

url = "http://localhost:8030/forwarding/quotes"
data = {
    "origin": {
        "country": "Mexico",
        "city": "Veracruz",
        "port_code": "MXVER"
    },
    "destination": {
        "country": "Honduras",
        "city": "Comayagua",
        "port_code": "HNCMY"
    },
    "cargo": {
        "description": "Electronics",
        "weight_kg": 1000,
        "volume_m3": 5.0,
        "quantity": 10,
        "unit_type": "CTN",
        "value_usd": 50000
    },
    "transportation_mode": "maritime",
    "insurance_required": True
}

response = requests.post(url, json=data)
quote = response.json()
```

### Create a Booking

```python
url = "http://localhost:8030/forwarding/bookings"
data = {
    "quote_id": "Q12345678",
    "selected_option_id": "option_id",
    "shipper_info": {
        "name": "Shipper Company",
        "email": "shipper@example.com",
        "phone": "+1234567890"
    },
    "consignee_info": {
        "name": "Consignee Company",
        "email": "consignee@example.com",
        "phone": "+0987654321"
    },
    "payment_terms": "NET 30"
}

response = requests.post(url, json=data)
booking = response.json()
```

### Track a Shipment

```python
url = "http://localhost:8030/tracking/shipment/S12345678"
response = requests.get(url)
tracking_info = response.json()
```

## 🔒 Security

- Rate limiting to prevent abuse
- CORS middleware for cross-origin requests
- Input validation with Pydantic
- Secure file upload handling

## 🧪 Testing

### Run Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run specific test file
pytest tests/test_health.py

# Run with verbose output
pytest tests/ -v
```

### Test Structure

- `tests/conftest.py` - Pytest fixtures and configuration
- `tests/test_health.py` - Health check endpoint tests
- `tests/test_quotes.py` - Quote endpoint tests
- `tests/test_validation.py` - Validation utility tests

### Test Coverage

The test suite includes:
- Health check endpoints
- Quote creation and retrieval
- Input validation
- Error handling
- Security validations

## 📊 Status Codes

- `200` - Success
- `201` - Created
- `400` - Bad Request
- `404` - Not Found
- `429` - Rate Limit Exceeded
- `500` - Internal Server Error

## ✨ Recent Improvements

### Prometheus Metrics
- Comprehensive metrics collection for HTTP requests, business operations, cache, and system health
- Prometheus-compatible endpoint at `/metrics` for monitoring integration
- Automatic metrics recording for all API operations

### Enhanced Health Checks
- Detailed health check endpoint (`/health`) with service status
- Readiness check endpoint (`/ready`) for deployment orchestration
- Individual service health monitoring (cache, database, etc.)

### API Versioning
- Support for API versioning via headers (`Accept-Version`, `X-API-Version`)
- Version validation and error handling
- Ready for future API versions

### Performance Monitoring
- Integrated Prometheus metrics in performance middleware
- Automatic error tracking and reporting
- Request duration monitoring with slow request alerts

### Docker & Containerization
- Production-ready Dockerfile with health checks
- docker-compose setup with Redis
- Optimized builds and deployment

### Structured Logging
- Enhanced logging with request context (request_id, client info, timing)
- Dynamic log levels based on performance
- Production-ready structured format

### OpenAPI Documentation
- Complete request/response examples in Swagger/ReDoc
- Organized tags and detailed descriptions
- Better developer experience

### Integration Tests
- End-to-end workflow tests
- Complete test coverage for critical paths
- Error handling validation

## 🚧 Future Enhancements

- Database integration (PostgreSQL/MySQL)
- Real-time WebSocket updates
- Google Maps integration for route visualization
- AI-powered route optimization
- Email notifications
- SMS alerts
- Multi-language support
- Advanced analytics and reporting
- Integration with carrier APIs
- Mobile app support

## 📄 License

This project is part of the Blatam Academy system.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

For support, please open an issue in the repository.

---

**Built with FastAPI** - A modern, fast web framework for building APIs with Python.

