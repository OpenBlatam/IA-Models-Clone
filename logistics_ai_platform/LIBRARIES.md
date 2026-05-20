# Libraries & Dependencies Guide

This document describes the libraries used in the Logistics AI Platform and their purposes.

## Core Framework

### FastAPI (0.115.0)
- **Purpose**: Modern, fast web framework for building APIs
- **Why**: High performance, automatic API documentation, type hints support
- **Usage**: Main application framework

### Uvicorn (0.32.0)
- **Purpose**: ASGI server for running FastAPI
- **Why**: High performance, supports async/await, HTTP/2, WebSockets
- **Usage**: Production server with `uvicorn[standard]` for better performance

### Starlette (0.41.0)
- **Purpose**: Lightweight ASGI framework (FastAPI is built on it)
- **Why**: Provides core ASGI functionality
- **Usage**: Included as FastAPI dependency

## Data Validation & Settings

### Pydantic (2.9.2)
- **Purpose**: Data validation using Python type annotations
- **Why**: Fast, extensible, integrates perfectly with FastAPI
- **Usage**: Request/response models, settings validation

### Pydantic Settings (2.5.2)
- **Purpose**: Settings management using Pydantic
- **Why**: Type-safe configuration, environment variable support
- **Usage**: Application configuration (`config/settings.py`)

### Pydantic Extra Types (2.6.0)
- **Purpose**: Additional field types for Pydantic
- **Why**: Extended validation types (URLs, IP addresses, etc.)
- **Usage**: Enhanced validation in schemas

## Database

### SQLAlchemy (2.0.36)
- **Purpose**: SQL toolkit and ORM
- **Why**: Powerful ORM, async support, database abstraction
- **Usage**: Database models and queries

### Alembic (1.13.2)
- **Purpose**: Database migration tool
- **Why**: Version control for database schema
- **Usage**: Database migrations

### AsyncPG (0.30.0)
- **Purpose**: Async PostgreSQL driver
- **Why**: Fast, native async support for PostgreSQL
- **Usage**: PostgreSQL database connections

### AIOMySQL (0.2.0)
- **Purpose**: Async MySQL driver
- **Why**: Async support for MySQL databases
- **Usage**: MySQL database connections

### AIOSQLite (0.20.0)
- **Purpose**: Async SQLite driver
- **Why**: Async support for SQLite (development/testing)
- **Usage**: SQLite database connections

### Psycopg2-binary (2.9.10)
- **Purpose**: PostgreSQL adapter (sync)
- **Why**: Required for Alembic migrations
- **Usage**: Database migrations

## Caching & Redis

### Redis (5.2.0)
- **Purpose**: In-memory data structure store
- **Why**: Fast caching, session storage, pub/sub
- **Usage**: Caching layer (`utils/cache.py`)

### Hiredis (2.3.2)
- **Purpose**: Fast Redis protocol parser
- **Why**: Performance improvement for Redis operations
- **Usage**: Included with Redis for better performance

### AIORedis (2.0.1)
- **Purpose**: Async Redis client
- **Why**: Async/await support for Redis operations
- **Usage**: Async Redis operations

## HTTP Client

### HTTPX (0.27.2)
- **Purpose**: Async HTTP client
- **Why**: Modern alternative to requests, async support
- **Usage**: External API calls, testing

### HTTPCore (1.0.5)
- **Purpose**: Low-level HTTP library
- **Why**: Foundation for HTTPX
- **Usage**: Included as HTTPX dependency

## Authentication & Security

### Python-JOSE (3.3.0)
- **Purpose**: JWT implementation
- **Why**: Secure token generation and validation
- **Usage**: Authentication tokens

### Passlib (1.7.4)
- **Purpose**: Password hashing library
- **Why**: Secure password hashing with bcrypt
- **Usage**: User password hashing

### Cryptography (43.0.1)
- **Purpose**: Cryptographic recipes and primitives
- **Why**: Security operations, encryption
- **Usage**: Security utilities

### Python-Multipart (0.0.12)
- **Purpose**: Multipart form data parsing
- **Why**: File uploads, form data
- **Usage**: File upload endpoints

## File Handling

### AIOFiles (24.1.0)
- **Purpose**: Async file operations
- **Why**: Non-blocking file I/O
- **Usage**: Document uploads, file operations

### Python-Magic (0.4.27)
- **Purpose**: File type detection
- **Why**: Validate file types, security
- **Usage**: Document validation

## Logging

### Loguru (0.7.2)
- **Purpose**: Modern logging library
- **Why**: Beautiful, structured logging, easy configuration
- **Usage**: Application logging (`utils/logger.py`)

### Structlog (24.4.0)
- **Purpose**: Structured logging
- **Why**: JSON logging, better for production
- **Usage**: Production logging (optional)

## Monitoring & Observability

### Prometheus Client (0.21.0)
- **Purpose**: Metrics collection
- **Why**: Application metrics, monitoring
- **Usage**: Performance metrics

## Task Queue

### Celery (5.4.0)
- **Purpose**: Distributed task queue
- **Why**: Background jobs, async processing
- **Usage**: Background tasks (optional)

## Email

### AIOSMTPLib (3.0.2)
- **Purpose**: Async SMTP client
- **Why**: Non-blocking email sending
- **Usage**: Email notifications

### Email-Validator (2.2.0)
- **Purpose**: Email validation
- **Why**: Validate email addresses
- **Usage**: User input validation

## Date & Time

### Python-Dateutil (2.9.0)
- **Purpose**: Date/time utilities
- **Why**: Date parsing, timezone handling
- **Usage**: Date operations

### Pytz (2024.2)
- **Purpose**: Timezone definitions
- **Why**: Timezone support
- **Usage**: Timezone conversions

## Utilities

### Typing-Extensions (4.12.2)
- **Purpose**: Backported type hints
- **Why**: Extended type support for older Python versions
- **Usage**: Type hints

### OrJSON (3.10.7)
- **Purpose**: Fast JSON serialization
- **Why**: Performance improvement over standard json
- **Usage**: JSON serialization (`utils/json_serializer.py`)

### UJSON (5.10.0)
- **Purpose**: Ultra-fast JSON encoder/decoder
- **Why**: Alternative fast JSON library
- **Usage**: JSON operations (alternative)

## Testing

### Pytest (8.3.3)
- **Purpose**: Testing framework
- **Why**: Powerful, plugin ecosystem
- **Usage**: Unit and integration tests

### Pytest-Asyncio (0.24.0)
- **Purpose**: Async test support
- **Why**: Test async functions
- **Usage**: Async test fixtures

### Pytest-Cov (6.0.0)
- **Purpose**: Coverage reporting
- **Why**: Code coverage metrics
- **Usage**: Test coverage

### Faker (30.0.0)
- **Purpose**: Test data generation
- **Why**: Generate realistic test data
- **Usage**: Test fixtures

## Development Tools

### Black (24.8.0)
- **Purpose**: Code formatter
- **Why**: Consistent code style
- **Usage**: Code formatting

### Ruff (0.6.9)
- **Purpose**: Fast linter
- **Why**: Fast alternative to flake8/pylint
- **Usage**: Code linting

### MyPy (1.11.2)
- **Purpose**: Static type checker
- **Why**: Type checking
- **Usage**: Type validation

### Pre-Commit (3.8.0)
- **Purpose**: Git hooks framework
- **Why**: Pre-commit checks
- **Usage**: Code quality checks

## Rate Limiting

### SlowAPI (0.1.9)
- **Purpose**: Rate limiting for FastAPI
- **Why**: API rate limiting
- **Usage**: Rate limiting middleware

## Background Tasks

### APScheduler (3.10.4)
- **Purpose**: Advanced Python Scheduler
- **Why**: Scheduled tasks, cron jobs
- **Usage**: Background scheduling

## Geospatial

### Geopy (2.4.1)
- **Purpose**: Geocoding and distance calculations
- **Why**: Location services, distance calculations
- **Usage**: Logistics calculations (`utils/geospatial.py`)

### Shapely (2.0.5)
- **Purpose**: Geometric operations
- **Why**: Spatial calculations
- **Usage**: Geographic operations

## Performance

### MSGPack (1.1.0)
- **Purpose**: Binary serialization
- **Why**: Faster than JSON for some use cases
- **Usage**: Binary data serialization

## Health Checks

### Healthcheck (1.3.3)
- **Purpose**: Health check endpoints
- **Why**: Application health monitoring
- **Usage**: Health endpoints

## Installation

### Development
```bash
pip install -r requirements-dev.txt
```

### Production
```bash
pip install -r requirements-prod.txt
```

### Minimal
```bash
pip install -r requirements.txt
```

## Version Management

All versions are pinned for reproducibility. Update carefully and test thoroughly.

## Security

Regularly update dependencies to patch security vulnerabilities:
```bash
pip install --upgrade pip
pip list --outdated
safety check
```













