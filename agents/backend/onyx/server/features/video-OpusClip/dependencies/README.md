# Dependencies Module for Video-OpusClip

Comprehensive dependency management and injection system for the Video-OpusClip project.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dependency Container](#dependency-container)
- [Service Registration](#service-registration)
- [FastAPI Integration](#fastapi-integration)
- [Dependency Management](#dependency-management)
- [Best Practices](#best-practices)
- [Examples](#examples)
- [API Reference](#api-reference)

## 🎯 Overview

The Dependencies module provides a comprehensive dependency injection and management system for the Video-OpusClip project. It includes:

- **Dependency Injection Container**: Service lifetime management and dependency resolution
- **Service Registration**: Utilities for registering different types of services
- **FastAPI Integration**: Seamless integration with FastAPI dependency injection
- **Dependency Management**: Tools for managing Python package dependencies

## ✨ Features

### 🔧 Dependency Injection
- **Service Lifetime Management**: Singleton, Transient, and Scoped services
- **Automatic Dependency Resolution**: Automatic resolution of constructor dependencies
- **Circular Dependency Detection**: Built-in circular dependency detection
- **Factory Support**: Support for factory functions and custom instantiation

### 🚀 FastAPI Integration
- **Seamless Integration**: Native FastAPI dependency injection support
- **Authentication Dependencies**: Built-in authentication and authorization dependencies
- **Service Dependencies**: Easy injection of business services
- **Dependency Overrides**: Support for dependency overrides in testing

### 📦 Dependency Management
- **Requirements Management**: Multiple requirements files for different environments
- **Dependency Checking**: Check installed vs required versions
- **Dependency Updates**: Automated dependency updates
- **Conflict Detection**: Detect and resolve dependency conflicts

## 🛠️ Installation

The dependencies module is included in the Video-OpusClip project. To install the project dependencies:

```bash
# Install main dependencies
pip install -r requirements/requirements.txt

# Install development dependencies
pip install -r requirements/requirements-dev.txt

# Install testing dependencies
pip install -r requirements/requirements-test.txt

# Install production dependencies
pip install -r requirements/requirements-prod.txt
```

## 🚀 Quick Start

### Basic Dependency Injection

```python
from dependencies import DependencyContainer, register_singleton, resolve

# Create container
container = DependencyContainer()

# Register services
container.register_singleton("ILogger", ConsoleLogger)
container.register_transient("EmailService", EmailService)

# Resolve services
email_service = container.resolve("EmailService")
email_service.send_email("user@example.com", "Hello!")
```

### FastAPI Integration

```python
from fastapi import FastAPI, Depends
from dependencies import FastAPIDependencyContainer, create_auth_dependency

# Create FastAPI container
container = FastAPIDependencyContainer()

# Register services
container.register_singleton("AuthService", AuthService)
container.register_singleton("UserService", UserService)

# Create FastAPI app
app = FastAPI()

# Create dependencies
auth_dependency = create_auth_dependency(container)

@app.get("/protected")
async def protected_route(user=Depends(auth_dependency)):
    return {"message": f"Hello {user.username}!"}
```

### Dependency Management

```python
from dependencies import DependencyManager

# Create dependency manager
manager = DependencyManager()

# Check dependencies
status = manager.check_dependencies()
print(f"Installed: {status['summary']['installed']}")
print(f"Outdated: {status['summary']['outdated']}")
print(f"Missing: {status['summary']['missing']}")

# Update dependencies
results = manager.update_dependencies()
print(f"Updated: {len(results['updated'])} packages")
```

## 🔧 Dependency Container

### Service Lifetimes

```python
from dependencies import ServiceLifetime

# Singleton - One instance for the entire application
container.register_singleton("DatabaseConnection", DatabaseConnection)

# Transient - New instance every time
container.register_transient("EmailService", EmailService)

# Scoped - One instance per scope
container.register_scoped("UserSession", UserSession)
```

### Factory Functions

```python
# Register with factory function
def create_database_connection(config):
    return DatabaseConnection(config.database_url)

container.register_singleton(
    "DatabaseConnection",
    factory=create_database_connection
)
```

### Dependency Resolution

```python
# Automatic dependency resolution
class EmailService:
    def __init__(self, logger: ILogger, config: Config):
        self.logger = logger
        self.config = config

# Dependencies are automatically resolved
email_service = container.resolve("EmailService")
```

## 🚀 Service Registration

### Database Services

```python
from dependencies import register_database_services

# Register all database services
container = register_database_services(container)

# Available services:
# - DatabaseConnection
# - DatabaseSession
# - DatabaseTransaction
# - DatabaseMigration
# - DatabaseHealthCheck
```

### Authentication Services

```python
from dependencies import register_auth_services

# Register all authentication services
container = register_auth_services(container)

# Available services:
# - AuthService
# - JWTService
# - PasswordService
# - PermissionService
# - RoleService
# - UserService
```

### Logging Services

```python
from dependencies import register_logging_services

# Register all logging services
container = register_logging_services(container)

# Available services:
# - LoggingService
# - StructuredLogger
# - LogFormatter
# - LogHandler
# - LogFilter
```

## 🌐 FastAPI Integration

### Authentication Dependencies

```python
from dependencies import create_auth_dependency, create_permission_dependency

# Create authentication dependency
auth_dependency = create_auth_dependency(container)

# Create permission dependency
permission_dependency = create_permission_dependency(container, "read_users")

@app.get("/users")
async def get_users(user=Depends(auth_dependency)):
    return {"users": []}

@app.post("/users")
async def create_user(user=Depends(permission_dependency)):
    return {"message": "User created"}
```

### Service Dependencies

```python
from dependencies import create_fastapi_dependency

# Create service dependencies
user_service_dep = create_fastapi_dependency(container, "UserService")
email_service_dep = create_fastapi_dependency(container, "EmailService")

@app.post("/users")
async def create_user(
    user_data: UserCreate,
    user_service=Depends(user_service_dep),
    email_service=Depends(email_service_dep)
):
    user = user_service.create_user(user_data)
    email_service.send_welcome_email(user.email)
    return user
```

### Dependency Overrides

```python
# Override dependencies for testing
container.override_dependency("AuthService", MockAuthService)
container.override_dependency("DatabaseConnection", MockDatabaseConnection)
```

## 📦 Dependency Management

### Checking Dependencies

```python
from dependencies import DependencyManager

manager = DependencyManager()

# Check all dependencies
status = manager.check_dependencies()

print(f"Total: {status['summary']['total']}")
print(f"Installed: {status['summary']['installed']}")
print(f"Outdated: {status['summary']['outdated']}")
print(f"Missing: {status['summary']['missing']}")

# Check specific dependency
dependency = manager.get_dependency_info("fastapi")
print(f"FastAPI: {dependency.installed_version} (required: {dependency.version})")
```

### Installing Dependencies

```python
# Install all dependencies
results = manager.install_dependencies()

# Install specific types
results = manager.install_dependencies(DependencyType.DEVELOPMENT)
results = manager.install_dependencies(DependencyType.TESTING)
results = manager.install_dependencies(DependencyType.PRODUCTION)

# Upgrade existing packages
results = manager.install_dependencies(upgrade=True)

# Force reinstall
results = manager.install_dependencies(force=True)
```

### Updating Dependencies

```python
# Update all outdated dependencies
results = manager.update_dependencies()

# Update specific types
results = manager.update_dependencies(DependencyType.DEVELOPMENT)

# Dry run (show what would be updated)
results = manager.update_dependencies(dry_run=True)
```

### Managing Dependencies

```python
# Add new dependency
manager.add_dependency(
    name="new-package",
    version="1.0.0",
    dep_type=DependencyType.PRODUCTION,
    description="New package for the project"
)

# Remove dependency
manager.remove_dependency("old-package")

# Generate requirements file
content = manager.generate_requirements_file(DependencyType.PRODUCTION)

# Export dependencies
json_export = manager.export_dependencies("json")
yaml_export = manager.export_dependencies("yaml")
toml_export = manager.export_dependencies("toml")
```

## 📋 Best Practices

### 1. Service Registration

```python
# ✅ Good: Register services by type
container.register_singleton(ILogger, ConsoleLogger)
container.register_transient(EmailService, EmailService)

# ❌ Bad: Register by string
container.register_singleton("logger", ConsoleLogger)
```

### 2. Dependency Lifetimes

```python
# ✅ Good: Use appropriate lifetimes
container.register_singleton(DatabaseConnection)  # Expensive to create
container.register_transient(EmailService)       # Lightweight
container.register_scoped(UserSession)           # Per request

# ❌ Bad: Use singleton for everything
container.register_singleton(EmailService)  # Unnecessary
```

### 3. Factory Functions

```python
# ✅ Good: Use factories for complex initialization
def create_database_connection(config: Config):
    return DatabaseConnection(
        host=config.database_host,
        port=config.database_port,
        credentials=config.database_credentials
    )

container.register_singleton(
    DatabaseConnection,
    factory=create_database_connection
)

# ❌ Bad: Complex initialization in constructor
class DatabaseConnection:
    def __init__(self):
        # Complex initialization logic here
        pass
```

### 4. FastAPI Integration

```python
# ✅ Good: Create dependencies once and reuse
auth_dependency = create_auth_dependency(container)
permission_dependency = create_permission_dependency(container, "admin")

@app.get("/admin")
async def admin_route(user=Depends(permission_dependency)):
    return {"message": "Admin access"}

# ❌ Bad: Create dependencies inline
@app.get("/admin")
async def admin_route(user=Depends(create_permission_dependency(container, "admin"))):
    return {"message": "Admin access"}
```

### 5. Error Handling

```python
# ✅ Good: Handle dependency resolution errors
try:
    service = container.resolve(ServiceType)
except Exception as e:
    logger.error(f"Failed to resolve {ServiceType}: {e}")
    # Handle gracefully

# ❌ Bad: Assume dependencies always resolve
service = container.resolve(ServiceType)  # May fail
```

## 📚 Examples

### Complete Application Setup

```python
from fastapi import FastAPI
from dependencies import (
    FastAPIDependencyContainer,
    register_database_services,
    register_auth_services,
    register_logging_services,
    create_auth_dependency,
    create_permission_dependency
)

# Create container
container = FastAPIDependencyContainer()

# Register services
container = register_database_services(container)
container = register_auth_services(container)
container = register_logging_services(container)

# Create FastAPI app
app = FastAPI()

# Create dependencies
auth_dependency = create_auth_dependency(container)
admin_dependency = create_permission_dependency(container, "admin")

# Routes
@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/protected")
async def protected(user=Depends(auth_dependency)):
    return {"message": f"Hello {user.username}!"}

@app.get("/admin")
async def admin(user=Depends(admin_dependency)):
    return {"message": "Admin access granted"}
```

### Testing with Dependency Overrides

```python
import pytest
from dependencies import FastAPIDependencyContainer

@pytest.fixture
def test_container():
    container = FastAPIDependencyContainer()
    
    # Register test services
    container.register_singleton("AuthService", MockAuthService)
    container.register_singleton("DatabaseConnection", MockDatabaseConnection)
    
    return container

def test_protected_route(test_container):
    auth_dependency = create_auth_dependency(test_container)
    
    # Test with mock dependencies
    user = auth_dependency()
    assert user.username == "test_user"
```

### Dependency Management Workflow

```python
from dependencies import DependencyManager

# Initialize
manager = DependencyManager()

# Check current state
status = manager.check_dependencies()
print(f"Missing: {status['summary']['missing']}")

# Install missing dependencies
if status['summary']['missing'] > 0:
    results = manager.install_dependencies()
    print(f"Installed: {len(results['success'])} packages")

# Update outdated dependencies
if status['summary']['outdated'] > 0:
    results = manager.update_dependencies()
    print(f"Updated: {len(results['updated'])} packages")

# Validate
validation = manager.validate_dependencies()
if not validation['valid']:
    print(f"Conflicts: {validation['conflicts']}")
```

## 📖 API Reference

### DependencyContainer

```python
class DependencyContainer:
    def register_singleton(self, service_type, implementation_type=None, factory=None, **metadata)
    def register_transient(self, service_type, implementation_type=None, factory=None, **metadata)
    def register_scoped(self, service_type, implementation_type=None, factory=None, **metadata)
    def resolve(self, service_type)
    def resolve_all(self, service_type)
    def create_scope(self)
    def validate(self)
```

### FastAPIDependencyContainer

```python
class FastAPIDependencyContainer(DependencyContainer):
    def register_fastapi_dependency(self, name, dependency_func, override=False)
    def override_dependency(self, name, dependency_func)
    def get_fastapi_dependency(self, name)
    def get_all_fastapi_dependencies(self)
```

### DependencyManager

```python
class DependencyManager:
    def install_dependencies(self, dep_type=None, upgrade=False, force=False)
    def uninstall_dependencies(self, package_names, yes=False)
    def check_dependencies(self)
    def update_dependencies(self, dep_type=None, dry_run=False)
    def generate_requirements_file(self, dep_type, output_file=None)
    def export_dependencies(self, format="json")
    def validate_dependencies(self)
    def add_dependency(self, name, version, dep_type, description=None)
    def remove_dependency(self, name)
```

### Utility Functions

```python
# Global functions
register_singleton(service_type, implementation_type=None, factory=None, **metadata)
register_transient(service_type, implementation_type=None, factory=None, **metadata)
register_scoped(service_type, implementation_type=None, factory=None, **metadata)
resolve(service_type)
resolve_all(service_type)
get_service_provider()

# FastAPI integration
create_auth_dependency(container)
create_permission_dependency(container, required_permission)
create_role_dependency(container, required_role)
create_logging_dependency(container)
create_cache_dependency(container)
create_database_dependency(container)
create_storage_dependency(container)
create_monitoring_dependency(container)
setup_fastapi_dependencies(container)

# Service registration
register_database_services(container)
register_auth_services(container)
register_logging_services(container)
register_cache_services(container)
register_queue_services(container)
register_storage_services(container)
register_monitoring_services(container)
```

## 🤝 Contributing

When contributing to the dependencies module:

1. **Follow the existing patterns** for service registration and dependency injection
2. **Add comprehensive tests** for new functionality
3. **Update documentation** for new features
4. **Use type hints** for all new functions and classes
5. **Follow the error handling patterns** established in the module

## 📄 License

This module is part of the Video-OpusClip project and follows the same license terms. 