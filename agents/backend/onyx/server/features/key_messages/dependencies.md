# Dependencies Documentation

## Overview

The Key Messages feature uses a modular dependency structure to support different use cases and environments. This document explains the dependency organization and provides guidance on which requirements file to use.

## Dependency Files

### Core Files

- **`requirements.txt`** - Main production dependencies with version pinning
- **`requirements-minimal.txt`** - Minimal dependencies for basic functionality
- **`requirements-dev.txt`** - Development tools and debugging dependencies
- **`requirements-test.txt`** - Comprehensive testing and mocking dependencies
- **`requirements-cyber.txt`** - Cybersecurity and penetration testing tools
- **`requirements-prod.txt`** - Production-optimized dependencies

### Configuration Files

- **`pyproject.toml`** - Modern Python packaging configuration
- **`setup.py`** - Traditional Python packaging configuration
- **`Makefile`** - Development automation and dependency management

## Installation Options

### Basic Installation

```bash
# Install core dependencies only
pip install -r requirements.txt

# Install minimal dependencies
pip install -r requirements-minimal.txt
```

### Development Installation

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install all dependencies for development
make install-all
```

### Environment-Specific Installation

```bash
# Testing environment
pip install -r requirements-test.txt

# Cybersecurity environment
pip install -r requirements-cyber.txt

# Production environment
pip install -r requirements-prod.txt
```

### Using pip with Extras

```bash
# Install with development extras
pip install -e .[dev]

# Install with testing extras
pip install -e .[test]

# Install with cybersecurity extras
pip install -e .[cyber]

# Install with machine learning extras
pip install -e .[ml]

# Install with all extras
pip install -e .[all]
```

## Dependency Categories

### Core Dependencies

Essential packages required for basic functionality:

- **FastAPI** - Web framework
- **Pydantic** - Data validation
- **HTTPX/AioHTTP** - HTTP clients
- **Redis** - Caching and session storage
- **Structlog** - Structured logging

### Machine Learning Dependencies

AI/ML capabilities:

- **PyTorch** - Deep learning framework
- **Transformers** - Pre-trained models
- **NumPy/Pandas** - Data processing
- **Scikit-learn** - Machine learning utilities
- **Matplotlib/Seaborn** - Visualization

### Cybersecurity Dependencies

Security and penetration testing tools:

- **Cryptography** - Encryption and hashing
- **Paramiko** - SSH client/server
- **Nmap-python** - Network scanning
- **BeautifulSoup** - Web scraping
- **Selenium** - Web automation
- **Yara-python** - Malware detection

### Development Dependencies

Development and debugging tools:

- **Pytest** - Testing framework
- **Black** - Code formatting
- **Flake8** - Linting
- **MyPy** - Type checking
- **IPython** - Interactive shell
- **Memory-profiler** - Performance analysis

### Testing Dependencies

Comprehensive testing tools:

- **Pytest extensions** - Various testing utilities
- **Responses** - HTTP mocking
- **Factory-boy** - Test data generation
- **Coverage** - Code coverage
- **Locust** - Load testing

## Version Management

### Version Pinning Strategy

All dependencies use version pinning with upper bounds:

```txt
package>=1.0.0,<2.0.0
```

This ensures:
- Minimum required version for features
- Upper bound to prevent breaking changes
- Reproducible builds

### Security Considerations

- **Regular updates** - Dependencies are updated regularly
- **Vulnerability scanning** - Safety and bandit for security checks
- **Minimal attack surface** - Only necessary dependencies included
- **Version constraints** - Prevents known vulnerable versions

## Environment-Specific Configurations

### Development Environment

```bash
# Install development dependencies
make install-dev

# Run development server
make run-dev

# Run tests
make test

# Format code
make format
```

### Production Environment

```bash
# Install production dependencies
pip install -r requirements-prod.txt

# Run production server
make run-prod

# Security checks
make check-security
```

### Testing Environment

```bash
# Install testing dependencies
pip install -r requirements-test.txt

# Run all tests
make test

# Run specific test categories
make test-unit
make test-integration
make test-cyber
```

## Dependency Management Commands

### Using Makefile

```bash
# Install dependencies
make install-all

# Update dependencies
make update-deps

# Check for vulnerabilities
make check-deps

# Clean build artifacts
make clean
```

### Using pip

```bash
# Install with specific requirements
pip install -r requirements-cyber.txt

# Install with extras
pip install -e .[dev,test,cyber]

# Update specific package
pip install --upgrade package-name

# Check outdated packages
pip list --outdated
```

## Troubleshooting

### Common Issues

1. **Version Conflicts**
   ```bash
   # Check for conflicts
   pip check
   
   # Resolve conflicts
   pip install --upgrade conflicting-package
   ```

2. **Missing Dependencies**
   ```bash
   # Install missing dependencies
   pip install -r requirements.txt --force-reinstall
   ```

3. **Security Vulnerabilities**
   ```bash
   # Check for vulnerabilities
   make check-security
   
   # Update vulnerable packages
   pip install --upgrade vulnerable-package
   ```

### Platform-Specific Issues

#### Windows
- Some packages may require Visual C++ build tools
- Use `pip install --only-binary=all` for problematic packages

#### macOS
- Some packages may require Xcode command line tools
- Use Homebrew for system dependencies

#### Linux
- Install system dependencies: `sudo apt-get install python3-dev`
- Use virtual environment for isolation

## Best Practices

### Dependency Management

1. **Keep dependencies minimal** - Only include necessary packages
2. **Pin versions** - Use version constraints for reproducibility
3. **Regular updates** - Update dependencies regularly
4. **Security scanning** - Scan for vulnerabilities
5. **Documentation** - Document dependency purposes

### Environment Management

1. **Use virtual environments** - Isolate project dependencies
2. **Environment-specific files** - Use appropriate requirements files
3. **CI/CD integration** - Automate dependency management
4. **Monitoring** - Monitor dependency health

### Security

1. **Regular audits** - Audit dependencies regularly
2. **Vulnerability scanning** - Use automated security tools
3. **Minimal permissions** - Use least privilege principle
4. **Secure defaults** - Configure secure defaults

## Migration Guide

### Upgrading Dependencies

1. **Backup current state**
   ```bash
   pip freeze > requirements-current.txt
   ```

2. **Update dependencies**
   ```bash
   make update-deps
   ```

3. **Test thoroughly**
   ```bash
   make test
   make check-security
   ```

4. **Update documentation**
   - Update version constraints
   - Document breaking changes
   - Update migration notes

### Breaking Changes

When upgrading dependencies with breaking changes:

1. **Review changelog** - Check for breaking changes
2. **Update code** - Adapt to new APIs
3. **Test extensively** - Ensure functionality
4. **Update documentation** - Reflect changes

## Support

For dependency-related issues:

1. Check the troubleshooting section
2. Review package documentation
3. Check for known issues
4. Create an issue with detailed information

## Contributing

When adding new dependencies:

1. **Justify necessity** - Explain why the dependency is needed
2. **Check alternatives** - Consider lighter alternatives
3. **Update documentation** - Document the new dependency
4. **Test thoroughly** - Ensure compatibility
5. **Security review** - Check for security implications 