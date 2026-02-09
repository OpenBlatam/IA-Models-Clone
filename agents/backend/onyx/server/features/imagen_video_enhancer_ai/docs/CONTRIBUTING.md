# Contributing Guide - Imagen Video Enhancer AI

Thank you for your interest in contributing to Imagen Video Enhancer AI!

## Getting Started

1. Fork the repository
2. Clone your fork
3. Create a virtual environment
4. Install dependencies: `pip install -r requirements.txt`
5. Install development dependencies: `pip install pytest pytest-asyncio black flake8 mypy`
6. Run tests: `pytest`

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Follow the code style (use `black` for formatting)
- Add type hints to all functions
- Write docstrings for public APIs
- Add tests for new features

### 3. Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=imagen_video_enhancer_ai --cov-report=html

# Run specific test file
pytest tests/test_validators.py
```

### 4. Check Code Quality

```bash
# Format code
black imagen_video_enhancer_ai/

# Lint code
flake8 imagen_video_enhancer_ai/

# Type check
mypy imagen_video_enhancer_ai/
```

### 5. Commit Changes

```bash
git add .
git commit -m "feat: add new feature"
```

Follow [Conventional Commits](https://www.conventionalcommits.org/):
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation
- `refactor:` - Code refactoring
- `test:` - Tests
- `chore:` - Maintenance

### 6. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

## Code Style

### Python Style

- Follow PEP 8
- Use `black` for formatting (line length: 100)
- Use type hints for all functions
- Maximum line length: 100 characters

### Naming Conventions

- Classes: `PascalCase`
- Functions/methods: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private methods: `_leading_underscore`

### Documentation

- Use docstrings for all public functions/classes
- Follow Google-style docstrings
- Include type information in docstrings

Example:
```python
def process_file(
    file_path: str,
    options: Optional[Dict[str, Any]] = None
) -> ProcessingResult:
    """
    Process a file with given options.
    
    Args:
        file_path: Path to the file to process
        options: Optional processing options
        
    Returns:
        ProcessingResult with processing outcome
        
    Raises:
        ValidationError: If file is invalid
        ProcessingError: If processing fails
    """
```

## Testing Guidelines

### Test Structure

- One test file per module
- Use descriptive test names
- Test both success and error cases
- Use fixtures from `conftest.py`

### Example Test

```python
import pytest
from imagen_video_enhancer_ai.utils.validators import FileValidator

def test_validate_image_file_success(sample_image_path):
    """Test successful image file validation."""
    result = FileValidator.validate_image_file(sample_image_path)
    assert result is True

def test_validate_image_file_invalid_extension():
    """Test validation with invalid extension."""
    with pytest.raises(ValidationError):
        FileValidator.validate_image_file("test.txt")
```

## Adding New Features

### 1. Service Handler

1. Create handler in `core/services/`
2. Inherit from `BaseServiceHandler`
3. Implement required methods
4. Register in `ServiceHandlerRegistry`
5. Add tests

### 2. API Endpoint

1. Create route in `api/routes/`
2. Add request/response models in `api/models.py`
3. Register route in `api/enhancer_api.py`
4. Add tests
5. Update API documentation

### 3. Utility Function

1. Add to appropriate `utils/` module
2. Add type hints
3. Add docstring
4. Add tests
5. Update documentation if needed

## Pull Request Process

1. **Update Documentation**: Update relevant docs
2. **Add Tests**: Ensure all new code is tested
3. **Update CHANGELOG**: Add entry to CHANGELOG.md
4. **Check CI**: Ensure all CI checks pass
5. **Request Review**: Request review from maintainers

## Code Review Guidelines

### For Contributors

- Be responsive to feedback
- Make requested changes promptly
- Ask questions if unclear
- Keep PRs focused and small

### For Reviewers

- Be constructive and respectful
- Explain reasoning for suggestions
- Approve when satisfied
- Request changes when needed

## Questions?

- Open an issue for bugs
- Open a discussion for questions
- Check existing documentation
- Review existing code examples

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.




