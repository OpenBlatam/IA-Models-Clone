# Contributing to Audio Separator

Thank you for your interest in contributing to Audio Separator!

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/audio-separator.git`
3. Create a branch: `git checkout -b feature/your-feature-name`
4. Install in development mode: `pip install -e ".[dev]"`

## Development Guidelines

### Code Style

- Follow PEP 8 style guidelines
- Use Black for code formatting: `black .`
- Use type hints where appropriate
- Write docstrings for all public functions and classes

### Testing

- Write tests for new features
- Run tests: `pytest`
- Ensure all tests pass before submitting

### Submitting Changes

1. Ensure your code passes all tests
2. Update documentation if needed
3. Commit your changes with clear messages
4. Push to your fork
5. Create a pull request

## Adding New Models

To add a new separation model:

1. Create a new model class in `audio_separator/model/` that inherits from `BaseSeparatorModel`
2. Implement the required methods: `forward()` and `separate()`
3. Add the model to `model_builder.py`
4. Update documentation
5. Add tests

## Reporting Issues

When reporting issues, please include:
- Description of the problem
- Steps to reproduce
- Expected behavior
- Actual behavior
- Environment details (OS, Python version, etc.)

## Questions?

Feel free to open an issue for questions or discussions.

