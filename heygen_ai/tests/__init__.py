"""
Test package initialization for HeyGen AI features.
Sets up proper import paths for testing.
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path for proper imports
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# Also add the current directory for relative imports
sys.path.insert(0, str(current_dir))

from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
HeyGen AI FastAPI Service - Comprehensive Testing Framework
===========================================================

A comprehensive testing framework following clean architecture principles.

This testing framework provides:
- Unit tests for each layer (domain, application, infrastructure, presentation)
- Integration tests for complete workflows
- Performance and load testing
- Mocking and stubbing for external dependencies
- Test data factories and utilities
- Coverage reporting and quality metrics

Directory Structure:
tests/
├── unit/                    # Unit tests by layer
│   ├── domain/             # Domain layer tests
│   ├── application/        # Application layer tests  
│   ├── infrastructure/     # Infrastructure layer tests
│   └── presentation/       # Presentation layer tests
├── integration/            # Integration tests
├── performance/            # Performance and load tests
├── fixtures/               # Test fixtures and data
├── utils/                  # Test utilities and helpers
└── conftest.py            # Global pytest configuration

Test Categories:
- unit: Fast, isolated tests for individual components
- integration: Tests for component interactions and workflows
- performance: Load testing and performance benchmarks
- e2e: End-to-end system tests
- slow: Tests that take longer than 5 seconds
"""

__version__ = "1.0.0"
__author__ = "HeyGen AI Team" 