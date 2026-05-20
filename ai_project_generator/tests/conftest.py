"""
Pytest configuration and fixtures for AI Project Generator tests
Enhanced with better utilities and helpers
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
import asyncio
import json
from datetime import datetime
from unittest.mock import Mock, MagicMock

from ..core.project_generator import ProjectGenerator
from ..core.backend_generator import BackendGenerator
from ..core.frontend_generator import FrontendGenerator
from ..core.continuous_generator import ContinuousGenerator


# ============================================================================
# Pytest Configuration
# ============================================================================

pytest_plugins = ["pytest_asyncio"]


# ============================================================================
# Directory and Path Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests with automatic cleanup"""
    temp_path = Path(tempfile.mkdtemp(prefix="ai_project_test_"))
    yield temp_path
    # Enhanced cleanup with error handling
    try:
        shutil.rmtree(temp_path, ignore_errors=True)
    except Exception:
        pass


@pytest.fixture
def test_data_dir(temp_dir):
    """Create a dedicated test data directory"""
    data_dir = temp_dir / "test_data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


@pytest.fixture
def sample_project_structure(temp_dir):
    """Create a sample project structure for testing"""
    project_dir = temp_dir / "sample_project"
    project_dir.mkdir()
    
    # Backend structure
    (project_dir / "backend").mkdir()
    (project_dir / "backend" / "app").mkdir()
    (project_dir / "backend" / "app" / "api").mkdir()
    (project_dir / "backend" / "app" / "core").mkdir()
    (project_dir / "backend" / "main.py").write_text("from fastapi import FastAPI\napp = FastAPI()")
    (project_dir / "backend" / "requirements.txt").write_text("fastapi\nuvicorn")
    
    # Frontend structure
    (project_dir / "frontend").mkdir()
    (project_dir / "frontend" / "src").mkdir()
    (project_dir / "frontend" / "package.json").write_text('{"name": "test", "version": "1.0.0"}')
    (project_dir / "frontend" / "vite.config.ts").write_text("export default {}")
    
    # Root files
    (project_dir / "README.md").write_text("# Test Project")
    (project_dir / ".gitignore").write_text("__pycache__/\n*.pyc")
    
    return project_dir


# ============================================================================
# Generator Fixtures
# ============================================================================

@pytest.fixture
def project_generator(temp_dir):
    """Create a ProjectGenerator instance with temporary directory"""
    return ProjectGenerator(base_output_dir=str(temp_dir / "generated_projects"))


@pytest.fixture
def backend_generator(temp_dir):
    """Create a BackendGenerator instance with temporary directory"""
    return BackendGenerator(output_dir=str(temp_dir / "backend"))


@pytest.fixture
def frontend_generator(temp_dir):
    """Create a FrontendGenerator instance with temporary directory"""
    return FrontendGenerator(output_dir=str(temp_dir / "frontend"))


@pytest.fixture
def continuous_generator(temp_dir):
    """Create a ContinuousGenerator instance with temporary directory"""
    return ContinuousGenerator(base_output_dir=str(temp_dir / "generated_projects"))


# ============================================================================
# Data Fixtures
# ============================================================================

@pytest.fixture
def sample_project_info():
    """Sample project information for testing"""
    return {
        "name": "test_project",
        "version": "1.0.0",
        "author": "Test Author",
        "description": "A test AI project",
        "ai_type": "chat",
        "backend_framework": "fastapi",
        "frontend_framework": "react"
    }


@pytest.fixture
def sample_keywords():
    """Sample keywords extracted from description"""
    return {
        "ai_type": "chat",
        "features": ["api", "websocket"],
        "requires_auth": False,
        "requires_database": False,
        "requires_api": True,
        "requires_ml": False,
        "requires_streaming": False,
        "requires_websocket": True,
        "requires_file_upload": False,
        "requires_cache": False,
        "requires_queue": False,
        "model_providers": ["openai"],
        "complexity": "medium",
        "is_deep_learning": False,
        "is_transformer": False,
        "is_diffusion": False,
        "is_llm": False,
        "requires_pytorch": False,
        "requires_gradio": False,
        "requires_training": False,
        "requires_fine_tuning": False,
        "model_architecture": None,
    }


@pytest.fixture
def sample_description():
    """Sample project description"""
    return "A chat AI system that responds to user questions using OpenAI"


@pytest.fixture
def sample_descriptions():
    """Multiple sample descriptions for testing"""
    return [
        "A chat AI system that responds to user questions",
        "An image recognition system that detects objects",
        "A voice assistant that processes audio commands",
        "A recommendation system for e-commerce",
        "A deep learning model for text classification"
    ]


@pytest.fixture
def mock_project_data():
    """Mock project data for testing"""
    return {
        "project_id": "test-123",
        "description": "Test project",
        "project_name": "test_project",
        "author": "Test Author",
        "status": "queued",
        "created_at": datetime.now().isoformat(),
        "priority": 1,
        "ai_type": "chat",
        "backend_framework": "fastapi",
        "frontend_framework": "react"
    }


# ============================================================================
# Async Fixtures
# ============================================================================

@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session"""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Helper Functions
# ============================================================================

def assert_project_structure(project_path: Path, required_files: List[str] = None):
    """Helper to assert project structure exists"""
    if required_files is None:
        required_files = ["README.md"]
    
    for file_path in required_files:
        full_path = project_path / file_path
        assert full_path.exists(), f"Required file {file_path} not found in {project_path}"


def assert_valid_json(file_path: Path):
    """Helper to assert a file contains valid JSON"""
    assert file_path.exists(), f"File {file_path} does not exist"
    try:
        json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        pytest.fail(f"File {file_path} contains invalid JSON: {e}")


def assert_valid_python(file_path: Path):
    """Helper to assert a file contains valid Python syntax"""
    assert file_path.exists(), f"File {file_path} does not exist"
    try:
        compile(file_path.read_text(encoding="utf-8"), str(file_path), "exec")
    except SyntaxError as e:
        pytest.fail(f"File {file_path} contains invalid Python: {e}")


def create_test_file(directory: Path, filename: str, content: str = ""):
    """Helper to create a test file"""
    file_path = directory / filename
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")
    return file_path


# ============================================================================
# Additional Fixtures
# ============================================================================

@pytest.fixture
def mock_async_function():
    """Fixture for creating async mock functions"""
    from unittest.mock import AsyncMock
    return AsyncMock


@pytest.fixture
def performance_timer():
    """Fixture for performance timing"""
    import time
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
            return self.elapsed
        
        @property
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
    
    return Timer()


@pytest.fixture
def sample_api_request():
    """Sample API request data"""
    return {
        "description": "A chat AI system",
        "project_name": "test_chat_ai",
        "author": "Test Author",
        "ai_type": "chat",
        "backend_framework": "fastapi",
        "frontend_framework": "react"
    }


@pytest.fixture
def sample_api_response():
    """Sample API response data"""
    return {
        "project_id": "test-123",
        "status": "completed",
        "project_path": "/path/to/project",
        "created_at": datetime.now().isoformat()
    }


# ============================================================================
# Pytest Markers
# ============================================================================

def pytest_configure(config):
    """Register custom markers"""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "async: marks tests as async tests")
    config.addinivalue_line("markers", "security: marks tests as security tests")
    config.addinivalue_line("markers", "performance: marks tests as performance tests")

