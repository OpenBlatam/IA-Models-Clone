"""
Advanced fixtures for complex test scenarios
"""

import pytest
import asyncio
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, MagicMock, AsyncMock
import tempfile
import shutil


@pytest.fixture
def complex_project_structure(temp_dir):
    """Create a complex project structure with all features"""
    project_dir = temp_dir / "complex_project"
    project_dir.mkdir()
    
    # Backend with all features
    backend = project_dir / "backend"
    backend.mkdir()
    
    (backend / "app").mkdir()
    (backend / "app" / "api").mkdir()
    (backend / "app" / "api" / "v1").mkdir()
    (backend / "app" / "core").mkdir()
    (backend / "app" / "models").mkdir()
    (backend / "app" / "services").mkdir()
    (backend / "app" / "utils").mkdir()
    (backend / "app" / "auth").mkdir()
    (backend / "app" / "database").mkdir()
    
    (backend / "main.py").write_text("""
from fastapi import FastAPI
from app.api.v1 import router

app = FastAPI()
app.include_router(router)
""")
    
    (backend / "requirements.txt").write_text("""
fastapi>=0.100.0
uvicorn>=0.20.0
pydantic>=2.0.0
sqlalchemy>=2.0.0
""")
    
    # Frontend with all features
    frontend = project_dir / "frontend"
    frontend.mkdir()
    
    (frontend / "src").mkdir()
    (frontend / "src" / "components").mkdir()
    (frontend / "src" / "pages").mkdir()
    (frontend / "src" / "services").mkdir()
    (frontend / "src" / "utils").mkdir()
    (frontend / "src" / "hooks").mkdir()
    
    (frontend / "package.json").write_text("""
{
  "name": "complex-project",
  "version": "1.0.0",
  "dependencies": {
    "react": "^18.0.0",
    "react-dom": "^18.0.0",
    "axios": "^1.0.0"
  }
}
""")
    
    (frontend / "vite.config.ts").write_text("""
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
})
""")
    
    # Root files
    (project_dir / "README.md").write_text("# Complex Project\n\nA full-stack AI project")
    (project_dir / ".gitignore").write_text("__pycache__/\n*.pyc\nnode_modules/")
    (project_dir / ".env.example").write_text("API_KEY=your_key_here")
    
    return project_dir


@pytest.fixture
def mock_external_services():
    """Mock external services (OpenAI, GitHub, etc.)"""
    mocks = {
        "openai": MagicMock(),
        "github": MagicMock(),
        "slack": MagicMock(),
        "email": MagicMock(),
    }
    
    # Configure OpenAI mock
    mocks["openai"].chat.completions.create = AsyncMock(return_value={
        "choices": [{
            "message": {
                "content": "Mocked response"
            }
        }]
    })
    
    # Configure GitHub mock
    mocks["github"].create_repo = AsyncMock(return_value={
        "id": 123,
        "name": "test-repo",
        "html_url": "https://github.com/test/test-repo"
    })
    
    return mocks


@pytest.fixture
def sample_projects_batch(temp_dir):
    """Create a batch of sample projects for testing"""
    projects = []
    
    for i in range(5):
        project_dir = temp_dir / f"project_{i}"
        project_dir.mkdir()
        
        (project_dir / "README.md").write_text(f"# Project {i}\n\nTest project {i}")
        (project_dir / "backend").mkdir()
        (project_dir / "backend" / "main.py").write_text(f"# Backend {i}")
        (project_dir / "frontend").mkdir()
        (project_dir / "frontend" / "package.json").write_text(f'{{"name": "project_{i}"}}')
        
        projects.append({
            "id": f"proj-{i}",
            "name": f"project_{i}",
            "path": str(project_dir)
        })
    
    return projects


@pytest.fixture
def error_scenarios():
    """Common error scenarios for testing"""
    return {
        "invalid_path": Path("/nonexistent/path"),
        "invalid_json": '{"invalid": json}',
        "invalid_python": "def invalid syntax here",
        "empty_string": "",
        "none_value": None,
        "large_string": "a" * 10000,
        "special_chars": "!@#$%^&*()",
        "unicode_string": "测试项目 🚀",
    }


@pytest.fixture
def performance_benchmarks():
    """Performance benchmark data"""
    return {
        "fast_threshold": 1.0,  # seconds
        "medium_threshold": 5.0,  # seconds
        "slow_threshold": 10.0,  # seconds
        "memory_limit_mb": 500,
        "cpu_limit_percent": 80,
    }


@pytest.fixture
def test_data_provider():
    """Provider for various test data"""
    class TestDataProvider:
        @staticmethod
        def get_ai_types():
            return ["chat", "vision", "audio", "nlp", "video", "recommendation"]
        
        @staticmethod
        def get_backend_frameworks():
            return ["fastapi", "flask", "django"]
        
        @staticmethod
        def get_frontend_frameworks():
            return ["react", "vue", "angular"]
        
        @staticmethod
        def get_features():
            return ["auth", "database", "websocket", "file_upload", "cache", "queue"]
        
        @staticmethod
        def get_project_configs():
            return [
                {"ai_type": "chat", "features": ["auth"]},
                {"ai_type": "vision", "features": ["file_upload"]},
                {"ai_type": "audio", "features": ["websocket"]},
            ]
    
    return TestDataProvider()


@pytest.fixture
def async_context_manager():
    """Async context manager for testing"""
    class AsyncContext:
        async def __aenter__(self):
            return self
        
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return False
    
    return AsyncContext

