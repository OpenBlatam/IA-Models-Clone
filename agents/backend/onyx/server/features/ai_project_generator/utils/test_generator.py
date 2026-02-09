"""
Test Generator - Generador de Tests Automáticos
================================================

Genera tests automáticos para los proyectos generados.
Optimizado siguiendo mejores prácticas de Python y FastAPI.
"""

import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


def _generate_test_main() -> str:
    """
    Genera test_main.py (función pura).
    
    Returns:
        Contenido del archivo test_main.py
    """
    return '''"""Tests for main API"""

import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_root():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "status" in response.json()


def test_health():
    """Test health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_api_docs():
    """Test API documentation"""
    response = client.get("/docs")
    assert response.status_code == 200
'''


def _generate_test_ai_endpoints() -> str:
    """
    Genera test_ai_endpoints.py (función pura).
    
    Returns:
        Contenido del archivo test_ai_endpoints.py
    """
    return '''"""Tests for AI endpoints"""

import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_ai_status():
    """Test AI status endpoint"""
    response = client.get("/api/v1/ai/status")
    assert response.status_code == 200
    assert "status" in response.json()


def test_ai_process():
    """Test AI process endpoint"""
    response = client.post(
        "/api/v1/ai/process",
        json={
            "prompt": "Test prompt",
            "context": {},
            "parameters": {}
        }
    )
    assert response.status_code == 200
    assert "result" in response.json()


def test_ai_process_invalid():
    """Test AI process with invalid input"""
    response = client.post(
        "/api/v1/ai/process",
        json={}  # Missing required field
    )
    assert response.status_code == 422  # Validation error
'''


def _generate_conftest() -> str:
    """
    Genera conftest.py (función pura).
    
    Returns:
        Contenido del archivo conftest.py
    """
    return '''"""Pytest configuration"""

import pytest
import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def test_client():
    """Test client fixture"""
    from fastapi.testclient import TestClient
    from main import app
    return TestClient(app)
'''


def _generate_pytest_ini() -> str:
    """
    Genera pytest.ini (función pura).
    
    Returns:
        Contenido del archivo pytest.ini
    """
    return '''[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short
'''


def _generate_app_test() -> str:
    """
    Genera App.test.tsx (función pura).
    
    Returns:
        Contenido del archivo App.test.tsx
    """
    return '''import { render, screen } from '@testing-library/react'
import { BrowserRouter } from 'react-router-dom'
import App from '../App'

test('renders app', () => {
  render(
    <BrowserRouter>
      <App />
    </BrowserRouter>
  )
  const linkElement = screen.getByText(/AI Project/i)
  expect(linkElement).toBeInTheDocument()
})
'''


def _generate_setup_tests() -> str:
    """
    Genera setupTests.ts (función pura).
    
    Returns:
        Contenido del archivo setupTests.ts
    """
    return '''import '@testing-library/jest-dom'
'''


def _write_file_safe(file_path: Path, content: str, encoding: str = "utf-8") -> None:
    """
    Escribir archivo de forma segura (función pura).
    
    Args:
        file_path: Ruta del archivo
        content: Contenido a escribir
        encoding: Codificación del archivo
        
    Raises:
        IOError: Si no se puede escribir el archivo
    """
    try:
        file_path.write_text(content, encoding=encoding)
    except IOError as e:
        logger.error(f"Failed to write file {file_path}: {e}")
        raise


class TestGenerator:
    """
    Generador de tests automáticos.
    Optimizado con funciones puras y mejor manejo de errores.
    """
    
    def __init__(self) -> None:
        """Inicializa el generador de tests."""
        pass
    
    async def generate_backend_tests(
        self,
        project_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """
        Genera tests para el backend.
        
        Args:
            project_dir: Directorio del proyecto
            keywords: Keywords extraídos
            project_info: Información del proyecto
            
        Raises:
            ValueError: Si los parámetros son inválidos
            IOError: Si no se pueden escribir los archivos
        """
        if not project_dir:
            raise ValueError("project_dir cannot be None")
        
        if not keywords:
            raise ValueError("keywords cannot be empty")
        
        if not project_info:
            raise ValueError("project_info cannot be empty")
        
        tests_dir = project_dir / "tests"
        tests_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            _write_file_safe(tests_dir / "test_main.py", _generate_test_main())
            _write_file_safe(tests_dir / "test_ai_endpoints.py", _generate_test_ai_endpoints())
            _write_file_safe(tests_dir / "conftest.py", _generate_conftest())
            _write_file_safe(tests_dir / "pytest.ini", _generate_pytest_ini())
            
            logger.info("Backend tests generated successfully")
        except IOError as e:
            logger.error(f"Failed to generate backend tests: {e}")
            raise
    
    async def generate_frontend_tests(
        self,
        project_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """
        Genera tests para el frontend.
        
        Args:
            project_dir: Directorio del proyecto
            keywords: Keywords extraídos
            project_info: Información del proyecto
            
        Raises:
            ValueError: Si los parámetros son inválidos
            IOError: Si no se pueden escribir los archivos
        """
        if not project_dir:
            raise ValueError("project_dir cannot be None")
        
        if not keywords:
            raise ValueError("keywords cannot be empty")
        
        if not project_info:
            raise ValueError("project_info cannot be empty")
        
        tests_dir = project_dir / "src" / "__tests__"
        tests_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            _write_file_safe(tests_dir / "App.test.tsx", _generate_app_test())
            _write_file_safe(tests_dir / "setupTests.ts", _generate_setup_tests())
            
            logger.info("Frontend tests generated successfully")
        except IOError as e:
            logger.error(f"Failed to generate frontend tests: {e}")
            raise
