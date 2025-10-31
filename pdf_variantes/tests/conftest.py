"""
PDF Variantes Test Configuration
Configuración de pruebas para el sistema PDF Variantes
"""

import pytest
import asyncio
import os
import tempfile
from pathlib import Path
from typing import Dict, Any

# Configuración de pytest
@pytest.fixture(scope="session")
def event_loop():
    """Crear event loop para toda la sesión"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def test_settings():
    """Configuración de testing"""
    from utils.config import TestingSettings
    return TestingSettings()

@pytest.fixture
def temp_dir():
    """Directorio temporal para testing"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

@pytest.fixture
def mock_database():
    """Base de datos mock para testing"""
    return "sqlite:///test.db"

@pytest.fixture
def mock_redis():
    """Redis mock para testing"""
    return "redis://localhost:6379/1"

@pytest.fixture
def sample_pdf_content():
    """Contenido de PDF de ejemplo"""
    return b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n/Contents 4 0 R\n>>\nendobj\n4 0 obj\n<<\n/Length 44\n>>\nstream\nBT\n/F1 12 Tf\n72 720 Td\n(Hello World) Tj\nET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000204 00000 n \ntrailer\n<<\n/Size 5\n/Root 1 0 R\n>>\nstartxref\n297\n%%EOF"

@pytest.fixture
def sample_text_content():
    """Contenido de texto de ejemplo"""
    return "Este es un documento de ejemplo para testing. Contiene información sobre inteligencia artificial y procesamiento de documentos."

@pytest.fixture
def sample_document_data():
    """Datos de documento de ejemplo"""
    return {
        "title": "Test Document",
        "filename": "test.pdf",
        "file_path": "/path/to/test.pdf",
        "file_size": 1024,
        "content_type": "application/pdf",
        "content": "Test document content"
    }

@pytest.fixture
def sample_variant_data():
    """Datos de variante de ejemplo"""
    return {
        "document_id": "test_doc_id",
        "content": "Test variant content",
        "similarity_score": 0.8,
        "creativity_score": 0.7,
        "quality_score": 0.9
    }

@pytest.fixture
def sample_user_data():
    """Datos de usuario de ejemplo"""
    return {
        "username": "testuser",
        "email": "test@example.com",
        "hashed_password": "hashed_password"
    }

# Configuración de entorno de testing
@pytest.fixture(autouse=True)
def setup_test_environment():
    """Configurar entorno de testing"""
    # Configurar variables de entorno para testing
    os.environ["ENVIRONMENT"] = "testing"
    os.environ["DATABASE_URL"] = "sqlite:///test.db"
    os.environ["REDIS_URL"] = "redis://localhost:6379/1"
    os.environ["ENABLE_CACHE"] = "false"
    os.environ["ENABLE_METRICS"] = "false"
    os.environ["ENABLE_ANALYTICS"] = "false"
    os.environ["ENABLE_BLOCKCHAIN"] = "false"
    os.environ["ENABLE_PLUGINS"] = "false"
    os.environ["LOG_LEVEL"] = "WARNING"

# Configuración de cobertura
@pytest.fixture(scope="session")
def coverage_config():
    """Configuración de cobertura"""
    return {
        "source": ["pdf_variantes"],
        "omit": [
            "*/tests/*",
            "*/test_*",
            "*/__pycache__/*",
            "*/migrations/*"
        ],
        "fail_under": 80
    }

# Configuración de marcadores
def pytest_configure(config):
    """Configurar marcadores de pytest"""
    config.addinivalue_line(
        "markers", "unit: marca tests unitarios"
    )
    config.addinivalue_line(
        "markers", "integration: marca tests de integración"
    )
    config.addinivalue_line(
        "markers", "slow: marca tests lentos"
    )
    config.addinivalue_line(
        "markers", "ai: marca tests que requieren IA"
    )
    config.addinivalue_line(
        "markers", "blockchain: marca tests que requieren blockchain"
    )

# Configuración de reportes
def pytest_html_report_title(report):
    """Configurar título del reporte HTML"""
    report.title = "PDF Variantes Test Report"

def pytest_html_results_table_header(cells):
    """Configurar encabezado de tabla de resultados"""
    cells.insert(1, html.th("Description"))
    cells.insert(2, html.th("Duration"))

def pytest_html_results_table_row(report, cells):
    """Configurar fila de tabla de resultados"""
    cells.insert(1, html.td(report.description))
    cells.insert(2, html.td(report.duration))

# Configuración de logging
@pytest.fixture(autouse=True)
def setup_logging():
    """Configurar logging para tests"""
    import logging
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

# Configuración de timeouts
@pytest.fixture
def timeout_config():
    """Configuración de timeouts"""
    return {
        "default": 30,  # 30 segundos por defecto
        "slow": 120,    # 2 minutos para tests lentos
        "ai": 300,      # 5 minutos para tests de IA
        "blockchain": 600  # 10 minutos para tests de blockchain
    }

# Configuración de mocks
@pytest.fixture
def mock_external_services():
    """Mock de servicios externos"""
    return {
        "openai": "mock_openai",
        "anthropic": "mock_anthropic",
        "huggingface": "mock_huggingface",
        "redis": "mock_redis",
        "database": "mock_database"
    }

# Configuración de datos de prueba
@pytest.fixture
def test_data():
    """Datos de prueba"""
    return {
        "users": [
            {
                "username": "user1",
                "email": "user1@example.com",
                "hashed_password": "hash1"
            },
            {
                "username": "user2", 
                "email": "user2@example.com",
                "hashed_password": "hash2"
            }
        ],
        "documents": [
            {
                "title": "Document 1",
                "filename": "doc1.pdf",
                "content": "Content 1"
            },
            {
                "title": "Document 2",
                "filename": "doc2.pdf", 
                "content": "Content 2"
            }
        ],
        "variants": [
            {
                "content": "Variant 1",
                "similarity_score": 0.8
            },
            {
                "content": "Variant 2",
                "similarity_score": 0.9
            }
        ]
    }

# Configuración de limpieza
@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Limpiar archivos de prueba"""
    yield
    # Limpiar archivos temporales
    import shutil
    temp_dirs = [
        "temp",
        "test_files",
        "uploads",
        "variants",
        "exports"
    ]
    for temp_dir in temp_dirs:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

# Configuración de base de datos de prueba
@pytest.fixture
def test_database():
    """Base de datos de prueba"""
    return "sqlite:///test.db"

# Configuración de Redis de prueba
@pytest.fixture
def test_redis():
    """Redis de prueba"""
    return "redis://localhost:6379/1"

# Configuración de archivos de prueba
@pytest.fixture
def test_files_dir():
    """Directorio de archivos de prueba"""
    return Path(__file__).parent / "test_files"

# Configuración de configuración de prueba
@pytest.fixture
def test_config():
    """Configuración de prueba"""
    return {
        "app_name": "PDF Variantes Test",
        "debug": True,
        "environment": "testing",
        "database_url": "sqlite:///test.db",
        "redis_url": "redis://localhost:6379/1",
        "log_level": "WARNING",
        "enable_cache": False,
        "enable_metrics": False,
        "enable_analytics": False
    }
