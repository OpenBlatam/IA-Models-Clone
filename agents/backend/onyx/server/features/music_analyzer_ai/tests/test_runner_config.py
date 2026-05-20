"""
Configuración avanzada para el runner de tests
"""

import pytest
import os


def pytest_configure(config):
    """Configuración de pytest"""
    # Agregar markers personalizados
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "api: marks tests as API tests"
    )
    config.addinivalue_line(
        "markers", "security: marks tests as security tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modificar items de colección de tests"""
    # Agregar markers automáticos basados en nombres de archivo
    for item in items:
        # Tests en test_performance.py son de performance
        if "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        
        # Tests en test_security.py son de seguridad
        if "security" in str(item.fspath):
            item.add_marker(pytest.mark.security)
        
        # Tests en test_integration.py son de integración
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Tests en test_api.py son de API
        if "test_api" in str(item.fspath):
            item.add_marker(pytest.mark.api)


def pytest_runtest_setup(item):
    """Setup antes de cada test"""
    # Verificar variables de entorno requeridas
    if "requires_spotify" in item.keywords:
        if not os.getenv("SPOTIFY_CLIENT_ID"):
            pytest.skip("SPOTIFY_CLIENT_ID not set")


def pytest_runtest_teardown(item):
    """Teardown después de cada test"""
    # Limpiar recursos si es necesario
    pass


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Generar reporte personalizado"""
    outcome = yield
    rep = outcome.get_result()
    
    # Agregar información adicional al reporte
    if rep.when == "call":
        # Puedes agregar lógica personalizada aquí
        pass
    
    return rep

