"""
Testing Assertions - Assertions personalizadas para MCP
=======================================================

Assertions personalizadas para facilitar testing de respuestas MCP.
"""

import logging
from typing import Dict, Any, Optional
from ..models import MCPResponse

logger = logging.getLogger(__name__)


def assert_mcp_response(
    response: Dict[str, Any],
    expected_success: Optional[bool] = None,
    expected_data: Optional[Any] = None,
    expected_error: Optional[str] = None
) -> None:
    """
    Assert que una respuesta MCP tenga la estructura correcta.
    
    Args:
        response: Respuesta a validar
        expected_success: Si se espera éxito (opcional)
        expected_data: Datos esperados (opcional)
        expected_error: Error esperado (opcional)
        
    Raises:
        AssertionError: Si la respuesta no cumple las expectativas
    """
    assert isinstance(response, dict), "Response must be a dictionary"
    assert "success" in response, "Response must have 'success' field"
    assert isinstance(response["success"], bool), "success must be a boolean"
    
    if expected_success is not None:
        assert response["success"] == expected_success, \
            f"Expected success={expected_success}, got {response['success']}"
    
    if response["success"]:
        assert "data" in response, "Successful response must have 'data' field"
        if expected_data is not None:
            assert response["data"] == expected_data, \
                f"Expected data={expected_data}, got {response['data']}"
    else:
        assert "error" in response, "Error response must have 'error' field"
        if expected_error is not None:
            assert expected_error in str(response["error"]), \
                f"Expected error containing '{expected_error}', got '{response['error']}'"
    
    assert "timestamp" in response, "Response must have 'timestamp' field"
    assert "metadata" in response, "Response must have 'metadata' field"


def assert_mcp_success(
    response: Dict[str, Any],
    expected_data: Optional[Any] = None
) -> None:
    """
    Assert que una respuesta MCP sea exitosa.
    
    Args:
        response: Respuesta a validar
        expected_data: Datos esperados (opcional)
        
    Raises:
        AssertionError: Si la respuesta no es exitosa
    """
    assert_mcp_response(response, expected_success=True, expected_data=expected_data)


def assert_mcp_error(
    response: Dict[str, Any],
    expected_error: Optional[str] = None,
    error_type: Optional[str] = None
) -> None:
    """
    Assert que una respuesta MCP sea un error.
    
    Args:
        response: Respuesta a validar
        expected_error: Mensaje de error esperado (opcional)
        error_type: Tipo de error esperado en metadata (opcional)
        
    Raises:
        AssertionError: Si la respuesta no es un error
    """
    assert_mcp_response(response, expected_success=False, expected_error=expected_error)
    
    if error_type is not None:
        assert "metadata" in response, "Response must have 'metadata' field"
        metadata = response["metadata"]
        assert "error_type" in metadata, "Error response metadata must have 'error_type'"
        assert metadata["error_type"] == error_type, \
            f"Expected error_type={error_type}, got {metadata['error_type']}"


def assert_mcp_authorized(response: Dict[str, Any]) -> None:
    """
    Assert que una respuesta indique autorización exitosa.
    
    Args:
        response: Respuesta a validar
        
    Raises:
        AssertionError: Si la respuesta no indica autorización
    """
    assert_mcp_success(response)
    assert "metadata" in response
    # Verificar que no haya error de autorización
    metadata = response.get("metadata", {})
    assert metadata.get("error_type") != "authorization_error", \
        "Response should not have authorization error"


def assert_mcp_forbidden(response: Dict[str, Any], expected_message: Optional[str] = None) -> None:
    """
    Assert que una respuesta indique acceso prohibido.
    
    Args:
        response: Respuesta a validar
        expected_message: Mensaje esperado (opcional)
        
    Raises:
        AssertionError: Si la respuesta no indica acceso prohibido
    """
    assert_mcp_error(response, expected_error=expected_message, error_type="authorization_error")


def assert_mcp_not_found(response: Dict[str, Any], resource_id: Optional[str] = None) -> None:
    """
    Assert que una respuesta indique recurso no encontrado.
    
    Args:
        response: Respuesta a validar
        resource_id: ID del recurso esperado en metadata (opcional)
        
    Raises:
        AssertionError: Si la respuesta no indica recurso no encontrado
    """
    assert_mcp_error(response, error_type="resource_not_found")
    
    if resource_id is not None:
        metadata = response.get("metadata", {})
        assert metadata.get("resource_id") == resource_id, \
            f"Expected resource_id={resource_id} in metadata"


def assert_mcp_validation_error(response: Dict[str, Any], field: Optional[str] = None) -> None:
    """
    Assert que una respuesta indique error de validación.
    
    Args:
        response: Respuesta a validar
        field: Campo que falló la validación (opcional)
        
    Raises:
        AssertionError: Si la respuesta no indica error de validación
    """
    assert_mcp_error(response, error_type="validation_error")
    
    if field is not None:
        metadata = response.get("metadata", {})
        details = metadata.get("details", {})
        if isinstance(details, dict):
            assert "field" in details or field in str(details), \
                f"Expected field '{field}' in validation error details"

