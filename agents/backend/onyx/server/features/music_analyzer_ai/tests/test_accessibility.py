"""
Tests de accesibilidad y usabilidad
"""

import pytest
from unittest.mock import Mock


class TestErrorMessages:
    """Tests de mensajes de error accesibles"""
    
    def test_error_message_readability(self):
        """Test de legibilidad de mensajes de error"""
        def get_readable_error(error_type, context=None):
            messages = {
                "not_found": "The requested track was not found. Please check the track ID and try again.",
                "invalid_input": f"Invalid input provided. {context or 'Please check your input and try again.'}",
                "server_error": "An internal server error occurred. Please try again later."
            }
            return messages.get(error_type, "An error occurred.")
        
        error1 = get_readable_error("not_found")
        assert "not found" in error1.lower()
        assert len(error1) > 20  # Mensaje descriptivo
        
        error2 = get_readable_error("invalid_input", "Track ID must be a string")
        assert "invalid" in error2.lower()
        assert "track id" in error2.lower()
    
    def test_error_message_actionability(self):
        """Test de que los mensajes de error son accionables"""
        def get_actionable_error(error_type):
            messages = {
                "missing_field": "The 'track_id' field is required. Please provide a valid track ID.",
                "invalid_range": "The 'tempo' value must be between 0 and 300. Please provide a valid tempo value.",
                "rate_limit": "Too many requests. Please wait a moment and try again."
            }
            return messages.get(error_type, "An error occurred.")
        
        error = get_actionable_error("missing_field")
        assert "required" in error.lower()
        assert "provide" in error.lower()


class TestResponseFormat:
    """Tests de formato de respuesta accesible"""
    
    def test_consistent_response_structure(self):
        """Test de estructura de respuesta consistente"""
        def create_response(success, data=None, error=None):
            response = {
                "success": success,
                "timestamp": "2024-01-01T00:00:00Z"
            }
            
            if success:
                response["data"] = data or {}
            else:
                response["error"] = error or {"message": "An error occurred"}
            
            return response
        
        success_response = create_response(True, {"result": "data"})
        assert success_response["success"] == True
        assert "data" in success_response
        
        error_response = create_response(False, error={"message": "Error"})
        assert error_response["success"] == False
        assert "error" in error_response
    
    def test_response_with_metadata(self):
        """Test de respuesta con metadata útil"""
        def create_enriched_response(data):
            return {
                "data": data,
                "metadata": {
                    "version": "1.0",
                    "timestamp": "2024-01-01T00:00:00Z",
                    "request_id": "req_123"
                }
            }
        
        response = create_enriched_response({"result": "test"})
        
        assert "data" in response
        assert "metadata" in response
        assert "version" in response["metadata"]


class TestInputValidation:
    """Tests de validación de entrada accesible"""
    
    def test_clear_validation_messages(self):
        """Test de mensajes de validación claros"""
        def validate_with_clear_messages(data):
            errors = []
            
            if "track_id" not in data:
                errors.append({
                    "field": "track_id",
                    "message": "Track ID is required",
                    "code": "REQUIRED_FIELD"
                })
            
            if "track_id" in data and not isinstance(data["track_id"], str):
                errors.append({
                    "field": "track_id",
                    "message": "Track ID must be a string",
                    "code": "INVALID_TYPE"
                })
            
            return {"valid": len(errors) == 0, "errors": errors}
        
        result1 = validate_with_clear_messages({})
        assert result1["valid"] == False
        assert len(result1["errors"]) > 0
        assert "field" in result1["errors"][0]
        assert "message" in result1["errors"][0]
        assert "code" in result1["errors"][0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

