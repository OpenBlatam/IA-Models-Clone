"""
Tests de logging y auditoría
"""

import pytest
from unittest.mock import Mock, patch
import logging
from io import StringIO


class TestLogging:
    """Tests de logging"""
    
    def test_log_info(self):
        """Test de log de información"""
        log_output = StringIO()
        handler = logging.StreamHandler(log_output)
        logger = logging.getLogger("test")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
        logger.info("Test info message")
        
        output = log_output.getvalue()
        assert "Test info message" in output
    
    def test_log_error(self):
        """Test de log de errores"""
        log_output = StringIO()
        handler = logging.StreamHandler(log_output)
        logger = logging.getLogger("test")
        logger.addHandler(handler)
        logger.setLevel(logging.ERROR)
        
        try:
            raise ValueError("Test error")
        except ValueError as e:
            logger.error(f"Error occurred: {e}")
        
        output = log_output.getvalue()
        assert "Error occurred" in output
        assert "Test error" in output
    
    def test_log_with_context(self):
        """Test de log con contexto"""
        log_output = StringIO()
        handler = logging.StreamHandler(log_output)
        logger = logging.getLogger("test")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
        context = {"user_id": "123", "track_id": "456"}
        logger.info("Track analyzed", extra=context)
        
        output = log_output.getvalue()
        assert "Track analyzed" in output
    
    def test_log_levels(self):
        """Test de diferentes niveles de log"""
        log_output = StringIO()
        handler = logging.StreamHandler(log_output)
        logger = logging.getLogger("test")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        
        output = log_output.getvalue()
        assert "Debug message" in output
        assert "Info message" in output
        assert "Warning message" in output
        assert "Error message" in output


class TestAuditLogging:
    """Tests de auditoría"""
    
    def test_audit_user_action(self):
        """Test de auditoría de acción de usuario"""
        audit_log = []
        
        def audit_action(user_id, action, resource, details=None):
            audit_log.append({
                "user_id": user_id,
                "action": action,
                "resource": resource,
                "details": details,
                "timestamp": "2024-01-01T00:00:00Z"
            })
        
        audit_action("user123", "analyze", "track", {"track_id": "456"})
        
        assert len(audit_log) == 1
        assert audit_log[0]["action"] == "analyze"
        assert audit_log[0]["user_id"] == "user123"
    
    def test_audit_data_access(self):
        """Test de auditoría de acceso a datos"""
        access_log = []
        
        def audit_data_access(user_id, data_type, access_type):
            access_log.append({
                "user_id": user_id,
                "data_type": data_type,
                "access_type": access_type,
                "timestamp": "2024-01-01T00:00:00Z"
            })
        
        audit_data_access("user123", "track_analysis", "read")
        
        assert len(access_log) == 1
        assert access_log[0]["data_type"] == "track_analysis"
        assert access_log[0]["access_type"] == "read"
    
    def test_audit_error_events(self):
        """Test de auditoría de eventos de error"""
        error_log = []
        
        def audit_error(error_type, error_message, context=None):
            error_log.append({
                "error_type": error_type,
                "error_message": error_message,
                "context": context,
                "timestamp": "2024-01-01T00:00:00Z"
            })
        
        audit_error("ValidationError", "Invalid track ID", {"track_id": "invalid"})
        
        assert len(error_log) == 1
        assert error_log[0]["error_type"] == "ValidationError"


class TestStructuredLogging:
    """Tests de logging estructurado"""
    
    def test_structured_log_format(self):
        """Test de formato de log estructurado"""
        def create_structured_log(level, message, **kwargs):
            log_entry = {
                "level": level,
                "message": message,
                "timestamp": "2024-01-01T00:00:00Z"
            }
            log_entry.update(kwargs)
            return log_entry
        
        log = create_structured_log(
            "INFO",
            "Track analyzed",
            track_id="123",
            user_id="456",
            duration_ms=180000
        )
        
        assert log["level"] == "INFO"
        assert log["message"] == "Track analyzed"
        assert log["track_id"] == "123"
        assert log["user_id"] == "456"
    
    def test_log_performance_metrics(self):
        """Test de log de métricas de performance"""
        def log_performance(operation, duration_ms, **metrics):
            log_entry = {
                "operation": operation,
                "duration_ms": duration_ms,
                "timestamp": "2024-01-01T00:00:00Z"
            }
            log_entry.update(metrics)
            return log_entry
        
        log = log_performance(
            "analyze_track",
            1500,
            memory_mb=256,
            cpu_percent=45.2
        )
        
        assert log["operation"] == "analyze_track"
        assert log["duration_ms"] == 1500
        assert log["memory_mb"] == 256


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

