"""
Error Handling Testing Helpers
Specialized helpers for error handling testing
"""

from typing import Any, Dict, List, Optional, Type, Callable
from unittest.mock import Mock, AsyncMock
import traceback
import asyncio
from datetime import datetime


class ErrorHandlingTestHelpers:
    """Helpers for error handling testing"""
    
    @staticmethod
    def create_mock_error_handler(
        handled_errors: Optional[List[Dict[str, Any]]] = None
    ) -> Mock:
        """Create mock error handler"""
        errors = handled_errors or []
        handler = Mock()
        
        def handle_side_effect(error: Exception, context: Dict[str, Any] = None):
            errors.append({
                "error": type(error).__name__,
                "message": str(error),
                "context": context or {},
                "timestamp": datetime.utcnow().isoformat(),
                "traceback": traceback.format_exc()
            })
            return {
                "handled": True,
                "error_id": f"error-{len(errors)}"
            }
        
        handler.handle = Mock(side_effect=handle_side_effect)
        handler.handle_async = AsyncMock(side_effect=handle_side_effect)
        handler.errors = errors
        return handler
    
    @staticmethod
    def assert_error_handled(
        handler: Mock,
        error_type: Type[Exception],
        context: Optional[Dict[str, Any]] = None
    ):
        """Assert error was handled"""
        assert handler.handle.called or handler.handle_async.called, \
            f"Error {error_type.__name__} was not handled"
        
        if hasattr(handler, "errors"):
            matching = [
                e for e in handler.errors
                if e["error"] == error_type.__name__
            ]
            assert len(matching) > 0, f"No errors of type {error_type.__name__} found"


class ErrorRecoveryHelpers:
    """Helpers for error recovery testing"""
    
    @staticmethod
    def create_mock_recovery_strategy(
        recovery_actions: Optional[List[Callable]] = None
    ) -> Mock:
        """Create mock recovery strategy"""
        actions = recovery_actions or []
        strategy = Mock()
        
        async def recover_side_effect(error: Exception):
            for action in actions:
                if asyncio.iscoroutinefunction(action):
                    await action(error)
                else:
                    action(error)
            return {"recovered": True}
        
        strategy.recover = AsyncMock(side_effect=recover_side_effect)
        strategy.can_recover = Mock(return_value=True)
        return strategy
    
    @staticmethod
    def assert_recovery_attempted(strategy: Mock):
        """Assert recovery was attempted"""
        assert strategy.recover.called, "Recovery was not attempted"


class GracefulDegradationHelpers:
    """Helpers for graceful degradation testing"""
    
    @staticmethod
    def create_mock_fallback_handler(
        fallback_results: Optional[List[Any]] = None
    ) -> Mock:
        """Create mock fallback handler"""
        results = fallback_results or []
        handler = Mock()
        
        async def fallback_side_effect(*args, **kwargs):
            result = results.pop(0) if results else {"fallback": True}
            return result
        
        handler.fallback = AsyncMock(side_effect=fallback_side_effect)
        handler.is_available = Mock(return_value=True)
        return handler
    
    @staticmethod
    def assert_fallback_used(handler: Mock):
        """Assert fallback was used"""
        assert handler.fallback.called, "Fallback was not used"


class ErrorClassificationHelpers:
    """Helpers for error classification testing"""
    
    @staticmethod
    def classify_error(error: Exception) -> Dict[str, Any]:
        """Classify error by type and severity"""
        error_type = type(error).__name__
        
        # Classify by error type
        if "Timeout" in error_type or "timeout" in str(error).lower():
            severity = "warning"
            category = "timeout"
        elif "Connection" in error_type or "connection" in str(error).lower():
            severity = "error"
            category = "network"
        elif "Validation" in error_type or "validation" in str(error).lower():
            severity = "warning"
            category = "validation"
        elif "Permission" in error_type or "permission" in str(error).lower():
            severity = "critical"
            category = "security"
        else:
            severity = "error"
            category = "unknown"
        
        return {
            "type": error_type,
            "severity": severity,
            "category": category,
            "message": str(error)
        }
    
    @staticmethod
    def assert_error_classified(
        classification: Dict[str, Any],
        expected_severity: Optional[str] = None,
        expected_category: Optional[str] = None
    ):
        """Assert error was classified correctly"""
        assert "severity" in classification, "Classification missing severity"
        assert "category" in classification, "Classification missing category"
        
        if expected_severity:
            assert classification["severity"] == expected_severity, \
                f"Severity {classification['severity']} does not match expected {expected_severity}"
        
        if expected_category:
            assert classification["category"] == expected_category, \
                f"Category {classification['category']} does not match expected {expected_category}"


# Convenience exports
create_mock_error_handler = ErrorHandlingTestHelpers.create_mock_error_handler
assert_error_handled = ErrorHandlingTestHelpers.assert_error_handled

create_mock_recovery_strategy = ErrorRecoveryHelpers.create_mock_recovery_strategy
assert_recovery_attempted = ErrorRecoveryHelpers.assert_recovery_attempted

create_mock_fallback_handler = GracefulDegradationHelpers.create_mock_fallback_handler
assert_fallback_used = GracefulDegradationHelpers.assert_fallback_used

classify_error = ErrorClassificationHelpers.classify_error
assert_error_classified = ErrorClassificationHelpers.assert_error_classified

