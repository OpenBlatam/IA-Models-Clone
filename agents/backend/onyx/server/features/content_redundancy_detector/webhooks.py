"""
Webhooks Module - Backward Compatibility Wrapper
================================================

This file provides backward compatibility for code that imports from 'webhooks'
instead of 'webhooks.*'. All functionality has been moved to the modular structure
in the webhooks/ directory.

Features:
- Robust import handling with multiple fallback strategies
- Graceful degradation if module unavailable
- Enterprise-grade error handling
- Production-ready implementation

The recommended import style is now:
    from webhooks import send_webhook, WebhookEvent, WebhookEndpoint

But the old style still works:
    from webhooks import send_webhook
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
from enum import Enum

logger = logging.getLogger(__name__)

# Try to import from modular webhooks package
_webhooks_imported = False
_import_error = None

# Strategy 1: Relative import (when used as package)
try:
    from .webhooks import (
        WebhookEvent,
        WebhookPayload,
        WebhookEndpoint,
        WebhookDelivery,
        CircuitBreaker,
        WebhookManager,
        webhook_manager,
        send_webhook,
        register_webhook_endpoint,
        unregister_webhook_endpoint,
        get_webhook_endpoints,
        get_webhook_deliveries,
        get_webhook_stats,
    )
    _webhooks_imported = True
    logger.debug("Webhooks module imported via relative import")
except ImportError as e1:
    _import_error = e1
    
    # Strategy 2: Absolute import (when used standalone)
    try:
        _current_dir = Path(__file__).parent
        _webhooks_dir = _current_dir / 'webhooks'
        
        # Verify webhooks directory exists
        if _webhooks_dir.exists() and _webhooks_dir.is_dir():
            # Add parent directory to path if needed
            _parent_path = str(_current_dir)
            if _parent_path not in sys.path:
                sys.path.insert(0, _parent_path)
            
            from webhooks import (
                WebhookEvent,
                WebhookPayload,
                WebhookEndpoint,
                WebhookDelivery,
                CircuitBreaker,
                WebhookManager,
                webhook_manager,
                send_webhook,
                register_webhook_endpoint,
                unregister_webhook_endpoint,
                get_webhook_endpoints,
                get_webhook_deliveries,
                get_webhook_stats,
            )
            _webhooks_imported = True
            logger.debug("Webhooks module imported via absolute import")
        else:
            raise ImportError(f"Webhooks directory not found: {_webhooks_dir}")
            
    except ImportError as e2:
        _import_error = e2
        
        # Strategy 3: Fallback implementations (graceful degradation)
        logger.warning(
            f"Webhooks module not available. Using fallback implementations. "
            f"Errors: relative={e1}, absolute={e2}"
        )
        
        # Fallback WebhookEvent Enum
        class WebhookEvent(Enum):
            """Fallback WebhookEvent enum - Limited functionality"""
            ANALYSIS_COMPLETED = "analysis_completed"
            SIMILARITY_COMPLETED = "similarity_completed"
            QUALITY_COMPLETED = "quality_completed"
            BATCH_COMPLETED = "batch_completed"
            BATCH_FAILED = "batch_failed"
            SYSTEM_ERROR = "system_error"
            RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
        
        # Fallback data classes (minimal implementations)
        from dataclasses import dataclass
        from typing import Optional as Opt
        
        @dataclass
        class WebhookPayload:
            """Fallback WebhookPayload - Minimal implementation"""
            event: str
            timestamp: float
            data: Dict[str, Any]
            request_id: Optional[str] = None
            user_id: Optional[str] = None
        
        @dataclass
        class WebhookEndpoint:
            """Fallback WebhookEndpoint - Minimal implementation"""
            id: str
            url: str
            events: List[WebhookEvent]
            secret: Optional[str] = None
            timeout: int = 30
            retry_count: int = 3
            is_active: bool = True
        
        @dataclass
        class WebhookDelivery:
            """Fallback WebhookDelivery - Minimal implementation"""
            id: str
            endpoint_id: str
            event: str
            payload: WebhookPayload
            status: str = "pending"
            attempts: int = 0
            last_attempt: Optional[float] = None
            next_retry: Optional[float] = None
        
        # Fallback CircuitBreaker (no-op)
        class CircuitBreaker:
            """Fallback CircuitBreaker - No-op implementation"""
            def __init__(self, *args, **kwargs):
                self.state = "closed"
            def record_success(self):
                pass
            def record_failure(self):
                pass
            def can_proceed(self):
                return True
            def get_state(self):
                return "closed"
        
        # Fallback WebhookManager (no-op)
        class WebhookManager:
            """Fallback WebhookManager - No-op implementation"""
            def __init__(self, *args, **kwargs):
                self._endpoints = {}
                self._is_running = False
            async def start(self):
                logger.warning("WebhookManager fallback: start() called but module unavailable")
                self._is_running = True
            async def stop(self):
                self._is_running = False
            def register_endpoint_sync(self, endpoint):
                logger.warning(f"WebhookManager fallback: Cannot register endpoint {endpoint.id}")
            def unregister_endpoint(self, endpoint_id):
                return False
            def get_endpoints(self):
                return []
            def get_deliveries(self, limit=100):
                return []
            def get_delivery_stats(self):
                return {
                    "status": "disabled",
                    "reason": "Webhooks module not available",
                    "total_deliveries": 0
                }
            async def send_webhook(self, *args, **kwargs):
                logger.debug("WebhookManager fallback: send_webhook() called but disabled")
                return {"status": "disabled", "reason": "Module unavailable"}
        
        # Fallback functions
        webhook_manager = WebhookManager()
        
        async def send_webhook(event: WebhookEvent, data: Dict[str, Any], 
                              request_id: Optional[str] = None, 
                              user_id: Optional[str] = None) -> Dict[str, Any]:
            """Fallback send_webhook - Logs and returns disabled status"""
            logger.debug(f"Fallback send_webhook called: {event.value if hasattr(event, 'value') else event}")
            return {
                "status": "disabled",
                "reason": "Webhooks module not available",
                "message": "Webhook functionality disabled - module unavailable"
            }
        
        def register_webhook_endpoint(endpoint: WebhookEndpoint) -> None:
            """Fallback register_webhook_endpoint - Logs warning"""
            logger.warning(f"Cannot register webhook endpoint {endpoint.id}: Module unavailable")
        
        def unregister_webhook_endpoint(endpoint_id: str) -> bool:
            """Fallback unregister_webhook_endpoint"""
            return False
        
        def get_webhook_endpoints() -> List[WebhookEndpoint]:
            """Fallback get_webhook_endpoints"""
            return []
        
        def get_webhook_deliveries(limit: int = 100) -> List[WebhookDelivery]:
            """Fallback get_webhook_deliveries"""
            return []
        
        def get_webhook_stats() -> Dict[str, Any]:
            """Fallback get_webhook_stats"""
            return {
                "status": "disabled",
                "reason": "Webhooks module not available",
                "available": False
            }

# Export all for backward compatibility
__all__ = [
    "WebhookEvent",
    "WebhookPayload",
    "WebhookEndpoint",
    "WebhookDelivery",
    "CircuitBreaker",
    "WebhookManager",
    "webhook_manager",
    "send_webhook",
    "register_webhook_endpoint",
    "unregister_webhook_endpoint",
    "get_webhook_endpoints",
    "get_webhook_deliveries",
    "get_webhook_stats",
]

# Metadata
__version__ = "3.0.0"
__module_available__ = _webhooks_imported

# Log final status
if _webhooks_imported:
    logger.info("Webhooks module loaded successfully")
else:
    logger.warning(
        f"Webhooks module using fallback implementations. "
        f"This limits functionality. Import error: {_import_error}"
    )
