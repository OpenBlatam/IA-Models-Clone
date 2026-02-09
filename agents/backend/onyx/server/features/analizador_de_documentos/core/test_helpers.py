"""
Test Helpers for Document Analyzer
===================================

Advanced testing utilities and helpers.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field
from unittest.mock import Mock, AsyncMock

logger = logging.getLogger(__name__)

class TestHelpers:
    """Advanced test helpers"""
    
    @staticmethod
    def create_mock_analyzer() -> Mock:
        """Create a mock document analyzer"""
        mock = Mock()
        mock.analyze_document = AsyncMock(return_value={
            "document_id": "test_doc",
            "summary": "Test summary",
            "classification": {"test": 0.9}
        })
        mock.classify_document = AsyncMock(return_value={"test": 0.9})
        mock.summarize_document = AsyncMock(return_value="Test summary")
        return mock
    
    @staticmethod
    def create_test_document(content: str = "Test document content") -> Dict[str, Any]:
        """Create a test document"""
        return {
            "document_id": f"test_doc_{int(time.time())}",
            "content": content,
            "document_type": "txt",
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "source": "test"
            }
        }
    
    @staticmethod
    async def run_async_test(func: Callable, timeout: float = 5.0) -> Any:
        """Run async test with timeout"""
        try:
            return await asyncio.wait_for(func(), timeout=timeout)
        except asyncio.TimeoutError:
            raise AssertionError(f"Test timed out after {timeout}s")
    
    @staticmethod
    def assert_performance(
        func: Callable,
        max_duration: float,
        *args,
        **kwargs
    ) -> bool:
        """Assert function completes within max duration"""
        start_time = time.time()
        if asyncio.iscoroutinefunction(func):
            result = asyncio.run(func(*args, **kwargs))
        else:
            result = func(*args, **kwargs)
        duration = time.time() - start_time
        
        if duration > max_duration:
            raise AssertionError(f"Function took {duration:.3f}s, expected < {max_duration}s")
        
        return True
    
    @staticmethod
    def generate_test_data(count: int = 10) -> List[Dict[str, Any]]:
        """Generate test data"""
        return [
            TestHelpers.create_test_document(f"Document {i}")
            for i in range(count)
        ]
    
    @staticmethod
    def create_mock_cache() -> Mock:
        """Create a mock cache"""
        cache = {}
        mock = Mock()
        
        def get(key: str):
            return cache.get(key)
        
        def set(key: str, value: Any, ttl: int = None):
            cache[key] = value
        
        def delete(key: str):
            if key in cache:
                del cache[key]
        
        mock.get = get
        mock.set = set
        mock.delete = delete
        mock.clear = lambda: cache.clear()
        
        return mock

# Global instance
test_helpers = TestHelpers()
















