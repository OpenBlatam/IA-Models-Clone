"""
Test Helpers
============

Utilities for testing and test data generation.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import random
import string

logger = logging.getLogger(__name__)

class TestDataGenerator:
    """Test data generator."""
    
    @staticmethod
    def generate_string(length: int = 10) -> str:
        """Generate random string."""
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    
    @staticmethod
    def generate_document(
        document_id: Optional[str] = None,
        content_length: int = 100
    ) -> Dict[str, Any]:
        """Generate test document."""
        return {
            "id": document_id or TestDataGenerator.generate_string(12),
            "query": f"Test query {TestDataGenerator.generate_string(10)}",
            "content": TestDataGenerator.generate_string(content_length),
            "timestamp": datetime.now().isoformat(),
            "model_used": "test-model",
            "generation_time": random.uniform(0.5, 2.0),
            "quality_score": random.uniform(0.7, 1.0),
            "metadata": {
                "test": True,
                "generated_at": datetime.now().isoformat()
            }
        }
    
    @staticmethod
    def generate_task_id() -> str:
        """Generate test task ID."""
        return f"test_task_{TestDataGenerator.generate_string(8)}"
    
    @staticmethod
    def generate_batch_documents(count: int = 10) -> List[Dict[str, Any]]:
        """Generate batch of test documents."""
        return [
            TestDataGenerator.generate_document()
            for _ in range(count)
        ]

class AsyncTestHelper:
    """Helper for async testing."""
    
    @staticmethod
    async def run_with_timeout(
        coro,
        timeout: float = 5.0,
        default: Any = None
    ) -> Any:
        """Run coroutine with timeout."""
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"Operation timed out after {timeout}s")
            return default
    
    @staticmethod
    async def wait_for_condition(
        condition: Callable,
        timeout: float = 10.0,
        check_interval: float = 0.1
    ) -> bool:
        """Wait for condition to be true."""
        start_time = asyncio.get_event_loop().time()
        
        while True:
            if condition():
                return True
            
            if asyncio.get_event_loop().time() - start_time > timeout:
                return False
            
            await asyncio.sleep(check_interval)

class MockHelper:
    """Helper for mocking."""
    
    @staticmethod
    def mock_truthgpt_response(content: str = "Generated content") -> Dict[str, Any]:
        """Mock TruthGPT response."""
        return {
            "content": {
                "content": content,
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "generation_time": 1.5
                }
            },
            "analysis": {
                "quality_score": 0.85
            },
            "metadata": {
                "model": "mock-model",
                "tokens": 100
            }
        }
    
    @staticmethod
    def mock_error_response(error_type: str = "TimeoutError") -> Exception:
        """Mock error response."""
        if error_type == "TimeoutError":
            return TimeoutError("Mock timeout error")
        elif error_type == "ConnectionError":
            return ConnectionError("Mock connection error")
        else:
            return Exception(f"Mock {error_type}")
















