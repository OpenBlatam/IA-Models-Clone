"""
Test utilities and data factories for copywriting service tests.
"""
import sys
import os
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import uuid
import random
import string

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    CopywritingInput,
    CopywritingOutput,
    Feedback,
    SectionFeedback,
    CopyVariantHistory,
    FeedbackType,
    CopyTone,
    ContentType,
    Platform,
    Language,
    UseCase,
    CreativityLevel,
    get_settings
)


class TestDataFactory:
    """Factory for creating test data."""
    
    @staticmethod
    def create_copywriting_input(
        product_description: str = None,
        target_platform: str = None,
        content_type: str = None,
        tone: str = None,
        use_case: str = None,
        **overrides
    ) -> CopywritingInput:
        """Create a CopywritingInput with default or custom values."""
        defaults = {
            "product_description": product_description or "Test product description",
            "target_platform": target_platform or "instagram",
            "content_type": content_type or "social_post",
            "tone": tone or "inspirational",
            "use_case": use_case or "product_launch",
            "target_audience": "Test audience",
            "key_points": ["Quality", "Innovation", "Value"],
            "instructions": "Create engaging content",
            "restrictions": ["No price mentions"],
            "creativity_level": "balanced",
            "language": "es"
        }
        defaults.update(overrides)
        return CopywritingInput(**defaults)
    
    @staticmethod
    def create_copywriting_output(
        variants: List[Dict[str, Any]] = None,
        model_used: str = None,
        generation_time: float = None,
        **overrides
    ) -> CopywritingOutput:
        """Create a CopywritingOutput with default or custom values."""
        if variants is None:
            variants = [
                {
                    "variant_id": "test_variant_1",
                    "headline": "Test Headline",
                    "primary_text": "Test primary text content",
                    "call_to_action": "Learn More",
                    "hashtags": ["#test", "#example"]
                }
            ]
        
        defaults = {
            "variants": variants,
            "model_used": model_used or "gpt-3.5-turbo",
            "generation_time": generation_time or 1.5,
            "extra_metadata": {"tokens_used": 100, "test": True}
        }
        defaults.update(overrides)
        return CopywritingOutput(**defaults)
    
    @staticmethod
    def create_feedback(
        feedback_type: str = None,
        score: float = None,
        comments: str = None,
        **overrides
    ) -> Feedback:
        """Create a Feedback with default or custom values."""
        defaults = {
            "type": feedback_type or "human",
            "score": score or 0.8,
            "comments": comments or "Test feedback comment",
            "user_id": "test_user_123"
        }
        defaults.update(overrides)
        return Feedback(**defaults)
    
    @staticmethod
    def create_section_feedback(
        section: str = None,
        feedback: Feedback = None,
        suggestions: List[str] = None,
        **overrides
    ) -> SectionFeedback:
        """Create a SectionFeedback with default or custom values."""
        if feedback is None:
            feedback = TestDataFactory.create_feedback()
        
        defaults = {
            "section": section or "headline",
            "feedback": feedback,
            "suggestions": suggestions or ["Improve clarity", "Add more emotion"]
        }
        defaults.update(overrides)
        return SectionFeedback(**defaults)
    
    @staticmethod
    def create_copy_variant_history(
        variant_id: str = None,
        previous_versions: List[str] = None,
        change_log: List[str] = None,
        **overrides
    ) -> CopyVariantHistory:
        """Create a CopyVariantHistory with default or custom values."""
        defaults = {
            "variant_id": variant_id or f"variant_{uuid.uuid4().hex[:8]}",
            "previous_versions": previous_versions or ["v1.0", "v1.1"],
            "change_log": change_log or ["Updated headline", "Modified CTA"],
            "created_at": datetime.now()
        }
        defaults.update(overrides)
        return CopyVariantHistory(**defaults)
    
    @staticmethod
    def create_batch_inputs(count: int = 3) -> List[CopywritingInput]:
        """Create a batch of CopywritingInput objects."""
        platforms = ["instagram", "facebook", "twitter", "linkedin"]
        tones = ["inspirational", "professional", "casual", "playful"]
        content_types = ["social_post", "ad_copy", "email_subject", "blog_title"]
        use_cases = ["product_launch", "brand_awareness", "lead_generation", "sales_conversion"]
        
        inputs = []
        for i in range(count):
            input_data = TestDataFactory.create_copywriting_input(
                product_description=f"Test product {i+1}",
                target_platform=platforms[i % len(platforms)],
                tone=tones[i % len(tones)],
                content_type=content_types[i % len(content_types)],
                use_case=use_cases[i % len(use_cases)],
                target_audience=f"Audience {i+1}",
                key_points=[f"Point {j+1}" for j in range(3)],
                instructions=f"Instructions for product {i+1}",
                restrictions=["No price", "Keep it short"],
                creativity_level=random.choice(["conservative", "balanced", "creative", "innovative"]),
                language=random.choice(["es", "en", "fr"])
            )
            inputs.append(input_data)
        
        return inputs
    
    @staticmethod
    def create_random_string(length: int = 10) -> str:
        """Create a random string of specified length."""
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    
    @staticmethod
    def create_random_email() -> str:
        """Create a random email address."""
        username = TestDataFactory.create_random_string(8)
        domain = TestDataFactory.create_random_string(6)
        return f"{username}@{domain}.com"
    
    @staticmethod
    def create_random_phone() -> str:
        """Create a random phone number."""
        return f"+1{random.randint(1000000000, 9999999999)}"


class MockAIService:
    """Mock AI service for testing."""
    
    def __init__(self, delay: float = 0.1, should_fail: bool = False, response_data: Optional[Dict] = None):
        self.delay = delay
        self.should_fail = should_fail
        self.call_count = 0
        self.response_data = response_data or {
            "variants": [{"headline": "Mock Headline", "primary_text": "Mock Content"}],
            "model_used": "gpt-3.5-turbo",
            "generation_time": 0.1,
            "extra_metadata": {"tokens_used": 50}
        }
    
    async def mock_call(self, request: CopywritingInput, model: str) -> Dict[str, Any]:
        """Mock AI model call."""
        self.call_count += 1
        await asyncio.sleep(self.delay)
        
        if self.should_fail:
            raise Exception("Mock AI service error")
        
        response = self.response_data.copy()
        response["model_used"] = model
        response["generation_time"] = self.delay
        response["extra_metadata"]["call_count"] = self.call_count
        
        # Customize response based on request
        if hasattr(request, 'product_description'):
            response["variants"][0]["headline"] = f"Mock Headline for {request.product_description}"
            response["variants"][0]["primary_text"] = f"Mock content for {request.target_platform}"
        
        return response


class TestAssertions:
    """Custom assertions for copywriting tests."""
    
    @staticmethod
    def assert_valid_copywriting_input(input_obj: CopywritingInput):
        """Assert that a CopywritingInput is valid."""
        assert isinstance(input_obj, CopywritingInput)
        assert input_obj.product_description is not None
        assert input_obj.target_platform is not None
        assert input_obj.content_type is not None
        assert input_obj.tone is not None
        assert input_obj.use_case is not None
    
    @staticmethod
    def assert_valid_copywriting_output(output: CopywritingOutput):
        """Assert that a CopywritingOutput is valid."""
        assert isinstance(output, CopywritingOutput)
        assert hasattr(output, 'variants') or hasattr(output, 'results')
        
        # Handle both single response and batch response formats
        if hasattr(output, 'variants'):
            variants = output.variants
        else:
            variants = output.results if hasattr(output, 'results') else []
        
        assert isinstance(variants, list)
        assert len(variants) > 0
        
        for variant in variants:
            assert isinstance(variant, dict)
            assert "headline" in variant
            assert "primary_text" in variant
            assert isinstance(variant["headline"], str)
            assert isinstance(variant["primary_text"], str)
            assert len(variant["headline"]) > 0
            assert len(variant["primary_text"]) > 0
    
    @staticmethod
    def assert_valid_feedback(feedback: Feedback):
        """Assert that a Feedback is valid."""
        assert isinstance(feedback, Feedback)
        assert feedback.type is not None
        assert feedback.score is None or 0.0 <= feedback.score <= 1.0
        assert feedback.comments is None or isinstance(feedback.comments, str)
    
    @staticmethod
    def assert_valid_section_feedback(section_feedback: SectionFeedback):
        """Assert that a SectionFeedback is valid."""
        assert isinstance(section_feedback, SectionFeedback)
        assert section_feedback.section is not None
        assert isinstance(section_feedback.feedback, Feedback)
        assert section_feedback.suggestions is None or isinstance(section_feedback.suggestions, list)
    
    @staticmethod
    def assert_performance_threshold(execution_time: float, max_time: float):
        """Assert that execution time is within threshold."""
        assert execution_time <= max_time, f"Execution time {execution_time}s exceeds threshold {max_time}s"
    
    @staticmethod
    def assert_error_response(response, expected_status_code: int, expected_error_contains: str = None):
        """Assert that an error response is valid."""
        assert response.status_code == expected_status_code
        
        if expected_error_contains:
            response_data = response.json()
            assert expected_error_contains in response_data.get("detail", "")


class TestConfig:
    """Test configuration constants."""
    
    # Performance thresholds
    MAX_RESPONSE_TIME = 5.0  # seconds
    MAX_BATCH_RESPONSE_TIME = 30.0  # seconds
    MAX_MEMORY_USAGE = 100  # MB
    
    # Test data limits
    MAX_BATCH_SIZE = 10
    MAX_STRING_LENGTH = 1000
    MAX_KEY_POINTS = 15
    MAX_RESTRICTIONS = 10
    
    # Mock service settings
    MOCK_DELAY = 0.1
    MOCK_FAILURE_RATE = 0.1
    
    # Test file paths
    TEST_DATA_DIR = "test_data"
    TEMP_DIR = "temp"
    LOG_DIR = "logs"


class PerformanceMixin:
    """Mixin for performance testing utilities."""
    
    def measure_execution_time(self, func, *args, **kwargs):
        """Measure execution time of a function."""
        import time
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time
    
    async def measure_async_execution_time(self, coro):
        """Measure execution time of an async function."""
        import time
        start_time = time.time()
        result = await coro
        end_time = time.time()
        return result, end_time - start_time
    
    def assert_performance_threshold(self, execution_time: float, max_time: float):
        """Assert that execution time is within threshold."""
        TestAssertions.assert_performance_threshold(execution_time, max_time)


class SecurityMixin:
    """Mixin for security testing utilities."""
    
    def get_malicious_inputs(self) -> List[str]:
        """Get list of malicious input strings for testing."""
        return [
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --",
            "../../../etc/passwd",
            "{{7*7}}",
            "javascript:alert('xss')",
            "data:text/html,<script>alert('xss')</script>",
            "eval('malicious_code')",
            "require('child_process').exec('rm -rf /')",
            "{{config.items()}}",
            "{{''.__class__.__mro__[2].__subclasses__()}}"
        ]
    
    def get_sql_injection_inputs(self) -> List[str]:
        """Get list of SQL injection test inputs."""
        return [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "' UNION SELECT * FROM users --",
            "'; INSERT INTO users VALUES ('hacker', 'password'); --",
            "' OR 1=1 --",
            "admin'--",
            "admin'/*",
            "' OR 'x'='x",
            "') OR ('1'='1",
            "' OR 1=1 LIMIT 1 --"
        ]
    
    def get_xss_inputs(self) -> List[str]:
        """Get list of XSS test inputs."""
        return [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "<svg onload=alert('XSS')>",
            "javascript:alert('XSS')",
            "<iframe src=javascript:alert('XSS')></iframe>",
            "<body onload=alert('XSS')>",
            "<input onfocus=alert('XSS') autofocus>",
            "<select onfocus=alert('XSS') autofocus>",
            "<textarea onfocus=alert('XSS') autofocus>",
            "<keygen onfocus=alert('XSS') autofocus>"
        ]


# Import asyncio for async functions
import asyncio
