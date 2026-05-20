from typing import Dict, Any
import logging

from ...domain.interfaces import IImageProcessor
from ..circuit_breaker import CircuitBreaker, CircuitBreakerOpenError
from ....utils.retry import retry, RetryConfig

logger = logging.getLogger(__name__)

# Circuit breaker for image processor (external service)
IMAGE_PROCESSOR_CB = CircuitBreaker(
    failure_threshold=5,
    success_threshold=2,
    timeout=30.0,
    name="image_processor"
)

# Retry config for image processing
IMAGE_PROCESSING_RETRY = RetryConfig(
    max_attempts=2,
    initial_delay=1.0,
    max_delay=5.0,
    exponential_base=2.0,
    jitter=True
)


class ImageProcessorAdapter(IImageProcessor):
    """
    Image processor adapter with circuit breaker and retry logic
    """
    
    def __init__(self, image_processor_service):
        self.service = image_processor_service
        self.circuit_breaker = IMAGE_PROCESSOR_CB
    
    @retry(config=IMAGE_PROCESSING_RETRY)
    async def process(self, image_data: bytes) -> Dict[str, Any]:
        """
        Process image with circuit breaker and retry protection
        """
        try:
            result = await self.circuit_breaker.call(
                self.service.process_image,
                image_data
            )
            return result
        except CircuitBreakerOpenError as e:
            logger.error(f"Image processor circuit breaker is open: {e}")
            raise RuntimeError("Image processing service temporarily unavailable") from e
    
    async def validate(self, image_data: bytes) -> bool:
        """
        Validate image (lightweight operation, no circuit breaker needed)
        """
        try:
            return await self.service.validate_image(image_data)
        except Exception as e:
            logger.warning(f"Image validation failed: {e}")
            return False

