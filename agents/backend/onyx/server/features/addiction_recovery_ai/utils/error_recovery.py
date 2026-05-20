"""
Advanced Error Recovery and Resilience
"""

import torch
import logging
from typing import Dict, List, Optional, Any, Callable
from functools import wraps
import time

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        """
        Initialize circuit breaker
        
        Args:
            failure_threshold: Number of failures before opening
            recovery_timeout: Time to wait before attempting recovery
            expected_exception: Exception type to catch
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
        
        logger.info("CircuitBreaker initialized")
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Call function with circuit breaker
        
        Args:
            func: Function to call
            *args: Function arguments
            **kwargs: Function keyword arguments
        
        Returns:
            Function result
        """
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half_open"
                logger.info("Circuit breaker: half-open")
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if should attempt reset"""
        if self.last_failure_time is None:
            return True
        
        return (time.time() - self.last_failure_time) >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful call"""
        if self.state == "half_open":
            self.state = "closed"
            logger.info("Circuit breaker: closed (recovered)")
        
        self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(f"Circuit breaker: OPEN (failures: {self.failure_count})")


class RetryHandler:
    """Advanced retry handler with exponential backoff"""
    
    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0
    ):
        """
        Initialize retry handler
        
        Args:
            max_retries: Maximum retry attempts
            initial_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
            exponential_base: Exponential backoff base
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
    
    def retry(self, func: Callable, *args, **kwargs) -> Any:
        """
        Retry function with exponential backoff
        
        Args:
            func: Function to retry
            *args: Function arguments
            **kwargs: Function keyword arguments
        
        Returns:
            Function result
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    delay = min(
                        self.initial_delay * (self.exponential_base ** attempt),
                        self.max_delay
                    )
                    logger.warning(f"Retry {attempt + 1}/{self.max_retries} after {delay:.2f}s: {e}")
                    time.sleep(delay)
                else:
                    logger.error(f"All retries exhausted: {e}")
        
        raise last_exception


class GracefulDegradation:
    """Graceful degradation for model failures"""
    
    def __init__(
        self,
        fallback_model: Optional[torch.nn.Module] = None,
        fallback_function: Optional[Callable] = None
    ):
        """
        Initialize graceful degradation
        
        Args:
            fallback_model: Fallback model
            fallback_function: Fallback function
        """
        self.fallback_model = fallback_model
        self.fallback_function = fallback_function
    
    def predict_with_fallback(
        self,
        primary_model: torch.nn.Module,
        inputs: torch.Tensor,
        fallback_value: float = 0.5
    ) -> float:
        """
        Predict with fallback
        
        Args:
            primary_model: Primary model
            inputs: Input tensor
            fallback_value: Fallback value if all fail
        
        Returns:
            Prediction result
        """
        # Try primary model
        try:
            primary_model.eval()
            with torch.no_grad():
                output = primary_model(inputs)
                return output.item()
        except Exception as e:
            logger.warning(f"Primary model failed: {e}, trying fallback")
        
        # Try fallback model
        if self.fallback_model is not None:
            try:
                self.fallback_model.eval()
                with torch.no_grad():
                    output = self.fallback_model(inputs)
                    return output.item()
            except Exception as e:
                logger.warning(f"Fallback model failed: {e}")
        
        # Try fallback function
        if self.fallback_function is not None:
            try:
                return self.fallback_function(inputs)
            except Exception as e:
                logger.warning(f"Fallback function failed: {e}")
        
        # Return default
        logger.warning(f"All fallbacks failed, using default: {fallback_value}")
        return fallback_value

