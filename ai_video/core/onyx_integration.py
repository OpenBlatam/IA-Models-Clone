from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import time
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
import logging
from onyx.core.functions import process_document, validate_user_access, format_response, handle_error
from onyx.utils.logger import setup_logger, OnyxLoggingAdapter
from onyx.utils.threadpool_concurrency import ThreadSafeDict, run_functions_in_parallel, FunctionCall
from onyx.utils.timing import time_function
from onyx.utils.retry_wrapper import retry_wrapper
from onyx.utils.telemetry import TelemetryLogger
from onyx.utils.encryption import encrypt_data, decrypt_data
from onyx.utils.file import get_file_extension, get_file_size
from onyx.utils.text_processing import clean_text, extract_keywords
from onyx.utils.gpu_utils import get_gpu_info, is_gpu_available
from onyx.utils.error_handling import handle_exception
from onyx.llm.factory import get_default_llms, get_llm, get_default_llm_with_vision
from onyx.llm.interfaces import LLM
from onyx.llm.utils import get_max_input_tokens_from_llm_provider
from onyx.db.engine import get_session_with_current_tenant
from onyx.db.models import Persona
from .exceptions import AIVideoError, PluginError, ValidationError
from .models import VideoRequest, VideoResponse, PluginConfig
            from onyx.llm.factory import get_llms_for_persona
            import base64
            from onyx.utils.threadpool_concurrency import run_with_timeout as onyx_timeout
            from onyx.utils.threadpool_concurrency import run_in_background as onyx_background
from typing import Any, List, Dict, Optional
"""
AI Video System - Onyx Integration

Integration module that adapts the AI Video system to use Onyx's existing
functions, utilities, and infrastructure for seamless operation within
the Onyx ecosystem.
"""


# Onyx imports

# Local imports

logger = setup_logger(__name__)


@dataclass
class OnyxIntegrationConfig:
    """Configuration for Onyx integration."""
    use_onyx_logging: bool = True
    use_onyx_llm: bool = True
    use_onyx_telemetry: bool = True
    use_onyx_encryption: bool = True
    use_onyx_threading: bool = True
    use_onyx_retry: bool = True
    use_onyx_gpu: bool = True
    max_workers: int = 10
    timeout_seconds: int = 300
    retry_attempts: int = 3


class OnyxLogger:
    """
    Onyx-compatible logging wrapper.
    
    Provides structured logging with Onyx's logging system including
    request IDs, tenant information, and context variables.
    """
    
    def __init__(self, name: str = "ai_video"):
        
    """__init__ function."""
self.logger = setup_logger(name)
        self.telemetry = TelemetryLogger() if OnyxIntegrationConfig().use_onyx_telemetry else None
    
    def info(self, message: str, **kwargs):
        """Log info message with Onyx context."""
        self.logger.info(message, extra=kwargs)
        if self.telemetry:
            self.telemetry.log_info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with Onyx context."""
        self.logger.warning(message, extra=kwargs)
        if self.telemetry:
            self.telemetry.log_warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with Onyx context."""
        self.logger.error(message, extra=kwargs)
        if self.telemetry:
            self.telemetry.log_error(message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with Onyx context."""
        self.logger.debug(message, extra=kwargs)
    
    def notice(self, message: str, **kwargs):
        """Log notice message with Onyx context."""
        self.logger.notice(message, extra=kwargs)


class OnyxLLMManager:
    """
    Onyx LLM manager for AI Video operations.
    
    Provides access to Onyx's LLM infrastructure including
    default models, vision models, and persona-specific models.
    """
    
    def __init__(self) -> Any:
        self.config = OnyxIntegrationConfig()
        self._llm_cache: ThreadSafeDict[str, LLM] = ThreadSafeDict()
        self.logger = OnyxLogger("ai_video.llm")
    
    async def get_default_llm(self, temperature: Optional[float] = None) -> LLM:
        """Get default LLM from Onyx."""
        try:
            llms = get_default_llms(temperature=temperature)
            return llms[0]  # Return main LLM
        except Exception as e:
            self.logger.error(f"Failed to get default LLM: {e}")
            raise AIVideoError(f"LLM initialization failed: {e}")
    
    async def get_vision_llm(self, temperature: Optional[float] = None) -> Optional[LLM]:
        """Get vision-capable LLM from Onyx."""
        try:
            return get_default_llm_with_vision(temperature=temperature)
        except Exception as e:
            self.logger.warning(f"Vision LLM not available: {e}")
            return None
    
    async def get_persona_llm(self, persona: Persona, temperature: Optional[float] = None) -> LLM:
        """Get LLM configured for specific persona."""
        try:
            llms = get_llms_for_persona(persona, temperature=temperature)
            return llms[0]  # Return main LLM
        except Exception as e:
            self.logger.error(f"Failed to get persona LLM: {e}")
            return await self.get_default_llm(temperature)
    
    async def generate_text(self, prompt: str, llm: Optional[LLM] = None, **kwargs) -> str:
        """Generate text using Onyx LLM."""
        if llm is None:
            llm = await self.get_default_llm()
        
        try:
            response = await llm.agenerate(prompt, **kwargs)
            return response.generations[0][0].text
        except Exception as e:
            self.logger.error(f"Text generation failed: {e}")
            raise AIVideoError(f"Text generation failed: {e}")
    
    async def generate_with_vision(self, prompt: str, image_data: bytes, **kwargs) -> str:
        """Generate text with vision using Onyx LLM."""
        llm = await self.get_vision_llm()
        if llm is None:
            raise AIVideoError("Vision LLM not available")
        
        try:
            # Convert image to base64 or use appropriate format
            image_b64 = base64.b64encode(image_data).decode('utf-8')
            
            # Create vision prompt
            vision_prompt = f"{prompt}\n\nImage: data:image/jpeg;base64,{image_b64}"
            response = await llm.agenerate(vision_prompt, **kwargs)
            return response.generations[0][0].text
        except Exception as e:
            self.logger.error(f"Vision generation failed: {e}")
            raise AIVideoError(f"Vision generation failed: {e}")


class OnyxTaskManager:
    """
    Onyx-compatible task manager using Onyx's threading utilities.
    
    Provides parallel execution, timeout handling, and background
    task management using Onyx's threadpool concurrency utilities.
    """
    
    def __init__(self) -> Any:
        self.config = OnyxIntegrationConfig()
        self.logger = OnyxLogger("ai_video.tasks")
        self._active_tasks: ThreadSafeDict[str, Any] = ThreadSafeDict()
    
    async def run_parallel(self, tasks: List[Callable], max_workers: Optional[int] = None) -> List[Any]:
        """Run tasks in parallel using Onyx's threading utilities."""
        try:
            # Convert async tasks to sync for Onyx's threading
            function_calls = [
                FunctionCall(self._wrap_async_task(task))
                for task in tasks
            ]
            
            results = run_functions_in_parallel(
                function_calls,
                allow_failures=False,
                max_workers=max_workers or self.config.max_workers
            )
            
            return results
        except Exception as e:
            self.logger.error(f"Parallel execution failed: {e}")
            raise AIVideoError(f"Task execution failed: {e}")
    
    def _wrap_async_task(self, task: Callable) -> Callable:
        """Wrap async task for sync execution."""
        def sync_wrapper(*args, **kwargs) -> Any:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Create new event loop for thread
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(task(*args, **kwargs))
                    finally:
                        new_loop.close()
                else:
                    return loop.run_until_complete(task(*args, **kwargs))
            except Exception as e:
                self.logger.error(f"Task execution error: {e}")
                raise
        
        return sync_wrapper
    
    async def run_with_timeout(self, task: Callable, timeout: float, *args, **kwargs) -> Any:
        """Run task with timeout using Onyx's timeout utilities."""
        try:
            
            result = onyx_timeout(timeout, task, *args, **kwargs)
            return result
        except Exception as e:
            self.logger.error(f"Timeout task failed: {e}")
            raise AIVideoError(f"Task timeout: {e}")
    
    async def run_in_background(self, task: Callable, *args, **kwargs) -> str:
        """Run task in background using Onyx's background utilities."""
        try:
            
            task_id = f"ai_video_{int(time.time())}"
            background_task = onyx_background(task, *args, **kwargs)
            
            self._active_tasks[task_id] = background_task
            return task_id
        except Exception as e:
            self.logger.error(f"Background task failed: {e}")
            raise AIVideoError(f"Background task failed: {e}")


class OnyxSecurityManager:
    """
    Onyx-compatible security manager.
    
    Provides encryption, validation, and access control using
    Onyx's security utilities.
    """
    
    def __init__(self) -> Any:
        self.config = OnyxIntegrationConfig()
        self.logger = OnyxLogger("ai_video.security")
    
    async def validate_access(self, user_id: str, resource_id: str) -> bool:
        """Validate user access using Onyx's access control."""
        try:
            return await validate_user_access(user_id, resource_id)
        except Exception as e:
            self.logger.error(f"Access validation failed: {e}")
            return False
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data using Onyx's encryption."""
        if not self.config.use_onyx_encryption:
            return data
        
        try:
            return encrypt_data(data)
        except Exception as e:
            self.logger.error(f"Encryption failed: {e}")
            return data
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data using Onyx's encryption."""
        if not self.config.use_onyx_encryption:
            return encrypted_data
        
        try:
            return decrypt_data(encrypted_data)
        except Exception as e:
            self.logger.error(f"Decryption failed: {e}")
            return encrypted_data
    
    def validate_input(self, input_text: str, max_length: int = 10000) -> tuple[bool, str]:
        """Validate input using Onyx's text processing."""
        try:
            # Clean text using Onyx utilities
            cleaned_text = clean_text(input_text)
            
            if len(cleaned_text) > max_length:
                return False, f"Input too long (max {max_length} characters)"
            
            if not cleaned_text.strip():
                return False, "Input cannot be empty"
            
            return True, cleaned_text
        except Exception as e:
            self.logger.error(f"Input validation failed: {e}")
            return False, f"Validation error: {e}"


class OnyxPerformanceManager:
    """
    Onyx-compatible performance manager.
    
    Provides performance monitoring, caching, and optimization
    using Onyx's performance utilities.
    """
    
    def __init__(self) -> Any:
        self.config = OnyxIntegrationConfig()
        self.logger = OnyxLogger("ai_video.performance")
        self._cache: ThreadSafeDict[str, Any] = ThreadSafeDict()
    
    def time_operation(self, operation_name: str):
        """Time operation using Onyx's timing utilities."""
        return time_function(operation_name)
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information using Onyx's GPU utilities."""
        if not self.config.use_onyx_gpu:
            return {"available": False}
        
        try:
            return get_gpu_info()
        except Exception as e:
            self.logger.warning(f"GPU info not available: {e}")
            return {"available": False}
    
    def is_gpu_available(self) -> bool:
        """Check GPU availability using Onyx's GPU utilities."""
        if not self.config.use_onyx_gpu:
            return False
        
        try:
            return is_gpu_available()
        except Exception as e:
            self.logger.warning(f"GPU check failed: {e}")
            return False
    
    def cache_get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        return self._cache.get(key)
    
    def cache_set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache with optional TTL."""
        self._cache[key] = {
            "value": value,
            "timestamp": time.time(),
            "ttl": ttl
        }
    
    def cache_cleanup(self) -> Any:
        """Clean up expired cache entries."""
        current_time = time.time()
        expired_keys = []
        
        for key, entry in self._cache.items():
            if entry.get("ttl") and (current_time - entry["timestamp"]) > entry["ttl"]:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._cache[key]


class OnyxRetryManager:
    """
    Onyx-compatible retry manager.
    
    Provides retry logic using Onyx's retry utilities.
    """
    
    def __init__(self) -> Any:
        self.config = OnyxIntegrationConfig()
        self.logger = OnyxLogger("ai_video.retry")
    
    def retry_operation(self, max_attempts: Optional[int] = None):
        """Retry operation using Onyx's retry utilities."""
        if not self.config.use_onyx_retry:
            return lambda func: func
        
        attempts = max_attempts or self.config.retry_attempts
        
        def decorator(func) -> Any:
            return retry_wrapper(
                func,
                max_attempts=attempts,
                backoff_factor=2,
                exceptions=(Exception,)
            )
        
        return decorator


class OnyxFileManager:
    """
    Onyx-compatible file manager.
    
    Provides file operations using Onyx's file utilities.
    """
    
    def __init__(self) -> Any:
        self.logger = OnyxLogger("ai_video.files")
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get file information using Onyx's file utilities."""
        try:
            return {
                "extension": get_file_extension(file_path),
                "size": get_file_size(file_path),
                "path": file_path
            }
        except Exception as e:
            self.logger.error(f"File info failed: {e}")
            return {"error": str(e)}
    
    def validate_file(self, file_path: str, allowed_extensions: List[str]) -> tuple[bool, str]:
        """Validate file using Onyx's file utilities."""
        try:
            file_info = self.get_file_info(file_path)
            
            if "error" in file_info:
                return False, file_info["error"]
            
            extension = file_info["extension"]
            if extension not in allowed_extensions:
                return False, f"File extension {extension} not allowed"
            
            return True, "File valid"
        except Exception as e:
            self.logger.error(f"File validation failed: {e}")
            return False, f"Validation error: {e}"


class OnyxIntegrationManager:
    """
    Main Onyx integration manager.
    
    Coordinates all Onyx integrations and provides a unified interface
    for the AI Video system to use Onyx's capabilities.
    """
    
    def __init__(self, config: Optional[OnyxIntegrationConfig] = None):
        
    """__init__ function."""
self.config = config or OnyxIntegrationConfig()
        self.logger = OnyxLogger("ai_video.integration")
        
        # Initialize managers
        self.llm_manager = OnyxLLMManager()
        self.task_manager = OnyxTaskManager()
        self.security_manager = OnyxSecurityManager()
        self.performance_manager = OnyxPerformanceManager()
        self.retry_manager = OnyxRetryManager()
        self.file_manager = OnyxFileManager()
    
    async def initialize(self) -> None:
        """Initialize Onyx integration."""
        try:
            self.logger.info("Initializing Onyx integration")
            
            # Test LLM availability
            await self.llm_manager.get_default_llm()
            
            # Test GPU availability
            gpu_info = self.performance_manager.get_gpu_info()
            if gpu_info.get("available"):
                self.logger.info(f"GPU available: {gpu_info}")
            else:
                self.logger.info("GPU not available, using CPU")
            
            self.logger.info("Onyx integration initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Onyx integration initialization failed: {e}")
            raise AIVideoError(f"Integration initialization failed: {e}")
    
    async async def process_video_request(self, request: VideoRequest) -> VideoResponse:
        """Process video request using Onyx integration."""
        try:
            # Validate access
            if not await self.security_manager.validate_access(request.user_id, request.request_id):
                raise ValidationError("Access denied")
            
            # Validate input
            is_valid, cleaned_text = self.security_manager.validate_input(
                request.input_text, max_length=10000
            )
            if not is_valid:
                raise ValidationError(f"Input validation failed: {cleaned_text}")
            
            # Generate video using Onyx LLM
            with self.performance_manager.time_operation("video_generation"):
                llm = await self.llm_manager.get_default_llm()
                prompt = f"Generate a video script for: {cleaned_text}"
                script = await self.llm_manager.generate_text(prompt, llm)
            
            # Create response
            response = VideoResponse(
                request_id=request.request_id,
                status="completed",
                output_url="generated_video.mp4",
                metadata={
                    "script": script,
                    "generated_at": datetime.now().isoformat(),
                    "onyx_integration": True
                }
            )
            
            self.logger.info(f"Video request processed successfully: {request.request_id}")
            return response
            
        except Exception as e:
            self.logger.error(f"Video request processing failed: {e}")
            raise AIVideoError(f"Video processing failed: {e}")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get system status with Onyx integration info."""
        try:
            status = {
                "onyx_integration": True,
                "llm_available": True,
                "gpu_available": self.performance_manager.is_gpu_available(),
                "gpu_info": self.performance_manager.get_gpu_info(),
                "cache_size": len(self.performance_manager._cache),
                "active_tasks": len(self.task_manager._active_tasks),
                "timestamp": datetime.now().isoformat()
            }
            
            return status
            
        except Exception as e:
            self.logger.error(f"Status check failed: {e}")
            return {"error": str(e)}


# Global Onyx integration instance
onyx_integration = OnyxIntegrationManager()


# Integration decorators
def use_onyx_llm(func) -> Any:
    """Decorator to use Onyx LLM for function."""
    async def wrapper(*args, **kwargs) -> Any:
        llm = await onyx_integration.llm_manager.get_default_llm()
        return await func(*args, llm=llm, **kwargs)
    return wrapper


def use_onyx_retry(max_attempts: Optional[int] = None):
    """Decorator to use Onyx retry for function."""
    return onyx_integration.retry_manager.retry_operation(max_attempts)


def use_onyx_timing(operation_name: str):
    """Decorator to use Onyx timing for function."""
    def decorator(func) -> Any:
        def wrapper(*args, **kwargs) -> Any:
            with onyx_integration.performance_manager.time_operation(operation_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# Utility functions
async def initialize_onyx_integration() -> None:
    """Initialize Onyx integration."""
    await onyx_integration.initialize()


async def get_onyx_llm() -> LLM:
    """Get Onyx LLM instance."""
    return await onyx_integration.llm_manager.get_default_llm()


async def process_with_onyx(request: VideoRequest) -> VideoResponse:
    """Process video request with Onyx integration."""
    return await onyx_integration.process_video_request(request)


def get_onyx_status() -> Dict[str, Any]:
    """Get Onyx integration status."""
    return asyncio.run(onyx_integration.get_system_status()) 