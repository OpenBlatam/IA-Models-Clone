"""
Base Processor - Base class for bulk document processors
========================================================

Provides common functionality shared between standard and enhanced processors.
"""

"""
Base Processor - Base class for bulk document processors
========================================================

Provides common functionality shared between standard and enhanced processors.

This module contains the BaseBulkProcessor class which serves as the foundation
for all bulk document processors. It handles common initialization, configuration,
and callback management.
"""

import logging
from typing import Dict, List, Optional, Any, Callable
from concurrent.futures import ThreadPoolExecutor

from ..config.openrouter_config import OpenRouterConfig
from ..config.bul_config import BULConfig
from ..utils.document_processor import DocumentProcessor
from .callback_manager import CallbackManager
from .langchain_setup import LangChainSetup

logger = logging.getLogger(__name__)


class BaseBulkProcessor:
    """
    Base class for bulk document processors with common functionality.
    
    This class provides the foundation for all bulk document processors,
    handling common initialization, configuration management, and callback
    registration. Subclasses should extend this to add processor-specific
    functionality.
    
    Attributes:
        config: BUL configuration instance
        openrouter_config: OpenRouter configuration instance
        document_processor: Document processor instance
        is_running: Whether the processor is currently running
        executor: Thread pool executor for concurrent operations
        callbacks: Callback manager for event handling
        output_parser: LangChain output parser (set by _setup_langchain_base)
    """
    
    def __init__(self, config: Optional[BULConfig] = None) -> None:
        """
        Initialize base processor components.
        
        Sets up configuration, document processor, thread pool executor,
        and callback manager. Subclasses should call super().__init__()
        before adding their own initialization.
        
        Args:
            config: Optional BUL configuration. If not provided, uses default.
        """
        self.config: BULConfig = config or BULConfig()
        self.openrouter_config: OpenRouterConfig = OpenRouterConfig()
        self.document_processor: DocumentProcessor = DocumentProcessor(self.config)
        
        self.is_running: bool = False
        self.executor: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=self.config.processing.max_concurrent_tasks
        )
        self.callbacks: CallbackManager = CallbackManager()
        
        logger.debug("Base processor components initialized")
    
    def _setup_langchain_base(self) -> None:
        """
        Setup base LangChain components.
        
        This method sets up the output parser. Subclasses should override
        _setup_langchain() to add model-specific setup (e.g., LLM initialization).
        
        Note:
            This method is called automatically during initialization.
            Subclasses should call super()._setup_langchain_base() in their
            _setup_langchain() method.
        """
        self.output_parser = LangChainSetup.create_output_parser()
        logger.debug("Base LangChain components setup complete")
    
    def stop_processing(self) -> None:
        """
        Stop all processing.
        
        Sets the is_running flag to False, which should cause processing
        loops to exit gracefully. Subclasses may override this to add
        additional cleanup logic.
        """
        self.is_running = False
        logger.info("Processing stop requested")
    
    def set_document_callback(self, callback: Callable[[Any, Any], None]) -> None:
        """
        Set callback for when documents are generated.
        
        Args:
            callback: Callable that receives (task, processed_doc) when a
                     document is successfully generated. Can be async or sync.
        """
        self.callbacks.set_document_callback(callback)
    
    def set_task_callback(self, callback: Callable[[Any], None]) -> None:
        """
        Set callback for when tasks are completed.
        
        Args:
            callback: Callable that receives (task) when a task is completed.
                     Can be async or sync.
        """
        self.callbacks.set_task_callback(callback)
    
    def set_error_callback(self, callback: Callable[[Exception], None]) -> None:
        """
        Set callback for when errors occur.
        
        Args:
            callback: Callable that receives (error) or (task, error) when an
                     error occurs. Can be async or sync.
        """
        self.callbacks.set_error_callback(callback)
    
    def cleanup(self) -> None:
        """
        Cleanup resources.
        
        Shuts down the thread pool executor and performs any necessary cleanup.
        Should be called when the processor is no longer needed.
        
        Note:
            This method does not stop processing. Call stop_processing() first
            if processing is still running.
        """
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
        logger.debug("Processor cleanup complete")

