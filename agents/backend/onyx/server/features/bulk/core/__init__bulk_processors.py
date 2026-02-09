"""
Bulk Processing Core - Main exports
====================================

Core modules for bulk document processing with modular architecture.
"""

from .base_processor import BaseBulkProcessor
from .continuous_processor import ContinuousProcessor
from .truthgpt_bulk_processor import (
    TruthGPTBulkProcessor,
    BulkDocumentRequest,
    DocumentGenerationTask,
    BulkGenerationResult,
    get_global_truthgpt_processor,
    start_global_truthgpt_processor,
    stop_global_truthgpt_processor
)
from .enhanced_truthgpt_processor import (
    EnhancedTruthGPTProcessor,
    EnhancedBulkDocumentRequest,
    EnhancedDocumentTask,
    EnhancedProcessingMetrics,
    get_global_enhanced_processor
)
from .processor_factory import (
    ProcessorFactory,
    get_global_processor,
    start_global_processor,
    stop_global_processor
)
from .processor_config import ProcessorConfig, DEFAULT_CONFIG
from .processor_metrics import ProcessingMetrics
from .callback_manager import CallbackManager
from .task_creator import TaskCreator
from .task_error_handler import TaskErrorHandler
from .task_processor import TaskProcessor
from .langchain_setup import LangChainSetup
from .prompt_templates import PromptTemplates
from .stats_helper import StatsHelper
from .request_validator import RequestValidator
from .request_submitter import RequestSubmitter
from .request_query_helper import RequestQueryHelper
from .task_queue_helper import TaskQueueHelper
from .metrics_updater import MetricsUpdater
from .processing_loop import ProcessingLoop
from .content_generator import ContentGenerator
from .constants import (
    MAX_DOCUMENTS_LIMIT,
    MIN_DOCUMENTS,
    DEFAULT_MAX_DOCUMENTS,
    DEFAULT_PRIORITY,
    TASK_STATUS_PENDING,
    TASK_STATUS_PROCESSING,
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
    REQUEST_STATUS_ACTIVE,
    REQUEST_STATUS_COMPLETED
)
from .idle_manager import IdleManager
from .maintenance_manager import MaintenanceManager
from .signal_handler import SignalHandler

__all__ = [
    "BaseBulkProcessor",
    "ContinuousProcessor",
    "TruthGPTBulkProcessor",
    "EnhancedTruthGPTProcessor",
    "BulkDocumentRequest",
    "EnhancedBulkDocumentRequest",
    "DocumentGenerationTask",
    "EnhancedDocumentTask",
    "BulkGenerationResult",
    "EnhancedProcessingMetrics",
    "ProcessingMetrics",
    "ProcessorFactory",
    "ProcessorConfig",
    "DEFAULT_CONFIG",
    "CallbackManager",
    "TaskCreator",
    "TaskErrorHandler",
    "TaskProcessor",
    "LangChainSetup",
    "PromptTemplates",
    "StatsHelper",
    "RequestValidator",
    "RequestSubmitter",
    "RequestQueryHelper",
    "TaskQueueHelper",
    "MetricsUpdater",
    "ProcessingLoop",
    "ContentGenerator",
    "IdleManager",
    "MaintenanceManager",
    "SignalHandler",
    "get_global_processor",
    "start_global_processor",
    "stop_global_processor",
    "get_global_truthgpt_processor",
    "start_global_truthgpt_processor",
    "stop_global_truthgpt_processor",
    "get_global_enhanced_processor",
    "MAX_DOCUMENTS_LIMIT",
    "MIN_DOCUMENTS",
    "DEFAULT_MAX_DOCUMENTS",
    "DEFAULT_PRIORITY",
    "TASK_STATUS_PENDING",
    "TASK_STATUS_PROCESSING",
    "TASK_STATUS_COMPLETED",
    "TASK_STATUS_FAILED",
    "REQUEST_STATUS_ACTIVE",
    "REQUEST_STATUS_COMPLETED"
]

