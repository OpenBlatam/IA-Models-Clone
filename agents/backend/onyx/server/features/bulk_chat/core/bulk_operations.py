"""
Bulk Operations - Operaciones Masivas
=====================================

Módulo para realizar operaciones masivas sobre sesiones, mensajes y otros elementos
del sistema de forma eficiente y paralela.
"""

import asyncio
import logging
import json
import zipfile
import io
import functools
import time
from typing import List, Dict, Any, Optional, Callable, Union, Tuple, AsyncGenerator
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from .chat_engine import ContinuousChatEngine
from .chat_session import ChatSession, ChatState
from .session_storage import SessionStorage
from .conversation_analyzer import ConversationAnalyzer
from .exporters import ConversationExporter
from .metrics import MetricsCollector

# Importar optimizaciones de performance
try:
    from .bulk_operations_performance import (
        ultra_fast_batch_process,
        FastBulkProcessor,
        batch_process_optimized,
        _get_optimal_workers,
        fast_json_dumps,
        fast_json_loads,
        fast_serialize,
        fast_deserialize,
        BulkConnectionPool,
        BulkVectorizedProcessor,
        BulkProfiler,
        BulkOptimizedCache,
        BulkParallelExecutor,
        BulkMemoryOptimizer,
        BulkJITCompiler,
        BulkStreamProcessor,
        BulkAsyncIterator,
        BulkSmartCache,
        BulkGPUAccelerator,
        BulkDistributedProcessor,
        BulkIOOptimizer,
        BulkMultiProcessExecutor,
        BulkNetworkOptimizer,
        BulkDatabaseOptimizer,
        BulkAdaptiveBatcher,
        BulkLoadPredictor,
        BulkCompressionAdvanced,
        BulkRateController,
        BulkResourceMonitor,
        BulkIntelligentScheduler,
        BulkAutoTuner,
        BulkStreamingProcessor,
        BulkPredictiveAnalyzer,
        BulkFaultTolerance,
        BulkWorkloadBalancer,
        BulkIntelligentBatching,
        BulkPredictiveCache,
        BulkMemoryPool,
        BulkLockFreeQueue,
        BulkZeroCopyProcessor,
        BulkBatchAggregator,
        BulkHyperOptimizer,
        BulkSmartAllocator,
        BulkAdaptiveThrottler,
        BulkParallelPipeline,
        BulkCodeOptimizer,
        BulkLazyEvaluator,
        BulkAsyncBatchCollector,
        BulkSmartFilter,
        BulkIncrementalProcessor,
        BulkSmartSorter,
        BulkConcurrentHashMap,
        BulkLockFreeCounter,
        BulkCircularBuffer,
        BulkFastHash,
        BulkObjectPool,
        BulkEventEmitter,
        BulkDebouncer,
        BulkThrottler,
        BulkPriorityQueue,
        BulkRateLimiterAdvanced,
        BulkDataStructureOptimizer,
        BulkMemoryEfficientIterator,
        BulkAsyncSemaphorePool,
        BulkProfilerAdvanced,
        BulkDataTransformer,
        BulkDataValidator,
        BulkDataAggregator,
        BulkRetryManager,
        BulkBatchSplitter,
        BulkDataDeduplicator,
        BulkDataFormatter,
        BulkDataParser,
        BulkAsyncQueue,
        BulkAsyncBarrier,
        BulkAsyncCondition,
        BulkDistributedCache,
        BulkSearchIndex,
        BulkLogAggregator,
        BulkDataSerializer,
        BulkTaskScheduler,
        BulkLoadBalancer,
        BulkCircuitBreakerAdvanced,
        BulkHealthChecker,
        BulkMetricsCollector,
        BulkEventBus,
        BulkStateMachine,
        BulkWorkflowEngine,
        BulkSecurityManager,
        BulkStringProcessor,
        BulkDateTimeProcessor,
        BulkConfigManager,
        BulkTestingUtilities,
        BulkValidationAdvanced,
        BulkDataSanitizer,
        BulkResourceTracker,
        BulkErrorHandler,
        BulkAsyncContextManager,
        BulkBatchWindow,
        BulkRateCalculator,
        BulkAsyncLockManager,
        BulkAsyncPool,
        BulkAsyncGenerator,
        BulkAsyncCache,
        BulkAsyncSemaphoreGroup,
        BulkAsyncTimer,
        bulk_retry,
        bulk_timeout,
        bulk_rate_limit,
        bulk_cache,
        bulk_log_execution,
        BulkAsyncLogger,
        BulkAsyncCounter,
        BulkAsyncMutex,
        BulkAsyncFuturePool,
        BulkAsyncObserver,
        BulkAsyncCommand,
        BulkAsyncCommandQueue,
        BulkAsyncBatchProcessor,
        BulkAsyncThrottle,
        BulkAsyncDebounce,
        BulkAsyncWaitGroup,
        BulkAsyncBarrierAdvanced,
        BulkAsyncReadWriteLock,
        BulkAsyncBoundedSemaphore,
        BulkAsyncOnce,
        BulkAsyncLazy,
        BulkAsyncSingleFlight,
        BulkAsyncTimeout,
        BulkAsyncRetry,
        BulkDataChunker,
        BulkDataFlattener,
        BulkDataGrouper,
        BulkDataMapper,
        BulkDataReducer,
        BulkDataFilter,
        BulkAsyncHTTPClient,
        BulkAsyncFileHandler,
        BulkAsyncStorage,
        BulkAsyncQueueAdvanced,
        BulkAsyncRateLimiter,
        BulkDataComparator,
        BulkDataMerger,
        BulkDataSorter,
        BulkDataSearcher,
        BulkDataStatistics,
        BulkDataValidatorAdvanced,
        BulkDataNormalizer,
        BulkDataSampler,
        BulkDataTransformerAdvanced,
        BulkAsyncMonitor,
        BulkAsyncNotifier,
        BulkDataCompressor,
        BulkAsyncStreamProcessor,
        BulkAsyncBuffer,
        BulkAsyncBatchCollectorAdvanced,
        BulkAsyncChannel,
        BulkAsyncFanOut,
        BulkAsyncFanIn,
        BulkAsyncWorkerPool,
        BulkAsyncPipeline,
        BulkAsyncTee,
        BulkAsyncBroadcast,
        BulkAsyncLoadBalancerAdvanced,
        BulkDataPartitioner,
        BulkDataClustering,
        BulkDataWindow,
        BulkDataAggregatorAdvanced,
        BulkDataJoiner,
        BulkDataPivot,
        BulkAsyncDatabasePool,
        BulkAsyncTaskQueue,
        BulkAsyncEventStore,
        BulkAsyncSchedulerAdvanced,
        BulkDataSerializerAdvanced,
        BulkDataValidatorAdvancedPlus,
        BulkDataTransformerPipeline,
        BulkDataCompressorAdvancedPlus,
        BulkSecurityManagerAdvanced,
        BulkMetricsCollectorAdvanced,
        BulkAsyncLoggerAdvanced,
        BulkTestingUtilitiesAdvanced,
        BulkConfigManagerAdvanced,
        BulkObservabilityManager,
        BulkResilienceManager,
        BulkIntegrationManager,
        BulkReportingManager,
        BulkBackupManager,
        BulkSyncManager,
        BulkPerformanceAnalyzer,
        BulkDependencyManager,
        BulkMigrationManager,
        BulkAuditManager,
        BulkNetworkManager,
        BulkTestingFramework,
        BulkDocumentationGenerator,
        BulkDeploymentManager,
        BulkAlertingManager,
        BulkCacheManagerAdvanced,
        BulkMessageQueueAdvanced,
        BulkWorkflowOrchestrator,
        BulkVersionManager,
        BulkExportImportManager,
        BulkLogAnalyzer,
        BulkBenchmarkManager,
        BulkServiceDiscovery,
        BulkHealthCheckManager,
        BulkRateLimiterAdvanced,
        BulkMLPredictor,
        BulkAdvancedSearch,
        BulkDataGenerator,
        BulkDataTransformerAdvanced,
        BulkValidationFramework,
        BulkSecurityAdvanced,
        BulkCommunicationManager,
        BulkDataSyncManager,
        BulkReplicationManager,
        BulkDistributedBackup,
        BulkTimeSeriesAnalyzer,
        BulkTextAnalyzer,
        BulkStateManager,
        BulkResourceManager,
        BulkTaskManagerAdvanced,
        BulkGraphAnalyzer,
        BulkCodeAnalyzer,
        BulkDependencyAnalyzer,
        BulkPerformanceProfiler,
        BulkPatternRecognizer,
        BulkOptimizationEngine,
        BulkSignalProcessor,
        BulkMLTrainer,
        BulkDataMiner,
        BulkSimulationEngine,
        BulkFeatureExtractor,
        BulkAnomalyDetector,
        BulkRecommenderSystem,
        BulkEventProcessor,
        BulkImageProcessor,
        BulkNetworkAnalyzer,
        BulkIoTManager,
        BulkDataVisualizer,
        BulkAPIGateway,
        BulkDataWarehouse,
        BulkBlockchainManager,
        BulkKnowledgeBase,
        BulkNLPProcessor,
        BulkGeoManager,
        BulkAudioProcessor,
        BulkFileManager,
        BulkDataLake,
        BulkStreamingEngine,
        BulkNotificationManager,
        BulkContentDeliveryNetwork,
        BulkMicroservicesOrchestrator,
        BulkDataPipeline,
        BulkRealTimeProcessor,
        BulkCryptographyAdvanced,
        BulkVideoProcessor,
        BulkBehaviorAnalyzer,
        BulkMemoryManagerAdvanced,
        BulkDocumentProcessor,
        BulkPerformanceAnalyzerAdvanced,
        BulkQueueManagerAdvanced,
        BulkDistributedSync,
        BulkMarketAnalyzer,
        BulkResourceManagerAdvanced,
        BulkBinaryProcessor,
        BulkWebPerformanceAnalyzer,
        BulkSessionManagerAdvanced,
        BulkDataEnricher,
        BulkDataQualityChecker,
        BulkDataCatalog,
        BulkDataGovernance,
        BulkDataLineage,
        BulkDataRetention,
        BulkDataClassification,
        BulkDataMasking,
        BulkDataArchive,
        HAS_ORJSON,
        HAS_MSGPACK,
        HAS_NUMPY
    )
    PERFORMANCE_OPTIMIZATIONS_AVAILABLE = True
except ImportError:
    # Fallback si no está disponible
    ultra_fast_batch_process = None
    FastBulkProcessor = None
    batch_process_optimized = None
    _get_optimal_workers = lambda: 10
    fast_json_dumps = None
    fast_json_loads = None
    fast_serialize = None
    fast_deserialize = None
    BulkConnectionPool = None
    BulkVectorizedProcessor = None
    BulkProfiler = None
    BulkOptimizedCache = None
    BulkParallelExecutor = None
    BulkMemoryOptimizer = None
    BulkJITCompiler = None
    BulkStreamProcessor = None
    BulkAsyncIterator = None
    BulkSmartCache = None
    BulkGPUAccelerator = None
    BulkDistributedProcessor = None
    BulkIOOptimizer = None
    BulkMultiProcessExecutor = None
    BulkNetworkOptimizer = None
    BulkDatabaseOptimizer = None
    BulkAdaptiveBatcher = None
    BulkLoadPredictor = None
    BulkCompressionAdvanced = None
    BulkRateController = None
    BulkResourceMonitor = None
    BulkIntelligentScheduler = None
    BulkAutoTuner = None
    BulkStreamingProcessor = None
    BulkPredictiveAnalyzer = None
    BulkFaultTolerance = None
    BulkWorkloadBalancer = None
    BulkMultiLevelCache = None
    BulkMemoryPool = None
    BulkFastSerializer = None
    BulkBatchAggregator = None
    BulkHyperOptimizer = None
    BulkSmartAllocator = None
    BulkAdaptiveThrottler = None
    BulkParallelPipeline = None
    BulkCodeOptimizer = None
    BulkLazyEvaluator = None
    BulkAsyncBatchCollector = None
    BulkSmartFilter = None
    BulkIncrementalProcessor = None
    BulkSmartSorter = None
    BulkConcurrentHashMap = None
    BulkLockFreeCounter = None
    BulkCircularBuffer = None
    BulkFastHash = None
    BulkObjectPool = None
    BulkEventEmitter = None
    BulkDebouncer = None
    BulkThrottler = None
    BulkPriorityQueue = None
    BulkRateLimiterAdvanced = None
    BulkDataStructureOptimizer = None
    BulkMemoryEfficientIterator = None
    BulkAsyncSemaphorePool = None
    BulkProfilerAdvanced = None
    BulkDataTransformer = None
    BulkDataValidator = None
    BulkDataAggregator = None
    BulkRetryManager = None
    BulkBatchSplitter = None
    BulkDataDeduplicator = None
    BulkDataFormatter = None
    BulkDataParser = None
    BulkAsyncQueue = None
    BulkAsyncBarrier = None
    BulkAsyncCondition = None
    BulkDistributedCache = None
    BulkSearchIndex = None
    BulkLogAggregator = None
    BulkDataSerializer = None
    BulkTaskScheduler = None
    BulkLoadBalancer = None
    BulkCircuitBreakerAdvanced = None
    BulkHealthChecker = None
    BulkMetricsCollector = None
    BulkEventBus = None
    BulkStateMachine = None
    BulkWorkflowEngine = None
    BulkSecurityManager = None
    BulkStringProcessor = None
    BulkDateTimeProcessor = None
    BulkConfigManager = None
    BulkTestingUtilities = None
    BulkValidationAdvanced = None
    BulkDataSanitizer = None
    BulkResourceTracker = None
    BulkErrorHandler = None
    BulkAsyncContextManager = None
    BulkBatchWindow = None
    BulkRateCalculator = None
    BulkAsyncLockManager = None
    BulkAsyncPool = None
    BulkAsyncGenerator = None
    BulkAsyncCache = None
    BulkAsyncSemaphoreGroup = None
    BulkAsyncTimer = None
    bulk_retry = None
    bulk_timeout = None
    bulk_rate_limit = None
    bulk_cache = None
    bulk_log_execution = None
    BulkAsyncLogger = None
    BulkAsyncCounter = None
    BulkAsyncMutex = None
    BulkAsyncFuturePool = None
    BulkAsyncObserver = None
    BulkAsyncCommand = None
    BulkAsyncCommandQueue = None
    BulkAsyncBatchProcessor = None
    BulkAsyncThrottle = None
    BulkAsyncDebounce = None
    BulkAsyncWaitGroup = None
    BulkAsyncBarrierAdvanced = None
    BulkAsyncReadWriteLock = None
    BulkAsyncBoundedSemaphore = None
    BulkAsyncOnce = None
    BulkAsyncLazy = None
    BulkAsyncSingleFlight = None
    BulkAsyncTimeout = None
    BulkAsyncRetry = None
    BulkDataChunker = None
    BulkDataFlattener = None
    BulkDataGrouper = None
    BulkDataMapper = None
    BulkDataReducer = None
    BulkDataFilter = None
    BulkAsyncHTTPClient = None
    BulkAsyncFileHandler = None
    BulkAsyncStorage = None
    BulkAsyncQueueAdvanced = None
    BulkAsyncRateLimiter = None
    BulkDataComparator = None
    BulkDataMerger = None
    BulkDataSorter = None
    BulkDataSearcher = None
    BulkDataStatistics = None
    BulkDataValidatorAdvanced = None
    BulkDataNormalizer = None
    BulkDataSampler = None
    BulkDataTransformerAdvanced = None
    BulkAsyncMonitor = None
    BulkAsyncNotifier = None
    BulkDataCompressor = None
    BulkAsyncStreamProcessor = None
    BulkAsyncBuffer = None
    BulkAsyncBatchCollectorAdvanced = None
    BulkAsyncChannel = None
    BulkAsyncFanOut = None
    BulkAsyncFanIn = None
    BulkAsyncWorkerPool = None
    BulkAsyncPipeline = None
    BulkAsyncTee = None
    BulkAsyncBroadcast = None
    BulkAsyncLoadBalancerAdvanced = None
    BulkDataPartitioner = None
    BulkDataClustering = None
    BulkDataWindow = None
    BulkDataAggregatorAdvanced = None
    BulkDataJoiner = None
    BulkDataPivot = None
    BulkAsyncDatabasePool = None
    BulkAsyncTaskQueue = None
    BulkAsyncEventStore = None
    BulkAsyncSchedulerAdvanced = None
    BulkDataSerializerAdvanced = None
    BulkDataValidatorAdvancedPlus = None
    BulkDataTransformerPipeline = None
    BulkDataCompressorAdvancedPlus = None
    BulkSecurityManagerAdvanced = None
    BulkMetricsCollectorAdvanced = None
    BulkAsyncLoggerAdvanced = None
    BulkTestingUtilitiesAdvanced = None
    BulkConfigManagerAdvanced = None
    BulkObservabilityManager = None
    BulkResilienceManager = None
    BulkIntegrationManager = None
    BulkReportingManager = None
    BulkBackupManager = None
    BulkSyncManager = None
    BulkPerformanceAnalyzer = None
    BulkDependencyManager = None
    BulkMigrationManager = None
    BulkAuditManager = None
    BulkNetworkManager = None
    BulkTestingFramework = None
    BulkDocumentationGenerator = None
    BulkDeploymentManager = None
    BulkAlertingManager = None
    BulkCacheManagerAdvanced = None
    BulkMessageQueueAdvanced = None
    BulkWorkflowOrchestrator = None
    BulkVersionManager = None
    BulkExportImportManager = None
    BulkLogAnalyzer = None
    BulkBenchmarkManager = None
    BulkServiceDiscovery = None
    BulkHealthCheckManager = None
    BulkRateLimiterAdvanced = None
    BulkMLPredictor = None
    BulkAdvancedSearch = None
    BulkDataGenerator = None
    BulkDataTransformerAdvanced = None
    BulkValidationFramework = None
    BulkSecurityAdvanced = None
    BulkCommunicationManager = None
    BulkDataSyncManager = None
    BulkReplicationManager = None
    BulkDistributedBackup = None
    BulkTimeSeriesAnalyzer = None
    BulkTextAnalyzer = None
    BulkStateManager = None
    BulkResourceManager = None
    BulkTaskManagerAdvanced = None
    BulkGraphAnalyzer = None
    BulkCodeAnalyzer = None
    BulkDependencyAnalyzer = None
    BulkPerformanceProfiler = None
    BulkPatternRecognizer = None
    BulkOptimizationEngine = None
    BulkSignalProcessor = None
    BulkMLTrainer = None
    BulkDataMiner = None
    BulkSimulationEngine = None
    BulkFeatureExtractor = None
    BulkAnomalyDetector = None
    BulkRecommenderSystem = None
    BulkEventProcessor = None
    BulkImageProcessor = None
    BulkNetworkAnalyzer = None
    BulkIoTManager = None
    BulkDataVisualizer = None
    BulkAPIGateway = None
    BulkDataWarehouse = None
    BulkBlockchainManager = None
    BulkKnowledgeBase = None
    BulkNLPProcessor = None
    BulkGeoManager = None
    BulkAudioProcessor = None
    BulkFileManager = None
    BulkDataLake = None
    BulkStreamingEngine = None
    BulkNotificationManager = None
    BulkContentDeliveryNetwork = None
    BulkMicroservicesOrchestrator = None
    BulkDataPipeline = None
    BulkRealTimeProcessor = None
    BulkCryptographyAdvanced = None
    BulkVideoProcessor = None
    BulkBehaviorAnalyzer = None
    BulkMemoryManagerAdvanced = None
    BulkDocumentProcessor = None
    BulkPerformanceAnalyzerAdvanced = None
    BulkQueueManagerAdvanced = None
    BulkDistributedSync = None
    BulkMarketAnalyzer = None
    BulkResourceManagerAdvanced = None
    BulkBinaryProcessor = None
    BulkWebPerformanceAnalyzer = None
    BulkSessionManagerAdvanced = None
    BulkDataEnricher = None
    BulkDataQualityChecker = None
    BulkDataCatalog = None
    BulkDataGovernance = None
    BulkDataLineage = None
    BulkDataRetention = None
    BulkDataClassification = None
    BulkDataMasking = None
    BulkDataArchive = None
    BulkPerformanceTracker = None
    HAS_ORJSON = False
    HAS_MSGPACK = False
    HAS_NUMPY = False
    PERFORMANCE_OPTIMIZATIONS_AVAILABLE = False

from .robust_helpers import (
    RobustRetry,
    CircuitBreaker,
    RateLimiter,
    circuit_breaker,
    rate_limiter,
    validate_input,
    safe_json_dumps,
    safe_json_loads,
    generate_id
)

logger = logging.getLogger(__name__)


# ============================================================================
# DECORADORES AVANZADOS - Decoradores para métricas, caching y optimización
# ============================================================================

def bulk_metrics_decorator(operation_name: Optional[str] = None):
    """Decorador para medir métricas de operaciones bulk."""
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Registrar métricas si hay un monitor disponible
                if args and hasattr(args[0], 'performance_monitor'):
                    monitor = args[0].performance_monitor
                    if hasattr(monitor, 'record_operation'):
                        items_processed = result.processed if hasattr(result, 'processed') else 0
                        success_rate = result.processed / result.total if hasattr(result, 'total') and result.total > 0 else 1.0
                        monitor.record_operation(op_name, duration, items_processed, success_rate)
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"Error in {op_name}: {e}")
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Registrar métricas si hay un monitor disponible
                if args and hasattr(args[0], 'performance_monitor'):
                    monitor = args[0].performance_monitor
                    if hasattr(monitor, 'record_operation'):
                        items_processed = result.processed if hasattr(result, 'processed') else 0
                        success_rate = result.processed / result.total if hasattr(result, 'total') and result.total > 0 else 1.0
                        monitor.record_operation(op_name, duration, items_processed, success_rate)
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"Error in {op_name}: {e}")
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def bulk_cache_decorator(ttl: int = 300, key_func: Optional[Callable] = None):
    """Decorador para cachear resultados de operaciones bulk."""
    cache: Dict[str, Tuple[Any, float]] = {}
    
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generar key de cache
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            
            # Verificar cache
            if cache_key in cache:
                result, timestamp = cache[cache_key]
                if time.time() - timestamp < ttl:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return result
                else:
                    del cache[cache_key]
            
            # Ejecutar función
            result = await func(*args, **kwargs)
            
            # Guardar en cache
            cache[cache_key] = (result, time.time())
            
            # Limpiar cache viejo
            if len(cache) > 1000:
                current_time = time.time()
                cache_keys_to_delete = [
                    k for k, (_, ts) in cache.items()
                    if current_time - ts > ttl
                ]
                for k in cache_keys_to_delete:
                    del cache[k]
            
            return result
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Generar key de cache
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            
            # Verificar cache
            if cache_key in cache:
                result, timestamp = cache[cache_key]
                if time.time() - timestamp < ttl:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return result
                else:
                    del cache[cache_key]
            
            # Ejecutar función
            result = func(*args, **kwargs)
            
            # Guardar en cache
            cache[cache_key] = (result, time.time())
            
            # Limpiar cache viejo
            if len(cache) > 1000:
                current_time = time.time()
                cache_keys_to_delete = [
                    k for k, (_, ts) in cache.items()
                    if current_time - ts > ttl
                ]
                for k in cache_keys_to_delete:
                    del cache[k]
            
            return result
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def bulk_retry_decorator(max_retries: int = 3, backoff_factor: float = 2.0):
    """Decorador para retry automático con backoff exponencial."""
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_error = None
            delay = 1.0
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_retries:
                        await asyncio.sleep(delay)
                        delay *= backoff_factor
                    else:
                        logger.error(f"Failed after {max_retries} retries: {e}")
                        raise last_error
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_error = None
            delay = 1.0
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_retries:
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        logger.error(f"Failed after {max_retries} retries: {e}")
                        raise last_error
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def bulk_rate_limit_decorator(max_calls: int = 100, period: float = 60.0):
    """Decorador para rate limiting."""
    calls: Dict[str, List[float]] = {}
    
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            key = f"{func.__name__}"
            current_time = time.time()
            
            # Limpiar llamadas antiguas
            if key in calls:
                calls[key] = [t for t in calls[key] if current_time - t < period]
            else:
                calls[key] = []
            
            # Verificar rate limit
            if len(calls[key]) >= max_calls:
                raise Exception(f"Rate limit exceeded for {func.__name__}")
            
            # Registrar llamada
            calls[key].append(current_time)
            
            return await func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            key = f"{func.__name__}"
            current_time = time.time()
            
            # Limpiar llamadas antiguas
            if key in calls:
                calls[key] = [t for t in calls[key] if current_time - t < period]
            else:
                calls[key] = []
            
            # Verificar rate limit
            if len(calls[key]) >= max_calls:
                raise Exception(f"Rate limit exceeded for {func.__name__}")
            
            # Registrar llamada
            calls[key].append(current_time)
            
            return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# ============================================================================
# HELPER FUNCTIONS - Funciones utilitarias para operaciones bulk
# ============================================================================

async def batch_process(
    items: List[Any],
    operation: Callable,
    batch_size: int = 100,
    max_workers: int = 10,
    progress_callback: Optional[Callable] = None
) -> List[Any]:
    """
    Procesar items en batches con control de concurrencia - OPTIMIZADO.
    
    Args:
        items: Lista de items a procesar
        operation: Función async para procesar cada item
        batch_size: Tamaño de cada batch
        max_workers: Máximo de workers concurrentes
        progress_callback: Callback para reportar progreso (progress, total, processed)
    
    Returns:
        Lista de resultados
    """
    # OPTIMIZACIÓN: Early return si no hay items
    if not items:
        return []
    
    # OPTIMIZACIÓN: Usar ultra_fast_batch_process si está disponible
    if PERFORMANCE_OPTIMIZATIONS_AVAILABLE and ultra_fast_batch_process:
        try:
            return await ultra_fast_batch_process(
                items,
                operation,
                batch_size=batch_size,
                max_workers=max_workers,
                progress_callback=progress_callback
            )
        except Exception as e:
            logger.warning(f"ultra_fast_batch_process failed, falling back: {e}")
    
    # Fallback a implementación estándar
    total = len(items)
    processed = 0
    
    # OPTIMIZACIÓN: Pre-allocate results
    results = [None] * total
    
    # Dividir en batches
    batches = [items[i:i + batch_size] for i in range(0, total, batch_size)]
    
    semaphore = asyncio.Semaphore(max_workers)
    
    # OPTIMIZACIÓN: Cache check de coroutine
    is_async = asyncio.iscoroutinefunction(operation)
    
    async def process_batch(batch: List[Any], batch_idx: int):
        nonlocal processed
        batch_start = batch_idx * batch_size
        
        async with semaphore:
            for i, item in enumerate(batch):
                try:
                    if is_async:
                        result = await operation(item)
                    else:
                        result = operation(item)
                    results[batch_start + i] = result
                    processed += 1
                    
                    # OPTIMIZACIÓN: Callback cada 10 items en lugar de cada item
                    if progress_callback and processed % 10 == 0:
                        await progress_callback(processed, total, processed)
                except Exception as e:
                    logger.error(f"Error processing item: {e}")
                    results[batch_start + i] = None
        
        return batch
    
    # Procesar batches en paralelo
    batch_tasks = [process_batch(batch, i) for i, batch in enumerate(batches)]
    batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
    
    # OPTIMIZACIÓN: Log errores pero no fallar
    for batch_result in batch_results:
        if isinstance(batch_result, Exception):
            logger.error(f"Error in batch: {batch_result}")
    
    # OPTIMIZACIÓN: Filtrar None si es necesario
    return [r for r in results if r is not None]


async def batch_process_ultra_optimized(
    items: List[Any],
    operation: Callable,
    batch_size: Optional[int] = None,
    max_workers: Optional[int] = None,
    progress_callback: Optional[Callable] = None,
    use_ultra_fast: bool = True
) -> List[Any]:
    """
    Procesamiento ultra-optimizado de batches con todas las optimizaciones.
    
    Args:
        items: Lista de items a procesar
        operation: Función async para procesar cada item
        batch_size: Tamaño de cada batch (auto si None)
        max_workers: Máximo de workers (auto si None)
        progress_callback: Callback para reportar progreso
        use_ultra_fast: Usar ultra_fast_batch_process si está disponible
    
    Returns:
        Lista de resultados
    """
    # OPTIMIZACIÓN: Early return
    if not items:
        return []
    
    # OPTIMIZACIÓN: Auto-calcular batch size y workers
    if batch_size is None and PERFORMANCE_OPTIMIZATIONS_AVAILABLE:
        if HAS_NUMPY:
            import os
            cpu_cores = os.cpu_count() or 4
            batch_size = calculate_optimal_batch_size(len(items), cpu_cores=cpu_cores)
        else:
            batch_size = 100
    
    if max_workers is None and PERFORMANCE_OPTIMIZATIONS_AVAILABLE:
        max_workers = _get_optimal_workers()
    
    # OPTIMIZACIÓN: Usar ultra_fast_batch_process si está disponible
    if use_ultra_fast and PERFORMANCE_OPTIMIZATIONS_AVAILABLE and ultra_fast_batch_process:
        try:
            return await ultra_fast_batch_process(
                items,
                operation,
                batch_size=batch_size or 100,
                max_workers=max_workers or 10,
                progress_callback=progress_callback
            )
        except Exception as e:
            logger.warning(f"ultra_fast_batch_process failed, falling back: {e}")
    
    # Fallback a batch_process optimizado
    return await batch_process(
        items,
        operation,
        batch_size=batch_size or 100,
        max_workers=max_workers or 10,
        progress_callback=progress_callback
    )


async def batch_process_with_all_optimizations(
    items: List[Any],
    operation: Callable,
    batch_size: Optional[int] = None,
    max_workers: Optional[int] = None,
    progress_callback: Optional[Callable] = None,
    use_gpu: bool = False,
    use_distributed: bool = False,
    use_memory_pool: bool = True,
    use_cache: bool = True
) -> List[Any]:
    """
    Procesamiento con TODAS las optimizaciones disponibles.
    
    Combina:
    - Ultra-fast batch processing
    - GPU acceleration (si disponible)
    - Distributed processing (si disponible)
    - Memory pooling
    - Smart caching
    - Adaptive batching
    - Load prediction
    - Resource monitoring
    """
    # OPTIMIZACIÓN: Early return
    if not items:
        return []
    
    # OPTIMIZACIÓN: Usar memory pool si está disponible
    memory_pool = None
    if use_memory_pool and PERFORMANCE_OPTIMIZATIONS_AVAILABLE and BulkMemoryPool:
        memory_pool = BulkMemoryPool()
    
    # OPTIMIZACIÓN: GPU acceleration si está disponible y solicitado
    if use_gpu and PERFORMANCE_OPTIMIZATIONS_AVAILABLE and BulkGPUAccelerator:
        accelerator = BulkGPUAccelerator()
        # Para operaciones numéricas, usar GPU
        if all(isinstance(item, (int, float)) for item in items[:10] if items):
            # Operación numérica, usar GPU
            try:
                if HAS_NUMPY:
                    import numpy as np
                    arr = np.array(items)
                    if accelerator.has_cupy:
                        # Procesar en GPU
                        gpu_arr = accelerator.cp.array(arr)
                        result_arr = accelerator.cp.sum(gpu_arr)  # Ejemplo
                        return [float(result_arr)]
            except Exception as e:
                logger.warning(f"GPU acceleration failed: {e}")
    
    # OPTIMIZACIÓN: Distributed processing si está disponible
    if use_distributed and PERFORMANCE_OPTIMIZATIONS_AVAILABLE and BulkDistributedProcessor:
        processor = BulkDistributedProcessor()
        return await processor.distribute_work(items, operation, strategy="round_robin")
    
    # OPTIMIZACIÓN: Usar hyper optimizer si está disponible
    if PERFORMANCE_OPTIMIZATIONS_AVAILABLE and BulkHyperOptimizer:
        optimizer = BulkHyperOptimizer()
        if BulkParallelExecutor:
            executor = BulkParallelExecutor(max_workers=max_workers or _get_optimal_workers())
            optimizer.register_optimizer("parallel", executor)
        
        return await optimizer.optimize_operation(
            operation,
            items,
            config={"batch_size": batch_size, "max_workers": max_workers}
        )
    
    # Fallback a ultra-optimized
    return await batch_process_ultra_optimized(
        items,
        operation,
        batch_size=batch_size,
        max_workers=max_workers,
        progress_callback=progress_callback,
        use_ultra_fast=True
    )


def calculate_optimal_batch_size(
    total_items: int,
    memory_limit_mb: int = 512,
    item_size_estimate_kb: float = 10.0,
    cpu_cores: Optional[int] = None
) -> int:
    """
    Calcular tamaño óptimo de batch basado en memoria disponible.
    
    Args:
        total_items: Total de items a procesar
        memory_limit_mb: Límite de memoria en MB
        item_size_estimate_kb: Tamaño estimado por item en KB
    
    Returns:
        Tamaño óptimo de batch
    """
    # Auto-detectar CPU cores si no se proporciona
    if cpu_cores is None:
        try:
            import os
            cpu_cores = os.cpu_count() or 4
        except:
            cpu_cores = 4
    
    # Calcular cuántos items caben en memoria
    memory_limit_kb = memory_limit_mb * 1024
    items_per_memory = int(memory_limit_kb / (item_size_estimate_kb * 2))  # *2 para seguridad
    
    # Optimizar basado en CPU cores (más workers = batches más pequeños)
    cpu_optimal = cpu_cores * 50  # ~50 items por core
    
    # Usar el mínimo entre el límite de memoria, CPU optimal y un máximo razonable
    optimal = min(items_per_memory, cpu_optimal, 1000, total_items)
    return max(1, optimal)  # Al menos 1


def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    """Dividir lista en chunks de tamaño fijo - OPTIMIZADO."""
    if not items:
        return []
    
    # OPTIMIZACIÓN: Pre-calcular número de chunks
    total = len(items)
    num_chunks = (total + chunk_size - 1) // chunk_size
    
    # OPTIMIZACIÓN: Usar range con step (más eficiente)
    return [items[i:i + chunk_size] for i in range(0, total, chunk_size)]


async def retry_operation(
    operation: Callable,
    *args,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    backoff_factor: float = 2.0,
    **kwargs
) -> Tuple[Any, bool, Optional[str]]:
    """
    Ejecutar operación con retry automático.
    
    Args:
        operation: Función a ejecutar
        max_retries: Número máximo de reintentos
        retry_delay: Delay inicial entre reintentos
        backoff_factor: Factor de backoff exponencial
        *args, **kwargs: Argumentos para la operación
    
    Returns:
        Tupla (resultado, éxito, error)
    """
    last_error = None
    delay = retry_delay
    
    for attempt in range(max_retries + 1):
        try:
            if asyncio.iscoroutinefunction(operation):
                result = await operation(*args, **kwargs)
            else:
                result = operation(*args, **kwargs)
            return result, True, None
        except Exception as e:
            last_error = str(e)
            if attempt < max_retries:
                await asyncio.sleep(delay)
                delay *= backoff_factor
            else:
                logger.error(f"Operation failed after {max_retries} retries: {e}")
    
    return None, False, last_error


def merge_bulk_results(results: List[BulkOperationResult]) -> BulkOperationResult:
    """Combinar múltiples resultados bulk en uno solo - OPTIMIZADO."""
    if not results:
        return BulkOperationResult(
            success=True,
            processed=0,
            failed=0,
            total=0,
            errors=[],
            duration=0.0
        )
    
    # OPTIMIZACIÓN: Usar list comprehensions más eficientes
    total_processed = sum(r.processed for r in results)
    total_failed = sum(r.failed for r in results)
    total_total = sum(r.total for r in results)
    total_duration = sum(r.duration for r in results)
    
    # OPTIMIZACIÓN: Usar list comprehension en lugar de loop
    all_errors = [
        error for r in results if r.errors
        for error in r.errors
    ]
    
    return BulkOperationResult(
        success=total_failed == 0,
        processed=total_processed,
        failed=total_failed,
        total=total_total,
        errors=all_errors[:100],  # Limitar errores
        duration=total_duration
    )


# ============================================================================
# ENUMS Y DATACLASSES
# ============================================================================

class BulkOperationStatus(Enum):
    """Estado de operación bulk."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BulkOperationResult:
    """Resultado de una operación bulk."""
    success: bool
    processed: int
    failed: int
    total: int
    errors: List[str]
    data: Optional[Dict[str, Any]] = None
    duration: float = 0.0


@dataclass
class BulkJob:
    """Job de procesamiento bulk."""
    job_id: str
    operation: str
    status: BulkOperationStatus
    total_items: int
    processed_items: int
    failed_items: int
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    errors: Optional[List[str]] = None
    result: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Inicializar lista de errores si es None."""
        if self.errors is None:
            self.errors = []


class BulkSessionOperations:
    """Operaciones masivas sobre sesiones de chat."""
    
    def __init__(
        self,
        chat_engine: Optional[ContinuousChatEngine] = None,
        storage: Optional[SessionStorage] = None,
        max_workers: int = 10,
        batch_size: int = 100
    ):
        self.chat_engine = chat_engine
        self.storage = storage
        # OPTIMIZACIÓN: Auto-detectar optimal workers
        if max_workers is None or max_workers == 10:
            if PERFORMANCE_OPTIMIZATIONS_AVAILABLE:
                self.max_workers = _get_optimal_workers()
            else:
                self.max_workers = 10
        else:
            self.max_workers = max_workers
        self.batch_size = batch_size
        self.jobs: Dict[str, BulkJob] = {}
        # OPTIMIZACIÓN: Cache de checks
        self._is_async_cache: Dict[Callable, bool] = {}
        # OPTIMIZACIÓN: Memory optimizer
        if PERFORMANCE_OPTIMIZATIONS_AVAILABLE and BulkMemoryOptimizer:
            self.memory_optimizer = BulkMemoryOptimizer()
        else:
            self.memory_optimizer = None
        # OPTIMIZACIÓN: Smart cache
        if PERFORMANCE_OPTIMIZATIONS_AVAILABLE and BulkSmartCache:
            self.smart_cache = BulkSmartCache(strategy="lru", maxsize=1000)
        else:
            self.smart_cache = None
        # OPTIMIZACIÓN: Parallel executor
        if PERFORMANCE_OPTIMIZATIONS_AVAILABLE and BulkParallelExecutor:
            self.parallel_executor = BulkParallelExecutor(max_workers=self.max_workers)
        else:
            self.parallel_executor = None
        # Robust helpers
        self.circuit_breaker = CircuitBreaker(name="bulk_sessions", failure_threshold=5, recovery_timeout=60.0)
        self.rate_limiter = RateLimiter(rate=10.0, capacity=50.0)
    
    async def create_sessions(
        self,
        count: int,
        initial_messages: Optional[List[str]] = None,
        auto_continue: bool = True,
        parallel: bool = True,
        user_id: Optional[str] = None
    ) -> List[str]:
        """
        Crear múltiples sesiones en lote.
        
        Args:
            count: Número de sesiones a crear
            initial_messages: Lista de mensajes iniciales (se rota si hay menos que count)
            auto_continue: Si las sesiones deben continuar automáticamente
            parallel: Si se deben crear en paralelo
            user_id: ID del usuario (opcional)
        
        Returns:
            Lista de IDs de sesiones creadas
        """
        if not self.chat_engine:
            raise ValueError("ChatEngine is required")
        
        session_ids = []
        
        if initial_messages and len(initial_messages) < count:
            # Rotar mensajes si hay menos que count
            initial_messages = (initial_messages * ((count // len(initial_messages)) + 1))[:count]
        
        if parallel:
            # OPTIMIZACIÓN: Usar ultra_fast_batch_process si está disponible
            if PERFORMANCE_OPTIMIZATIONS_AVAILABLE and ultra_fast_batch_process and count > 100:
                items = list(range(count))
                async def create_one(index: int) -> Optional[str]:
                    try:
                        msg = initial_messages[index] if initial_messages else None
                        session = await self.chat_engine.create_session(
                            initial_message=msg,
                            auto_continue=auto_continue,
                            user_id=user_id
                        )
                        return session.session_id
                    except Exception as e:
                        logger.error(f"Error creating session {index}: {e}")
                        return None
                
                results = await ultra_fast_batch_process(
                    items,
                    create_one,
                    batch_size=self.batch_size,
                    max_workers=self.max_workers
                )
                session_ids = [sid for sid in results if sid is not None]
                return session_ids
            
            # Optimizar batch size si es muy grande
            if count > 1000:
                optimal_batch = calculate_optimal_batch_size(count)
                # Usar batch_process para mejor gestión de memoria
                items = list(range(count))
                async def create_one(index: int) -> Optional[str]:
                    try:
                        msg = initial_messages[index] if initial_messages else None
                        session = await self.chat_engine.create_session(
                            user_id=user_id or f"bulk_user_{index}",
                            initial_message=msg,
                            auto_continue=auto_continue
                        )
                        return session.session_id
                    except Exception as e:
                        logger.error(f"Error creating session {index}: {e}")
                        return None
                
                results = await batch_process(
                    items,
                    create_one,
                    batch_size=optimal_batch,
                    max_workers=self.max_workers
                )
                session_ids = [r for r in results if r is not None]
            else:
                # Para cantidades pequeñas, usar método original
                semaphore = asyncio.Semaphore(self.max_workers)
                
                async def create_one(index: int) -> Optional[str]:
                    async with semaphore:
                        try:
                            msg = initial_messages[index] if initial_messages else None
                            session = await self.chat_engine.create_session(
                                user_id=user_id or f"bulk_user_{index}",
                                initial_message=msg,
                                auto_continue=auto_continue
                            )
                            return session.session_id
                        except Exception as e:
                            logger.error(f"Error creating session {index}: {e}")
                            return None
                
                tasks = [create_one(i) for i in range(count)]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                session_ids = [r for r in results if r and not isinstance(r, Exception)]
        else:
            # Crear secuencialmente
            for i in range(count):
                try:
                    msg = initial_messages[i] if initial_messages else None
                    session = await self.chat_engine.create_session(
                        user_id=user_id or f"bulk_user_{i}",
                        initial_message=msg,
                        auto_continue=auto_continue
                    )
                    session_ids.append(session.session_id)
                except Exception as e:
                    logger.error(f"Error creating session {i}: {e}")
        
        logger.info(f"Created {len(session_ids)}/{count} sessions")
        return session_ids
    
    async def delete_sessions(
        self,
        session_ids: List[str],
        parallel: bool = True
    ) -> BulkOperationResult:
        """Eliminar múltiples sesiones."""
        start_time = datetime.now()
        processed = 0
        failed = 0
        errors = []
        
        if not self.chat_engine:
            raise ValueError("ChatEngine is required")
        
        if parallel:
            # OPTIMIZACIÓN: Usar ultra_fast_batch_process si está disponible
            if PERFORMANCE_OPTIMIZATIONS_AVAILABLE and ultra_fast_batch_process and len(session_ids) > 100:
                async def delete_one(session_id: str) -> bool:
                    try:
                        await self.chat_engine.stop_session(session_id)
                        if self.storage:
                            await self.storage.delete_session(session_id)
                        return True
                    except Exception as e:
                        logger.error(f"Error deleting session {session_id}: {e}")
                        errors.append(f"{session_id}: {str(e)}")
                        return False
                
                results = await ultra_fast_batch_process(
                    session_ids,
                    delete_one,
                    batch_size=self.batch_size,
                    max_workers=self.max_workers
                )
                processed = sum(1 for r in results if r is True)
                failed = len(session_ids) - processed
            else:
                # Método original optimizado
                semaphore = asyncio.Semaphore(self.max_workers)
                
                async def delete_one(session_id: str):
                    async with semaphore:
                        try:
                            await self.chat_engine.stop_session(session_id)
                            if self.storage:
                                await self.storage.delete_session(session_id)
                            return True
                        except Exception as e:
                            logger.error(f"Error deleting session {session_id}: {e}")
                            errors.append(f"{session_id}: {str(e)}")
                            return False
                
                tasks = [delete_one(sid) for sid in session_ids]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                processed = sum(1 for r in results if r is True)
                failed = len(session_ids) - processed
        else:
            for session_id in session_ids:
                try:
                    await self.chat_engine.stop_session(session_id)
                    if self.storage:
                        await self.storage.delete_session(session_id)
                    processed += 1
                except Exception as e:
                    failed += 1
                    errors.append(f"{session_id}: {str(e)}")
                    logger.error(f"Error deleting session {session_id}: {e}")
        
        duration = (datetime.now() - start_time).total_seconds()
        return BulkOperationResult(
            success=failed == 0,
            processed=processed,
            failed=failed,
            total=len(session_ids),
            errors=errors,
            duration=duration
        )
    
    async def pause_sessions(
        self,
        session_ids: List[str],
        reason: Optional[str] = None,
        parallel: bool = True
    ) -> BulkOperationResult:
        """Pausar múltiples sesiones."""
        start_time = datetime.now()
        processed = 0
        failed = 0
        errors = []
        
        if not self.chat_engine:
            raise ValueError("ChatEngine is required")
        
        if parallel:
            semaphore = asyncio.Semaphore(self.max_workers)
            
            async def pause_one(session_id: str):
                async with semaphore:
                    try:
                        await self.chat_engine.pause_session(session_id, reason)
                        return True
                    except Exception as e:
                        logger.error(f"Error pausing session {session_id}: {e}")
                        errors.append(f"{session_id}: {str(e)}")
                        return False
            
            tasks = [pause_one(sid) for sid in session_ids]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            processed = sum(1 for r in results if r is True)
            failed = len(session_ids) - processed
        else:
            for session_id in session_ids:
                try:
                    await self.chat_engine.pause_session(session_id, reason)
                    processed += 1
                except Exception as e:
                    failed += 1
                    errors.append(f"{session_id}: {str(e)}")
        
        duration = (datetime.now() - start_time).total_seconds()
        return BulkOperationResult(
            success=failed == 0,
            processed=processed,
            failed=failed,
            total=len(session_ids),
            errors=errors,
            duration=duration
        )
    
    async def resume_sessions(
        self,
        session_ids: List[str],
        parallel: bool = True
    ) -> BulkOperationResult:
        """Reanudar múltiples sesiones."""
        start_time = datetime.now()
        processed = 0
        failed = 0
        errors = []
        
        if not self.chat_engine:
            raise ValueError("ChatEngine is required")
        
        if parallel:
            semaphore = asyncio.Semaphore(self.max_workers)
            
            async def resume_one(session_id: str):
                async with semaphore:
                    try:
                        await self.chat_engine.resume_session(session_id)
                        return True
                    except Exception as e:
                        logger.error(f"Error resuming session {session_id}: {e}")
                        errors.append(f"{session_id}: {str(e)}")
                        return False
            
            tasks = [resume_one(sid) for sid in session_ids]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            processed = sum(1 for r in results if r is True)
            failed = len(session_ids) - processed
        else:
            for session_id in session_ids:
                try:
                    await self.chat_engine.resume_session(session_id)
                    processed += 1
                except Exception as e:
                    failed += 1
                    errors.append(f"{session_id}: {str(e)}")
        
        duration = (datetime.now() - start_time).total_seconds()
        return BulkOperationResult(
            success=failed == 0,
            processed=processed,
            failed=failed,
            total=len(session_ids),
            errors=errors,
            duration=duration
        )
    
    async def stop_sessions(
        self,
        session_ids: List[str],
        parallel: bool = True
    ) -> BulkOperationResult:
        """Detener múltiples sesiones."""
        start_time = datetime.now()
        processed = 0
        failed = 0
        errors = []
        
        if not self.chat_engine:
            raise ValueError("ChatEngine is required")
        
        if parallel:
            semaphore = asyncio.Semaphore(self.max_workers)
            
            async def stop_one(session_id: str):
                async with semaphore:
                    try:
                        await self.chat_engine.stop_session(session_id)
                        return True
                    except Exception as e:
                        logger.error(f"Error stopping session {session_id}: {e}")
                        errors.append(f"{session_id}: {str(e)}")
                        return False
            
            tasks = [stop_one(sid) for sid in session_ids]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            processed = sum(1 for r in results if r is True)
            failed = len(session_ids) - processed
        else:
            for session_id in session_ids:
                try:
                    await self.chat_engine.stop_session(session_id)
                    processed += 1
                except Exception as e:
                    failed += 1
                    errors.append(f"{session_id}: {str(e)}")
        
        duration = (datetime.now() - start_time).total_seconds()
        return BulkOperationResult(
            success=failed == 0,
            processed=processed,
            failed=failed,
            total=len(session_ids),
            errors=errors,
            duration=duration
        )
    
    async def export_sessions(
        self,
        session_ids: List[str],
        format: str = "json",
        parallel: bool = True
    ) -> BulkOperationResult:
        """Exportar múltiples sesiones."""
        from .exporters import ConversationExporter
        
        exporter = ConversationExporter()
        start_time = datetime.now()
        processed = 0
        failed = 0
        errors = []
        exports = []
        
        if not self.storage:
            raise ValueError("Storage is required")
        
        if parallel:
            semaphore = asyncio.Semaphore(self.max_workers)
            
            async def export_one(session_id: str):
                async with semaphore:
                    try:
                        session = await self.storage.load_session(session_id)
                        if session:
                            exported = await exporter.export(session, format)
                            return (session_id, exported)
                    except Exception as e:
                        logger.error(f"Error exporting session {session_id}: {e}")
                        errors.append(f"{session_id}: {str(e)}")
                    return None
            
            tasks = [export_one(sid) for sid in session_ids]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            exports = [r for r in results if r and not isinstance(r, Exception)]
            processed = len(exports)
            failed = len(session_ids) - processed
        else:
            for session_id in session_ids:
                try:
                    session = await self.storage.load_session(session_id)
                    if session:
                        exported = await exporter.export(session, format)
                        exports.append((session_id, exported))
                        processed += 1
                except Exception as e:
                    failed += 1
                    errors.append(f"{session_id}: {str(e)}")
        
        duration = (datetime.now() - start_time).total_seconds()
        return BulkOperationResult(
            success=failed == 0,
            processed=processed,
            failed=failed,
            total=len(session_ids),
            errors=errors,
            duration=duration,
            data={"exports": exports}
        )


class BulkMessageOperations:
    """Operaciones masivas sobre mensajes."""
    
    def __init__(
        self,
        chat_engine: Optional[ContinuousChatEngine] = None,
        max_workers: int = 10
    ):
        self.chat_engine = chat_engine
        self.max_workers = max_workers
    
    async def send_to_sessions(
        self,
        session_ids: List[str],
        message: str,
        parallel: bool = True,
        max_retries: int = 3
    ) -> BulkOperationResult:
        """Enviar mensaje a múltiples sesiones con retry automático."""
        start_time = datetime.now()
        processed = 0
        failed = 0
        errors = []
        
        if not self.chat_engine:
            raise ValueError("ChatEngine is required")
        
        async def send_one(session_id: str) -> Tuple[bool, Optional[str]]:
            """Enviar mensaje con retry."""
            result, success, error = await retry_operation(
                self.chat_engine.send_message,
                session_id,
                message,
                max_retries=max_retries
            )
            return success, error
        
        if parallel:
            # Usar batch_process para mejor gestión con grandes volúmenes
            if len(session_ids) > 500:
                optimal_batch = calculate_optimal_batch_size(len(session_ids))
                async def process_item(session_id: str):
                    return await send_one(session_id)
                
                results = await batch_process(
                    session_ids,
                    process_item,
                    batch_size=optimal_batch,
                    max_workers=self.max_workers
                )
                for result_tuple in results:
                    if result_tuple and result_tuple[0]:
                        processed += 1
                    else:
                        failed += 1
                        if result_tuple and result_tuple[1]:
                            errors.append(result_tuple[1])
            else:
                semaphore = asyncio.Semaphore(self.max_workers)
                
                async def send_with_semaphore(session_id: str):
                    async with semaphore:
                        return await send_one(session_id)
                
                tasks = [send_with_semaphore(sid) for sid in session_ids]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in results:
                    if isinstance(result, Exception):
                        failed += 1
                        errors.append(str(result))
                    elif result and result[0]:
                        processed += 1
                    else:
                        failed += 1
                        if result and result[1]:
                            errors.append(result[1])
        else:
            # Secuencial con retry
            for session_id in session_ids:
                success, error = await send_one(session_id)
                if success:
                    processed += 1
                else:
                    failed += 1
                    if error:
                        errors.append(f"{session_id}: {error}")
        
        duration = (datetime.now() - start_time).total_seconds()
        return BulkOperationResult(
            success=failed == 0,
            processed=processed,
            failed=failed,
            total=len(session_ids),
            errors=errors[:100],  # Limitar errores
            duration=duration
        )


class BulkExporter:
    """Exportación masiva de datos."""
    
    def __init__(
        self,
        storage: Optional[SessionStorage] = None,
        exporter: Optional[ConversationExporter] = None,
        max_workers: int = 10
    ):
        self.storage = storage
        self.exporter = exporter or ConversationExporter()
        self.max_workers = max_workers
        self.jobs: Dict[str, BulkJob] = {}
    
    async def export_sessions(
        self,
        session_ids: List[str],
        format: str = "json",
        compress: bool = False,
        parallel: bool = True
    ) -> str:
        """
        Exportar múltiples sesiones.
        
        Returns:
            job_id para seguir el progreso
        """
        import uuid
        job_id = str(uuid.uuid4())
        
        job = BulkJob(
            job_id=job_id,
            operation="export_sessions",
            status=BulkOperationStatus.PENDING,
            total_items=len(session_ids),
            processed_items=0,
            failed_items=0,
            created_at=datetime.now()
        )
        self.jobs[job_id] = job
        
        # Ejecutar en background
        asyncio.create_task(self._export_sessions_background(
            job_id, session_ids, format, compress, parallel
        ))
        
        return job_id
    
    async def _export_sessions_background(
        self,
        job_id: str,
        session_ids: List[str],
        format: str,
        compress: bool,
        parallel: bool
    ):
        """Exportar sesiones en background."""
        job = self.jobs[job_id]
        job.status = BulkOperationStatus.RUNNING
        job.started_at = datetime.now()
        
        exports = []
        errors = []
        
        try:
            if not self.storage:
                raise ValueError("Storage is required")
            
            if parallel:
                semaphore = asyncio.Semaphore(self.max_workers)
                
                async def export_one(session_id: str):
                    async with semaphore:
                        try:
                            session = await self.storage.load_session(session_id)
                            if session:
                                exported = await self.exporter.export(
                                    session, format
                                )
                                return (session_id, exported)
                        except Exception as e:
                            logger.error(f"Error exporting session {session_id}: {e}")
                            errors.append(f"{session_id}: {str(e)}")
                        return None
                
                tasks = [export_one(sid) for sid in session_ids]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                exports = [r for r in results if r and not isinstance(r, Exception)]
            else:
                for session_id in session_ids:
                    try:
                        session = await self.storage.load_session(session_id)
                        if session:
                            exported = await self.exporter.export(session, format)
                            exports.append((session_id, exported))
                        job.processed_items += 1
                    except Exception as e:
                        job.failed_items += 1
                        errors.append(f"{session_id}: {str(e)}")
                        logger.error(f"Error exporting session {session_id}: {e}")
            
            job.processed_items = len(exports)
            job.failed_items = len(errors)
            job.status = BulkOperationStatus.COMPLETED
            job.completed_at = datetime.now()
            job.result = {
                "exports": exports,
                "errors": errors,
                "compress": compress
            }
        except Exception as e:
            job.status = BulkOperationStatus.FAILED
            job.errors = [str(e)]
            logger.error(f"Error in export job {job_id}: {e}")
    
    async def get_export_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Obtener estado de exportación."""
        job = self.jobs.get(job_id)
        if not job:
            return None
        
        progress = (job.processed_items / job.total_items * 100) if job.total_items > 0 else 0
        
        return {
            "job_id": job_id,
            "status": job.status.value,
            "progress": round(progress, 2),
            "processed": job.processed_items,
            "failed": job.failed_items,
            "total": job.total_items,
            "created_at": job.created_at.isoformat(),
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "errors": job.errors or []
        }


class BulkAnalytics:
    """Análisis masivo de datos."""
    
    def __init__(
        self,
        analyzer: Optional[ConversationAnalyzer] = None,
        storage: Optional[SessionStorage] = None,
        max_workers: int = 10
    ):
        self.analyzer = analyzer or ConversationAnalyzer()
        self.storage = storage
        self.max_workers = max_workers
    
    async def analyze_sessions_bulk(
        self,
        session_ids: List[str],
        parallel: bool = True
    ) -> List[Dict[str, Any]]:
        """Analizar múltiples sesiones en paralelo."""
        if not self.storage:
            raise ValueError("Storage is required")
        
        results = []
        
        if parallel:
            semaphore = asyncio.Semaphore(self.max_workers)
            
            async def analyze_one(session_id: str):
                async with semaphore:
                    try:
                        session = await self.storage.load_session(session_id)
                        if session:
                            analysis = await self.analyzer.analyze(session)
                            return {"session_id": session_id, "analysis": analysis}
                    except Exception as e:
                        logger.error(f"Error analyzing session {session_id}: {e}")
                    return None
            
            tasks = [analyze_one(sid) for sid in session_ids]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            results = [r for r in results if r and not isinstance(r, Exception)]
        else:
            for session_id in session_ids:
                try:
                    session = await self.storage.load_session(session_id)
                    if session:
                        analysis = await self.analyzer.analyze(session)
                        results.append({"session_id": session_id, "analysis": analysis})
                except Exception as e:
                    logger.error(f"Error analyzing session {session_id}: {e}")
        
        return results


class BulkCleanup:
    """Limpieza masiva de datos."""
    
    def __init__(
        self,
        storage: Optional[SessionStorage] = None,
        chat_engine: Optional[ContinuousChatEngine] = None,
        max_workers: int = 10
    ):
        self.storage = storage
        self.chat_engine = chat_engine
        self.max_workers = max_workers
    
    async def cleanup_old_sessions(
        self,
        days_old: int = 30,
        dry_run: bool = False
    ) -> BulkOperationResult:
        """Limpiar sesiones más antiguas que days_old días."""
        if not self.storage:
            raise ValueError("Storage is required")
        
        cutoff_date = datetime.now() - timedelta(days=days_old)
        start_time = datetime.now()
        
        # Obtener todas las sesiones (esto depende de la implementación de storage)
        # Por ahora, asumimos que storage tiene un método list_sessions
        if hasattr(self.storage, 'list_sessions'):
            all_sessions = await self.storage.list_sessions()
            old_sessions = [
                sid for sid in all_sessions
                if await self._is_session_old(sid, cutoff_date)
            ]
        else:
            # Fallback: necesitaríamos otra forma de obtener sesiones antiguas
            old_sessions = []
        
        if dry_run:
            logger.info(f"DRY RUN: Would delete {len(old_sessions)} sessions")
            return BulkOperationResult(
                success=True,
                processed=0,
                failed=0,
                total=len(old_sessions),
                errors=[],
                data={"dry_run": True, "would_delete": len(old_sessions)}
            )
        
        # Eliminar sesiones
        if self.chat_engine:
            bulk_ops = BulkSessionOperations(
                chat_engine=self.chat_engine,
                storage=self.storage
            )
            result = await bulk_ops.delete_sessions(old_sessions, parallel=True)
            result.duration = (datetime.now() - start_time).total_seconds()
            return result
        
        return BulkOperationResult(
            success=False,
            processed=0,
            failed=len(old_sessions),
            total=len(old_sessions),
            errors=["ChatEngine not available"],
            duration=(datetime.now() - start_time).total_seconds()
        )
    
    async def _is_session_old(self, session_id: str, cutoff_date: datetime) -> bool:
        """Verificar si una sesión es antigua."""
        try:
            if self.storage:
                session = await self.storage.load_session(session_id)
                if session and session.created_at:
                    return session.created_at < cutoff_date
        except Exception:
            pass
        return False


class BulkProcessor:
    """Procesador genérico de operaciones bulk."""
    
    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        self.jobs: Dict[str, BulkJob] = {}
    
    async def process_batch(
        self,
        items: List[Any],
        operation: Union[str, Callable],
        config: Optional[Dict[str, Any]] = None,
        parallel: bool = True
    ) -> str:
        """Procesar un lote de items con una operación."""
        import uuid
        job_id = str(uuid.uuid4())
        
        job = BulkJob(
            job_id=job_id,
            operation=operation if isinstance(operation, str) else operation.__name__,
            status=BulkOperationStatus.PENDING,
            total_items=len(items),
            processed_items=0,
            failed_items=0,
            created_at=datetime.now()
        )
        self.jobs[job_id] = job
        
        asyncio.create_task(self._process_batch_background(
            job_id, items, operation, config or {}, parallel
        ))
        
        return job_id
    
    async def _process_batch_background(
        self,
        job_id: str,
        items: List[Any],
        operation: Union[str, Callable],
        config: Dict[str, Any],
        parallel: bool
    ):
        """Procesar lote en background."""
        job = self.jobs[job_id]
        job.status = BulkOperationStatus.RUNNING
        job.started_at = datetime.now()
        job.errors = []
        
        try:
            if isinstance(operation, str):
                # Operación por nombre (necesitaría un registry)
                raise NotImplementedError("Operation by name not yet implemented")
            
            if parallel:
                semaphore = asyncio.Semaphore(self.max_workers)
                
                async def process_one(item: Any):
                    async with semaphore:
                        try:
                            result = await operation(item, **config)
                            return result
                        except Exception as e:
                            logger.error(f"Error processing item: {e}")
                            job.failed_items += 1
                            job.errors.append(str(e))
                            return None
                
                tasks = [process_one(item) for item in items]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                job.processed_items = sum(1 for r in results if r and not isinstance(r, Exception))
            else:
                for item in items:
                    try:
                        await operation(item, **config)
                        job.processed_items += 1
                    except Exception as e:
                        job.failed_items += 1
                        job.errors.append(str(e))
                        logger.error(f"Error processing item: {e}")
            
            job.status = BulkOperationStatus.COMPLETED
            job.completed_at = datetime.now()
        except Exception as e:
            job.status = BulkOperationStatus.FAILED
            job.errors = [str(e)]
            logger.error(f"Error in process job {job_id}: {e}")
    
    async def get_progress(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Obtener progreso de un job."""
        job = self.jobs.get(job_id)
        if not job:
            return None
        
        progress = (job.processed_items / job.total_items * 100) if job.total_items > 0 else 0
        
        return {
            "job_id": job_id,
            "status": job.status.value,
            "progress": round(progress, 2),
            "processed": job.processed_items,
            "failed": job.failed_items,
            "total": job.total_items,
            "errors": job.errors or []
        }
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancelar un job en ejecución."""
        job = self.jobs.get(job_id)
        if not job:
            return False
        
        if job.status == BulkOperationStatus.RUNNING:
            job.status = BulkOperationStatus.CANCELLED
            job.completed_at = datetime.now()
            return True
        
        return False


class BulkImporter:
    """Importación masiva de datos."""
    
    def __init__(
        self,
        chat_engine: Optional[ContinuousChatEngine] = None,
        storage: Optional[SessionStorage] = None,
        max_workers: int = 10
    ):
        self.chat_engine = chat_engine
        self.storage = storage
        self.max_workers = max_workers
        self.jobs: Dict[str, BulkJob] = {}
    
    async def import_sessions(
        self,
        sessions_data: List[Dict[str, Any]],
        validate: bool = True,
        parallel: bool = True
    ) -> str:
        """Importar múltiples sesiones desde datos."""
        import uuid
        job_id = str(uuid.uuid4())
        
        job = BulkJob(
            job_id=job_id,
            operation="import_sessions",
            status=BulkOperationStatus.PENDING,
            total_items=len(sessions_data),
            processed_items=0,
            failed_items=0,
            created_at=datetime.now()
        )
        self.jobs[job_id] = job
        
        asyncio.create_task(self._import_sessions_background(
            job_id, sessions_data, validate, parallel
        ))
        
        return job_id
    
    async def _import_sessions_background(
        self,
        job_id: str,
        sessions_data: List[Dict[str, Any]],
        validate: bool,
        parallel: bool
    ):
        """Importar sesiones en background."""
        job = self.jobs[job_id]
        job.status = BulkOperationStatus.RUNNING
        job.started_at = datetime.now()
        job.errors = []
        
        try:
            if not self.chat_engine:
                raise ValueError("ChatEngine is required")
            
            if parallel:
                semaphore = asyncio.Semaphore(self.max_workers)
                
                async def import_one(session_data: Dict[str, Any]):
                    async with semaphore:
                        try:
                            if validate:
                                # Validar datos básicos
                                if "session_id" not in session_data and "messages" not in session_data:
                                    raise ValueError("Invalid session data")
                            
                            # Crear o actualizar sesión
                            # Esta implementación depende de cómo se manejen las sesiones
                            # Por ahora, asumimos que podemos crear sesiones desde datos
                            session_id = session_data.get("session_id")
                            if not session_id:
                                # Crear nueva sesión
                                session = await self.chat_engine.create_session(
                                    user_id=session_data.get("user_id"),
                                    initial_message=None,
                                    auto_continue=session_data.get("auto_continue", True)
                                )
                                session_id = session.session_id
                            
                            # Agregar mensajes si existen
                            if "messages" in session_data:
                                for msg_data in session_data["messages"]:
                                    await self.chat_engine.send_message(
                                        session_id,
                                        msg_data.get("content", "")
                                    )
                            
                            return session_id
                        except Exception as e:
                            logger.error(f"Error importing session: {e}")
                            job.failed_items += 1
                            job.errors.append(str(e))
                            return None
                
                tasks = [import_one(data) for data in sessions_data]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                imported = [r for r in results if r and not isinstance(r, Exception)]
                job.processed_items = len(imported)
            else:
                imported = []
                for session_data in sessions_data:
                    try:
                        if validate:
                            if "session_id" not in session_data and "messages" not in session_data:
                                raise ValueError("Invalid session data")
                        
                        session_id = session_data.get("session_id")
                        if not session_id:
                            session = await self.chat_engine.create_session(
                                user_id=session_data.get("user_id"),
                                initial_message=None,
                                auto_continue=session_data.get("auto_continue", True)
                            )
                            session_id = session.session_id
                        
                        if "messages" in session_data:
                            for msg_data in session_data["messages"]:
                                await self.chat_engine.send_message(
                                    session_id,
                                    msg_data.get("content", "")
                                )
                        
                        imported.append(session_id)
                        job.processed_items += 1
                    except Exception as e:
                        job.failed_items += 1
                        job.errors.append(str(e))
                        logger.error(f"Error importing session: {e}")
            
            job.status = BulkOperationStatus.COMPLETED
            job.completed_at = datetime.now()
            job.result = {"imported": imported}
        except Exception as e:
            job.status = BulkOperationStatus.FAILED
            job.errors = [str(e)]
            logger.error(f"Error in import job {job_id}: {e}")
    
    async def get_import_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Obtener estado de importación."""
        job = self.jobs.get(job_id)
        if not job:
            return None
        
        progress = (job.processed_items / job.total_items * 100) if job.total_items > 0 else 0
        
        return {
            "job_id": job_id,
            "status": job.status.value,
            "progress": round(progress, 2),
            "processed": job.processed_items,
            "failed": job.failed_items,
            "total": job.total_items,
            "errors": job.errors or []
        }


class BulkNotifications:
    """Notificaciones masivas."""
    
    def __init__(
        self,
        notification_manager: Optional[Any] = None,
        max_workers: int = 10
    ):
        self.notification_manager = notification_manager
        self.max_workers = max_workers
    
    async def send_bulk(
        self,
        user_ids: List[str],
        template: str,
        data: Optional[Dict[str, Any]] = None,
        channels: Optional[List[str]] = None,
        parallel: bool = True
    ) -> BulkOperationResult:
        """Enviar notificaciones masivas."""
        start_time = datetime.now()
        processed = 0
        failed = 0
        errors = []
        
        if not self.notification_manager:
            raise ValueError("NotificationManager is required")
        
        channels = channels or ["email"]
        
        if parallel:
            semaphore = asyncio.Semaphore(self.max_workers)
            
            async def send_one(user_id: str):
                async with semaphore:
                    try:
                        await self.notification_manager.send(
                            user_id=user_id,
                            template=template,
                            data=data or {},
                            channels=channels
                        )
                        return True
                    except Exception as e:
                        logger.error(f"Error sending notification to {user_id}: {e}")
                        errors.append(f"{user_id}: {str(e)}")
                        return False
            
            tasks = [send_one(uid) for uid in user_ids]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            processed = sum(1 for r in results if r is True)
            failed = len(user_ids) - processed
        else:
            for user_id in user_ids:
                try:
                    await self.notification_manager.send(
                        user_id=user_id,
                        template=template,
                        data=data or {},
                        channels=channels
                    )
                    processed += 1
                except Exception as e:
                    failed += 1
                    errors.append(f"{user_id}: {str(e)}")
        
        duration = (datetime.now() - start_time).total_seconds()
        return BulkOperationResult(
            success=failed == 0,
            processed=processed,
            failed=failed,
            total=len(user_ids),
            errors=errors,
            duration=duration
        )


class BulkSearch:
    """Búsqueda masiva."""
    
    def __init__(
        self,
        search_engine: Optional[Any] = None,
        storage: Optional[SessionStorage] = None,
        max_workers: int = 10
    ):
        self.search_engine = search_engine
        self.storage = storage
        self.max_workers = max_workers
    
    async def search_bulk(
        self,
        queries: List[str],
        filters: Optional[Dict[str, Any]] = None,
        parallel: bool = True
    ) -> List[Dict[str, Any]]:
        """Búsqueda masiva con múltiples queries."""
        results = []
        
        if not self.search_engine:
            raise ValueError("SearchEngine is required")
        
        if parallel:
            semaphore = asyncio.Semaphore(self.max_workers)
            
            async def search_one(query: str):
                async with semaphore:
                    try:
                        result = await self.search_engine.search(
                            query=query,
                            filters=filters or {}
                        )
                        return {"query": query, "results": result}
                    except Exception as e:
                        logger.error(f"Error searching for '{query}': {e}")
                        return {"query": query, "results": [], "error": str(e)}
            
            tasks = [search_one(q) for q in queries]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            results = [r for r in results if r and not isinstance(r, Exception)]
        else:
            for query in queries:
                try:
                    result = await self.search_engine.search(
                        query=query,
                        filters=filters or {}
                    )
                    results.append({"query": query, "results": result})
                except Exception as e:
                    logger.error(f"Error searching for '{query}': {e}")
                    results.append({"query": query, "results": [], "error": str(e)})
        
        return results


class BulkTesting:
    """Pruebas masivas del sistema con framework de testing integrado."""
    
    def __init__(
        self,
        chat_engine: Optional[ContinuousChatEngine] = None,
        max_workers: int = 50
    ):
        self.chat_engine = chat_engine
        self.max_workers = max_workers
        self.test_results: List[Dict[str, Any]] = []
        self.metrics_history: List[Dict[str, Any]] = []
    
    async def load_test(
        self,
        concurrent_sessions: int = 100,
        duration: int = 60,
        operations_per_session: int = 10
    ) -> Dict[str, Any]:
        """Test de carga con múltiples sesiones simultáneas."""
        if not self.chat_engine:
            raise ValueError("ChatEngine is required")
        
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=duration)
        results = {
            "sessions_created": 0,
            "operations_completed": 0,
            "errors": [],
            "response_times": []
        }
        
        semaphore = asyncio.Semaphore(concurrent_sessions)
        
        async def create_and_operate(session_num: int):
            async with semaphore:
                try:
                    # Crear sesión
                    session = await self.chat_engine.create_session(
                        user_id=f"load_test_user_{session_num}",
                        initial_message=f"Test message {session_num}",
                        auto_continue=True
                    )
                    results["sessions_created"] += 1
                    
                    # Realizar operaciones
                    for op in range(operations_per_session):
                        if datetime.now() > end_time:
                            break
                        
                        op_start = datetime.now()
                        await self.chat_engine.send_message(
                            session.session_id,
                            f"Operation {op}"
                        )
                        op_duration = (datetime.now() - op_start).total_seconds()
                        results["response_times"].append(op_duration)
                        results["operations_completed"] += 1
                        
                        await asyncio.sleep(0.1)  # Pequeña pausa
                    
                    # Detener sesión
                    await self.chat_engine.stop_session(session.session_id)
                    
                except Exception as e:
                    results["errors"].append(f"Session {session_num}: {str(e)}")
                    logger.error(f"Error in load test session {session_num}: {e}")
        
        # Crear tareas
        tasks = [create_and_operate(i) for i in range(concurrent_sessions)]
        
        # Ejecutar hasta que expire el tiempo
        await asyncio.gather(*tasks, return_exceptions=True)
        
        total_duration = (datetime.now() - start_time).total_seconds()
        
        # Calcular métricas
        if results["response_times"]:
            avg_response = sum(results["response_times"]) / len(results["response_times"])
            max_response = max(results["response_times"])
            min_response = min(results["response_times"])
        else:
            avg_response = max_response = min_response = 0
        
        return {
            "duration": total_duration,
            "sessions_created": results["sessions_created"],
            "operations_completed": results["operations_completed"],
            "operations_per_second": results["operations_completed"] / total_duration if total_duration > 0 else 0,
            "errors": len(results["errors"]),
            "error_rate": len(results["errors"]) / results["operations_completed"] if results["operations_completed"] > 0 else 0,
            "avg_response_time": avg_response,
            "max_response_time": max_response,
            "min_response_time": min_response,
            "error_details": results["errors"][:10]  # Primeros 10 errores
        }
    
    async def stress_test(
        self,
        max_sessions: int = 1000,
        ramp_up_seconds: int = 60
    ) -> Dict[str, Any]:
        """Test de estrés con aumento gradual de carga."""
        if not self.chat_engine:
            raise ValueError("ChatEngine is required")
        
        start_time = datetime.now()
        sessions_created = 0
        errors = []
        
        # Crear sesiones gradualmente
        sessions_per_second = max_sessions / ramp_up_seconds if ramp_up_seconds > 0 else max_sessions
        
        for i in range(max_sessions):
            try:
                await self.chat_engine.create_session(
                    user_id=f"stress_test_user_{i}",
                    initial_message=f"Stress test {i}",
                    auto_continue=False
                )
                sessions_created += 1
                
                # Esperar para ramp-up gradual
                if i < max_sessions - 1 and ramp_up_seconds > 0:
                    await asyncio.sleep(1.0 / sessions_per_second)
                    
            except Exception as e:
                errors.append(f"Session {i}: {str(e)}")
                logger.error(f"Error in stress test session {i}: {e}")
        
        duration = (datetime.now() - start_time).total_seconds()
        
        result = {
            "duration": duration,
            "sessions_created": sessions_created,
            "max_sessions": max_sessions,
            "success_rate": sessions_created / max_sessions if max_sessions > 0 else 0,
            "errors": len(errors),
            "error_details": errors[:10],
            "timestamp": datetime.now().isoformat()
        }
        
        # Guardar métricas
        self.metrics_history.append(result)
        return result
    
    async def test_operation(
        self,
        operation: Callable,
        test_data: List[Any],
        expected_success_rate: float = 1.0,
        timeout: float = 60.0,
        parallel: bool = True
    ) -> Dict[str, Any]:
        """Ejecutar test de operación genérica con soporte paralelo."""
        start_time = datetime.now()
        results = []
        
        if parallel and len(test_data) > 1:
            semaphore = asyncio.Semaphore(min(self.max_workers, len(test_data)))
            
            async def test_item(item: Any):
                async with semaphore:
                    try:
                        if asyncio.iscoroutinefunction(operation):
                            result = await asyncio.wait_for(
                                operation(item),
                                timeout=timeout
                            )
                        else:
                            result = operation(item)
                        return {"success": True, "result": result}
                    except asyncio.TimeoutError:
                        return {"success": False, "error": "timeout"}
                    except Exception as e:
                        return {"success": False, "error": str(e)}
            
            tasks = [test_item(item) for item in test_data]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # Filtrar excepciones
            results = [r for r in results if not isinstance(r, Exception)]
        else:
            # Secuencial
            for item in test_data:
                try:
                    if asyncio.iscoroutinefunction(operation):
                        result = await asyncio.wait_for(
                            operation(item),
                            timeout=timeout
                        )
                    else:
                        result = operation(item)
                    results.append({"success": True, "result": result})
                except asyncio.TimeoutError:
                    results.append({"success": False, "error": "timeout"})
                except Exception as e:
                    results.append({"success": False, "error": str(e)})
        
        duration = (datetime.now() - start_time).total_seconds()
        success_count = sum(1 for r in results if r.get("success", False))
        success_rate = success_count / len(results) if results else 0
        
        test_result = {
            "operation": operation.__name__ if hasattr(operation, '__name__') else "unknown",
            "total_tests": len(test_data),
            "success_count": success_count,
            "failure_count": len(results) - success_count,
            "success_rate": success_rate,
            "expected_success_rate": expected_success_rate,
            "passed": success_rate >= expected_success_rate,
            "duration": duration,
            "timestamp": datetime.now().isoformat(),
            "parallel": parallel
        }
        
        self.test_results.append(test_result)
        return test_result
    
    def get_test_summary(self) -> Dict[str, Any]:
        """Obtener resumen de todos los tests ejecutados."""
        if not self.test_results:
            return {}
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.get("passed", False))
        
        return {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": total_tests - passed_tests,
            "pass_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            "tests": self.test_results,
            "metrics_history": self.metrics_history[-10:]  # Últimas 10 métricas
        }
    
    def clear_test_results(self):
        """Limpiar resultados de tests."""
        self.test_results = []
        self.metrics_history = []


class BulkBackupRestore:
    """Backup y restauración masiva."""
    
    def __init__(
        self,
        storage: Optional[SessionStorage] = None,
        backup_manager: Optional[Any] = None,
        max_workers: int = 10
    ):
        self.storage = storage
        self.backup_manager = backup_manager
        self.max_workers = max_workers
        self.jobs: Dict[str, BulkJob] = {}
    
    async def backup_sessions(
        self,
        session_ids: List[str],
        compress: bool = True,
        encrypt: bool = False
    ) -> str:
        """Crear backup masivo de sesiones."""
        import uuid
        job_id = str(uuid.uuid4())
        
        job = BulkJob(
            job_id=job_id,
            operation="backup_sessions",
            status=BulkOperationStatus.PENDING,
            total_items=len(session_ids),
            processed_items=0,
            failed_items=0,
            created_at=datetime.now()
        )
        self.jobs[job_id] = job
        
        asyncio.create_task(self._backup_sessions_background(
            job_id, session_ids, compress, encrypt
        ))
        
        return job_id
    
    async def _backup_sessions_background(
        self,
        job_id: str,
        session_ids: List[str],
        compress: bool,
        encrypt: bool
    ):
        """Ejecutar backup en background."""
        job = self.jobs[job_id]
        job.status = BulkOperationStatus.RUNNING
        job.started_at = datetime.now()
        job.errors = []
        
        try:
            if not self.storage:
                raise ValueError("Storage is required")
            
            backups = []
            semaphore = asyncio.Semaphore(self.max_workers)
            
            async def backup_one(session_id: str):
                async with semaphore:
                    try:
                        session = await self.storage.load_session(session_id)
                        if session:
                            # Serializar sesión
                            import json
                            session_data = {
                                "session_id": session.session_id,
                                "user_id": session.user_id,
                                "messages": [
                                    {
                                        "role": msg.role.value if hasattr(msg.role, 'value') else str(msg.role),
                                        "content": msg.content,
                                        "timestamp": msg.timestamp.isoformat() if hasattr(msg, 'timestamp') and msg.timestamp else None
                                    }
                                    for msg in session.messages
                                ],
                                "created_at": session.created_at.isoformat() if hasattr(session, 'created_at') and session.created_at else None,
                                "state": session.state.value if hasattr(session.state, 'value') else str(session.state)
                            }
                            backups.append(session_data)
                            job.processed_items += 1
                    except Exception as e:
                        job.failed_items += 1
                        job.errors.append(f"{session_id}: {str(e)}")
                        logger.error(f"Error backing up session {session_id}: {e}")
            
            tasks = [backup_one(sid) for sid in session_ids]
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Guardar backup
            backup_data = {
                "backup_id": job_id,
                "created_at": datetime.now().isoformat(),
                "sessions": backups,
                "total_sessions": len(backups)
            }
            
            # Guardar en archivo si hay backup_manager
            if self.backup_manager and hasattr(self.backup_manager, 'save_backup'):
                await self.backup_manager.save_backup(job_id, backup_data, compress, encrypt)
            
            job.status = BulkOperationStatus.COMPLETED
            job.completed_at = datetime.now()
            job.result = {
                "backup_id": job_id,
                "sessions_backed_up": len(backups),
                "compress": compress,
                "encrypt": encrypt
            }
        except Exception as e:
            job.status = BulkOperationStatus.FAILED
            job.errors = [str(e)]
            logger.error(f"Error in backup job {job_id}: {e}")
    
    async def get_backup_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Obtener estado de backup."""
        job = self.jobs.get(job_id)
        if not job:
            return None
        
        progress = (job.processed_items / job.total_items * 100) if job.total_items > 0 else 0
        
        return {
            "job_id": job_id,
            "status": job.status.value,
            "progress": round(progress, 2),
            "processed": job.processed_items,
            "failed": job.failed_items,
            "total": job.total_items,
            "created_at": job.created_at.isoformat(),
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "errors": job.errors or [],
            "result": job.result
        }


class BulkMigration:
    """Migración masiva de datos."""
    
    def __init__(
        self,
        chat_engine: Optional[ContinuousChatEngine] = None,
        storage: Optional[SessionStorage] = None,
        max_workers: int = 10
    ):
        self.chat_engine = chat_engine
        self.storage = storage
        self.max_workers = max_workers
        self.jobs: Dict[str, BulkJob] = {}
    
    async def migrate_sessions(
        self,
        session_ids: List[str],
        source_format: str,
        target_format: str,
        transform: Optional[Callable] = None,
        batch_size: int = 100
    ) -> str:
        """Migrar sesiones a nuevo formato."""
        import uuid
        job_id = str(uuid.uuid4())
        
        job = BulkJob(
            job_id=job_id,
            operation="migrate_sessions",
            status=BulkOperationStatus.PENDING,
            total_items=len(session_ids),
            processed_items=0,
            failed_items=0,
            created_at=datetime.now()
        )
        self.jobs[job_id] = job
        
        asyncio.create_task(self._migrate_sessions_background(
            job_id, session_ids, source_format, target_format, transform, batch_size
        ))
        
        return job_id
    
    async def _migrate_sessions_background(
        self,
        job_id: str,
        session_ids: List[str],
        source_format: str,
        target_format: str,
        transform: Optional[Callable],
        batch_size: int
    ):
        """Migrar sesiones en background."""
        job = self.jobs[job_id]
        job.status = BulkOperationStatus.RUNNING
        job.started_at = datetime.now()
        job.errors = []
        
        migrated = []
        
        try:
            if not self.storage:
                raise ValueError("Storage is required")
            
            # Procesar en batches
            for i in range(0, len(session_ids), batch_size):
                batch = session_ids[i:i + batch_size]
                
                semaphore = asyncio.Semaphore(self.max_workers)
                
                async def migrate_one(session_id: str):
                    async with semaphore:
                        try:
                            session = await self.storage.load_session(session_id)
                            if session:
                                if transform:
                                    session = await transform(session, source_format, target_format)
                                migrated.append(session_id)
                                return session_id
                        except Exception as e:
                            logger.error(f"Error migrating session {session_id}: {e}")
                            job.failed_items += 1
                            job.errors.append(f"{session_id}: {str(e)}")
                            return None
                
                tasks = [migrate_one(sid) for sid in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                job.processed_items += sum(1 for r in results if r and not isinstance(r, Exception))
            
            job.status = BulkOperationStatus.COMPLETED
            job.completed_at = datetime.now()
            job.result = {"migrated": migrated, "source": source_format, "target": target_format}
        except Exception as e:
            job.status = BulkOperationStatus.FAILED
            job.errors = [str(e)]
            logger.error(f"Error in migration job {job_id}: {e}")
    
    async def get_migration_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Obtener estado de migración."""
        job = self.jobs.get(job_id)
        if not job:
            return None
        
        progress = (job.processed_items / job.total_items * 100) if job.total_items > 0 else 0
        
        return {
            "job_id": job_id,
            "status": job.status.value,
            "progress": round(progress, 2),
            "processed": job.processed_items,
            "failed": job.failed_items,
            "total": job.total_items,
            "errors": job.errors or []
        }


class BulkMetrics:
    """Métricas y estadísticas de operaciones bulk."""
    
    def __init__(self):
        self.operation_stats: Dict[str, Dict[str, Any]] = {}
        self.job_history: List[Dict[str, Any]] = []
    
    def record_operation(
        self,
        operation: str,
        success: bool,
        processed: int,
        failed: int,
        duration: float
    ):
        """Registrar estadísticas de operación."""
        if operation not in self.operation_stats:
            self.operation_stats[operation] = {
                "total_operations": 0,
                "total_processed": 0,
                "total_failed": 0,
                "total_duration": 0.0,
                "success_count": 0,
                "failure_count": 0,
                "avg_duration": 0.0,
                "avg_processed": 0.0
            }
        
        stats = self.operation_stats[operation]
        stats["total_operations"] += 1
        stats["total_processed"] += processed
        stats["total_failed"] += failed
        stats["total_duration"] += duration
        
        if success:
            stats["success_count"] += 1
        else:
            stats["failure_count"] += 1
        
        stats["avg_duration"] = stats["total_duration"] / stats["total_operations"]
        stats["avg_processed"] = stats["total_processed"] / stats["total_operations"]
        
        self.job_history.append({
            "operation": operation,
            "success": success,
            "processed": processed,
            "failed": failed,
            "duration": duration,
            "timestamp": datetime.now().isoformat()
        })
        
        if len(self.job_history) > 1000:
            self.job_history = self.job_history[-1000:]
    
    def get_stats(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """Obtener estadísticas."""
        if operation:
            return self.operation_stats.get(operation, {})
        return self.operation_stats
    
    def get_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Obtener historial de operaciones."""
        return self.job_history[-limit:]
    
    def get_summary(self) -> Dict[str, Any]:
        """Obtener resumen de todas las operaciones."""
        total_ops = sum(s["total_operations"] for s in self.operation_stats.values())
        total_processed = sum(s["total_processed"] for s in self.operation_stats.values())
        total_failed = sum(s["total_failed"] for s in self.operation_stats.values())
        
        return {
            "total_operations": total_ops,
            "total_processed": total_processed,
            "total_failed": total_failed,
            "success_rate": (total_processed / (total_processed + total_failed) * 100) if (total_processed + total_failed) > 0 else 0,
            "operations": self.operation_stats
        }


class BulkScheduler:
    """Programador de operaciones bulk recurrentes."""
    
    def __init__(self):
        self.scheduled_jobs: Dict[str, Dict[str, Any]] = {}
        self.running_tasks: Dict[str, asyncio.Task] = {}
    
    async def schedule_recurring(
        self,
        job_id: str,
        operation: Callable,
        schedule: str,  # cron-like: "0 2 * * *"
        config: Optional[Dict[str, Any]] = None,
        enabled: bool = True
    ):
        """Programar operación recurrente."""
        self.scheduled_jobs[job_id] = {
            "operation": operation,
            "schedule": schedule,
            "config": config or {},
            "enabled": enabled,
            "last_run": None,
            "next_run": None,
            "run_count": 0
        }
        
        if enabled:
            await self._start_scheduler(job_id)
    
    async def _start_scheduler(self, job_id: str):
        """Iniciar scheduler para un job."""
        if job_id in self.running_tasks:
            return
        
        async def run_scheduled():
            while True:
                try:
                    job = self.scheduled_jobs.get(job_id)
                    if not job or not job["enabled"]:
                        break
                    
                    await asyncio.sleep(60)
                    
                    if self._should_run(job):
                        await job["operation"](**job["config"])
                        job["last_run"] = datetime.now()
                        job["run_count"] += 1
                        job["next_run"] = self._calculate_next_run(job["schedule"])
                
                except Exception as e:
                    logger.error(f"Error in scheduled job {job_id}: {e}")
                    await asyncio.sleep(60)
        
        task = asyncio.create_task(run_scheduled())
        self.running_tasks[job_id] = task
    
    def _should_run(self, job: Dict[str, Any]) -> bool:
        """Verificar si un job debe ejecutarse."""
        return True
    
    def _calculate_next_run(self, schedule: str) -> datetime:
        """Calcular próximo tiempo de ejecución."""
        return datetime.now() + timedelta(hours=24)
    
    async def disable_job(self, job_id: str):
        """Deshabilitar job programado."""
        if job_id in self.scheduled_jobs:
            self.scheduled_jobs[job_id]["enabled"] = False
        
        if job_id in self.running_tasks:
            self.running_tasks[job_id].cancel()
            del self.running_tasks[job_id]
    
    async def enable_job(self, job_id: str):
        """Habilitar job programado."""
        if job_id in self.scheduled_jobs:
            self.scheduled_jobs[job_id]["enabled"] = True
            await self._start_scheduler(job_id)
    
    def list_jobs(self) -> List[Dict[str, Any]]:
        """Listar jobs programados."""
        return [
            {
                "job_id": jid,
                "schedule": job["schedule"],
                "enabled": job["enabled"],
                "last_run": job["last_run"].isoformat() if job["last_run"] else None,
                "next_run": job["next_run"].isoformat() if job["next_run"] else None,
                "run_count": job["run_count"]
            }
            for jid, job in self.scheduled_jobs.items()
        ]


class BulkRateLimiter:
    """Rate limiter para operaciones bulk."""
    
    def __init__(
        self,
        max_operations_per_minute: int = 100,
        max_operations_per_hour: int = 1000
    ):
        self.max_per_minute = max_operations_per_minute
        self.max_per_hour = max_operations_per_hour
        self.operation_times: Dict[str, List[datetime]] = {}
    
    async def check_rate_limit(
        self,
        operation: str,
        user_id: Optional[str] = None
    ) -> Tuple[bool, Optional[str]]:
        """Verificar rate limit."""
        key = f"{operation}:{user_id}" if user_id else operation
        now = datetime.now()
        
        if key not in self.operation_times:
            self.operation_times[key] = []
        
        minute_ago = now - timedelta(minutes=1)
        hour_ago = now - timedelta(hours=1)
        
        self.operation_times[key] = [
            t for t in self.operation_times[key]
            if t > hour_ago
        ]
        
        recent_minute = [t for t in self.operation_times[key] if t > minute_ago]
        recent_hour = self.operation_times[key]
        
        if len(recent_minute) >= self.max_per_minute:
            return False, f"Rate limit exceeded: {self.max_per_minute} operations per minute"
        
        if len(recent_hour) >= self.max_per_hour:
            return False, f"Rate limit exceeded: {self.max_per_hour} operations per hour"
        
        self.operation_times[key].append(now)
        
        return True, None
    
    def get_stats(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """Obtener estadísticas de rate limiting."""
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        hour_ago = now - timedelta(hours=1)
        
        stats = {}
        for key, times in self.operation_times.items():
            if operation and not key.startswith(operation):
                continue
            
            recent_minute = len([t for t in times if t > minute_ago])
            recent_hour = len([t for t in times if t > hour_ago])
            
            stats[key] = {
                "last_minute": recent_minute,
                "last_hour": recent_hour,
                "limit_per_minute": self.max_per_minute,
                "limit_per_hour": self.max_per_hour
            }
        
        return stats


class BulkValidator:
    """Validador de operaciones bulk."""
    
    def __init__(self):
        self.validation_rules: Dict[str, List[Callable]] = {}
    
    def register_rule(self, operation: str, rule: Callable):
        """Registrar regla de validación para una operación."""
        if operation not in self.validation_rules:
            self.validation_rules[operation] = []
        self.validation_rules[operation].append(rule)
    
    async def validate_operation(
        self,
        operation: str,
        data: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """Validar datos antes de ejecutar operación."""
        if operation not in self.validation_rules:
            return True, None
        
        for rule in self.validation_rules[operation]:
            try:
                result = await rule(data) if asyncio.iscoroutinefunction(rule) else rule(data)
                if not result:
                    return False, f"Validation failed for operation {operation}"
                if isinstance(result, tuple):
                    valid, error = result
                    if not valid:
                        return False, error
            except Exception as e:
                logger.error(f"Error in validation rule: {e}")
                return False, f"Validation error: {str(e)}"
        
        return True, None
    
    def validate_session_ids(self, session_ids: List[str]) -> Tuple[bool, Optional[str]]:
        """Validar lista de session IDs."""
        if not session_ids:
            return False, "Session IDs list cannot be empty"
        
        if len(session_ids) > 10000:
            return False, "Maximum 10000 sessions allowed per operation"
        
        if not all(isinstance(sid, str) and len(sid) > 0 for sid in session_ids):
            return False, "All session IDs must be non-empty strings"
        
        return True, None
    
    def validate_count(self, count: int, max_count: int = 10000) -> Tuple[bool, Optional[str]]:
        """Validar count para operaciones de creación."""
        if count <= 0:
            return False, "Count must be greater than 0"
        
        if count > max_count:
            return False, f"Maximum {max_count} items allowed per operation"
        
        return True, None


class BulkWebhooks:
    """Sistema de webhooks para notificaciones de progreso bulk."""
    
    def __init__(self, webhook_manager: Optional[Any] = None):
        self.webhook_manager = webhook_manager
        self.job_subscriptions: Dict[str, List[str]] = {}  # job_id -> [webhook_urls]
    
    def subscribe_to_job(self, job_id: str, webhook_url: str):
        """Suscribir webhook a un job."""
        if job_id not in self.job_subscriptions:
            self.job_subscriptions[job_id] = []
        if webhook_url not in self.job_subscriptions[job_id]:
            self.job_subscriptions[job_id].append(webhook_url)
    
    async def notify_progress(
        self,
        job_id: str,
        progress: float,
        status: str,
        data: Optional[Dict[str, Any]] = None
    ):
        """Notificar progreso a webhooks suscritos."""
        if job_id not in self.job_subscriptions:
            return
        
        payload = {
            "job_id": job_id,
            "progress": progress,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "data": data or {}
        }
        
        if self.webhook_manager:
            for webhook_url in self.job_subscriptions[job_id]:
                try:
                    await self.webhook_manager.send_webhook(
                        url=webhook_url,
                        payload=payload
                    )
                except Exception as e:
                    logger.error(f"Error sending webhook to {webhook_url}: {e}")


class BulkGrouping:
    """Agrupación y filtrado de operaciones bulk."""
    
    @staticmethod
    def group_by_user(sessions: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Agrupar sesiones por usuario."""
        grouped = {}
        for session in sessions:
            user_id = session.get("user_id", "unknown")
            session_id = session.get("session_id")
            if session_id:
                if user_id not in grouped:
                    grouped[user_id] = []
                grouped[user_id].append(session_id)
        return grouped
    
    @staticmethod
    def group_by_date_range(
        sessions: List[Dict[str, Any]],
        days: int = 7
    ) -> Dict[str, List[str]]:
        """Agrupar sesiones por rango de fechas."""
        cutoff_date = datetime.now() - timedelta(days=days)
        old_sessions = []
        new_sessions = []
        
        for session in sessions:
            created_at = session.get("created_at")
            session_id = session.get("session_id")
            
            if session_id and created_at:
                try:
                    if isinstance(created_at, str):
                        created_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    else:
                        created_date = created_at
                    
                    if created_date < cutoff_date:
                        old_sessions.append(session_id)
                    else:
                        new_sessions.append(session_id)
                except Exception:
                    new_sessions.append(session_id)
            elif session_id:
                new_sessions.append(session_id)
        
        return {
            f"older_than_{days}_days": old_sessions,
            f"newer_than_{days}_days": new_sessions
        }
    
    @staticmethod
    def filter_by_state(
        sessions: List[Dict[str, Any]],
        state: str
    ) -> List[str]:
        """Filtrar sesiones por estado."""
        filtered = []
        for session in sessions:
            session_state = session.get("state", "").lower()
            session_id = session.get("session_id")
            if session_id and session_state == state.lower():
                filtered.append(session_id)
        return filtered
    
    @staticmethod
    def filter_by_message_count(
        sessions: List[Dict[str, Any]],
        min_count: Optional[int] = None,
        max_count: Optional[int] = None
    ) -> List[str]:
        """Filtrar sesiones por número de mensajes."""
        filtered = []
        for session in sessions:
            message_count = session.get("message_count", 0)
            session_id = session.get("session_id")
            
            if session_id:
                if min_count is not None and message_count < min_count:
                    continue
                if max_count is not None and message_count > max_count:
                    continue
                filtered.append(session_id)
        
        return filtered


class BulkRetry:
    """Sistema de reintentos para operaciones bulk fallidas."""
    
    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        exponential_backoff: bool = True
    ):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.exponential_backoff = exponential_backoff
    
    async def execute_with_retry(
        self,
        operation: Callable,
        *args,
        **kwargs
    ) -> Tuple[Any, bool, Optional[str]]:
        """Ejecutar operación con reintentos."""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                if asyncio.iscoroutinefunction(operation):
                    result = await operation(*args, **kwargs)
                else:
                    result = operation(*args, **kwargs)
                return result, True, None
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay
                    if self.exponential_backoff:
                        delay = self.retry_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
        
        return None, False, last_error


class BulkBatchProcessor:
    """Procesador de batches con control avanzado."""
    
    def __init__(
        self,
        batch_size: int = 100,
        max_workers: int = 10,
        retry_config: Optional[Dict[str, Any]] = None
    ):
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.retry = BulkRetry(**(retry_config or {}))
    
    async def process_in_batches(
        self,
        items: List[Any],
        operation: Callable,
        progress_callback: Optional[Callable] = None,
        **kwargs
    ) -> BulkOperationResult:
        """Procesar items en batches con control de progreso."""
        start_time = datetime.now()
        total_items = len(items)
        processed = 0
        failed = 0
        errors = []
        
        # Dividir en batches
        batches = [
            items[i:i + self.batch_size]
            for i in range(0, total_items, self.batch_size)
        ]
        
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def process_batch(batch: List[Any], batch_num: int):
            nonlocal processed, failed
            
            async with semaphore:
                for item in batch:
                    result, success, error = await self.retry.execute_with_retry(
                        operation,
                        item,
                        **kwargs
                    )
                    
                    if success:
                        processed += 1
                    else:
                        failed += 1
                        errors.append(f"Item {item}: {error}")
                    
                    # Callback de progreso
                    if progress_callback:
                        progress = ((processed + failed) / total_items) * 100
                        await progress_callback(progress, processed, failed)
        
        # Procesar batches en paralelo
        tasks = [process_batch(batch, i) for i, batch in enumerate(batches)]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        duration = (datetime.now() - start_time).total_seconds()
        return BulkOperationResult(
            success=failed == 0,
            processed=processed,
            failed=failed,
            total=total_items,
            errors=errors[:100],  # Limitar errores
            duration=duration
        )


class BulkPerformanceOptimizer:
    """Optimizador de performance para operaciones bulk."""
    
    def __init__(self):
        self.performance_stats: Dict[str, List[float]] = {}
    
    def record_operation_time(self, operation: str, duration: float):
        """Registrar tiempo de operación."""
        if operation not in self.performance_stats:
            self.performance_stats[operation] = []
        
        self.performance_stats[operation].append(duration)
        
        # Mantener solo últimos 100 registros
        if len(self.performance_stats[operation]) > 100:
            self.performance_stats[operation] = self.performance_stats[operation][-100:]
    
    def get_optimal_batch_size(self, operation: str) -> int:
        """Calcular tamaño óptimo de batch basado en estadísticas."""
        if operation not in self.performance_stats:
            return 100  # Default
        
        times = self.performance_stats[operation]
        if not times:
            return 100
        
        avg_time = sum(times) / len(times)
        
        # Si las operaciones son rápidas (< 0.1s), aumentar batch size
        if avg_time < 0.1:
            return 200
        # Si son lentas (> 1s), reducir batch size
        elif avg_time > 1.0:
            return 50
        else:
            return 100
    
    def get_optimal_workers(self, operation: str) -> int:
        """Calcular número óptimo de workers."""
        if operation not in self.performance_stats:
            return 10  # Default
        
        times = self.performance_stats[operation]
        if not times:
            return 10
        
        avg_time = sum(times) / len(times)
        
        # Si las operaciones son rápidas, usar más workers
        if avg_time < 0.1:
            return 20
        # Si son lentas, usar menos workers
        elif avg_time > 1.0:
            return 5
        else:
            return 10
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de performance."""
        stats = {}
        for operation, times in self.performance_stats.items():
            if times:
                stats[operation] = {
                    "avg_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times),
                    "total_operations": len(times),
                    "optimal_batch_size": self.get_optimal_batch_size(operation),
                    "optimal_workers": self.get_optimal_workers(operation)
                }
        return stats


class BulkQueue:
    """Cola de trabajos bulk con prioridades."""
    
    def __init__(self):
        self.queue: List[Dict[str, Any]] = []
        self.processing: Dict[str, Dict[str, Any]] = {}
        self.completed: Dict[str, Dict[str, Any]] = {}
        self.failed: Dict[str, Dict[str, Any]] = {}
    
    def enqueue(
        self,
        job_id: str,
        operation: str,
        data: Dict[str, Any],
        priority: int = 5  # 1-10, 10 es más alta
    ):
        """Agregar trabajo a la cola."""
        job = {
            "job_id": job_id,
            "operation": operation,
            "data": data,
            "priority": priority,
            "created_at": datetime.now(),
            "status": "queued"
        }
        self.queue.append(job)
        # Ordenar por prioridad (mayor primero)
        self.queue.sort(key=lambda x: x["priority"], reverse=True)
    
    def dequeue(self) -> Optional[Dict[str, Any]]:
        """Obtener siguiente trabajo de la cola."""
        if not self.queue:
            return None
        
        job = self.queue.pop(0)
        job["status"] = "processing"
        job["started_at"] = datetime.now()
        self.processing[job["job_id"]] = job
        return job
    
    def complete(self, job_id: str, result: Optional[Dict[str, Any]] = None):
        """Marcar trabajo como completado."""
        if job_id in self.processing:
            job = self.processing.pop(job_id)
            job["status"] = "completed"
            job["completed_at"] = datetime.now()
            job["result"] = result
            self.completed[job_id] = job
    
    def fail(self, job_id: str, error: str):
        """Marcar trabajo como fallido."""
        if job_id in self.processing:
            job = self.processing.pop(job_id)
            job["status"] = "failed"
            job["completed_at"] = datetime.now()
            job["error"] = error
            self.failed[job_id] = job
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Obtener estado de la cola."""
        return {
            "queued": len(self.queue),
            "processing": len(self.processing),
            "completed": len(self.completed),
            "failed": len(self.failed),
            "total": len(self.queue) + len(self.processing) + len(self.completed) + len(self.failed)
        }


class BulkTransformation:
    """Transformaciones de datos para operaciones bulk."""
    
    @staticmethod
    def transform_session_format(
        sessions: List[Dict[str, Any]],
        target_format: str
    ) -> List[Dict[str, Any]]:
        """Transformar formato de sesiones."""
        if target_format == "v2":
            return [
                {
                    "id": s.get("session_id"),
                    "user": s.get("user_id"),
                    "state": s.get("state"),
                    "messages": s.get("messages", []),
                    "metadata": {
                        "created": s.get("created_at"),
                        "updated": s.get("updated_at")
                    }
                }
                for s in sessions
            ]
        return sessions
    
    @staticmethod
    def normalize_session_ids(
        session_ids: List[Union[str, int, Dict[str, Any]]]
    ) -> List[str]:
        """Normalizar session IDs a strings."""
        normalized = []
        for sid in session_ids:
            if isinstance(sid, str):
                normalized.append(sid)
            elif isinstance(sid, int):
                normalized.append(str(sid))
            elif isinstance(sid, dict):
                normalized.append(sid.get("session_id") or sid.get("id") or str(sid))
        return normalized
    
    @staticmethod
    def enrich_with_metadata(
        items: List[Dict[str, Any]],
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Enriquecer items con metadata."""
        return [
            {**item, "_metadata": metadata, "_enriched_at": datetime.now().isoformat()}
            for item in items
        ]


class BulkAggregation:
    """Agregación y análisis de resultados bulk."""
    
    @staticmethod
    def aggregate_by_operation(
        results: List[BulkOperationResult]
    ) -> Dict[str, Dict[str, Any]]:
        """Agregar resultados por tipo de operación."""
        aggregated = {}
        
        for result in results:
            # Necesitaría un campo "operation" en BulkOperationResult
            op_type = getattr(result, "operation", "unknown")
            if op_type not in aggregated:
                aggregated[op_type] = {
                    "total_operations": 0,
                    "total_processed": 0,
                    "total_failed": 0,
                    "total_duration": 0.0,
                    "success_count": 0,
                    "failure_count": 0
                }
            
            agg = aggregated[op_type]
            agg["total_operations"] += 1
            agg["total_processed"] += result.processed
            agg["total_failed"] += result.failed
            agg["total_duration"] += result.duration
            if result.success:
                agg["success_count"] += 1
            else:
                agg["failure_count"] += 1
        
        # Calcular promedios
        for op_type, agg in aggregated.items():
            if agg["total_operations"] > 0:
                agg["avg_duration"] = agg["total_duration"] / agg["total_operations"]
                agg["avg_processed"] = agg["total_processed"] / agg["total_operations"]
                agg["success_rate"] = (agg["success_count"] / agg["total_operations"]) * 100
        
        return aggregated
    
    @staticmethod
    def calculate_statistics(
        results: List[BulkOperationResult]
    ) -> Dict[str, Any]:
        """Calcular estadísticas generales."""
        if not results:
            return {}
        
        total_processed = sum(r.processed for r in results)
        total_failed = sum(r.failed for r in results)
        total_duration = sum(r.duration for r in results)
        success_count = sum(1 for r in results if r.success)
        
        durations = [r.duration for r in results]
        processed_counts = [r.processed for r in results]
        
        return {
            "total_operations": len(results),
            "total_processed": total_processed,
            "total_failed": total_failed,
            "total_duration": total_duration,
            "avg_duration": total_duration / len(results) if results else 0,
            "min_duration": min(durations) if durations else 0,
            "max_duration": max(durations) if durations else 0,
            "avg_processed": total_processed / len(results) if results else 0,
            "min_processed": min(processed_counts) if processed_counts else 0,
            "max_processed": max(processed_counts) if processed_counts else 0,
            "success_rate": (success_count / len(results) * 100) if results else 0,
            "throughput": total_processed / total_duration if total_duration > 0 else 0
        }


class BulkMonitoring:
    """Monitoreo avanzado de operaciones bulk."""
    
    def __init__(self):
        self.metrics: Dict[str, List[Dict[str, Any]]] = {}
        self.alerts: List[Dict[str, Any]] = []
    
    def record_metric(
        self,
        operation: str,
        metric_name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None
    ):
        """Registrar métrica."""
        if operation not in self.metrics:
            self.metrics[operation] = []
        
        self.metrics[operation].append({
            "metric": metric_name,
            "value": value,
            "tags": tags or {},
            "timestamp": datetime.now().isoformat()
        })
        
        # Mantener solo últimos 1000 registros por operación
        if len(self.metrics[operation]) > 1000:
            self.metrics[operation] = self.metrics[operation][-1000:]
    
    def check_threshold(
        self,
        operation: str,
        metric_name: str,
        threshold: float,
        operator: str = ">"  # >, <, >=, <=
    ) -> bool:
        """Verificar si métrica excede umbral."""
        if operation not in self.metrics:
            return False
        
        recent_metrics = [
            m for m in self.metrics[operation]
            if m["metric"] == metric_name
        ][-10:]  # Últimas 10 métricas
        
        if not recent_metrics:
            return False
        
        avg_value = sum(m["value"] for m in recent_metrics) / len(recent_metrics)
        
        if operator == ">":
            return avg_value > threshold
        elif operator == "<":
            return avg_value < threshold
        elif operator == ">=":
            return avg_value >= threshold
        elif operator == "<=":
            return avg_value <= threshold
        
        return False
    
    def create_alert(
        self,
        operation: str,
        metric_name: str,
        message: str,
        severity: str = "warning"
    ):
        """Crear alerta."""
        alert = {
            "operation": operation,
            "metric": metric_name,
            "message": message,
            "severity": severity,
            "timestamp": datetime.now().isoformat()
        }
        self.alerts.append(alert)
        
        # Mantener solo últimos 100 alertas
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
    
    def get_alerts(
        self,
        operation: Optional[str] = None,
        severity: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Obtener alertas."""
        filtered = self.alerts
        
        if operation:
            filtered = [a for a in filtered if a["operation"] == operation]
        
        if severity:
            filtered = [a for a in filtered if a["severity"] == severity]
        
        return filtered
    
    def get_metrics_summary(self, operation: str) -> Dict[str, Any]:
        """Obtener resumen de métricas."""
        if operation not in self.metrics:
            return {}
        
        metrics_by_name: Dict[str, List[float]] = {}
        
        for metric_record in self.metrics[operation]:
            name = metric_record["metric"]
            if name not in metrics_by_name:
                metrics_by_name[name] = []
            metrics_by_name[name].append(metric_record["value"])
        
        summary = {}
        for name, values in metrics_by_name.items():
            summary[name] = {
                "count": len(values),
                "avg": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "latest": values[-1] if values else None
            }
        
        return summary


class BulkThrottle:
    """Throttling inteligente para operaciones bulk."""
    
    def __init__(
        self,
        max_operations_per_second: float = 10.0,
        burst_size: int = 50
    ):
        self.max_ops_per_sec = max_operations_per_second
        self.burst_size = burst_size
        self.operation_times: List[datetime] = []
        self.tokens = burst_size
        self.last_refill = datetime.now()
    
    async def acquire(self) -> bool:
        """Adquirir token para operación."""
        now = datetime.now()
        
        # Refill tokens basado en tiempo transcurrido
        elapsed = (now - self.last_refill).total_seconds()
        tokens_to_add = elapsed * self.max_ops_per_sec
        
        if tokens_to_add > 0:
            self.tokens = min(self.burst_size, self.tokens + tokens_to_add)
            self.last_refill = now
        
        # Consumir token
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            self.operation_times.append(now)
            return True
        
        # Calcular tiempo de espera
        wait_time = (1.0 - self.tokens) / self.max_ops_per_sec
        if wait_time > 0:
            await asyncio.sleep(wait_time)
            self.tokens = 0.0
            self.operation_times.append(datetime.now())
            return True
        
        return False
    
    def get_current_rate(self) -> float:
        """Obtener tasa actual de operaciones por segundo."""
        now = datetime.now()
        one_second_ago = now - timedelta(seconds=1)
        
        recent_ops = [t for t in self.operation_times if t > one_second_ago]
        
        # Limpiar tiempos antiguos
        self.operation_times = [t for t in self.operation_times if t > one_second_ago]
        
        return len(recent_ops)


class BulkCircuitBreaker:
    """Circuit breaker para operaciones bulk."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 3
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        
        self.state = "closed"  # closed, open, half_open
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.half_open_calls = 0
    
    async def call(self, operation: Callable, *args, **kwargs) -> Tuple[Any, bool]:
        """Ejecutar operación a través del circuit breaker."""
        # Verificar estado
        if self.state == "open":
            if self.last_failure_time:
                elapsed = (datetime.now() - self.last_failure_time).total_seconds()
                if elapsed >= self.recovery_timeout:
                    self.state = "half_open"
                    self.half_open_calls = 0
                else:
                    return None, False
        
        # Ejecutar operación
        try:
            if asyncio.iscoroutinefunction(operation):
                result = await operation(*args, **kwargs)
            else:
                result = operation(*args, **kwargs)
            
            # Éxito
            if self.state == "half_open":
                self.half_open_calls += 1
                if self.half_open_calls >= self.half_open_max_calls:
                    self.state = "closed"
                    self.failure_count = 0
            
            self.failure_count = 0
            return result, True
        
        except Exception as e:
            # Falla
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if self.state == "half_open":
                self.state = "open"
            elif self.failure_count >= self.failure_threshold:
                self.state = "open"
            
            logger.error(f"Circuit breaker: Operation failed: {e}")
            return None, False
    
    def get_state(self) -> Dict[str, Any]:
        """Obtener estado del circuit breaker."""
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "half_open_calls": self.half_open_calls if self.state == "half_open" else 0
        }
    
    def reset(self):
        """Resetear circuit breaker."""
        self.state = "closed"
        self.failure_count = 0
        self.last_failure_time = None
        self.half_open_calls = 0


class BulkCache:
    """Sistema de cache para resultados de operaciones bulk."""
    
    def __init__(self, ttl: int = 3600, max_size: int = 1000):
        self.ttl = ttl
        self.max_size = max_size
        self.cache: Dict[str, Dict[str, Any]] = {}
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Obtener valor del cache."""
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        created_at = entry.get("created_at")
        
        if created_at:
            if isinstance(created_at, str):
                created = datetime.fromisoformat(created_at)
            else:
                created = created_at
            
            elapsed = (datetime.now() - created).total_seconds()
            if elapsed > self.ttl:
                del self.cache[key]
                return None
        
        return entry.get("value")
    
    def set(self, key: str, value: Dict[str, Any]):
        """Guardar valor en cache."""
        # Limpiar si excede tamaño máximo
        if len(self.cache) >= self.max_size:
            # Eliminar entrada más antigua
            oldest_key = min(
                self.cache.keys(),
                key=lambda k: self.cache[k].get("created_at", datetime.min)
            )
            del self.cache[oldest_key]
        
        self.cache[key] = {
            "value": value,
            "created_at": datetime.now().isoformat()
        }
    
    def invalidate(self, key: Optional[str] = None):
        """Invalidar cache."""
        if key:
            if key in self.cache:
                del self.cache[key]
        else:
            self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del cache."""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "ttl": self.ttl
        }


class BulkAudit:
    """Sistema de auditoría para operaciones bulk."""
    
    def __init__(self):
        self.audit_log: List[Dict[str, Any]] = []
    
    def log_operation(
        self,
        operation: str,
        user_id: Optional[str],
        details: Dict[str, Any],
        result: Optional[str] = None
    ):
        """Registrar operación en auditoría."""
        log_entry = {
            "operation": operation,
            "user_id": user_id,
            "details": details,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        
        self.audit_log.append(log_entry)
        
        # Mantener solo últimos 10000 registros
        if len(self.audit_log) > 10000:
            self.audit_log = self.audit_log[-10000:]
        
        logger.info(f"Audit: {operation} by {user_id}: {result}")
    
    def get_audit_log(
        self,
        operation: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Obtener log de auditoría."""
        filtered = self.audit_log
        
        if operation:
            filtered = [entry for entry in filtered if entry["operation"] == operation]
        
        if user_id:
            filtered = [entry for entry in filtered if entry["user_id"] == user_id]
        
        return filtered[-limit:]
    
    def get_audit_summary(self) -> Dict[str, Any]:
        """Obtener resumen de auditoría."""
        if not self.audit_log:
            return {}
        
        operations = {}
        users = {}
        
        for entry in self.audit_log:
            op = entry["operation"]
            user = entry.get("user_id", "unknown")
            
            if op not in operations:
                operations[op] = 0
            operations[op] += 1
            
            if user not in users:
                users[user] = 0
            users[user] += 1
        
        return {
            "total_operations": len(self.audit_log),
            "unique_operations": len(operations),
            "unique_users": len(users),
            "operations": operations,
            "top_users": dict(sorted(users.items(), key=lambda x: x[1], reverse=True)[:10])
        }


class BulkOrchestrator:
    """Orquestador de operaciones bulk complejas."""
    
    def __init__(
        self,
        chat_engine: Optional[ContinuousChatEngine] = None,
        storage: Optional[SessionStorage] = None
    ):
        self.chat_engine = chat_engine
        self.storage = storage
        self.validator = BulkValidator()
        self.metrics = BulkMetrics()
        self.audit = BulkAudit()
        self.performance_optimizer = BulkPerformanceOptimizer()
    
    async def execute_workflow(
        self,
        workflow: List[Dict[str, Any]],
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Ejecutar workflow de operaciones bulk.
        
        workflow: [
            {
                "operation": "create_sessions",
                "config": {"count": 100},
                "validate": True,
                "on_success": "next",
                "on_failure": "stop"
            },
            ...
        ]
        """
        results = []
        workflow_id = f"workflow_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        self.audit.log_operation(
            operation="bulk_workflow",
            user_id=user_id,
            details={"workflow_id": workflow_id, "steps": len(workflow)},
            result="started"
        )
        
        for i, step in enumerate(workflow):
            step_name = step.get("operation", f"step_{i}")
            config = step.get("config", {})
            
            try:
                # Validar si es requerido
                if step.get("validate", False):
                    valid, error = await self.validator.validate_operation(
                        step_name,
                        config
                    )
                    if not valid:
                        if step.get("on_failure") == "stop":
                            break
                        continue
                
                # Ejecutar operación
                start_time = datetime.now()
                result = await self._execute_step(step_name, config)
                duration = (datetime.now() - start_time).total_seconds()
                
                # Registrar métricas
                self.metrics.record_operation(
                    operation=step_name,
                    success=result.get("success", False),
                    processed=result.get("processed", 0),
                    failed=result.get("failed", 0),
                    duration=duration
                )
                
                # Registrar performance
                self.performance_optimizer.record_operation_time(
                    step_name,
                    duration
                )
                
                results.append({
                    "step": i,
                    "operation": step_name,
                    "result": result,
                    "duration": duration
                })
                
                # Manejar resultado
                if not result.get("success") and step.get("on_failure") == "stop":
                    break
                
            except Exception as e:
                logger.error(f"Error in workflow step {i}: {e}")
                if step.get("on_failure") == "stop":
                    break
                results.append({
                    "step": i,
                    "operation": step_name,
                    "error": str(e),
                    "success": False
                })
        
        self.audit.log_operation(
            operation="bulk_workflow",
            user_id=user_id,
            details={"workflow_id": workflow_id},
            result="completed"
        )
        
        return {
            "workflow_id": workflow_id,
            "steps": len(workflow),
            "completed": len(results),
            "results": results
        }
    
    async def _execute_step(self, operation: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecutar un paso del workflow."""
        # Esta es una implementación simplificada
        # En producción, usaría un registry de operaciones
        return {"success": False, "message": f"Operation {operation} not implemented"}


class BulkHealthChecker:
    """Health checker para operaciones bulk."""
    
    def __init__(self):
        self.checks: Dict[str, Callable] = {}
    
    def register_check(self, name: str, check: Callable):
        """Registrar check de salud."""
        self.checks[name] = check
    
    async def check_all(self) -> Dict[str, Any]:
        """Ejecutar todos los checks."""
        results = {}
        overall_healthy = True
        
        for name, check in self.checks.items():
            try:
                if asyncio.iscoroutinefunction(check):
                    result = await check()
                else:
                    result = check()
                
                healthy = result if isinstance(result, bool) else result.get("healthy", False)
                results[name] = {
                    "healthy": healthy,
                    "details": result if not isinstance(result, bool) else {}
                }
                
                if not healthy:
                    overall_healthy = False
            except Exception as e:
                results[name] = {
                    "healthy": False,
                    "error": str(e)
                }
                overall_healthy = False
        
        return {
            "healthy": overall_healthy,
            "checks": results,
            "timestamp": datetime.now().isoformat()
        }


class BulkErrorHandler:
    """Manejador avanzado de errores para operaciones bulk."""
    
    def __init__(self):
        self.error_handlers: Dict[str, List[Callable]] = {}
        self.error_history: List[Dict[str, Any]] = []
    
    def register_handler(self, error_type: str, handler: Callable):
        """Registrar handler para tipo de error."""
        if error_type not in self.error_handlers:
            self.error_handlers[error_type] = []
        self.error_handlers[error_type].append(handler)
    
    async def handle_error(
        self,
        error: Exception,
        context: Dict[str, Any]
    ) -> Optional[Any]:
        """Manejar error con handlers registrados."""
        error_type = type(error).__name__
        error_message = str(error)
        
        # Registrar en historial
        self.error_history.append({
            "error_type": error_type,
            "error_message": error_message,
            "context": context,
            "timestamp": datetime.now().isoformat()
        })
        
        # Mantener solo últimos 1000 errores
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-1000:]
        
        # Ejecutar handlers
        handlers = self.error_handlers.get(error_type, [])
        handlers.extend(self.error_handlers.get("all", []))
        
        results = []
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(error, context)
                else:
                    result = handler(error, context)
                results.append(result)
            except Exception as e:
                logger.error(f"Error in error handler: {e}")
        
        return results[0] if results else None
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de errores."""
        if not self.error_history:
            return {}
        
        error_counts = {}
        for entry in self.error_history:
            error_type = entry["error_type"]
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        return {
            "total_errors": len(self.error_history),
            "unique_error_types": len(error_counts),
            "error_counts": error_counts,
            "recent_errors": self.error_history[-10:]
        }


class BulkConfig:
    """Configuración centralizada para operaciones bulk."""
    
    def __init__(self):
        self.config: Dict[str, Any] = {
            "max_workers": 10,
            "batch_size": 100,
            "max_retries": 3,
            "retry_delay": 1.0,
            "timeout": 300.0,
            "max_items_per_operation": 10000,
            "enable_metrics": True,
            "enable_audit": True,
            "enable_cache": True,
            "cache_ttl": 3600,
            "rate_limit_per_minute": 100,
            "rate_limit_per_hour": 1000
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Obtener valor de configuración."""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        """Establecer valor de configuración."""
        self.config[key] = value
    
    def update(self, updates: Dict[str, Any]):
        """Actualizar múltiples valores."""
        self.config.update(updates)
    
    def validate(self) -> Tuple[bool, Optional[str]]:
        """Validar configuración."""
        if self.config["max_workers"] <= 0:
            return False, "max_workers must be > 0"
        if self.config["batch_size"] <= 0:
            return False, "batch_size must be > 0"
        if self.config["max_items_per_operation"] <= 0:
            return False, "max_items_per_operation must be > 0"
        return True, None
    
    def get_all(self) -> Dict[str, Any]:
        """Obtener toda la configuración."""
        return self.config.copy()


class BulkFactory:
    """Factory para crear instancias de clases bulk."""
    
    @staticmethod
    def create_session_operations(
        chat_engine: Optional[ContinuousChatEngine] = None,
        storage: Optional[SessionStorage] = None,
        config: Optional[BulkConfig] = None
    ) -> BulkSessionOperations:
        """Crear instancia de BulkSessionOperations."""
        cfg = config or BulkConfig()
        return BulkSessionOperations(
            chat_engine=chat_engine,
            storage=storage,
            max_workers=cfg.get("max_workers", 10),
            batch_size=cfg.get("batch_size", 100)
        )
    
    @staticmethod
    def create_full_stack(
        chat_engine: Optional[ContinuousChatEngine] = None,
        storage: Optional[SessionStorage] = None,
        config: Optional[BulkConfig] = None
    ) -> Dict[str, Any]:
        """Crear stack completo de operaciones bulk."""
        cfg = config or BulkConfig()
        
        return {
            "config": cfg,
            "sessions": BulkFactory.create_session_operations(chat_engine, storage, cfg),
            "messages": BulkMessageOperations(
                chat_engine=chat_engine,
                max_workers=cfg.get("max_workers", 10)
            ),
            "exporter": BulkExporter(
                storage=storage,
                max_workers=cfg.get("max_workers", 10)
            ),
            "analytics": BulkAnalytics(
                storage=storage,
                max_workers=cfg.get("max_workers", 10)
            ),
            "cleanup": BulkCleanup(
                storage=storage,
                chat_engine=chat_engine,
                max_workers=cfg.get("max_workers", 10)
            ),
            "processor": BulkProcessor(
                max_workers=cfg.get("max_workers", 10)
            ),
            "importer": BulkImporter(
                chat_engine=chat_engine,
                storage=storage,
                max_workers=cfg.get("max_workers", 10)
            ),
            "validator": BulkValidator(),
            "metrics": BulkMetrics(),
            "performance_optimizer": BulkPerformanceOptimizer(),
            "queue": BulkQueue(),
            "cache": BulkCache(
                ttl=cfg.get("cache_ttl", 3600),
                max_size=1000
            ),
            "rate_limiter": BulkRateLimiter(
                max_operations_per_minute=cfg.get("rate_limit_per_minute", 100),
                max_operations_per_hour=cfg.get("rate_limit_per_hour", 1000)
            ),
            "audit": BulkAudit(),
            "monitoring": BulkMonitoring(),
            "orchestrator": BulkOrchestrator(
                chat_engine=chat_engine,
                storage=storage
            ),
            "health_checker": BulkHealthChecker(),
            "error_handler": BulkErrorHandler()
        }


# Exportar todas las clases y funciones helper
__all__ = [
    # Helper functions
    "batch_process",
    "calculate_optimal_batch_size",
    "chunk_list",
    "retry_operation",
    "merge_bulk_results",
    # Enums y Dataclasses
    "BulkOperationStatus",
    "BulkOperationResult",
    "BulkJob",
    "BulkSessionOperations",
    "BulkMessageOperations",
    "BulkExporter",
    "BulkAnalytics",
    "BulkCleanup",
    "BulkProcessor",
    "BulkImporter",
    "BulkNotifications",
    "BulkSearch",
    "BulkBackupRestore",
    "BulkMigration",
    "BulkMetrics",
    "BulkScheduler",
    "BulkRateLimiter",
    "BulkValidator",
    "BulkWebhooks",
    "BulkGrouping",
    "BulkRetry",
    "BulkBatchProcessor",
    "BulkPerformanceOptimizer",
    "BulkQueue",
    "BulkTransformation",
    "BulkAggregation",
    "BulkMonitoring",
    "BulkThrottle",
    "BulkCircuitBreaker",
    "BulkCache",
    "BulkAudit",
    "BulkOrchestrator",
    "BulkHealthChecker",
    "BulkErrorHandler",
    "BulkConfig",
    "BulkFactory",
    "BulkTesting",
    "BulkSecurity",
    "BulkCompression",
    "BulkStreaming",
    "BulkAsyncQueue",
    "BulkLock",
    "BulkProgressTracker",
    "BulkResourceManager",
    "BulkAutoCreator",
    "BulkAutoExpander",
    "BulkAutoProcessor",
    "BulkAutoMaintainer",
    "BulkInfiniteGenerator",
    "BulkSelfSustaining",
    # Decoradores
    "bulk_metrics_decorator",
    "bulk_cache_decorator",
    "bulk_retry_decorator",
    "bulk_rate_limit_decorator",
    # Clases avanzadas de mejoras
    "BulkLoadBalancer",
    "BulkFailoverManager",
    "BulkAdvancedMetrics",
    "BulkSynchronizer",
    "BulkReplicator",
    "BulkAdvancedAnalytics",
    "BulkOptimizationEngine",
    "BulkEventBus",
    "BulkDataValidator",
    "BulkDataTransformer",
    "BulkRouter",
    "BulkCompressionAdvanced",
    "BulkSecurityAdvanced",
    # Nuevas clases avanzadas
    "BulkRealTimeMetrics",
    "BulkAdvancedCache",
    "BulkPriorityQueue",
    "BulkEnhancedValidator",
    "BulkDashboard",
    # Decoradores y utilidades
    "track_metrics",
    "cache_result",
    "validate_input",
    "BulkBenchmark",
    "BulkAutoTuner",
    # Sistemas avanzados
    "BulkAdaptiveRateLimiter",
    "BulkLoadBalancer",
    "BulkLoadPredictor",
    "BulkAutoScaler",
    # Sistemas de observabilidad y optimización
    "BulkEventSourcing",
    "BulkObservability",
    "BulkCostOptimizer",
    "BulkAnomalyDetector"
]


# BulkTesting ya está definida arriba (línea 1172), esta clase duplicada fue eliminada


class BulkSecurity:
    """Sistema de seguridad para operaciones bulk."""
    
    def __init__(self):
        self.permissions: Dict[str, List[str]] = {}  # user_id -> [operations]
        self.blocked_operations: List[str] = []
    
    def check_permission(
        self,
        user_id: str,
        operation: str
    ) -> Tuple[bool, Optional[str]]:
        """Verificar permiso para operación."""
        if operation in self.blocked_operations:
            return False, f"Operation {operation} is blocked"
        
        if user_id in self.permissions:
            allowed_ops = self.permissions[user_id]
            if "*" in allowed_ops or operation in allowed_ops:
                return True, None
        
        return False, f"User {user_id} not authorized for {operation}"
    
    def grant_permission(self, user_id: str, operation: str):
        """Otorgar permiso."""
        if user_id not in self.permissions:
            self.permissions[user_id] = []
        if operation not in self.permissions[user_id]:
            self.permissions[user_id].append(operation)
    
    def revoke_permission(self, user_id: str, operation: str):
        """Revocar permiso."""
        if user_id in self.permissions:
            if operation in self.permissions[user_id]:
                self.permissions[user_id].remove(operation)
    
    def block_operation(self, operation: str):
        """Bloquear operación globalmente."""
        if operation not in self.blocked_operations:
            self.blocked_operations.append(operation)
    
    def sanitize_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitizar input de datos."""
        sanitized = {}
        for key, value in data.items():
            if isinstance(value, str):
                # Remover caracteres peligrosos
                sanitized[key] = value.replace("<script>", "").replace("</script>", "")
            elif isinstance(value, dict):
                sanitized[key] = self.sanitize_input(value)
            elif isinstance(value, list):
                sanitized[key] = [
                    self.sanitize_input(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                sanitized[key] = value
        return sanitized


class BulkCompression:
    """Sistema de compresión para datos bulk."""
    
    @staticmethod
    def compress_data(data: Dict[str, Any], method: str = "gzip") -> bytes:
        """Comprimir datos."""
        import gzip
        import json
        
        json_data = json.dumps(data).encode('utf-8')
        
        if method == "gzip":
            return gzip.compress(json_data)
        else:
            return json_data
    
    @staticmethod
    def decompress_data(compressed_data: bytes, method: str = "gzip") -> Dict[str, Any]:
        """Descomprimir datos."""
        import gzip
        import json
        
        if method == "gzip":
            json_data = gzip.decompress(compressed_data)
        else:
            json_data = compressed_data
        
        return json.loads(json_data.decode('utf-8'))
    
    @staticmethod
    def compress_sessions(
        sessions: List[Dict[str, Any]],
        method: str = "gzip"
    ) -> bytes:
        """Comprimir múltiples sesiones."""
        return BulkCompression.compress_data({"sessions": sessions}, method)
    
    @staticmethod
    def get_compression_ratio(original: bytes, compressed: bytes) -> float:
        """Calcular ratio de compresión."""
        if len(original) == 0:
            return 0.0
        return (1 - len(compressed) / len(original)) * 100


class BulkStreaming:
    """Sistema de streaming para resultados bulk."""
    
    def __init__(self):
        self.streams: Dict[str, asyncio.Queue] = {}
    
    def create_stream(self, stream_id: str, max_size: int = 1000):
        """Crear stream."""
        self.streams[stream_id] = asyncio.Queue(maxsize=max_size)
    
    async def push_to_stream(
        self,
        stream_id: str,
        data: Dict[str, Any]
    ) -> bool:
        """Enviar datos al stream."""
        if stream_id not in self.streams:
            return False
        
        try:
            await self.streams[stream_id].put(data)
            return True
        except Exception as e:
            logger.error(f"Error pushing to stream {stream_id}: {e}")
            return False
    
    async def read_from_stream(
        self,
        stream_id: str,
        timeout: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """Leer datos del stream."""
        if stream_id not in self.streams:
            return None
        
        try:
            if timeout:
                return await asyncio.wait_for(
                    self.streams[stream_id].get(),
                    timeout=timeout
                )
            else:
                return await self.streams[stream_id].get()
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            logger.error(f"Error reading from stream {stream_id}: {e}")
            return None
    
    async def stream_generator(
        self,
        stream_id: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Generador asíncrono para stream."""
        while True:
            data = await self.read_from_stream(stream_id)
            if data is None:
                break
            yield data
    
    def close_stream(self, stream_id: str):
        """Cerrar stream."""
        if stream_id in self.streams:
            del self.streams[stream_id]


class BulkAsyncQueue:
    """Cola asíncrona avanzada para operaciones bulk."""
    
    def __init__(self, maxsize: int = 1000):
        self.queue = asyncio.Queue(maxsize=maxsize)
        self.priority_queue: List[Tuple[int, Any]] = []  # (priority, item)
    
    async def enqueue(
        self,
        item: Any,
        priority: int = 5
    ):
        """Agregar item a la cola con prioridad."""
        self.priority_queue.append((priority, item))
        self.priority_queue.sort(key=lambda x: x[0], reverse=True)
        
        # Mover a cola principal
        if self.priority_queue:
            _, top_item = self.priority_queue.pop(0)
            await self.queue.put(top_item)
    
    async def dequeue(self) -> Any:
        """Obtener siguiente item de la cola."""
        return await self.queue.get()
    
    def qsize(self) -> int:
        """Tamaño de la cola."""
        return self.queue.qsize()
    
    def empty(self) -> bool:
        """Verificar si la cola está vacía."""
        return self.queue.empty()
    
    async def drain(self) -> List[Any]:
        """Vaciar cola y retornar todos los items."""
        items = []
        while not self.queue.empty():
            items.append(await self.queue.get())
        return items


class BulkLock:
    """Sistema de locks para operaciones concurrentes."""
    
    def __init__(self):
        self.locks: Dict[str, asyncio.Lock] = {}
        self.lock_holders: Dict[str, str] = {}  # resource -> holder
    
    async def acquire(
        self,
        resource: str,
        holder: str,
        timeout: Optional[float] = None
    ) -> bool:
        """Adquirir lock para recurso."""
        if resource not in self.locks:
            self.locks[resource] = asyncio.Lock()
        
        try:
            if timeout:
                acquired = await asyncio.wait_for(
                    self.locks[resource].acquire(),
                    timeout=timeout
                )
            else:
                await self.locks[resource].acquire()
                acquired = True
            
            if acquired:
                self.lock_holders[resource] = holder
            return acquired
        except asyncio.TimeoutError:
            return False
    
    async def release(self, resource: str, holder: str):
        """Liberar lock."""
        if resource in self.locks:
            if self.lock_holders.get(resource) == holder:
                self.locks[resource].release()
                del self.lock_holders[resource]
    
    async def __aenter__(self):
        """Context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # Cleanup si es necesario
        pass


class BulkProgressTracker:
    """Tracker avanzado de progreso para operaciones bulk."""
    
    def __init__(self):
        self.trackers: Dict[str, Dict[str, Any]] = {}
    
    def create_tracker(
        self,
        tracker_id: str,
        total: int,
        description: str = ""
    ):
        """Crear tracker de progreso."""
        self.trackers[tracker_id] = {
            "total": total,
            "processed": 0,
            "failed": 0,
            "description": description,
            "started_at": datetime.now(),
            "checkpoints": []
        }
    
    def update(
        self,
        tracker_id: str,
        processed: int = 1,
        failed: int = 0
    ):
        """Actualizar progreso."""
        if tracker_id not in self.trackers:
            return
        
        tracker = self.trackers[tracker_id]
        tracker["processed"] += processed
        tracker["failed"] += failed
    
    def checkpoint(
        self,
        tracker_id: str,
        message: str = ""
    ):
        """Crear checkpoint."""
        if tracker_id not in self.trackers:
            return
        
        tracker = self.trackers[tracker_id]
        progress = (tracker["processed"] / tracker["total"] * 100) if tracker["total"] > 0 else 0
        
        tracker["checkpoints"].append({
            "progress": progress,
            "processed": tracker["processed"],
            "failed": tracker["failed"],
            "message": message,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_progress(self, tracker_id: str) -> Optional[Dict[str, Any]]:
        """Obtener progreso."""
        if tracker_id not in self.trackers:
            return None
        
        tracker = self.trackers[tracker_id]
        progress = (tracker["processed"] / tracker["total"] * 100) if tracker["total"] > 0 else 0
        
        elapsed = (datetime.now() - tracker["started_at"]).total_seconds()
        remaining = tracker["total"] - tracker["processed"]
        estimated_time = (elapsed / tracker["processed"] * remaining) if tracker["processed"] > 0 else 0
        
        return {
            "tracker_id": tracker_id,
            "description": tracker["description"],
            "progress": round(progress, 2),
            "processed": tracker["processed"],
            "failed": tracker["failed"],
            "total": tracker["total"],
            "elapsed": round(elapsed, 2),
            "estimated_remaining": round(estimated_time, 2),
            "checkpoints": tracker["checkpoints"]
        }
    
    def complete(self, tracker_id: str):
        """Completar tracker."""
        if tracker_id in self.trackers:
            tracker = self.trackers[tracker_id]
            tracker["completed_at"] = datetime.now()
            duration = (tracker["completed_at"] - tracker["started_at"]).total_seconds()
            tracker["duration"] = duration


class BulkResourceManager:
    """Gestor de recursos para operaciones bulk."""
    
    def __init__(self):
        self.resources: Dict[str, Dict[str, Any]] = {}
        self.usage: Dict[str, List[datetime]] = {}
    
    def register_resource(
        self,
        resource_id: str,
        resource_type: str,
        capacity: int,
        config: Optional[Dict[str, Any]] = None
    ):
        """Registrar recurso."""
        self.resources[resource_id] = {
            "type": resource_type,
            "capacity": capacity,
            "available": capacity,
            "used": 0,
            "config": config or {}
        }
        self.usage[resource_id] = []
    
    async def acquire_resource(
        self,
        resource_id: str,
        amount: int = 1,
        timeout: Optional[float] = None
    ) -> bool:
        """Adquirir recurso."""
        if resource_id not in self.resources:
            return False
        
        resource = self.resources[resource_id]
        start_time = datetime.now()
        
        while resource["available"] < amount:
            if timeout:
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed >= timeout:
                    return False
            await asyncio.sleep(0.1)
        
        resource["available"] -= amount
        resource["used"] += amount
        self.usage[resource_id].append(datetime.now())
        
        return True
    
    def release_resource(self, resource_id: str, amount: int = 1):
        """Liberar recurso."""
        if resource_id not in self.resources:
            return
        
        resource = self.resources[resource_id]
        resource["available"] = min(
            resource["capacity"],
            resource["available"] + amount
        )
        resource["used"] = max(0, resource["used"] - amount)
    
    def get_resource_status(self, resource_id: str) -> Optional[Dict[str, Any]]:
        """Obtener estado de recurso."""
        if resource_id not in self.resources:
            return None
        
        resource = self.resources[resource_id]
        usage_count = len([
            t for t in self.usage[resource_id]
            if (datetime.now() - t).total_seconds() < 3600  # Última hora
        ])
        
        return {
            "resource_id": resource_id,
            "type": resource["type"],
            "capacity": resource["capacity"],
            "available": resource["available"],
            "used": resource["used"],
            "usage_rate": (resource["used"] / resource["capacity"] * 100) if resource["capacity"] > 0 else 0,
            "recent_usage": usage_count
        }
    
    def get_all_resources_status(self) -> Dict[str, Dict[str, Any]]:
        """Obtener estado de todos los recursos."""
        return {
            rid: self.get_resource_status(rid)
            for rid in self.resources.keys()
        }


# Importar mejoras desde módulo separado
try:
    from .bulk_operations_improvements import (
        BulkPerformanceMonitor,
        BulkAdaptiveOptimizer,
        BulkSmartBatching,
        BulkIntelligentRetry,
        BulkPredictiveScaling,
        BulkMLOptimizer,
        BulkDistributedProcessor,
        BulkPatternAnalyzer,
        BulkAutoTuner,
        BulkLoadBalancer,
        BulkFailoverManager,
        BulkAdvancedMetrics,
        BulkSynchronizer,
        BulkReplicator,
        BulkAdvancedAnalytics,
        BulkOptimizationEngine,
        BulkEventBus,
        BulkDataValidator,
        BulkDataTransformer,
        BulkRouter,
        BulkCompressionAdvanced,
        BulkSecurityAdvanced
    )
except ImportError:
    # Si no se puede importar, definir clases básicas aquí
    class BulkPerformanceMonitor:
        def __init__(self):
            self.metrics = {}
        
        def record_operation(self, *args, **kwargs):
            pass
        
        def get_performance_stats(self, *args, **kwargs):
            return {}
    
    class BulkAdaptiveOptimizer:
        def __init__(self):
            self.config = {"max_workers": 10, "batch_size": 100}
        
        def optimize_config(self, *args, **kwargs):
            return self.config.copy()
    
    class BulkSmartBatching:
        def __init__(self):
            self.batch_history = {}
        
        def calculate_dynamic_batch_size(self, *args, **kwargs):
            return 100
    
    class BulkIntelligentRetry:
        def __init__(self):
            self.error_patterns = {}
        
        def analyze_error(self, *args, **kwargs):
            return {"should_retry": True, "retry_delay": 1.0}
    
    class BulkPredictiveScaling:
        def __init__(self):
            self.demand_patterns = {}
        
        def predict_demand(self, *args, **kwargs):
            return {"recommended_workers": 5, "confidence": 0.0}
    
    class BulkMLOptimizer:
        def __init__(self):
            self.models = {}
        
        def predict_optimal_config(self, *args, **kwargs):
            return {"max_workers": 10, "batch_size": 100}
    
    class BulkDistributedProcessor:
        def __init__(self):
            self.nodes = {}
        
        async def distribute_operation(self, *args, **kwargs):
            return []
    
    class BulkPatternAnalyzer:
        def __init__(self):
            self.patterns = {}
        
        def analyze_pattern(self, *args, **kwargs):
            return {"pattern_detected": False}
    
    class BulkAutoTuner:
        def __init__(self):
            self.best_configs = {}
        
        async def auto_tune(self, *args, **kwargs):
            return {"best_config": None, "best_score": 0}
    
    class BulkLoadBalancer:
        def __init__(self):
            self.backends = {}
        
        def select_backend(self, *args, **kwargs):
            return None
    
    class BulkFailoverManager:
        def __init__(self):
            self.primary_operations = {}
        
        async def execute_with_failover(self, *args, **kwargs):
            return None
    
    class BulkAdvancedMetrics:
        def __init__(self):
            self.metrics = {}
        
        def record_metric(self, *args, **kwargs):
            pass
    
    class BulkSynchronizer:
        def __init__(self):
            self.locks = {}
        
        async def acquire_lock(self, *args, **kwargs):
            return True
    
    class BulkReplicator:
        def __init__(self):
            self.replicas = {}
        
        async def replicate_operation(self, *args, **kwargs):
            return []
    
    class BulkAdvancedAnalytics:
        def __init__(self):
            self.analytics_data = {}
        
        def record_operation_data(self, *args, **kwargs):
            pass
    
    class BulkOptimizationEngine:
        def __init__(self):
            self.optimization_rules = {}
        
        async def optimize_operation(self, *args, **kwargs):
            return {}
    
    class BulkEventBus:
        def __init__(self):
            self.subscribers = {}
        
        async def publish(self, *args, **kwargs):
            pass
    
    class BulkDataValidator:
        def __init__(self):
            self.validation_rules = {}
        
        async def validate_data(self, *args, **kwargs):
            return True, []
    
    class BulkDataTransformer:
        def __init__(self):
            self.transformers = {}
        
        async def transform_data(self, *args, **kwargs):
            return args[1] if len(args) > 1 else None
    
    class BulkRouter:
        def __init__(self):
            self.routes = {}
        
        async def route(self, *args, **kwargs):
            return None
    
    class BulkCompressionAdvanced:
        def __init__(self):
            self.compression_methods = {}
        
        async def compress(self, *args, **kwargs):
            return b""
    
    class BulkSecurityAdvanced:
        def __init__(self):
            self.encryption_keys = {}
        
        def generate_key(self, *args, **kwargs):
            return b""


# ============================================================================
# NUEVAS CLASES AVANZADAS
# ============================================================================

class BulkRealTimeMetrics:
    """Sistema de métricas en tiempo real para operaciones bulk."""
    
    def __init__(self, window_size_seconds: int = 60):
        self.window_size = window_size_seconds
        self.metrics: Dict[str, List[Dict[str, Any]]] = {}
        self.aggregated: Dict[str, Dict[str, Any]] = {}
        self.start_time = datetime.now()
    
    def record_operation(
        self,
        operation_type: str,
        duration: float,
        success: bool,
        items_processed: int = 1,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Registrar una operación."""
        timestamp = datetime.now()
        metric = {
            "timestamp": timestamp,
            "duration": duration,
            "success": success,
            "items_processed": items_processed,
            "metadata": metadata or {}
        }
        
        if operation_type not in self.metrics:
            self.metrics[operation_type] = []
        
        self.metrics[operation_type].append(metric)
        
        # Limpiar métricas antiguas
        cutoff = timestamp - timedelta(seconds=self.window_size)
        self.metrics[operation_type] = [
            m for m in self.metrics[operation_type]
            if m["timestamp"] > cutoff
        ]
        
        # Actualizar agregados
        self._update_aggregates(operation_type)
    
    def _update_aggregates(self, operation_type: str):
        """Actualizar métricas agregadas."""
        if operation_type not in self.metrics:
            return
        
        metrics_list = self.metrics[operation_type]
        if not metrics_list:
            return
        
        total = len(metrics_list)
        successful = sum(1 for m in metrics_list if m["success"])
        total_items = sum(m["items_processed"] for m in metrics_list)
        durations = [m["duration"] for m in metrics_list]
        
        self.aggregated[operation_type] = {
            "total_operations": total,
            "successful_operations": successful,
            "failed_operations": total - successful,
            "success_rate": (successful / total * 100) if total > 0 else 0,
            "total_items_processed": total_items,
            "avg_duration": sum(durations) / len(durations) if durations else 0,
            "min_duration": min(durations) if durations else 0,
            "max_duration": max(durations) if durations else 0,
            "operations_per_second": total / self.window_size if self.window_size > 0 else 0,
            "items_per_second": total_items / self.window_size if self.window_size > 0 else 0,
            "last_updated": datetime.now().isoformat()
        }
    
    def get_metrics(self, operation_type: Optional[str] = None) -> Dict[str, Any]:
        """Obtener métricas."""
        if operation_type:
            return self.aggregated.get(operation_type, {})
        return {
            "operations": self.aggregated,
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
            "total_operation_types": len(self.aggregated)
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Obtener estado de salud basado en métricas."""
        if not self.aggregated:
            return {"status": "unknown", "message": "No metrics available"}
        
        # Calcular health score
        total_ops = sum(a["total_operations"] for a in self.aggregated.values())
        total_success = sum(a["successful_operations"] for a in self.aggregated.values())
        overall_success_rate = (total_success / total_ops * 100) if total_ops > 0 else 100
        
        avg_duration = sum(
            a["avg_duration"] for a in self.aggregated.values()
        ) / len(self.aggregated) if self.aggregated else 0
        
        status = "healthy"
        if overall_success_rate < 90:
            status = "degraded"
        if overall_success_rate < 70:
            status = "unhealthy"
        if avg_duration > 10:
            status = "slow"
        
        return {
            "status": status,
            "overall_success_rate": round(overall_success_rate, 2),
            "avg_duration": round(avg_duration, 2),
            "total_operations": total_ops,
            "operation_types": len(self.aggregated)
        }


class BulkAdvancedCache:
    """Sistema de caché avanzado con TTL y LRU."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, datetime] = {}
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Obtener valor del caché."""
        if key not in self.cache:
            self.misses += 1
            return None
        
        entry = self.cache[key]
        
        # Verificar TTL
        if datetime.now() > entry["expires_at"]:
            del self.cache[key]
            if key in self.access_times:
                del self.access_times[key]
            self.misses += 1
            return None
        
        # Actualizar tiempo de acceso (LRU)
        self.access_times[key] = datetime.now()
        self.hits += 1
        return entry["value"]
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ):
        """Guardar valor en caché."""
        ttl = ttl or self.default_ttl
        
        # Si está lleno, eliminar LRU
        if len(self.cache) >= self.max_size and key not in self.cache:
            self._evict_lru()
        
        expires_at = datetime.now() + timedelta(seconds=ttl)
        self.cache[key] = {
            "value": value,
            "expires_at": expires_at,
            "created_at": datetime.now()
        }
        self.access_times[key] = datetime.now()
    
    def _evict_lru(self):
        """Eliminar entrada menos recientemente usada."""
        if not self.access_times:
            # Si no hay access_times, eliminar la más antigua del cache
            if self.cache:
                oldest_key = min(
                    self.cache.keys(),
                    key=lambda k: self.cache[k]["created_at"]
                )
                del self.cache[oldest_key]
            return
        
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        if lru_key in self.cache:
            del self.cache[lru_key]
        if lru_key in self.access_times:
            del self.access_times[lru_key]
    
    def invalidate(self, key: str):
        """Invalidar entrada específica."""
        if key in self.cache:
            del self.cache[key]
        if key in self.access_times:
            del self.access_times[key]
    
    def clear(self):
        """Limpiar todo el caché."""
        self.cache.clear()
        self.access_times.clear()
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del caché."""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(hit_rate, 2),
            "usage_percent": round((len(self.cache) / self.max_size * 100), 2)
        }


class BulkPriorityQueue:
    """Cola de prioridades para operaciones bulk."""
    
    def __init__(self):
        self.queue: List[Tuple[int, float, Dict[str, Any]]] = []  # (priority, timestamp, operation)
        self.priority_levels = {
            "critical": 10,
            "high": 7,
            "medium": 5,
            "low": 3,
            "background": 1
        }
    
    def enqueue(
        self,
        operation: Dict[str, Any],
        priority: Union[str, int] = "medium"
    ):
        """Agregar operación a la cola."""
        if isinstance(priority, str):
            priority_value = self.priority_levels.get(priority, 5)
        else:
            priority_value = priority
        
        timestamp = datetime.now().timestamp()
        self.queue.append((priority_value, timestamp, operation))
        # Ordenar por prioridad (mayor primero) y luego por timestamp
        self.queue.sort(key=lambda x: (-x[0], x[1]))
    
    def dequeue(self) -> Optional[Dict[str, Any]]:
        """Obtener siguiente operación de mayor prioridad."""
        if not self.queue:
            return None
        
        _, _, operation = self.queue.pop(0)
        return operation
    
    def peek(self) -> Optional[Dict[str, Any]]:
        """Ver siguiente operación sin removerla."""
        if not self.queue:
            return None
        return self.queue[0][2]
    
    def size(self) -> int:
        """Tamaño de la cola."""
        return len(self.queue)
    
    def get_by_priority(self, priority: Union[str, int]) -> List[Dict[str, Any]]:
        """Obtener todas las operaciones de una prioridad."""
        if isinstance(priority, str):
            priority_value = self.priority_levels.get(priority, 5)
        else:
            priority_value = priority
        
        return [op for p, _, op in self.queue if p == priority_value]
    
    def clear(self):
        """Limpiar cola."""
        self.queue.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de la cola."""
        priority_counts = {}
        for p, _, _ in self.queue:
            priority_counts[p] = priority_counts.get(p, 0) + 1
        
        return {
            "total": len(self.queue),
            "by_priority": priority_counts,
            "oldest_waiting": (
                datetime.fromtimestamp(self.queue[-1][1]) if self.queue else None
            )
        }


class BulkEnhancedValidator:
    """Sistema de validación mejorado para operaciones bulk."""
    
    def __init__(self):
        self.rules: Dict[str, List[Callable]] = {}
        self.validation_cache: Dict[str, Tuple[bool, Optional[str]]] = {}
    
    def add_rule(
        self,
        operation_type: str,
        validator: Callable,
        cache_result: bool = True
    ):
        """Añadir regla de validación."""
        if operation_type not in self.rules:
            self.rules[operation_type] = []
        self.rules[operation_type].append(validator)
    
    def validate(
        self,
        operation_type: str,
        data: Dict[str, Any],
        use_cache: bool = True
    ) -> Tuple[bool, Optional[str]]:
        """Validar datos."""
        # Verificar caché
        if use_cache:
            cache_key = f"{operation_type}:{hash(str(sorted(data.items())))}"
            if cache_key in self.validation_cache:
                return self.validation_cache[cache_key]
        
        # Ejecutar validaciones
        if operation_type not in self.rules:
            return True, None
        
        for validator in self.rules[operation_type]:
            try:
                if asyncio.iscoroutinefunction(validator):
                    # Para async validators, necesitaríamos await
                    # Por ahora, asumimos que son síncronos o manejados externamente
                    result = validator(data)
                else:
                    result = validator(data)
                
                if not result:
                    error_msg = f"Validation failed for {operation_type}"
                    if use_cache:
                        self.validation_cache[cache_key] = (False, error_msg)
                    return False, error_msg
            except Exception as e:
                error_msg = f"Validation error: {str(e)}"
                if use_cache:
                    self.validation_cache[cache_key] = (False, error_msg)
                return False, error_msg
        
        # Cachear resultado exitoso
        if use_cache:
            cache_key = f"{operation_type}:{hash(str(sorted(data.items())))}"
            self.validation_cache[cache_key] = (True, None)
        
        return True, None
    
    def validate_batch(
        self,
        operation_type: str,
        items: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validar lote de items."""
        results = {
            "valid": [],
            "invalid": [],
            "total": len(items)
        }
        
        for i, item in enumerate(items):
            is_valid, error = self.validate(operation_type, item)
            if is_valid:
                results["valid"].append({"index": i, "data": item})
            else:
                results["invalid"].append({"index": i, "data": item, "error": error})
        
        results["valid_count"] = len(results["valid"])
        results["invalid_count"] = len(results["invalid"])
        results["validity_rate"] = (
            results["valid_count"] / results["total"] * 100
            if results["total"] > 0 else 0
        )
        
        return results
    
    def clear_cache(self):
        """Limpiar caché de validación."""
        self.validation_cache.clear()


class BulkDashboard:
    """Dashboard de monitoreo integrado para operaciones bulk."""
    
    def __init__(
        self,
        metrics: Optional[BulkRealTimeMetrics] = None,
        cache: Optional[BulkAdvancedCache] = None,
        priority_queue: Optional[BulkPriorityQueue] = None
    ):
        self.metrics = metrics or BulkRealTimeMetrics()
        self.cache = cache or BulkAdvancedCache()
        self.priority_queue = priority_queue or BulkPriorityQueue()
        self.alerts: List[Dict[str, Any]] = []
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Obtener datos completos del dashboard."""
        return {
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "real_time": self.metrics.get_metrics(),
                "health": self.metrics.get_health_status()
            },
            "cache": {
                "stats": self.cache.get_stats()
            },
            "queue": {
                "stats": self.priority_queue.get_stats(),
                "size": self.priority_queue.size()
            },
            "alerts": self.alerts[-10:],  # Últimos 10 alerts
            "summary": self._generate_summary()
        }
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generar resumen ejecutivo."""
        health = self.metrics.get_health_status()
        cache_stats = self.cache.get_stats()
        queue_stats = self.priority_queue.get_stats()
        
        return {
            "system_status": health.get("status", "unknown"),
            "success_rate": health.get("overall_success_rate", 0),
            "cache_hit_rate": cache_stats.get("hit_rate", 0),
            "pending_operations": queue_stats.get("total", 0),
            "performance": "excellent" if health.get("overall_success_rate", 0) > 95 else "good" if health.get("overall_success_rate", 0) > 85 else "needs_attention"
        }
    
    def add_alert(
        self,
        level: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """Añadir alerta."""
        alert = {
            "level": level,
            "message": message,
            "details": details or {},
            "timestamp": datetime.now().isoformat()
        }
        self.alerts.append(alert)
        
        # Mantener solo últimas 100 alertas
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
    
    def get_alerts(self, level: Optional[str] = None) -> List[Dict[str, Any]]:
        """Obtener alertas."""
        if level:
            return [a for a in self.alerts if a["level"] == level]
        return self.alerts


# ============================================================================
# DECORADORES Y UTILIDADES AVANZADAS
# ============================================================================

def track_metrics(metrics: Optional[BulkRealTimeMetrics] = None, operation_type: Optional[str] = None):
    """
    Decorador para tracking automático de métricas en operaciones bulk.
    
    Args:
        metrics: Instancia de BulkRealTimeMetrics (opcional, busca en contexto)
        operation_type: Tipo de operación (opcional, usa nombre de función)
    """
    def decorator(func: Callable):
        operation_name = operation_type or func.__name__
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            success = False
            items_processed = 0
            error = None
            
            try:
                result = await func(*args, **kwargs)
                
                # Intentar extraer items procesados del resultado
                if isinstance(result, BulkOperationResult):
                    items_processed = result.processed
                    success = result.success
                elif isinstance(result, (list, tuple)):
                    items_processed = len(result)
                    success = True
                elif isinstance(result, dict):
                    items_processed = result.get("processed", result.get("count", 1))
                    success = result.get("success", True)
                else:
                    items_processed = 1
                    success = True
                
                duration = time.time() - start_time
                
                # Registrar métrica si hay instancia disponible
                if metrics:
                    metrics.record_operation(
                        operation_type=operation_name,
                        duration=duration,
                        success=success,
                        items_processed=items_processed,
                        metadata={"function": func.__name__, "args_count": len(args)}
                    )
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                error = str(e)
                
                if metrics:
                    metrics.record_operation(
                        operation_type=operation_name,
                        duration=duration,
                        success=False,
                        items_processed=0,
                        metadata={"error": error, "function": func.__name__}
                    )
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            success = False
            items_processed = 0
            
            try:
                result = func(*args, **kwargs)
                
                if isinstance(result, BulkOperationResult):
                    items_processed = result.processed
                    success = result.success
                elif isinstance(result, (list, tuple)):
                    items_processed = len(result)
                    success = True
                elif isinstance(result, dict):
                    items_processed = result.get("processed", result.get("count", 1))
                    success = result.get("success", True)
                else:
                    items_processed = 1
                    success = True
                
                duration = time.time() - start_time
                
                if metrics:
                    metrics.record_operation(
                        operation_type=operation_name,
                        duration=duration,
                        success=success,
                        items_processed=items_processed,
                        metadata={"function": func.__name__}
                    )
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                
                if metrics:
                    metrics.record_operation(
                        operation_type=operation_name,
                        duration=duration,
                        success=False,
                        items_processed=0,
                        metadata={"error": str(e), "function": func.__name__}
                    )
                raise
        
        # Retornar wrapper apropiado
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def cache_result(cache: Optional[BulkAdvancedCache] = None, ttl: int = 3600, key_func: Optional[Callable] = None):
    """
    Decorador para cachear resultados de operaciones.
    
    Args:
        cache: Instancia de BulkAdvancedCache
        ttl: Tiempo de vida del caché en segundos
        key_func: Función para generar clave del caché (opcional)
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not cache:
                return await func(*args, **kwargs)
            
            # Generar clave del caché
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Intentar obtener del caché
            cached = cache.get(cache_key)
            if cached is not None:
                return cached
            
            # Ejecutar función y cachear resultado
            result = await func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            return result
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            if not cache:
                return func(*args, **kwargs)
            
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
            cached = cache.get(cache_key)
            if cached is not None:
                return cached
            
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            return result
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def validate_input(validator: Optional[BulkEnhancedValidator] = None, operation_type: Optional[str] = None):
    """
    Decorador para validar inputs antes de ejecutar operaciones.
    
    Args:
        validator: Instancia de BulkEnhancedValidator
        operation_type: Tipo de operación para validación
    """
    def decorator(func: Callable):
        operation_name = operation_type or func.__name__
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            if validator:
                # Validar argumentos (simplificado - validar kwargs como dict)
                data_to_validate = kwargs.copy()
                if args:
                    data_to_validate["_args"] = args
                
                is_valid, error = validator.validate(operation_name, data_to_validate)
                if not is_valid:
                    raise ValueError(f"Validation failed: {error}")
            
            return await func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            if validator:
                data_to_validate = kwargs.copy()
                if args:
                    data_to_validate["_args"] = args
                
                is_valid, error = validator.validate(operation_name, data_to_validate)
                if not is_valid:
                    raise ValueError(f"Validation failed: {error}")
            
            return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


class BulkBenchmark:
    """Sistema de benchmarking para operaciones bulk."""
    
    def __init__(self):
        self.benchmarks: Dict[str, List[Dict[str, Any]]] = {}
    
    async def benchmark_operation(
        self,
        operation: Callable,
        operation_name: str,
        test_data: List[Any],
        iterations: int = 3,
        warmup: int = 1
    ) -> Dict[str, Any]:
        """Ejecutar benchmark de una operación."""
        # Warmup
        for _ in range(warmup):
            if asyncio.iscoroutinefunction(operation):
                await operation(*test_data[0] if test_data else [])
            else:
                operation(*test_data[0] if test_data else [])
        
        # Benchmark
        times = []
        for i in range(iterations):
            start = time.time()
            if asyncio.iscoroutinefunction(operation):
                await operation(*test_data[i % len(test_data)] if test_data else [])
            else:
                operation(*test_data[i % len(test_data)] if test_data else [])
            times.append(time.time() - start)
        
        results = {
            "operation": operation_name,
            "iterations": iterations,
            "avg_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
            "total_time": sum(times),
            "times": times,
            "timestamp": datetime.now().isoformat()
        }
        
        if operation_name not in self.benchmarks:
            self.benchmarks[operation_name] = []
        self.benchmarks[operation_name].append(results)
        
        return results
    
    def compare_operations(
        self,
        operation1_name: str,
        operation2_name: str
    ) -> Dict[str, Any]:
        """Comparar dos operaciones."""
        ops1 = self.benchmarks.get(operation1_name, [])
        ops2 = self.benchmarks.get(operation2_name, [])
        
        if not ops1 or not ops2:
            return {"error": "Not enough benchmark data"}
        
        avg1 = sum(op["avg_time"] for op in ops1) / len(ops1)
        avg2 = sum(op["avg_time"] for op in ops2) / len(ops2)
        
        faster = operation1_name if avg1 < avg2 else operation2_name
        speedup = max(avg1, avg2) / min(avg1, avg2)
        
        return {
            operation1_name: {"avg_time": avg1, "runs": len(ops1)},
            operation2_name: {"avg_time": avg2, "runs": len(ops2)},
            "faster": faster,
            "speedup": round(speedup, 2),
            "comparison": f"{faster} is {speedup:.2f}x faster"
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Obtener resumen de benchmarks."""
        summary = {}
        for op_name, benchmarks in self.benchmarks.items():
            if benchmarks:
                avg_time = sum(b["avg_time"] for b in benchmarks) / len(benchmarks)
                summary[op_name] = {
                    "avg_time": avg_time,
                    "runs": len(benchmarks),
                    "min_avg": min(b["avg_time"] for b in benchmarks),
                    "max_avg": max(b["avg_time"] for b in benchmarks)
                }
        return summary


class BulkAutoTuner:
    """Sistema de auto-tuning de parámetros para optimizar rendimiento."""
    
    def __init__(
        self,
        metrics: Optional[BulkRealTimeMetrics] = None,
        benchmark: Optional[BulkBenchmark] = None
    ):
        self.metrics = metrics or BulkRealTimeMetrics()
        self.benchmark = benchmark or BulkBenchmark()
        self.tuning_history: List[Dict[str, Any]] = []
    
    async def tune_batch_size(
        self,
        operation: Callable,
        test_data: List[Any],
        min_batch: int = 10,
        max_batch: int = 1000,
        step: int = 50,
        iterations: int = 3
    ) -> Dict[str, Any]:
        """Auto-ajustar tamaño de batch para una operación."""
        best_batch = min_batch
        best_time = float('inf')
        results = []
        
        for batch_size in range(min_batch, max_batch + 1, step):
            # Wrapper para operación con batch size específico
            async def batched_operation():
                batches = [test_data[i:i + batch_size] for i in range(0, len(test_data), batch_size)]
                for batch in batches:
                    if asyncio.iscoroutinefunction(operation):
                        await operation(batch)
                    else:
                        operation(batch)
            
            # Benchmark
            benchmark_result = await self.benchmark.benchmark_operation(
                batched_operation,
                f"batch_{batch_size}",
                [None],
                iterations=iterations
            )
            
            avg_time = benchmark_result["avg_time"]
            results.append({
                "batch_size": batch_size,
                "avg_time": avg_time,
                "throughput": len(test_data) / avg_time if avg_time > 0 else 0
            })
            
            if avg_time < best_time:
                best_time = avg_time
                best_batch = batch_size
        
        tuning_result = {
            "best_batch_size": best_batch,
            "best_time": best_time,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
        self.tuning_history.append(tuning_result)
        return tuning_result
    
    async def tune_workers(
        self,
        operation: Callable,
        test_data: List[Any],
        min_workers: int = 1,
        max_workers: int = 50,
        step: int = 5
    ) -> Dict[str, Any]:
        """Auto-ajustar número de workers."""
        best_workers = min_workers
        best_throughput = 0
        results = []
        
        for num_workers in range(min_workers, max_workers + 1, step):
            semaphore = asyncio.Semaphore(num_workers)
            
            async def worker_operation(data):
                async with semaphore:
                    if asyncio.iscoroutinefunction(operation):
                        return await operation(data)
                    return operation(data)
            
            start = time.time()
            tasks = [worker_operation(item) for item in test_data]
            await asyncio.gather(*tasks)
            duration = time.time() - start
            
            throughput = len(test_data) / duration if duration > 0 else 0
            results.append({
                "workers": num_workers,
                "duration": duration,
                "throughput": throughput
            })
            
            if throughput > best_throughput:
                best_throughput = throughput
                best_workers = num_workers
        
        tuning_result = {
            "best_workers": best_workers,
            "best_throughput": best_throughput,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
        self.tuning_history.append(tuning_result)
        return tuning_result
    
    def get_tuning_recommendations(self) -> Dict[str, Any]:
        """Obtener recomendaciones de tuning basadas en historial."""
        if not self.tuning_history:
            return {"message": "No tuning history available"}
        
        recommendations = {}
        for tuning in self.tuning_history:
            if "best_batch_size" in tuning:
                recommendations["batch_size"] = tuning["best_batch_size"]
            if "best_workers" in tuning:
                recommendations["workers"] = tuning["best_workers"]
        
        return {
            "recommendations": recommendations,
            "based_on": len(self.tuning_history),
            "last_tuning": self.tuning_history[-1].get("timestamp")
        }


# ============================================================================
# SISTEMAS AVANZADOS DE RESILIENCIA Y OPTIMIZACIÓN
# ============================================================================

class BulkAdaptiveRateLimiter:
    """Rate limiter adaptativo que ajusta límites según condiciones del sistema."""
    
    def __init__(
        self,
        initial_rate: int = 100,
        min_rate: int = 10,
        max_rate: int = 1000,
        adjustment_factor: float = 1.2
    ):
        self.current_rate = initial_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.adjustment_factor = adjustment_factor
        self.requests: List[datetime] = []
        self.error_history: List[bool] = []
        self.response_times: List[float] = []
    
    def can_proceed(self) -> bool:
        """Verificar si se puede procesar una petición."""
        now = datetime.now()
        # Limpiar requests antiguos (última hora)
        cutoff = now - timedelta(hours=1)
        self.requests = [r for r in self.requests if r > cutoff]
        
        # Verificar si estamos bajo el límite
        if len(self.requests) < self.current_rate:
            self.requests.append(now)
            return True
        return False
    
    def record_success(self, response_time: float):
        """Registrar éxito y ajustar rate si es necesario."""
        self.error_history.append(True)
        self.response_times.append(response_time)
        
        # Mantener solo últimas 100
        if len(self.error_history) > 100:
            self.error_history = self.error_history[-100:]
        if len(self.response_times) > 100:
            self.response_times = self.response_times[-100:]
        
        # Si todo va bien y hay margen, aumentar rate
        if len(self.error_history) >= 10:
            recent_success_rate = sum(self.error_history[-10:]) / 10
            avg_response_time = sum(self.response_times[-10:]) / 10 if self.response_times else 0
            
            if recent_success_rate > 0.95 and avg_response_time < 1.0:
                self.current_rate = min(
                    self.max_rate,
                    int(self.current_rate * self.adjustment_factor)
                )
    
    def record_failure(self):
        """Registrar fallo y reducir rate."""
        self.error_history.append(False)
        
        if len(self.error_history) > 100:
            self.error_history = self.error_history[-100:]
        
        # Reducir rate si hay muchos errores
        if len(self.error_history) >= 10:
            recent_success_rate = sum(self.error_history[-10:]) / 10
            if recent_success_rate < 0.8:
                self.current_rate = max(
                    self.min_rate,
                    int(self.current_rate / self.adjustment_factor)
                )
    
    def get_status(self) -> Dict[str, Any]:
        """Obtener estado del rate limiter."""
        success_rate = (
            sum(self.error_history) / len(self.error_history) * 100
            if self.error_history else 100
        )
        avg_response_time = (
            sum(self.response_times) / len(self.response_times)
            if self.response_times else 0
        )
        
        return {
            "current_rate": self.current_rate,
            "min_rate": self.min_rate,
            "max_rate": self.max_rate,
            "recent_requests": len(self.requests),
            "success_rate": round(success_rate, 2),
            "avg_response_time": round(avg_response_time, 3),
            "utilization": round((len(self.requests) / self.current_rate * 100), 2) if self.current_rate > 0 else 0
        }


class BulkLoadBalancer:
    """Load balancer inteligente para distribuir carga entre workers."""
    
    def __init__(self, initial_workers: int = 10):
        self.workers: Dict[int, Dict[str, Any]] = {}
        self.worker_count = initial_workers
        self.initialize_workers()
    
    def initialize_workers(self):
        """Inicializar workers."""
        for i in range(self.worker_count):
            self.workers[i] = {
                "id": i,
                "active_tasks": 0,
                "completed_tasks": 0,
                "failed_tasks": 0,
                "avg_response_time": 0.0,
                "last_used": datetime.now(),
                "status": "idle"
            }
    
    def select_worker(self) -> int:
        """Seleccionar worker con menor carga."""
        if not self.workers:
            return 0
        
        # Seleccionar worker con menor carga activa
        best_worker = min(
            self.workers.keys(),
            key=lambda w: (
                self.workers[w]["active_tasks"],
                -self.workers[w]["completed_tasks"]  # Preferir workers con más experiencia
            )
        )
        return best_worker
    
    def assign_task(self, worker_id: int):
        """Asignar tarea a worker."""
        if worker_id in self.workers:
            self.workers[worker_id]["active_tasks"] += 1
            self.workers[worker_id]["last_used"] = datetime.now()
            self.workers[worker_id]["status"] = "busy"
    
    def complete_task(self, worker_id: int, success: bool, response_time: float):
        """Marcar tarea como completada."""
        if worker_id in self.workers:
            worker = self.workers[worker_id]
            worker["active_tasks"] = max(0, worker["active_tasks"] - 1)
            
            if success:
                worker["completed_tasks"] += 1
            else:
                worker["failed_tasks"] += 1
            
            # Actualizar tiempo de respuesta promedio
            total_tasks = worker["completed_tasks"] + worker["failed_tasks"]
            if total_tasks > 0:
                worker["avg_response_time"] = (
                    (worker["avg_response_time"] * (total_tasks - 1) + response_time) / total_tasks
                )
            
            if worker["active_tasks"] == 0:
                worker["status"] = "idle"
    
    def add_worker(self) -> int:
        """Añadir nuevo worker."""
        new_id = max(self.workers.keys()) + 1 if self.workers else 0
        self.workers[new_id] = {
            "id": new_id,
            "active_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "avg_response_time": 0.0,
            "last_used": datetime.now(),
            "status": "idle"
        }
        self.worker_count += 1
        return new_id
    
    def remove_worker(self, worker_id: int) -> bool:
        """Remover worker (solo si no tiene tareas activas)."""
        if worker_id in self.workers:
            if self.workers[worker_id]["active_tasks"] == 0:
                del self.workers[worker_id]
                self.worker_count -= 1
                return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del load balancer."""
        total_active = sum(w["active_tasks"] for w in self.workers.values())
        total_completed = sum(w["completed_tasks"] for w in self.workers.values())
        total_failed = sum(w["failed_tasks"] for w in self.workers.values())
        avg_response_time = (
            sum(w["avg_response_time"] for w in self.workers.values()) / len(self.workers)
            if self.workers else 0
        )
        
        return {
            "worker_count": self.worker_count,
            "total_active_tasks": total_active,
            "total_completed": total_completed,
            "total_failed": total_failed,
            "success_rate": (
                (total_completed / (total_completed + total_failed) * 100)
                if (total_completed + total_failed) > 0 else 100
            ),
            "avg_response_time": round(avg_response_time, 3),
            "workers": {
                w_id: {
                    "status": w["status"],
                    "active": w["active_tasks"],
                    "completed": w["completed_tasks"],
                    "failed": w["failed_tasks"]
                }
                for w_id, w in self.workers.items()
            }
        }


class BulkLoadPredictor:
    """Sistema de predicción de carga basado en patrones históricos."""
    
    def __init__(self, window_size: int = 60):
        self.window_size = window_size  # minutos
        self.load_history: List[Dict[str, Any]] = []
        self.patterns: Dict[str, List[float]] = {}  # hora_del_dia -> [cargas]
    
    def record_load(self, load: float, timestamp: Optional[datetime] = None):
        """Registrar carga actual."""
        ts = timestamp or datetime.now()
        hour = ts.hour
        
        record = {
            "timestamp": ts,
            "load": load,
            "hour": hour
        }
        self.load_history.append(record)
        
        # Limpiar historial antiguo
        cutoff = ts - timedelta(minutes=self.window_size)
        self.load_history = [r for r in self.load_history if r["timestamp"] > cutoff]
        
        # Actualizar patrones
        hour_key = str(hour)
        if hour_key not in self.patterns:
            self.patterns[hour_key] = []
        self.patterns[hour_key].append(load)
        if len(self.patterns[hour_key]) > 100:
            self.patterns[hour_key] = self.patterns[hour_key][-100:]
    
    def predict_load(self, minutes_ahead: int = 5) -> Dict[str, Any]:
        """Predecir carga futura."""
        now = datetime.now()
        target_time = now + timedelta(minutes=minutes_ahead)
        target_hour = target_time.hour
        
        # Obtener patrones históricos para esta hora
        hour_key = str(target_hour)
        if hour_key not in self.patterns or not self.patterns[hour_key]:
            return {
                "predicted_load": 0.5,  # Default
                "confidence": 0.0,
                "based_on": 0,
                "message": "Insufficient historical data"
            }
        
        historical_loads = self.patterns[hour_key]
        avg_load = sum(historical_loads) / len(historical_loads)
        
        # Calcular tendencia reciente
        recent_loads = self.load_history[-10:] if len(self.load_history) >= 10 else self.load_history
        if recent_loads:
            recent_avg = sum(r["load"] for r in recent_loads) / len(recent_loads)
            # Combinar histórico y tendencia reciente
            predicted = (avg_load * 0.7) + (recent_avg * 0.3)
        else:
            predicted = avg_load
        
        confidence = min(1.0, len(historical_loads) / 50)  # Más datos = más confianza
        
        return {
            "predicted_load": round(predicted, 2),
            "confidence": round(confidence, 2),
            "based_on": len(historical_loads),
            "target_time": target_time.isoformat(),
            "historical_avg": round(avg_load, 2)
        }
    
    def get_load_pattern(self) -> Dict[str, Any]:
        """Obtener patrón de carga por hora."""
        hourly_avg = {}
        for hour_key, loads in self.patterns.items():
            if loads:
                hourly_avg[hour_key] = {
                    "avg_load": round(sum(loads) / len(loads), 2),
                    "samples": len(loads),
                    "min": round(min(loads), 2),
                    "max": round(max(loads), 2)
                }
        
        return {
            "hourly_patterns": hourly_avg,
            "current_load": self.load_history[-1]["load"] if self.load_history else 0,
            "total_samples": len(self.load_history)
        }


class BulkAutoScaler:
    """Sistema de auto-scaling basado en métricas y predicciones."""
    
    def __init__(
        self,
        load_balancer: Optional[BulkLoadBalancer] = None,
        load_predictor: Optional[BulkLoadPredictor] = None,
        metrics: Optional[BulkRealTimeMetrics] = None,
        min_workers: int = 1,
        max_workers: int = 100,
        scale_up_threshold: float = 0.8,
        scale_down_threshold: float = 0.3
    ):
        self.load_balancer = load_balancer or BulkLoadBalancer()
        self.load_predictor = load_predictor or BulkLoadPredictor()
        self.metrics = metrics or BulkRealTimeMetrics()
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.scaling_history: List[Dict[str, Any]] = []
    
    def evaluate_scaling(self) -> Dict[str, Any]:
        """Evaluar si es necesario hacer scaling."""
        # Obtener métricas actuales
        lb_stats = self.load_balancer.get_stats()
        health = self.metrics.get_health_status()
        
        # Calcular carga actual
        active_tasks = lb_stats["total_active_tasks"]
        worker_count = lb_stats["worker_count"]
        utilization = (active_tasks / worker_count) if worker_count > 0 else 0
        
        # Predecir carga futura
        prediction = self.load_predictor.predict_load(minutes_ahead=5)
        predicted_load = prediction["predicted_load"]
        
        action = "none"
        reason = ""
        new_worker_count = worker_count
        
        # Decisión de scaling
        if utilization > self.scale_up_threshold or predicted_load > 0.8:
            if worker_count < self.max_workers:
                action = "scale_up"
                new_worker_count = min(
                    self.max_workers,
                    worker_count + max(1, int(worker_count * 0.2))  # Aumentar 20%
                )
                reason = f"High utilization ({utilization:.2%}) or predicted load ({predicted_load:.2f})"
        elif utilization < self.scale_down_threshold and worker_count > self.min_workers:
            if predicted_load < 0.3:
                action = "scale_down"
                new_worker_count = max(
                    self.min_workers,
                    worker_count - max(1, int(worker_count * 0.1))  # Reducir 10%
                )
                reason = f"Low utilization ({utilization:.2%}) and predicted load ({predicted_load:.2f})"
        
        return {
            "action": action,
            "current_workers": worker_count,
            "recommended_workers": new_worker_count,
            "utilization": round(utilization, 2),
            "predicted_load": predicted_load,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        }
    
    async def execute_scaling(self) -> Dict[str, Any]:
        """Ejecutar scaling si es necesario."""
        evaluation = self.evaluate_scaling()
        
        if evaluation["action"] == "scale_up":
            workers_to_add = evaluation["recommended_workers"] - evaluation["current_workers"]
            for _ in range(workers_to_add):
                self.load_balancer.add_worker()
            
            self.scaling_history.append({
                **evaluation,
                "executed": True,
                "workers_added": workers_to_add
            })
        
        elif evaluation["action"] == "scale_down":
            workers_to_remove = evaluation["current_workers"] - evaluation["recommended_workers"]
            removed = 0
            for worker_id in list(self.load_balancer.workers.keys()):
                if removed >= workers_to_remove:
                    break
                if self.load_balancer.remove_worker(worker_id):
                    removed += 1
            
            self.scaling_history.append({
                **evaluation,
                "executed": True,
                "workers_removed": removed
            })
        
        return evaluation
    
    def get_scaling_history(self) -> List[Dict[str, Any]]:
        """Obtener historial de scaling."""
        return self.scaling_history[-50:]  # Últimos 50 eventos


# ============================================================================
# SISTEMAS AVANZADOS DE OBSERVABILIDAD Y OPTIMIZACIÓN
# ============================================================================

class BulkEventSourcing:
    """Sistema de event sourcing para operaciones bulk."""
    
    def __init__(self):
        self.events: List[Dict[str, Any]] = []
        self.event_streams: Dict[str, List[Dict[str, Any]]] = {}
    
    def record_event(
        self,
        event_type: str,
        aggregate_id: str,
        payload: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Registrar evento."""
        event = {
            "event_id": f"{aggregate_id}_{len(self.events)}",
            "event_type": event_type,
            "aggregate_id": aggregate_id,
            "payload": payload,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
            "version": 1
        }
        
        self.events.append(event)
        
        # Mantener en streams por aggregate
        if aggregate_id not in self.event_streams:
            self.event_streams[aggregate_id] = []
        self.event_streams[aggregate_id].append(event)
        
        # Limitar tamaño (mantener últimos 10000 eventos)
        if len(self.events) > 10000:
            self.events = self.events[-10000:]
    
    def get_events(
        self,
        aggregate_id: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Obtener eventos."""
        events = self.events
        
        if aggregate_id:
            events = self.event_streams.get(aggregate_id, [])
        
        if event_type:
            events = [e for e in events if e["event_type"] == event_type]
        
        return events[-limit:] if limit else events
    
    def replay_events(
        self,
        aggregate_id: str,
        handler: Callable
    ) -> List[Any]:
        """Replay eventos para reconstruir estado."""
        events = self.event_streams.get(aggregate_id, [])
        results = []
        
        for event in events:
            try:
                result = handler(event)
                results.append(result)
            except Exception as e:
                logger.error(f"Error replaying event {event['event_id']}: {e}")
        
        return results


class BulkObservability:
    """Sistema de observabilidad avanzado para operaciones bulk."""
    
    def __init__(self):
        self.traces: Dict[str, Dict[str, Any]] = {}
        self.spans: Dict[str, List[Dict[str, Any]]] = {}
        self.metrics_log: List[Dict[str, Any]] = []
        self.logs: List[Dict[str, Any]] = []
    
    def start_trace(
        self,
        trace_id: str,
        operation_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Iniciar trace distribuido."""
        self.traces[trace_id] = {
            "trace_id": trace_id,
            "operation_name": operation_name,
            "start_time": datetime.now(),
            "spans": [],
            "metadata": metadata or {},
            "status": "active"
        }
        self.spans[trace_id] = []
    
    def add_span(
        self,
        trace_id: str,
        span_name: str,
        duration: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Añadir span a trace."""
        span = {
            "span_name": span_name,
            "duration": duration,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        if trace_id in self.spans:
            self.spans[trace_id].append(span)
        if trace_id in self.traces:
            self.traces[trace_id]["spans"].append(span)
    
    def complete_trace(
        self,
        trace_id: str,
        status: str = "success",
        error: Optional[str] = None
    ):
        """Completar trace."""
        if trace_id in self.traces:
            trace = self.traces[trace_id]
            trace["end_time"] = datetime.now()
            trace["status"] = status
            trace["duration"] = (
                trace["end_time"] - trace["start_time"]
            ).total_seconds()
            
            if error:
                trace["error"] = error
    
    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Obtener trace completo."""
        return self.traces.get(trace_id)
    
    def log_event(
        self,
        level: str,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None
    ):
        """Registrar evento de log."""
        log_entry = {
            "level": level,
            "message": message,
            "context": context or {},
            "trace_id": trace_id,
            "timestamp": datetime.now().isoformat()
        }
        
        self.logs.append(log_entry)
        
        # Limitar tamaño
        if len(self.logs) > 10000:
            self.logs = self.logs[-10000:]
    
    def get_logs(
        self,
        level: Optional[str] = None,
        trace_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Obtener logs."""
        logs = self.logs
        
        if level:
            logs = [l for l in logs if l["level"] == level]
        if trace_id:
            logs = [l for l in logs if l.get("trace_id") == trace_id]
        
        return logs[-limit:] if limit else logs
    
    def get_observability_summary(self) -> Dict[str, Any]:
        """Obtener resumen de observabilidad."""
        active_traces = sum(1 for t in self.traces.values() if t.get("status") == "active")
        completed_traces = sum(1 for t in self.traces.values() if t.get("status") != "active")
        
        error_logs = sum(1 for l in self.logs if l["level"] in ["error", "critical"])
        
        return {
            "traces": {
                "total": len(self.traces),
                "active": active_traces,
                "completed": completed_traces
            },
            "spans": {
                "total": sum(len(spans) for spans in self.spans.values())
            },
            "logs": {
                "total": len(self.logs),
                "errors": error_logs
            }
        }


class BulkCostOptimizer:
    """Sistema de optimización de costos para operaciones bulk."""
    
    def __init__(self):
        self.cost_history: List[Dict[str, Any]] = []
        self.operation_costs: Dict[str, float] = {}
        self.optimizations: List[Dict[str, Any]] = []
    
    def record_operation_cost(
        self,
        operation_type: str,
        cost: float,
        items_processed: int,
        duration: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Registrar costo de operación."""
        cost_record = {
            "operation_type": operation_type,
            "cost": cost,
            "items_processed": items_processed,
            "cost_per_item": cost / items_processed if items_processed > 0 else 0,
            "duration": duration,
            "cost_per_second": cost / duration if duration > 0 else 0,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.cost_history.append(cost_record)
        
        # Actualizar costo promedio por operación
        operation_costs = [
            c["cost"] for c in self.cost_history
            if c["operation_type"] == operation_type
        ]
        if operation_costs:
            self.operation_costs[operation_type] = sum(operation_costs) / len(operation_costs)
        
        # Limitar historial
        if len(self.cost_history) > 10000:
            self.cost_history = self.cost_history[-10000:]
    
    def suggest_optimizations(
        self,
        operation_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Sugerir optimizaciones basadas en costos."""
        suggestions = []
        
        # Filtrar por tipo de operación
        costs = self.cost_history
        if operation_type:
            costs = [c for c in costs if c["operation_type"] == operation_type]
        
        if not costs:
            return suggestions
        
        # Analizar costos
        avg_cost_per_item = sum(c["cost_per_item"] for c in costs) / len(costs)
        avg_cost_per_second = sum(c["cost_per_second"] for c in costs) / len(costs)
        
        # Sugerir optimizaciones
        high_cost_ops = [
            c for c in costs
            if c["cost_per_item"] > avg_cost_per_item * 1.5
        ]
        
        if high_cost_ops:
            suggestions.append({
                "type": "reduce_batch_size",
                "reason": f"High cost per item detected: {avg_cost_per_item:.4f}",
                "recommendation": "Consider reducing batch size to improve efficiency",
                "potential_savings": "20-30%"
            })
        
        slow_ops = [
            c for c in costs
            if c["cost_per_second"] < avg_cost_per_second * 0.5
        ]
        
        if slow_ops:
            suggestions.append({
                "type": "increase_parallelism",
                "reason": f"Low throughput detected: {avg_cost_per_second:.4f} ops/sec",
                "recommendation": "Consider increasing parallelism to improve throughput",
                "potential_savings": "30-40%"
            })
        
        return suggestions
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Obtener resumen de costos."""
        if not self.cost_history:
            return {"message": "No cost data available"}
        
        total_cost = sum(c["cost"] for c in self.cost_history)
        total_items = sum(c["items_processed"] for c in self.cost_history)
        total_duration = sum(c["duration"] for c in self.cost_history)
        
        cost_by_operation = {}
        for cost_record in self.cost_history:
            op_type = cost_record["operation_type"]
            if op_type not in cost_by_operation:
                cost_by_operation[op_type] = {
                    "total_cost": 0,
                    "total_items": 0,
                    "count": 0
                }
            cost_by_operation[op_type]["total_cost"] += cost_record["cost"]
            cost_by_operation[op_type]["total_items"] += cost_record["items_processed"]
            cost_by_operation[op_type]["count"] += 1
        
        return {
            "total_cost": round(total_cost, 2),
            "total_items_processed": total_items,
            "total_duration": round(total_duration, 2),
            "avg_cost_per_item": round(total_cost / total_items, 4) if total_items > 0 else 0,
            "avg_cost_per_second": round(total_cost / total_duration, 4) if total_duration > 0 else 0,
            "by_operation": {
                op: {
                    "total_cost": round(data["total_cost"], 2),
                    "avg_cost_per_item": round(
                        data["total_cost"] / data["total_items"], 4
                    ) if data["total_items"] > 0 else 0,
                    "operations": data["count"]
                }
                for op, data in cost_by_operation.items()
            }
        }


class BulkAnomalyDetector:
    """Sistema de detección de anomalías en operaciones bulk."""
    
    def __init__(self, threshold_std: float = 2.0):
        self.threshold_std = threshold_std
        self.metrics_history: Dict[str, List[float]] = {}
        self.anomalies: List[Dict[str, Any]] = []
    
    def record_metric(
        self,
        metric_name: str,
        value: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Registrar métrica y detectar anomalías."""
        if metric_name not in self.metrics_history:
            self.metrics_history[metric_name] = []
        
        self.metrics_history[metric_name].append(value)
        
        # Mantener solo últimas 1000 valores
        if len(self.metrics_history[metric_name]) > 1000:
            self.metrics_history[metric_name] = self.metrics_history[metric_name][-1000:]
        
        # Detectar anomalía si hay suficientes datos
        if len(self.metrics_history[metric_name]) >= 10:
            is_anomaly, anomaly_score = self._detect_anomaly(metric_name, value)
            
            if is_anomaly:
                anomaly = {
                    "metric_name": metric_name,
                    "value": value,
                    "anomaly_score": anomaly_score,
                    "timestamp": datetime.now().isoformat(),
                    "metadata": metadata or {}
                }
                self.anomalies.append(anomaly)
                
                # Limitar tamaño
                if len(self.anomalies) > 1000:
                    self.anomalies = self.anomalies[-1000:]
    
    def _detect_anomaly(
        self,
        metric_name: str,
        value: float
    ) -> Tuple[bool, float]:
        """Detectar si un valor es anómalo."""
        history = self.metrics_history[metric_name]
        
        if len(history) < 10:
            return False, 0.0
        
        # Calcular media y desviación estándar
        mean = sum(history[:-1]) / len(history[:-1])  # Excluir valor actual
        variance = sum((x - mean) ** 2 for x in history[:-1]) / len(history[:-1])
        std_dev = variance ** 0.5
        
        if std_dev == 0:
            return False, 0.0
        
        # Calcular z-score
        z_score = abs((value - mean) / std_dev)
        
        # Es anomalía si está fuera de threshold_std desviaciones estándar
        is_anomaly = z_score > self.threshold_std
        
        return is_anomaly, z_score
    
    def get_anomalies(
        self,
        metric_name: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Obtener anomalías detectadas."""
        anomalies = self.anomalies
        
        if metric_name:
            anomalies = [a for a in anomalies if a["metric_name"] == metric_name]
        
        return anomalies[-limit:] if limit else anomalies
    
    def get_anomaly_summary(self) -> Dict[str, Any]:
        """Obtener resumen de anomalías."""
        if not self.anomalies:
            return {"message": "No anomalies detected"}
        
        # Agrupar por métrica
        by_metric = {}
        for anomaly in self.anomalies:
            metric = anomaly["metric_name"]
            if metric not in by_metric:
                by_metric[metric] = []
            by_metric[metric].append(anomaly)
        
        return {
            "total_anomalies": len(self.anomalies),
            "by_metric": {
                metric: {
                    "count": len(anomalies),
                    "latest": anomalies[-1] if anomalies else None,
                    "avg_score": sum(a["anomaly_score"] for a in anomalies) / len(anomalies)
                }
                for metric, anomalies in by_metric.items()
            },
            "recent_anomalies": self.anomalies[-10:]
        }


# ============================================================================
# SISTEMAS ENTERPRISE Y AVANZADOS
# ============================================================================

class BulkWorkflowEngine:
    """Motor de workflows para operaciones bulk complejas."""
    
    def __init__(self):
        self.workflows: Dict[str, Dict[str, Any]] = {}
        self.executions: Dict[str, Dict[str, Any]] = {}
    
    def register_workflow(
        self,
        workflow_id: str,
        steps: List[Dict[str, Any]],
        name: str = "",
        description: str = ""
    ):
        """Registrar workflow."""
        self.workflows[workflow_id] = {
            "workflow_id": workflow_id,
            "name": name or workflow_id,
            "description": description,
            "steps": steps,
            "created_at": datetime.now().isoformat()
        }
    
    async def execute_workflow(
        self,
        workflow_id: str,
        initial_data: Dict[str, Any],
        execution_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Ejecutar workflow."""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.workflows[workflow_id]
        exec_id = execution_id or f"exec_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        execution = {
            "execution_id": exec_id,
            "workflow_id": workflow_id,
            "status": "running",
            "current_step": 0,
            "data": initial_data,
            "steps_completed": [],
            "steps_failed": [],
            "started_at": datetime.now(),
            "completed_at": None
        }
        
        self.executions[exec_id] = execution
        
        try:
            for i, step in enumerate(workflow["steps"]):
                execution["current_step"] = i
                step_type = step.get("type", "operation")
                step_name = step.get("name", f"step_{i}")
                operation = step.get("operation")
                condition = step.get("condition")
                
                # Verificar condición si existe
                if condition:
                    if not self._evaluate_condition(condition, execution["data"]):
                        continue
                
                # Ejecutar operación
                if operation:
                    if asyncio.iscoroutinefunction(operation):
                        result = await operation(execution["data"])
                    else:
                        result = operation(execution["data"])
                    
                    execution["data"].update(result or {})
                
                execution["steps_completed"].append({
                    "step": i,
                    "name": step_name,
                    "type": step_type
                })
            
            execution["status"] = "completed"
            execution["completed_at"] = datetime.now()
            
        except Exception as e:
            execution["status"] = "failed"
            execution["error"] = str(e)
            execution["completed_at"] = datetime.now()
            raise
        
        return execution
    
    def _evaluate_condition(self, condition: Dict[str, Any], data: Dict[str, Any]) -> bool:
        """Evaluar condición."""
        condition_type = condition.get("type", "equals")
        field = condition.get("field")
        value = condition.get("value")
        
        if field not in data:
            return False
        
        field_value = data[field]
        
        if condition_type == "equals":
            return field_value == value
        elif condition_type == "greater_than":
            return field_value > value
        elif condition_type == "less_than":
            return field_value < value
        elif condition_type == "contains":
            return value in field_value if isinstance(field_value, (list, str)) else False
        
        return False
    
    def get_workflow(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Obtener workflow."""
        return self.workflows.get(workflow_id)
    
    def get_execution(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Obtener ejecución."""
        return self.executions.get(execution_id)


class BulkMultiTenancy:
    """Sistema de multi-tenancy para operaciones bulk."""
    
    def __init__(self):
        self.tenants: Dict[str, Dict[str, Any]] = {}
        self.tenant_resources: Dict[str, Dict[str, Any]] = {}
    
    def register_tenant(
        self,
        tenant_id: str,
        name: str,
        config: Optional[Dict[str, Any]] = None
    ):
        """Registrar tenant."""
        self.tenants[tenant_id] = {
            "tenant_id": tenant_id,
            "name": name,
            "config": config or {},
            "created_at": datetime.now().isoformat(),
            "status": "active"
        }
        
        # Inicializar recursos del tenant
        self.tenant_resources[tenant_id] = {
            "quota": config.get("quota", {}),
            "usage": {
                "sessions": 0,
                "operations": 0,
                "storage": 0
            },
            "limits": config.get("limits", {})
        }
    
    def check_quota(
        self,
        tenant_id: str,
        resource_type: str,
        amount: int = 1
    ) -> Tuple[bool, Optional[str]]:
        """Verificar si tenant tiene quota disponible."""
        if tenant_id not in self.tenant_resources:
            return False, "Tenant not found"
        
        resources = self.tenant_resources[tenant_id]
        quota = resources["quota"].get(resource_type, float('inf'))
        usage = resources["usage"].get(resource_type, 0)
        
        if usage + amount > quota:
            return False, f"Quota exceeded for {resource_type}"
        
        return True, None
    
    def record_usage(
        self,
        tenant_id: str,
        resource_type: str,
        amount: int = 1
    ):
        """Registrar uso de recursos."""
        if tenant_id in self.tenant_resources:
            resources = self.tenant_resources[tenant_id]
            resources["usage"][resource_type] = resources["usage"].get(resource_type, 0) + amount
    
    def get_tenant_stats(self, tenant_id: str) -> Optional[Dict[str, Any]]:
        """Obtener estadísticas del tenant."""
        if tenant_id not in self.tenants:
            return None
        
        tenant = self.tenants[tenant_id]
        resources = self.tenant_resources.get(tenant_id, {})
        
        return {
            "tenant": tenant,
            "quota": resources.get("quota", {}),
            "usage": resources.get("usage", {}),
            "limits": resources.get("limits", {}),
            "utilization": {
                resource: (
                    (usage / quota * 100) if quota != float('inf') and quota > 0 else 0
                )
                for resource, usage in resources.get("usage", {}).items()
                for quota in [resources.get("quota", {}).get(resource, float('inf'))]
            }
        }


class BulkDisasterRecovery:
    """Sistema de disaster recovery para operaciones bulk."""
    
    def __init__(self, backup_interval_minutes: int = 60):
        self.backup_interval = backup_interval_minutes
        self.checkpoints: Dict[str, Dict[str, Any]] = {}
        self.recovery_points: List[Dict[str, Any]] = []
        self.last_backup: Optional[datetime] = None
    
    def create_checkpoint(
        self,
        checkpoint_id: str,
        state: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Crear checkpoint de estado."""
        checkpoint = {
            "checkpoint_id": checkpoint_id,
            "state": state,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        }
        
        self.checkpoints[checkpoint_id] = checkpoint
        self.recovery_points.append(checkpoint)
        
        # Mantener solo últimos 100 checkpoints
        if len(self.recovery_points) > 100:
            self.recovery_points = self.recovery_points[-100:]
        
        self.last_backup = datetime.now()
    
    def get_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Obtener checkpoint."""
        return self.checkpoints.get(checkpoint_id)
    
    def get_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Obtener último checkpoint."""
        if not self.recovery_points:
            return None
        return self.recovery_points[-1]
    
    async def restore_from_checkpoint(
        self,
        checkpoint_id: str,
        restore_handler: Callable
    ) -> Dict[str, Any]:
        """Restaurar desde checkpoint."""
        checkpoint = self.get_checkpoint(checkpoint_id)
        if not checkpoint:
            raise ValueError(f"Checkpoint {checkpoint_id} not found")
        
        if asyncio.iscoroutinefunction(restore_handler):
            result = await restore_handler(checkpoint["state"])
        else:
            result = restore_handler(checkpoint["state"])
        
        return {
            "checkpoint_id": checkpoint_id,
            "restored_at": datetime.now().isoformat(),
            "result": result
        }
    
    def get_recovery_status(self) -> Dict[str, Any]:
        """Obtener estado de recovery."""
        return {
            "total_checkpoints": len(self.checkpoints),
            "last_backup": self.last_backup.isoformat() if self.last_backup else None,
            "backup_interval_minutes": self.backup_interval,
            "time_since_last_backup": (
                (datetime.now() - self.last_backup).total_seconds() / 60
                if self.last_backup else None
            ),
            "latest_checkpoint": self.get_latest_checkpoint()
        }


class BulkComplianceAudit:
    """Sistema de compliance y auditoría avanzado."""
    
    def __init__(self):
        self.audit_logs: List[Dict[str, Any]] = []
        self.compliance_rules: Dict[str, List[Callable]] = {}
        self.violations: List[Dict[str, Any]] = []
    
    def add_compliance_rule(
        self,
        rule_id: str,
        rule_name: str,
        validator: Callable,
        severity: str = "medium"
    ):
        """Añadir regla de compliance."""
        if rule_id not in self.compliance_rules:
            self.compliance_rules[rule_id] = []
        
        self.compliance_rules[rule_id].append({
            "rule_id": rule_id,
            "rule_name": rule_name,
            "validator": validator,
            "severity": severity
        })
    
    def audit_operation(
        self,
        operation_type: str,
        user_id: str,
        details: Dict[str, Any],
        result: Optional[Dict[str, Any]] = None
    ):
        """Registrar operación en auditoría."""
        audit_entry = {
            "operation_type": operation_type,
            "user_id": user_id,
            "details": details,
            "result": result,
            "timestamp": datetime.now().isoformat(),
            "compliance_status": "unknown"
        }
        
        # Verificar compliance
        violations = self._check_compliance(operation_type, details)
        if violations:
            audit_entry["compliance_status"] = "violation"
            audit_entry["violations"] = violations
            self.violations.extend(violations)
        else:
            audit_entry["compliance_status"] = "compliant"
        
        self.audit_logs.append(audit_entry)
        
        # Limitar tamaño
        if len(self.audit_logs) > 100000:
            self.audit_logs = self.audit_logs[-100000:]
    
    def _check_compliance(
        self,
        operation_type: str,
        details: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Verificar compliance."""
        violations = []
        
        for rule_id, rules in self.compliance_rules.items():
            for rule in rules:
                try:
                    validator = rule["validator"]
                    if asyncio.iscoroutinefunction(validator):
                        # Para async, necesitaríamos await
                        is_compliant = validator(details)
                    else:
                        is_compliant = validator(details)
                    
                    if not is_compliant:
                        violations.append({
                            "rule_id": rule_id,
                            "rule_name": rule["rule_name"],
                            "severity": rule["severity"],
                            "operation_type": operation_type,
                            "timestamp": datetime.now().isoformat()
                        })
                except Exception as e:
                    logger.error(f"Error checking compliance rule {rule_id}: {e}")
        
        return violations
    
    def get_audit_logs(
        self,
        user_id: Optional[str] = None,
        operation_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Obtener logs de auditoría."""
        logs = self.audit_logs
        
        if user_id:
            logs = [l for l in logs if l["user_id"] == user_id]
        if operation_type:
            logs = [l for l in logs if l["operation_type"] == operation_type]
        
        return logs[-limit:] if limit else logs
    
    def get_compliance_report(self) -> Dict[str, Any]:
        """Obtener reporte de compliance."""
        total_operations = len(self.audit_logs)
        compliant_ops = sum(1 for l in self.audit_logs if l["compliance_status"] == "compliant")
        violations = len(self.violations)
        
        return {
            "total_operations": total_operations,
            "compliant_operations": compliant_ops,
            "violations": violations,
            "compliance_rate": (
                (compliant_ops / total_operations * 100) if total_operations > 0 else 100
            ),
            "recent_violations": self.violations[-20:],
            "rules_count": sum(len(rules) for rules in self.compliance_rules.values())
        }


class BulkMLOptimizer:
    """Sistema de optimización basado en machine learning."""
    
    def __init__(self):
        self.models: Dict[str, Dict[str, Any]] = {}
        self.training_data: Dict[str, List[Dict[str, Any]]] = {}
        self.predictions: Dict[str, List[Dict[str, Any]]] = {}
    
    def record_training_data(
        self,
        model_name: str,
        features: Dict[str, float],
        target: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Registrar datos de entrenamiento."""
        if model_name not in self.training_data:
            self.training_data[model_name] = []
        
        self.training_data[model_name].append({
            "features": features,
            "target": target,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        })
        
        # Limitar tamaño
        if len(self.training_data[model_name]) > 10000:
            self.training_data[model_name] = self.training_data[model_name][-10000:]
    
    def train_model(
        self,
        model_name: str,
        model_type: str = "linear_regression"
    ) -> Dict[str, Any]:
        """Entrenar modelo (simplificado - implementación básica)."""
        if model_name not in self.training_data:
            return {"error": "No training data available"}
        
        data = self.training_data[model_name]
        if len(data) < 10:
            return {"error": "Insufficient training data"}
        
        # Modelo simplificado (en producción usarías scikit-learn, etc.)
        # Calcular promedio y tendencia básica
        targets = [d["target"] for d in data]
        avg_target = sum(targets) / len(targets)
        
        # Calcular pesos básicos para features
        feature_weights = {}
        if data:
            sample_features = data[0]["features"]
            for feature in sample_features.keys():
                # Correlación simple (simplificado)
                feature_weights[feature] = 1.0 / len(sample_features)
        
        self.models[model_name] = {
            "model_name": model_name,
            "model_type": model_type,
            "avg_target": avg_target,
            "feature_weights": feature_weights,
            "trained_at": datetime.now().isoformat(),
            "training_samples": len(data)
        }
        
        return {
            "model_name": model_name,
            "status": "trained",
            "training_samples": len(data),
            "avg_target": avg_target
        }
    
    def predict(
        self,
        model_name: str,
        features: Dict[str, float]
    ) -> Dict[str, Any]:
        """Hacer predicción."""
        if model_name not in self.models:
            return {"error": "Model not trained"}
        
        model = self.models[model_name]
        
        # Predicción simplificada
        prediction = model["avg_target"]
        for feature, value in features.items():
            weight = model["feature_weights"].get(feature, 0)
            prediction += weight * value
        
        prediction_record = {
            "model_name": model_name,
            "features": features,
            "prediction": prediction,
            "timestamp": datetime.now().isoformat()
        }
        
        if model_name not in self.predictions:
            self.predictions[model_name] = []
        self.predictions[model_name].append(prediction_record)
        
        return prediction_record
    
    def get_model_stats(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Obtener estadísticas del modelo."""
        if model_name not in self.models:
            return None
        
        model = self.models[model_name]
        training_count = len(self.training_data.get(model_name, []))
        prediction_count = len(self.predictions.get(model_name, []))
        
        return {
            **model,
            "training_samples_count": training_count,
            "predictions_count": prediction_count
        }


# ============================================================================
# SISTEMAS AVANZADOS DE OPTIMIZACIÓN Y GESTIÓN
# ============================================================================

class BulkDataPipeline:
    """Pipeline de procesamiento de datos para operaciones bulk."""
    
    def __init__(self, stages: Optional[List[Callable]] = None):
        self.stages = stages or []
        self.pipeline_history: List[Dict[str, Any]] = []
    
    def add_stage(self, stage: Callable, name: Optional[str] = None):
        """Agregar etapa al pipeline."""
        self.stages.append({
            "name": name or stage.__name__,
            "function": stage
        })
        return self
    
    async def execute(
        self,
        data: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Ejecutar pipeline completo."""
        context = context or {}
        result = data
        execution_log = []
        
        for i, stage_info in enumerate(self.stages):
            stage_name = stage_info["name"]
            stage_func = stage_info["function"]
            
            start_time = time.time()
            try:
                if asyncio.iscoroutinefunction(stage_func):
                    result = await stage_func(result, context)
                else:
                    result = stage_func(result, context)
                
                duration = time.time() - start_time
                execution_log.append({
                    "stage": stage_name,
                    "status": "success",
                    "duration": duration,
                    "input_size": len(str(result)) if isinstance(result, (list, dict, str)) else 1,
                    "output_size": len(str(result)) if isinstance(result, (list, dict, str)) else 1
                })
            except Exception as e:
                duration = time.time() - start_time
                execution_log.append({
                    "stage": stage_name,
                    "status": "error",
                    "error": str(e),
                    "duration": duration
                })
                raise
        
        self.pipeline_history.append({
            "timestamp": datetime.now().isoformat(),
            "stages": execution_log,
            "total_duration": sum(s["duration"] for s in execution_log)
        })
        
        return result
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del pipeline."""
        if not self.pipeline_history:
            return {"message": "No pipeline executions yet"}
        
        recent = self.pipeline_history[-100:]  # Últimas 100 ejecuciones
        
        stage_stats = {}
        for execution in recent:
            for stage_log in execution["stages"]:
                stage_name = stage_log["stage"]
                if stage_name not in stage_stats:
                    stage_stats[stage_name] = {
                        "executions": 0,
                        "successes": 0,
                        "errors": 0,
                        "total_duration": 0.0,
                        "avg_duration": 0.0
                    }
                
                stage_stats[stage_name]["executions"] += 1
                if stage_log["status"] == "success":
                    stage_stats[stage_name]["successes"] += 1
                else:
                    stage_stats[stage_name]["errors"] += 1
                stage_stats[stage_name]["total_duration"] += stage_log["duration"]
        
        # Calcular promedios
        for stats in stage_stats.values():
            if stats["executions"] > 0:
                stats["avg_duration"] = stats["total_duration"] / stats["executions"]
                stats["success_rate"] = stats["successes"] / stats["executions"]
        
        return {
            "total_executions": len(recent),
            "stage_stats": stage_stats,
            "avg_total_duration": sum(e["total_duration"] for e in recent) / len(recent) if recent else 0
        }


class BulkIntelligentScheduler:
    """Scheduler inteligente que aprende y optimiza basado en patrones."""
    
    def __init__(self):
        self.scheduled_jobs: Dict[str, Dict[str, Any]] = {}
        self.execution_history: List[Dict[str, Any]] = []
        self.pattern_analyzer: Dict[str, Any] = {}
        self.learning_enabled = True
    
    async def schedule_job(
        self,
        job_id: str,
        job_func: Callable,
        schedule_config: Dict[str, Any],
        priority: int = 5
    ) -> str:
        """Programar trabajo con configuración inteligente."""
        # Analizar patrones históricos
        optimal_time = None
        if self.learning_enabled and job_id in self.execution_history:
            optimal_time = self._find_optimal_time(job_id)
        
        schedule = {
            "job_id": job_id,
            "job_func": job_func,
            "schedule_config": schedule_config,
            "priority": priority,
            "optimal_time": optimal_time,
            "created_at": datetime.now().isoformat(),
            "next_execution": self._calculate_next_execution(schedule_config, optimal_time),
            "execution_count": 0,
            "success_count": 0,
            "failure_count": 0
        }
        
        self.scheduled_jobs[job_id] = schedule
        return job_id
    
    def _find_optimal_time(self, job_id: str) -> Optional[datetime]:
        """Encontrar tiempo óptimo basado en historial."""
        history = [h for h in self.execution_history if h.get("job_id") == job_id]
        
        if len(history) < 10:
            return None
        
        # Analizar tiempos de ejecución exitosos
        successful_runs = [h for h in history if h.get("status") == "success"]
        
        if not successful_runs:
            return None
        
        # Agrupar por hora del día
        hour_performance = {}
        for run in successful_runs:
            hour = run["timestamp"].hour
            if hour not in hour_performance:
                hour_performance[hour] = {"count": 0, "total_duration": 0.0}
            
            hour_performance[hour]["count"] += 1
            hour_performance[hour]["total_duration"] += run.get("duration", 0.0)
        
        # Encontrar hora con mejor performance (menor duración promedio)
        best_hour = min(
            hour_performance.items(),
            key=lambda x: x[1]["total_duration"] / x[1]["count"]
        )[0]
        
        return datetime.now().replace(hour=best_hour, minute=0, second=0)
    
    def _calculate_next_execution(
        self,
        schedule_config: Dict[str, Any],
        optimal_time: Optional[datetime]
    ) -> datetime:
        """Calcular próxima ejecución."""
        if optimal_time:
            return optimal_time
        
        schedule_type = schedule_config.get("type", "interval")
        
        if schedule_type == "interval":
            interval = schedule_config.get("interval_seconds", 3600)
            return datetime.now() + timedelta(seconds=interval)
        elif schedule_type == "cron":
            # Parsear cron expression (simplificado)
            cron_expr = schedule_config.get("cron", "0 * * * *")
            return self._parse_cron_next(cron_expr)
        else:
            return datetime.now() + timedelta(hours=1)
    
    def _parse_cron_next(self, cron_expr: str) -> datetime:
        """Parsear expresión cron y calcular próxima ejecución (simplificado)."""
        # Implementación básica
        parts = cron_expr.split()
        if len(parts) >= 5:
            minute, hour, day, month, weekday = parts[:5]
            
            now = datetime.now()
            next_time = now.replace(second=0, microsecond=0)
            
            if minute != "*":
                next_time = next_time.replace(minute=int(minute))
            if hour != "*":
                next_time = next_time.replace(hour=int(hour))
            
            # Si ya pasó el tiempo hoy, programar para mañana
            if next_time <= now:
                next_time += timedelta(days=1)
            
            return next_time
        
        return datetime.now() + timedelta(hours=1)
    
    async def execute_job(self, job_id: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Ejecutar trabajo programado."""
        if job_id not in self.scheduled_jobs:
            raise ValueError(f"Job {job_id} not found")
        
        job = self.scheduled_jobs[job_id]
        job_func = job["job_func"]
        context = context or {}
        
        start_time = time.time()
        try:
            if asyncio.iscoroutinefunction(job_func):
                result = await job_func(context)
            else:
                result = job_func(context)
            
            duration = time.time() - start_time
            
            # Actualizar estadísticas
            job["execution_count"] += 1
            job["success_count"] += 1
            job["next_execution"] = self._calculate_next_execution(
                job["schedule_config"],
                job.get("optimal_time")
            )
            
            # Registrar en historial
            self.execution_history.append({
                "job_id": job_id,
                "status": "success",
                "duration": duration,
                "timestamp": datetime.now(),
                "result": result
            })
            
            return {
                "status": "success",
                "duration": duration,
                "result": result
            }
        except Exception as e:
            duration = time.time() - start_time
            
            job["execution_count"] += 1
            job["failure_count"] += 1
            
            self.execution_history.append({
                "job_id": job_id,
                "status": "error",
                "duration": duration,
                "timestamp": datetime.now(),
                "error": str(e)
            })
            
            raise
    
    def get_job_recommendations(self, job_id: str) -> Dict[str, Any]:
        """Obtener recomendaciones para un trabajo."""
        if job_id not in self.scheduled_jobs:
            return {"message": "Job not found"}
        
        job = self.scheduled_jobs[job_id]
        history = [h for h in self.execution_history if h.get("job_id") == job_id]
        
        if not history:
            return {"message": "No execution history"}
        
        # Calcular métricas
        total_executions = len(history)
        success_rate = sum(1 for h in history if h.get("status") == "success") / total_executions
        avg_duration = sum(h.get("duration", 0) for h in history) / total_executions
        
        recommendations = []
        
        # Recomendación de frecuencia
        if success_rate < 0.8:
            recommendations.append({
                "type": "frequency",
                "message": "Consider reducing execution frequency due to low success rate",
                "current_rate": success_rate,
                "suggested_interval": job["schedule_config"].get("interval_seconds", 3600) * 2
            })
        
        # Recomendación de tiempo óptimo
        optimal_time = self._find_optimal_time(job_id)
        if optimal_time:
            recommendations.append({
                "type": "timing",
                "message": f"Optimal execution time: {optimal_time.hour}:00",
                "optimal_hour": optimal_time.hour
            })
        
        # Recomendación de prioridad
        if avg_duration > 300:  # Más de 5 minutos
            recommendations.append({
                "type": "priority",
                "message": "Consider increasing priority due to long execution time",
                "current_duration": avg_duration
            })
        
        return {
            "job_id": job_id,
            "stats": {
                "total_executions": total_executions,
                "success_rate": success_rate,
                "avg_duration": avg_duration,
                "failure_rate": 1 - success_rate
            },
            "recommendations": recommendations
        }


class BulkResourceOptimizer:
    """Optimizador de recursos para operaciones bulk."""
    
    def __init__(self):
        self.resource_usage: Dict[str, List[Dict[str, Any]]] = {}
        self.optimization_history: List[Dict[str, Any]] = {}
        self.current_allocations: Dict[str, Any] = {}
    
    def track_resource_usage(
        self,
        resource_type: str,
        operation_id: str,
        usage: Dict[str, float],
        timestamp: Optional[datetime] = None
    ):
        """Rastrear uso de recursos."""
        if resource_type not in self.resource_usage:
            self.resource_usage[resource_type] = []
        
        self.resource_usage[resource_type].append({
            "operation_id": operation_id,
            "usage": usage,
            "timestamp": timestamp or datetime.now()
        })
        
        # Mantener solo últimos 10000 registros
        if len(self.resource_usage[resource_type]) > 10000:
            self.resource_usage[resource_type] = self.resource_usage[resource_type][-10000:]
    
    def analyze_resource_efficiency(
        self,
        resource_type: str,
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """Analizar eficiencia de uso de recursos."""
        if resource_type not in self.resource_usage:
            return {"message": f"No data for resource type: {resource_type}"}
        
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        recent_usage = [
            u for u in self.resource_usage[resource_type]
            if u["timestamp"] >= cutoff_time
        ]
        
        if not recent_usage:
            return {"message": "No recent usage data"}
        
        # Calcular métricas
        total_usage = sum(sum(u["usage"].values()) for u in recent_usage)
        avg_usage = total_usage / len(recent_usage)
        
        # Identificar peaks y valleys
        usage_values = [sum(u["usage"].values()) for u in recent_usage]
        peak_usage = max(usage_values)
        min_usage = min(usage_values)
        
        # Calcular eficiencia (ratio de uso promedio vs peak)
        efficiency = (avg_usage / peak_usage) if peak_usage > 0 else 0.0
        
        # Recomendaciones
        recommendations = []
        
        if efficiency < 0.5:
            recommendations.append({
                "type": "over_provisioning",
                "message": "Resource may be over-provisioned",
                "efficiency": efficiency,
                "suggestion": "Consider reducing allocation"
            })
        
        if peak_usage > avg_usage * 2:
            recommendations.append({
                "type": "spike_handling",
                "message": "Significant usage spikes detected",
                "peak_ratio": peak_usage / avg_usage,
                "suggestion": "Consider implementing auto-scaling"
            })
        
        return {
            "resource_type": resource_type,
            "time_window_hours": time_window_hours,
            "metrics": {
                "total_operations": len(recent_usage),
                "avg_usage": avg_usage,
                "peak_usage": peak_usage,
                "min_usage": min_usage,
                "efficiency": efficiency
            },
            "recommendations": recommendations
        }
    
    def optimize_allocation(
        self,
        resource_type: str,
        target_efficiency: float = 0.8,
        max_allocation: Optional[float] = None
    ) -> Dict[str, Any]:
        """Optimizar asignación de recursos."""
        analysis = self.analyze_resource_efficiency(resource_type)
        
        if "message" in analysis:
            return analysis
        
        current_avg = analysis["metrics"]["avg_usage"]
        current_peak = analysis["metrics"]["peak_usage"]
        
        # Calcular asignación óptima
        # Usar peak * 1.2 como buffer, pero no más que max_allocation
        optimal_allocation = current_peak * 1.2
        
        if max_allocation:
            optimal_allocation = min(optimal_allocation, max_allocation)
        
        # Calcular ahorro potencial
        if resource_type in self.current_allocations:
            current_allocation = self.current_allocations[resource_type]
            potential_savings = current_allocation - optimal_allocation
            savings_percentage = (potential_savings / current_allocation) * 100 if current_allocation > 0 else 0
        else:
            current_allocation = None
            potential_savings = 0
            savings_percentage = 0
        
        optimization = {
            "resource_type": resource_type,
            "current_allocation": current_allocation,
            "recommended_allocation": optimal_allocation,
            "potential_savings": potential_savings,
            "savings_percentage": savings_percentage,
            "target_efficiency": target_efficiency,
            "estimated_efficiency": (current_avg / optimal_allocation) if optimal_allocation > 0 else 0
        }
        
        # Guardar en historial
        self.optimization_history[resource_type] = optimization
        
        return optimization
    
    def apply_optimization(self, resource_type: str) -> bool:
        """Aplicar optimización recomendada."""
        if resource_type not in self.optimization_history:
            return False
        
        optimization = self.optimization_history[resource_type]
        
        # Aplicar (esto dependería de la implementación real)
        self.current_allocations[resource_type] = optimization["recommended_allocation"]
        
        return True


class BulkDataQualityChecker:
    """Verificador de calidad de datos para operaciones bulk."""
    
    def __init__(self):
        self.quality_rules: Dict[str, List[Callable]] = {}
        self.quality_metrics: Dict[str, List[Dict[str, Any]]] = {}
        self.quality_history: List[Dict[str, Any]] = []
    
    def add_quality_rule(
        self,
        rule_name: str,
        rule_func: Callable,
        data_type: str = "general"
    ):
        """Agregar regla de calidad."""
        if data_type not in self.quality_rules:
            self.quality_rules[data_type] = []
        
        self.quality_rules[data_type].append({
            "name": rule_name,
            "function": rule_func
        })
    
    async def check_quality(
        self,
        data: Any,
        data_type: str = "general",
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Verificar calidad de datos."""
        context = context or {}
        rules = self.quality_rules.get(data_type, [])
        
        if not rules:
            return {
                "quality_score": 1.0,
                "message": "No quality rules defined",
                "checks": []
            }
        
        checks = []
        passed = 0
        failed = 0
        
        for rule_info in rules:
            rule_name = rule_info["name"]
            rule_func = rule_info["function"]
            
            try:
                if asyncio.iscoroutinefunction(rule_func):
                    result = await rule_func(data, context)
                else:
                    result = rule_func(data, context)
                
                is_valid = bool(result) if isinstance(result, (bool, int, float)) else result is not None
                
                checks.append({
                    "rule": rule_name,
                    "status": "passed" if is_valid else "failed",
                    "result": result
                })
                
                if is_valid:
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                checks.append({
                    "rule": rule_name,
                    "status": "error",
                    "error": str(e)
                })
                failed += 1
        
        total_checks = len(checks)
        quality_score = passed / total_checks if total_checks > 0 else 0.0
        
        quality_report = {
            "quality_score": quality_score,
            "total_checks": total_checks,
            "passed": passed,
            "failed": failed,
            "checks": checks,
            "timestamp": datetime.now().isoformat()
        }
        
        # Guardar en historial
        self.quality_history.append({
            **quality_report,
            "data_type": data_type,
            "data_sample": str(data)[:100] if isinstance(data, (str, list, dict)) else str(data)
        })
        
        # Mantener solo últimos 1000 registros
        if len(self.quality_history) > 1000:
            self.quality_history = self.quality_history[-1000:]
        
        return quality_report
    
    def get_quality_trends(
        self,
        data_type: Optional[str] = None,
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """Obtener tendencias de calidad."""
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        
        relevant_history = [
            h for h in self.quality_history
            if datetime.fromisoformat(h["timestamp"]) >= cutoff_time
        ]
        
        if data_type:
            relevant_history = [h for h in relevant_history if h.get("data_type") == data_type]
        
        if not relevant_history:
            return {"message": "No quality data in time window"}
        
        # Calcular tendencias
        scores = [h["quality_score"] for h in relevant_history]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        # Tendencias por regla
        rule_stats = {}
        for record in relevant_history:
            for check in record.get("checks", []):
                rule_name = check["rule"]
                if rule_name not in rule_stats:
                    rule_stats[rule_name] = {"passed": 0, "failed": 0, "errors": 0}
                
                if check["status"] == "passed":
                    rule_stats[rule_name]["passed"] += 1
                elif check["status"] == "failed":
                    rule_stats[rule_name]["failed"] += 1
                else:
                    rule_stats[rule_name]["errors"] += 1
        
        # Calcular tasas de éxito por regla
        rule_success_rates = {
            rule: stats["passed"] / (stats["passed"] + stats["failed"] + stats["errors"])
            if (stats["passed"] + stats["failed"] + stats["errors"]) > 0 else 0.0
            for rule, stats in rule_stats.items()
        }
        
        return {
            "time_window_hours": time_window_hours,
            "data_type": data_type or "all",
            "total_checks": len(relevant_history),
            "avg_quality_score": avg_score,
            "min_score": min(scores) if scores else 0.0,
            "max_score": max(scores) if scores else 0.0,
            "rule_statistics": rule_stats,
            "rule_success_rates": rule_success_rates,
            "trend": "improving" if len(scores) > 1 and scores[-1] > scores[0] else "declining"
        }


class BulkSmartBatching:
    """Sistema inteligente de batching que adapta tamaño según contexto."""
    
    def __init__(self):
        self.batch_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, List[float]] = {}
        self.optimal_batch_sizes: Dict[str, int] = {}
    
    async def process_with_smart_batching(
        self,
        items: List[Any],
        process_func: Callable,
        operation_type: str = "default",
        initial_batch_size: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Procesar items con batching inteligente."""
        context = context or {}
        
        # Determinar tamaño inicial de batch
        if initial_batch_size:
            batch_size = initial_batch_size
        elif operation_type in self.optimal_batch_sizes:
            batch_size = self.optimal_batch_sizes[operation_type]
        else:
            batch_size = self._calculate_initial_batch_size(items, operation_type)
        
        # Dividir en batches
        batches = chunk_list(items, batch_size)
        total_batches = len(batches)
        
        results = []
        total_processed = 0
        total_duration = 0.0
        batch_performances = []
        
        for i, batch in enumerate(batches):
            batch_start = time.time()
            
            try:
                # Procesar batch
                if asyncio.iscoroutinefunction(process_func):
                    batch_result = await process_func(batch, context)
                else:
                    batch_result = process_func(batch, context)
                
                batch_duration = time.time() - batch_start
                batch_performances.append({
                    "batch_num": i + 1,
                    "batch_size": len(batch),
                    "duration": batch_duration,
                    "items_per_second": len(batch) / batch_duration if batch_duration > 0 else 0,
                    "success": True
                })
                
                results.append(batch_result)
                total_processed += len(batch)
                total_duration += batch_duration
                
                # Ajustar tamaño de batch dinámicamente
                if i > 0 and i % 5 == 0:  # Cada 5 batches
                    recent_performance = batch_performances[-5:]
                    avg_throughput = sum(p["items_per_second"] for p in recent_performance) / len(recent_performance)
                    
                    # Si el throughput está mejorando, aumentar batch size
                    if len(batch_performances) >= 10:
                        prev_avg = sum(p["items_per_second"] for p in batch_performances[-10:-5]) / 5
                        if avg_throughput > prev_avg * 1.1:  # 10% mejor
                            batch_size = min(batch_size + 10, len(items) - total_processed)
                            logger.info(f"Increasing batch size to {batch_size} due to improved throughput")
                    elif avg_throughput < 10:  # Muy bajo throughput
                        batch_size = max(batch_size - 10, 1)
                        logger.info(f"Decreasing batch size to {batch_size} due to low throughput")
            except Exception as e:
                batch_duration = time.time() - batch_start
                batch_performances.append({
                    "batch_num": i + 1,
                    "batch_size": len(batch),
                    "duration": batch_duration,
                    "success": False,
                    "error": str(e)
                })
                logger.error(f"Error processing batch {i + 1}: {e}")
                # Reducir batch size después de error
                batch_size = max(batch_size - 5, 1)
        
        # Guardar en historial
        self.batch_history.append({
            "operation_type": operation_type,
            "total_items": len(items),
            "total_batches": total_batches,
            "final_batch_size": batch_size,
            "total_duration": total_duration,
            "avg_throughput": total_processed / total_duration if total_duration > 0 else 0,
            "batch_performances": batch_performances,
            "timestamp": datetime.now().isoformat()
        })
        
        # Actualizar tamaño óptimo
        if batch_performances:
            successful_batches = [p for p in batch_performances if p.get("success")]
            if successful_batches:
                best_batch = max(successful_batches, key=lambda x: x["items_per_second"])
                self.optimal_batch_sizes[operation_type] = best_batch["batch_size"]
        
        return {
            "total_items": len(items),
            "total_processed": total_processed,
            "total_batches": total_batches,
            "total_duration": total_duration,
            "avg_throughput": total_processed / total_duration if total_duration > 0 else 0,
            "results": results,
            "performance_summary": {
                "avg_batch_duration": sum(p["duration"] for p in batch_performances) / len(batch_performances),
                "best_batch_size": self.optimal_batch_sizes.get(operation_type),
                "success_rate": sum(1 for p in batch_performances if p.get("success")) / len(batch_performances)
            }
        }
    
    def _calculate_initial_batch_size(
        self,
        items: List[Any],
        operation_type: str
    ) -> int:
        """Calcular tamaño inicial de batch."""
        # Basado en tamaño de items
        if not items:
            return 100
        
        # Estimación simple basada en número de items
        if len(items) < 100:
            return max(1, len(items))
        elif len(items) < 1000:
            return 50
        elif len(items) < 10000:
            return 100
        else:
            return 200
    
    def get_batching_recommendations(
        self,
        operation_type: str
    ) -> Dict[str, Any]:
        """Obtener recomendaciones de batching."""
        relevant_history = [
            h for h in self.batch_history
            if h["operation_type"] == operation_type
        ]
        
        if not relevant_history:
            return {"message": "No history for this operation type"}
        
        # Analizar rendimiento por tamaño de batch
        batch_size_performance = {}
        for record in relevant_history:
            for perf in record.get("batch_performances", []):
                batch_size = perf["batch_size"]
                if batch_size not in batch_size_performance:
                    batch_size_performance[batch_size] = []
                
                if perf.get("success"):
                    batch_size_performance[batch_size].append(perf["items_per_second"])
        
        # Encontrar mejor tamaño
        best_batch_size = None
        best_avg_throughput = 0
        
        for batch_size, throughputs in batch_size_performance.items():
            if throughputs:
                avg_throughput = sum(throughputs) / len(throughputs)
                if avg_throughput > best_avg_throughput:
                    best_avg_throughput = avg_throughput
                    best_batch_size = batch_size
        
        return {
            "operation_type": operation_type,
            "recommended_batch_size": best_batch_size or self.optimal_batch_sizes.get(operation_type, 100),
            "expected_throughput": best_avg_throughput,
            "batch_size_analysis": {
                size: {
                    "avg_throughput": sum(throughputs) / len(throughputs) if throughputs else 0,
                    "samples": len(throughputs)
                }
                for size, throughputs in batch_size_performance.items()
            }
        }


class BulkDistributedCoordinator:
    """Coordinador distribuido para operaciones bulk en múltiples nodos."""
    
    def __init__(self, node_id: str, coordinator_url: Optional[str] = None):
        self.node_id = node_id
        self.coordinator_url = coordinator_url
        self.active_operations: Dict[str, Dict[str, Any]] = {}
        self.node_status: Dict[str, Any] = {
            "node_id": node_id,
            "status": "active",
            "capacity": 100,
            "current_load": 0,
            "last_heartbeat": datetime.now()
        }
    
    async def register_operation(
        self,
        operation_id: str,
        operation_config: Dict[str, Any],
        partition_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Registrar operación para coordinación distribuida."""
        partition = partition_key or operation_id
        
        operation = {
            "operation_id": operation_id,
            "node_id": self.node_id,
            "partition": partition,
            "config": operation_config,
            "status": "registered",
            "created_at": datetime.now().isoformat(),
            "progress": {
                "total": operation_config.get("total_items", 0),
                "processed": 0,
                "failed": 0
            }
        }
        
        self.active_operations[operation_id] = operation
        
        # Actualizar carga del nodo
        self.node_status["current_load"] += 1
        
        return operation
    
    async def update_operation_progress(
        self,
        operation_id: str,
        processed: int,
        failed: int = 0
    ):
        """Actualizar progreso de operación."""
        if operation_id not in self.active_operations:
            return
        
        operation = self.active_operations[operation_id]
        operation["progress"]["processed"] = processed
        operation["progress"]["failed"] = failed
        
        # Calcular porcentaje
        total = operation["progress"]["total"]
        if total > 0:
            operation["progress"]["percentage"] = (processed / total) * 100
        
        # Actualizar estado
        if processed >= total:
            operation["status"] = "completed"
            operation["completed_at"] = datetime.now().isoformat()
            self.node_status["current_load"] = max(0, self.node_status["current_load"] - 1)
    
    def get_node_status(self) -> Dict[str, Any]:
        """Obtener estado del nodo."""
        return {
            **self.node_status,
            "active_operations": len(self.active_operations),
            "available_capacity": self.node_status["capacity"] - self.node_status["current_load"]
        }
    
    def get_operation_status(self, operation_id: str) -> Optional[Dict[str, Any]]:
        """Obtener estado de operación."""
        return self.active_operations.get(operation_id)
    
    async def coordinate_distributed_operation(
        self,
        operation_id: str,
        items: List[Any],
        process_func: Callable,
        partition_strategy: str = "hash",
        num_partitions: int = 4
    ) -> Dict[str, Any]:
        """Coordinar operación distribuida."""
        # Dividir items en particiones
        if partition_strategy == "hash":
            partitions = self._hash_partition(items, num_partitions)
        elif partition_strategy == "round_robin":
            partitions = self._round_robin_partition(items, num_partitions)
        else:
            partitions = [items]  # Sin particionar
        
        # Procesar particiones
        partition_results = []
        for i, partition in enumerate(partitions):
            partition_id = f"{operation_id}_partition_{i}"
            
            try:
                if asyncio.iscoroutinefunction(process_func):
                    result = await process_func(partition, {"partition_id": partition_id})
                else:
                    result = process_func(partition, {"partition_id": partition_id})
                
                partition_results.append({
                    "partition_id": partition_id,
                    "items_processed": len(partition),
                    "status": "success",
                    "result": result
                })
            except Exception as e:
                partition_results.append({
                    "partition_id": partition_id,
                    "items_processed": 0,
                    "status": "error",
                    "error": str(e)
                })
        
        return {
            "operation_id": operation_id,
            "total_items": len(items),
            "partitions": len(partitions),
            "partition_results": partition_results,
            "success_rate": sum(1 for r in partition_results if r["status"] == "success") / len(partition_results) if partition_results else 0
        }
    
    def _hash_partition(self, items: List[Any], num_partitions: int) -> List[List[Any]]:
        """Particionar por hash."""
        partitions = [[] for _ in range(num_partitions)]
        
        for item in items:
            # Hash del item (simplificado)
            item_hash = hash(str(item)) if item is not None else 0
            partition_index = abs(item_hash) % num_partitions
            partitions[partition_index].append(item)
        
        return partitions
    
    def _round_robin_partition(self, items: List[Any], num_partitions: int) -> List[List[Any]]:
        """Particionar round-robin."""
        partitions = [[] for _ in range(num_partitions)]
        
        for i, item in enumerate(items):
            partition_index = i % num_partitions
            partitions[partition_index].append(item)
        
        return partitions


class BulkAdvancedAnalytics:
    """Analytics avanzado con machine learning para operaciones bulk."""
    
    def __init__(self):
        self.operation_patterns: Dict[str, List[Dict[str, Any]]] = {}
        self.predictions: Dict[str, Any] = {}
        self.ml_models: Dict[str, Any] = {}
    
    def record_operation_pattern(
        self,
        operation_type: str,
        pattern: Dict[str, Any]
    ):
        """Registrar patrón de operación."""
        if operation_type not in self.operation_patterns:
            self.operation_patterns[operation_type] = []
        
        self.operation_patterns[operation_type].append({
            **pattern,
            "timestamp": datetime.now().isoformat()
        })
        
        # Mantener solo últimos 1000 patrones por tipo
        if len(self.operation_patterns[operation_type]) > 1000:
            self.operation_patterns[operation_type] = self.operation_patterns[operation_type][-1000:]
    
    def predict_operation_duration(
        self,
        operation_type: str,
        item_count: int,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Predecir duración de operación."""
        context = context or {}
        
        if operation_type not in self.operation_patterns:
            return {
                "prediction": None,
                "confidence": 0.0,
                "message": "Insufficient data for prediction"
            }
        
        patterns = self.operation_patterns[operation_type]
        
        if len(patterns) < 10:
            return {
                "prediction": None,
                "confidence": 0.0,
                "message": "Need at least 10 historical patterns"
            }
        
        # Análisis simple basado en regresión lineal
        # En producción usaría un modelo ML real
        durations = [p.get("duration", 0) for p in patterns]
        item_counts = [p.get("item_count", 0) for p in patterns]
        
        if not durations or not item_counts:
            return {"prediction": None, "confidence": 0.0}
        
        # Calcular relación promedio
        avg_duration_per_item = sum(d / c for d, c in zip(durations, item_counts) if c > 0) / len([c for c in item_counts if c > 0])
        
        # Predecir
        predicted_duration = avg_duration_per_item * item_count
        
        # Calcular confianza basada en varianza
        variances = [abs(d / c - avg_duration_per_item) for d, c in zip(durations, item_counts) if c > 0]
        avg_variance = sum(variances) / len(variances) if variances else 0
        confidence = max(0.0, 1.0 - (avg_variance / avg_duration_per_item) if avg_duration_per_item > 0 else 0.0)
        
        prediction = {
            "predicted_duration": predicted_duration,
            "predicted_duration_seconds": predicted_duration,
            "confidence": min(confidence, 1.0),
            "based_on_samples": len(patterns),
            "item_count": item_count,
            "avg_duration_per_item": avg_duration_per_item
        }
        
        self.predictions[f"{operation_type}_{item_count}"] = prediction
        
        return prediction
    
    def detect_operation_anomalies(
        self,
        operation_type: str,
        threshold_std: float = 2.0
    ) -> List[Dict[str, Any]]:
        """Detectar anomalías en operaciones."""
        if operation_type not in self.operation_patterns:
            return []
        
        patterns = self.operation_patterns[operation_type]
        
        if len(patterns) < 20:
            return []
        
        # Calcular métricas
        durations = [p.get("duration", 0) for p in patterns]
        mean_duration = sum(durations) / len(durations)
        variance = sum((d - mean_duration) ** 2 for d in durations) / len(durations)
        std_dev = variance ** 0.5
        
        if std_dev == 0:
            return []
        
        # Detectar anomalías
        anomalies = []
        for i, pattern in enumerate(patterns):
            duration = pattern.get("duration", 0)
            z_score = abs((duration - mean_duration) / std_dev)
            
            if z_score > threshold_std:
                anomalies.append({
                    "pattern_index": i,
                    "duration": duration,
                    "z_score": z_score,
                    "pattern": pattern,
                    "timestamp": pattern.get("timestamp")
                })
        
        return anomalies
    
    def get_operation_insights(
        self,
        operation_type: str
    ) -> Dict[str, Any]:
        """Obtener insights sobre operaciones."""
        if operation_type not in self.operation_patterns:
            return {"message": "No data for this operation type"}
        
        patterns = self.operation_patterns[operation_type]
        
        if not patterns:
            return {"message": "No patterns recorded"}
        
        # Calcular estadísticas
        durations = [p.get("duration", 0) for p in patterns]
        item_counts = [p.get("item_count", 0) for p in patterns]
        success_rates = [p.get("success_rate", 1.0) for p in patterns]
        
        insights = {
            "operation_type": operation_type,
            "total_operations": len(patterns),
            "duration_stats": {
                "avg": sum(durations) / len(durations) if durations else 0,
                "min": min(durations) if durations else 0,
                "max": max(durations) if durations else 0,
                "p95": sorted(durations)[int(len(durations) * 0.95)] if durations else 0
            },
            "throughput_stats": {
                "avg_items_per_second": sum(c / d for c, d in zip(item_counts, durations) if d > 0) / len([d for d in durations if d > 0]) if durations else 0,
                "max_items_per_second": max(c / d for c, d in zip(item_counts, durations) if d > 0) if durations else 0
            },
            "reliability": {
                "avg_success_rate": sum(success_rates) / len(success_rates) if success_rates else 0,
                "min_success_rate": min(success_rates) if success_rates else 0
            },
            "trends": {
                "recent_avg_duration": sum(durations[-10:]) / len(durations[-10:]) if len(durations) >= 10 else sum(durations) / len(durations),
                "trend": "improving" if len(durations) >= 10 and durations[-1] < durations[-10] else "stable"
            }
        }
        
        return insights


# ============================================================================
# UTILIDADES FINALES Y EXPORTACIONES
# ============================================================================

__all__ = [
    # Decoradores
    "bulk_metrics_decorator",
    "bulk_cache_decorator",
    "bulk_retry_decorator",
    "bulk_rate_limit_decorator",
    
    # Funciones utilitarias
    "batch_process",
    "calculate_optimal_batch_size",
    "chunk_list",
    "retry_operation",
    "merge_bulk_results",
    
    # Clases principales
    "BulkOperationStatus",
    "BulkOperationResult",
    "BulkJob",
    "BulkSessionOperations",
    "BulkMessageOperations",
    "BulkExporter",
    "BulkAnalytics",
    "BulkCleanup",
    "BulkProcessor",
    "BulkImporter",
    "BulkNotifications",
    "BulkSearch",
    "BulkTesting",
    "BulkBackupRestore",
    "BulkMigration",
    "BulkMetrics",
    "BulkScheduler",
    "BulkRateLimiter",
    "BulkValidator",
    "BulkWebhooks",
    "BulkGrouping",
    "BulkRetry",
    "BulkBatchProcessor",
    "BulkPerformanceOptimizer",
    "BulkQueue",
    "BulkTransformation",
    "BulkAggregation",
    "BulkMonitoring",
    "BulkThrottle",
    "BulkCircuitBreaker",
    "BulkCache",
    "BulkAudit",
    "BulkOrchestrator",
    "BulkHealthChecker",
    "BulkErrorHandler",
    "BulkConfig",
    "BulkFactory",
    "BulkSecurity",
    "BulkCompression",
    "BulkStreaming",
    "BulkAsyncQueue",
    "BulkLock",
    "BulkProgressTracker",
    "BulkResourceManager",
    "BulkRealTimeMetrics",
    "BulkAdvancedCache",
    "BulkPriorityQueue",
    "BulkEnhancedValidator",
    "BulkDashboard",
    "BulkBenchmark",
    "BulkAutoTuner",
    "BulkAdaptiveRateLimiter",
    "BulkLoadBalancer",
    "BulkLoadPredictor",
    "BulkAutoScaler",
    "BulkEventSourcing",
    "BulkObservability",
    "BulkCostOptimizer",
    "BulkAnomalyDetector",
    # Nuevas clases agregadas
    "BulkDataPipeline",
    "BulkIntelligentScheduler",
    "BulkResourceOptimizer",
    "BulkDataQualityChecker",
    "BulkSmartBatching",
    "BulkDistributedCoordinator",
    "BulkAdvancedAnalytics",
]

