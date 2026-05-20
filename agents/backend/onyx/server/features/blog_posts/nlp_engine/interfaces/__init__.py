from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from .analyzers import (
from .cache import (
from .metrics import (
from .config import (
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
🔌 INTERFACES MODULE - Ports & Contracts
========================================

Define los contratos (interfaces/protocols) que deben implementar
las capas de infraestructura.
"""

# Analyzer interfaces
    IAnalyzer, 
    IAnalyzerFactory, 
    IAdvancedAnalyzer, 
    IConfigurableAnalyzer
)

# Cache interfaces
    ICacheRepository, 
    IDistributedCache, 
    ICacheKeyGenerator,
    ICacheEvictionPolicy,
    ICacheSerializer
)

# Metrics interfaces
    IMetricsCollector, 
    IPerformanceMonitor, 
    IHealthChecker,
    IAlertManager,
    IStructuredLogger,
    IMetricsExporter
)

# Config interfaces
    IConfigurationService, 
    IEnvironmentConfigLoader, 
    IFileConfigLoader,
    ISecretManager,
    IConfigValidator,
    IConfigMerger,
    IConfigTransformer
)

__all__ = [
    # Analyzer interfaces
    'IAnalyzer',
    'IAnalyzerFactory', 
    'IAdvancedAnalyzer', 
    'IConfigurableAnalyzer',
    
    # Cache interfaces
    'ICacheRepository',
    'IDistributedCache', 
    'ICacheKeyGenerator',
    'ICacheEvictionPolicy',
    'ICacheSerializer',
    
    # Metrics interfaces
    'IMetricsCollector',
    'IPerformanceMonitor', 
    'IHealthChecker',
    'IAlertManager',
    'IStructuredLogger',
    'IMetricsExporter',
    
    # Config interfaces
    'IConfigurationService',
    'IEnvironmentConfigLoader', 
    'IFileConfigLoader',
    'ISecretManager',
    'IConfigValidator',
    'IConfigMerger',
    'IConfigTransformer'
] 