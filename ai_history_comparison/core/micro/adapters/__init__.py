"""
Micro Adapters Module

Ultra-specialized adapter components for the AI History Comparison System.
Each adapter handles a specific interface transformation or protocol adaptation.
"""

from .base_adapter import BaseAdapter, AdapterRegistry
from .data_adapter import DataAdapter, DataTransformationAdapter
from .api_adapter import APIAdapter, RESTAdapter, GraphQLAdapter
from .protocol_adapter import ProtocolAdapter, HTTPAdapter, WebSocketAdapter
from .format_adapter import FormatAdapter, JSONAdapter, XMLAdapter, CSVAdapter
from .service_adapter import ServiceAdapter, MicroserviceAdapter
from .database_adapter import DatabaseAdapter, SQLAdapter, NoSQLAdapter
from .cache_adapter import CacheAdapter, RedisAdapter, MemoryAdapter
from .queue_adapter import QueueAdapter, MessageQueueAdapter, EventAdapter
from .file_adapter import FileAdapter, StorageAdapter, CloudAdapter
from .ai_adapter import AIAdapter, ModelAdapter, ProviderAdapter

__all__ = [
    'BaseAdapter', 'AdapterRegistry',
    'DataAdapter', 'DataTransformationAdapter',
    'APIAdapter', 'RESTAdapter', 'GraphQLAdapter',
    'ProtocolAdapter', 'HTTPAdapter', 'WebSocketAdapter',
    'FormatAdapter', 'JSONAdapter', 'XMLAdapter', 'CSVAdapter',
    'ServiceAdapter', 'MicroserviceAdapter',
    'DatabaseAdapter', 'SQLAdapter', 'NoSQLAdapter',
    'CacheAdapter', 'RedisAdapter', 'MemoryAdapter',
    'QueueAdapter', 'MessageQueueAdapter', 'EventAdapter',
    'FileAdapter', 'StorageAdapter', 'CloudAdapter',
    'AIAdapter', 'ModelAdapter', 'ProviderAdapter'
]





















