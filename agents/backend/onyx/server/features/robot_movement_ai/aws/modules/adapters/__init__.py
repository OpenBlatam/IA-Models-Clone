"""
Adapters (Hexagonal Architecture)
==================================

Adapters implement ports with specific technologies.
"""

from aws.modules.adapters.repository_adapters import (
    DynamoDBRepositoryAdapter,
    PostgreSQLRepositoryAdapter,
    InMemoryRepositoryAdapter,
)
from aws.modules.adapters.cache_adapters import (
    RedisCacheAdapter,
    MemcachedCacheAdapter,
    InMemoryCacheAdapter,
)
from aws.modules.adapters.messaging_adapters import (
    KafkaMessagingAdapter,
    RabbitMQMessagingAdapter,
    SQSMessagingAdapter,
)

__all__ = [
    "DynamoDBRepositoryAdapter",
    "PostgreSQLRepositoryAdapter",
    "InMemoryRepositoryAdapter",
    "RedisCacheAdapter",
    "MemcachedCacheAdapter",
    "InMemoryCacheAdapter",
    "KafkaMessagingAdapter",
    "RabbitMQMessagingAdapter",
    "SQSMessagingAdapter",
]















