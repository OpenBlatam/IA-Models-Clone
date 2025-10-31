"""
Document Workflow Chain v2.0 - Optimized & Refactored
====================================================

A high-performance, modular document workflow chain system with advanced features:
- Async/await throughout for better performance
- Type hints for better code quality
- Clean architecture with dependency injection
- Advanced caching and optimization
- Real-time monitoring and analytics
- Plugin system for extensibility
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Callable, Awaitable
from uuid import uuid4
import json
import hashlib
from collections import defaultdict, deque
import weakref

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Enhanced workflow status with more granular states"""
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"
    ARCHIVED = "archived"


class Priority(Enum):
    """Task priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class PerformanceMetrics:
    """Enhanced performance metrics"""
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    tokens_used: int = 0
    tokens_per_second: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    api_calls: int = 0
    errors: int = 0
    
    @property
    def duration(self) -> float:
        """Get duration in seconds"""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return (datetime.utcnow() - self.start_time).total_seconds()
    
    @property
    def cache_hit_rate(self) -> float:
        """Get cache hit rate"""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0


@dataclass
class DocumentNode:
    """Enhanced document node with better structure"""
    id: str = field(default_factory=lambda: str(uuid4()))
    title: str = ""
    content: str = ""
    prompt: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    
    # Enhanced metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    priority: Priority = Priority.NORMAL
    
    # Performance data
    metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    
    # Quality scores
    quality_score: Optional[float] = None
    readability_score: Optional[float] = None
    sentiment_score: Optional[float] = None
    seo_score: Optional[float] = None
    
    # Content analysis
    word_count: int = 0
    character_count: int = 0
    language: str = "en"
    topics: List[str] = field(default_factory=list)
    entities: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        """Post-initialization processing"""
        if self.content:
            self.word_count = len(self.content.split())
            self.character_count = len(self.content)
            self.updated_at = datetime.utcnow()
    
    def add_child(self, child_id: str) -> None:
        """Add child node"""
        if child_id not in self.children_ids:
            self.children_ids.append(child_id)
            self.updated_at = datetime.utcnow()
    
    def remove_child(self, child_id: str) -> None:
        """Remove child node"""
        if child_id in self.children_ids:
            self.children_ids.remove(child_id)
            self.updated_at = datetime.utcnow()
    
    def get_content_hash(self) -> str:
        """Get content hash for deduplication"""
        return hashlib.md5(self.content.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "prompt": self.prompt,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "metadata": self.metadata,
            "tags": self.tags,
            "priority": self.priority.value,
            "quality_score": self.quality_score,
            "readability_score": self.readability_score,
            "sentiment_score": self.sentiment_score,
            "seo_score": self.seo_score,
            "word_count": self.word_count,
            "character_count": self.character_count,
            "language": self.language,
            "topics": self.topics,
            "entities": self.entities,
            "metrics": {
                "duration": self.metrics.duration,
                "tokens_used": self.metrics.tokens_used,
                "tokens_per_second": self.metrics.tokens_per_second,
                "cache_hit_rate": self.metrics.cache_hit_rate,
                "api_calls": self.metrics.api_calls,
                "errors": self.metrics.errors
            }
        }


class CacheManager:
    """High-performance cache manager with TTL and LRU eviction"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, datetime] = {}
        self._expiry_times: Dict[str, datetime] = {}
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key in self._cache:
            if datetime.utcnow() < self._expiry_times[key]:
                self._access_times[key] = datetime.utcnow()
                self._hits += 1
                return self._cache[key]["value"]
            else:
                # Expired, remove it
                self._remove(key)
        
        self._misses += 1
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache"""
        if len(self._cache) >= self.max_size:
            self._evict_lru()
        
        ttl = ttl or self.default_ttl
        expiry = datetime.utcnow() + timedelta(seconds=ttl)
        
        self._cache[key] = {"value": value, "created_at": datetime.utcnow()}
        self._access_times[key] = datetime.utcnow()
        self._expiry_times[key] = expiry
    
    def _remove(self, key: str) -> None:
        """Remove key from cache"""
        self._cache.pop(key, None)
        self._access_times.pop(key, None)
        self._expiry_times.pop(key, None)
    
    def _evict_lru(self) -> None:
        """Evict least recently used item"""
        if not self._access_times:
            return
        
        lru_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
        self._remove(lru_key)
    
    def clear(self) -> None:
        """Clear all cache"""
        self._cache.clear()
        self._access_times.clear()
        self._expiry_times.clear()
        self._hits = 0
        self._misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total_requests if total_requests > 0 else 0.0,
            "memory_usage_mb": sum(len(str(v)) for v in self._cache.values()) / 1024 / 1024
        }


class EventBus:
    """Event bus for decoupled communication"""
    
    def __init__(self):
        self._listeners: Dict[str, List[Callable]] = defaultdict(list)
        self._async_listeners: Dict[str, List[Callable[[Any], Awaitable[None]]]] = defaultdict(list)
    
    def subscribe(self, event_type: str, callback: Callable) -> None:
        """Subscribe to event"""
        if asyncio.iscoroutinefunction(callback):
            self._async_listeners[event_type].append(callback)
        else:
            self._listeners[event_type].append(callback)
    
    def unsubscribe(self, event_type: str, callback: Callable) -> None:
        """Unsubscribe from event"""
        if callback in self._listeners[event_type]:
            self._listeners[event_type].remove(callback)
        if callback in self._async_listeners[event_type]:
            self._async_listeners[event_type].remove(callback)
    
    async def emit(self, event_type: str, data: Any = None) -> None:
        """Emit event to all listeners"""
        # Sync listeners
        for callback in self._listeners[event_type]:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Error in sync event listener: {e}")
        
        # Async listeners
        for callback in self._async_listeners[event_type]:
            try:
                await callback(data)
            except Exception as e:
                logger.error(f"Error in async event listener: {e}")


class PluginManager:
    """Plugin system for extensibility"""
    
    def __init__(self):
        self._plugins: Dict[str, Any] = {}
        self._hooks: Dict[str, List[Callable]] = defaultdict(list)
    
    def register_plugin(self, name: str, plugin: Any) -> None:
        """Register a plugin"""
        self._plugins[name] = plugin
        logger.info(f"Plugin registered: {name}")
    
    def unregister_plugin(self, name: str) -> None:
        """Unregister a plugin"""
        if name in self._plugins:
            del self._plugins[name]
            logger.info(f"Plugin unregistered: {name}")
    
    def add_hook(self, hook_name: str, callback: Callable) -> None:
        """Add a hook callback"""
        self._hooks[hook_name].append(callback)
    
    async def execute_hook(self, hook_name: str, *args, **kwargs) -> Any:
        """Execute all callbacks for a hook"""
        results = []
        for callback in self._hooks[hook_name]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    result = await callback(*args, **kwargs)
                else:
                    result = callback(*args, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Error in hook {hook_name}: {e}")
        return results


class WorkflowChain:
    """Enhanced workflow chain with better performance and features"""
    
    def __init__(self, name: str, description: str = ""):
        self.id = str(uuid4())
        self.name = name
        self.description = description
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self.status = WorkflowStatus.DRAFT
        self.settings: Dict[str, Any] = {}
        
        # Enhanced data structures
        self._nodes: Dict[str, DocumentNode] = {}
        self._node_index: Dict[str, List[str]] = defaultdict(list)  # For fast lookups
        self._cache = CacheManager()
        self._event_bus = EventBus()
        self._plugin_manager = PluginManager()
        
        # Performance tracking
        self._metrics = PerformanceMetrics()
        self._operation_count = 0
        
        # Weak references for memory efficiency
        self._weak_refs = weakref.WeakValueDictionary()
    
    async def add_node(self, node: DocumentNode) -> None:
        """Add node with enhanced performance"""
        start_time = time.time()
        
        # Check cache first
        cache_key = f"node:{node.id}"
        if self._cache.get(cache_key):
            return
        
        # Add to data structures
        self._nodes[node.id] = node
        self._node_index[node.title].append(node.id)
        
        # Update relationships
        if node.parent_id and node.parent_id in self._nodes:
            self._nodes[node.parent_id].add_child(node.id)
        
        # Cache the node
        self._cache.set(cache_key, node.to_dict())
        
        # Update metrics
        self._metrics.api_calls += 1
        self._operation_count += 1
        self.updated_at = datetime.utcnow()
        
        # Emit event
        await self._event_bus.emit("node_added", {"node_id": node.id, "chain_id": self.id})
        
        # Execute hooks
        await self._plugin_manager.execute_hook("after_node_add", node, self)
        
        duration = time.time() - start_time
        logger.debug(f"Added node {node.id} in {duration:.3f}s")
    
    async def get_node(self, node_id: str) -> Optional[DocumentNode]:
        """Get node with caching"""
        # Check cache first
        cache_key = f"node:{node_id}"
        cached = self._cache.get(cache_key)
        if cached:
            return DocumentNode(**cached)
        
        # Get from storage
        node = self._nodes.get(node_id)
        if node:
            # Cache it
            self._cache.set(cache_key, node.to_dict())
        
        return node
    
    async def get_nodes_by_title(self, title: str) -> List[DocumentNode]:
        """Get nodes by title with indexing"""
        node_ids = self._node_index.get(title, [])
        nodes = []
        
        for node_id in node_ids:
            node = await self.get_node(node_id)
            if node:
                nodes.append(node)
        
        return nodes
    
    async def get_chain_statistics(self) -> Dict[str, Any]:
        """Get comprehensive chain statistics"""
        nodes = list(self._nodes.values())
        
        if not nodes:
            return {
                "total_nodes": 0,
                "total_words": 0,
                "average_quality": 0.0,
                "cache_stats": self._cache.get_stats(),
                "performance_metrics": {
                    "duration": self._metrics.duration,
                    "operations": self._operation_count,
                    "api_calls": self._metrics.api_calls,
                    "errors": self._metrics.errors
                }
            }
        
        # Calculate statistics
        total_words = sum(node.word_count for node in nodes)
        quality_scores = [node.quality_score for node in nodes if node.quality_score is not None]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        
        # Get depth
        max_depth = 0
        for node in nodes:
            depth = await self._calculate_depth(node.id)
            max_depth = max(max_depth, depth)
        
        return {
            "total_nodes": len(nodes),
            "total_words": total_words,
            "average_quality": avg_quality,
            "max_depth": max_depth,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "cache_stats": self._cache.get_stats(),
            "performance_metrics": {
                "duration": self._metrics.duration,
                "operations": self._operation_count,
                "api_calls": self._metrics.api_calls,
                "errors": self._metrics.errors,
                "tokens_used": sum(node.metrics.tokens_used for node in nodes),
                "cache_hit_rate": self._cache.get_stats()["hit_rate"]
            }
        }
    
    async def _calculate_depth(self, node_id: str, visited: Optional[set] = None) -> int:
        """Calculate depth of a node"""
        if visited is None:
            visited = set()
        
        if node_id in visited:
            return 0  # Circular reference
        
        visited.add(node_id)
        node = await self.get_node(node_id)
        
        if not node or not node.parent_id:
            return 0
        
        return 1 + await self._calculate_depth(node.parent_id, visited)
    
    async def optimize(self) -> Dict[str, Any]:
        """Optimize the workflow chain"""
        start_time = time.time()
        
        # Clear expired cache entries
        self._cache.clear()
        
        # Rebuild indexes
        self._node_index.clear()
        for node in self._nodes.values():
            self._node_index[node.title].append(node.id)
        
        # Execute optimization hooks
        results = await self._plugin_manager.execute_hook("optimize_chain", self)
        
        duration = time.time() - start_time
        
        return {
            "optimization_time": duration,
            "cache_cleared": True,
            "indexes_rebuilt": True,
            "plugin_results": results
        }
    
    def subscribe_to_events(self, event_type: str, callback: Callable) -> None:
        """Subscribe to chain events"""
        self._event_bus.subscribe(event_type, callback)
    
    def add_plugin(self, name: str, plugin: Any) -> None:
        """Add a plugin to the chain"""
        self._plugin_manager.register_plugin(name, plugin)
    
    def add_hook(self, hook_name: str, callback: Callable) -> None:
        """Add a hook callback"""
        self._plugin_manager.add_hook(hook_name, callback)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "status": self.status.value,
            "settings": self.settings,
            "statistics": asyncio.run(self.get_chain_statistics())
        }


class WorkflowChainManager:
    """Manager for multiple workflow chains with advanced features"""
    
    def __init__(self):
        self._chains: Dict[str, WorkflowChain] = {}
        self._global_cache = CacheManager(max_size=5000)
        self._event_bus = EventBus()
        self._plugin_manager = PluginManager()
        self._metrics = PerformanceMetrics()
    
    async def create_chain(self, name: str, description: str = "") -> WorkflowChain:
        """Create a new workflow chain"""
        chain = WorkflowChain(name, description)
        self._chains[chain.id] = chain
        
        # Subscribe to chain events
        chain.subscribe_to_events("node_added", self._on_node_added)
        
        # Cache the chain
        self._global_cache.set(f"chain:{chain.id}", chain.to_dict())
        
        # Emit event
        await self._event_bus.emit("chain_created", {"chain_id": chain.id})
        
        logger.info(f"Created workflow chain: {chain.id}")
        return chain
    
    async def get_chain(self, chain_id: str) -> Optional[WorkflowChain]:
        """Get workflow chain"""
        # Check cache first
        cached = self._global_cache.get(f"chain:{chain_id}")
        if cached:
            return self._chains.get(chain_id)
        
        return self._chains.get(chain_id)
    
    async def list_chains(self, status: Optional[WorkflowStatus] = None) -> List[WorkflowChain]:
        """List all chains with optional filtering"""
        chains = list(self._chains.values())
        
        if status:
            chains = [chain for chain in chains if chain.status == status]
        
        return chains
    
    async def delete_chain(self, chain_id: str) -> bool:
        """Delete a workflow chain"""
        if chain_id in self._chains:
            chain = self._chains[chain_id]
            
            # Emit event
            await self._event_bus.emit("chain_deleted", {"chain_id": chain_id})
            
            # Remove from cache
            self._global_cache.set(f"chain:{chain_id}", None, ttl=1)
            
            # Delete chain
            del self._chains[chain_id]
            
            logger.info(f"Deleted workflow chain: {chain_id}")
            return True
        
        return False
    
    async def get_global_statistics(self) -> Dict[str, Any]:
        """Get global statistics"""
        chains = list(self._chains.values())
        
        total_nodes = sum(len(chain._nodes) for chain in chains)
        total_words = sum(
            sum(node.word_count for node in chain._nodes.values())
            for chain in chains
        )
        
        return {
            "total_chains": len(chains),
            "total_nodes": total_nodes,
            "total_words": total_words,
            "cache_stats": self._global_cache.get_stats(),
            "performance_metrics": {
                "duration": self._metrics.duration,
                "api_calls": self._metrics.api_calls,
                "errors": self._metrics.errors
            }
        }
    
    async def _on_node_added(self, data: Dict[str, Any]) -> None:
        """Handle node added event"""
        chain_id = data["chain_id"]
        node_id = data["node_id"]
        
        # Update global metrics
        self._metrics.api_calls += 1
        
        # Emit global event
        await self._event_bus.emit("global_node_added", {
            "chain_id": chain_id,
            "node_id": node_id
        })
    
    def subscribe_to_events(self, event_type: str, callback: Callable) -> None:
        """Subscribe to global events"""
        self._event_bus.subscribe(event_type, callback)
    
    def add_global_plugin(self, name: str, plugin: Any) -> None:
        """Add a global plugin"""
        self._plugin_manager.register_plugin(name, plugin)
        
        # Add to all existing chains
        for chain in self._chains.values():
            chain.add_plugin(name, plugin)


# Example usage and testing
async def main():
    """Example usage of the optimized workflow chain system"""
    
    # Create manager
    manager = WorkflowChainManager()
    
    # Create a chain
    chain = await manager.create_chain("AI Blog Series", "A series of AI-related blog posts")
    
    # Add some nodes
    node1 = DocumentNode(
        title="Introduction to AI",
        content="Artificial Intelligence is revolutionizing the world...",
        prompt="Write an introduction to AI"
    )
    
    node2 = DocumentNode(
        title="Machine Learning Basics",
        content="Machine Learning is a subset of AI...",
        prompt="Write about machine learning basics",
        parent_id=node1.id
    )
    
    await chain.add_node(node1)
    await chain.add_node(node2)
    
    # Get statistics
    stats = await chain.get_chain_statistics()
    print("Chain Statistics:", json.dumps(stats, indent=2))
    
    # Get global statistics
    global_stats = await manager.get_global_statistics()
    print("Global Statistics:", json.dumps(global_stats, indent=2))
    
    # Optimize the chain
    optimization_results = await chain.optimize()
    print("Optimization Results:", json.dumps(optimization_results, indent=2))


if __name__ == "__main__":
    asyncio.run(main())




