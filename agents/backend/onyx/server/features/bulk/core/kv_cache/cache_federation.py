"""
Cache federation system.

Provides federation capabilities for multiple cache clusters.
"""
from __future__ import annotations

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CacheCluster:
    """Cache cluster information."""
    id: str
    nodes: List[Any]
    metadata: Dict[str, Any]


class CacheFederation:
    """
    Cache federation manager.
    
    Manages federation of multiple cache clusters.
    """
    
    def __init__(self):
        """Initialize federation."""
        self.clusters: Dict[str, CacheCluster] = {}
        self.routing_table: Dict[str, str] = {}  # key -> cluster_id
    
    def register_cluster(self, cluster: CacheCluster) -> None:
        """
        Register cluster.
        
        Args:
            cluster: Cache cluster
        """
        self.clusters[cluster.id] = cluster
        logger.info(f"Registered cluster: {cluster.id}")
    
    def unregister_cluster(self, cluster_id: str) -> bool:
        """
        Unregister cluster.
        
        Args:
            cluster_id: Cluster ID
            
        Returns:
            True if unregistered
        """
        if cluster_id in self.clusters:
            del self.clusters[cluster_id]
            # Remove from routing table
            self.routing_table = {
                k: v for k, v in self.routing_table.items()
                if v != cluster_id
            }
            logger.info(f"Unregistered cluster: {cluster_id}")
            return True
        return False
    
    def route_to_cluster(self, key: Any) -> Optional[CacheCluster]:
        """
        Route key to cluster.
        
        Args:
            key: Cache key
            
        Returns:
            Cache cluster or None
        """
        key_str = str(key)
        
        # Check routing table
        if key_str in self.routing_table:
            cluster_id = self.routing_table[key_str]
            return self.clusters.get(cluster_id)
        
        # Route based on key hash
        import hashlib
        hash_value = int(hashlib.md5(key_str.encode()).hexdigest(), 16)
        cluster_ids = list(self.clusters.keys())
        
        if cluster_ids:
            cluster_index = hash_value % len(cluster_ids)
            cluster_id = cluster_ids[cluster_index]
            
            # Update routing table
            self.routing_table[key_str] = cluster_id
            
            return self.clusters[cluster_id]
        
        return None
    
    def get(self, key: Any) -> Optional[Any]:
        """
        Get value from federated cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        cluster = self.route_to_cluster(key)
        if cluster and cluster.nodes:
            # Try first node
            return cluster.nodes[0].get(key)
        return None
    
    def put(self, key: Any, value: Any) -> bool:
        """
        Put value to federated cache.
        
        Args:
            key: Cache key
            value: Value to cache
            
        Returns:
            True if successful
        """
        cluster = self.route_to_cluster(key)
        if cluster and cluster.nodes:
            # Put to first node (could replicate)
            cluster.nodes[0].put(key, value)
            return True
        return False
    
    def replicate(self, key: Any, value: Any, cluster_ids: Optional[List[str]] = None) -> int:
        """
        Replicate value across clusters.
        
        Args:
            key: Cache key
            value: Value to cache
            cluster_ids: Optional list of cluster IDs to replicate to
            
        Returns:
            Number of clusters replicated to
        """
        if cluster_ids is None:
            cluster_ids = list(self.clusters.keys())
        
        replicated = 0
        for cluster_id in cluster_ids:
            if cluster_id in self.clusters:
                cluster = self.clusters[cluster_id]
                if cluster.nodes:
                    cluster.nodes[0].put(key, value)
                    replicated += 1
        
        return replicated
    
    def get_cluster_stats(self, cluster_id: str) -> Optional[Dict[str, Any]]:
        """
        Get statistics for cluster.
        
        Args:
            cluster_id: Cluster ID
            
        Returns:
            Statistics or None
        """
        if cluster_id not in self.clusters:
            return None
        
        cluster = self.clusters[cluster_id]
        stats_list = []
        
        for node in cluster.nodes:
            node_stats = node.get_stats()
            stats_list.append(node_stats)
        
        # Aggregate stats
        return {
            "cluster_id": cluster_id,
            "num_nodes": len(cluster.nodes),
            "aggregated_stats": {
                "total_hits": sum(s.get("hits", 0) for s in stats_list),
                "total_misses": sum(s.get("misses", 0) for s in stats_list),
                "total_memory_mb": sum(s.get("memory_mb", 0.0) for s in stats_list)
            },
            "node_stats": stats_list
        }
    
    def list_clusters(self) -> List[str]:
        """
        List all cluster IDs.
        
        Returns:
            List of cluster IDs
        """
        return list(self.clusters.keys())

