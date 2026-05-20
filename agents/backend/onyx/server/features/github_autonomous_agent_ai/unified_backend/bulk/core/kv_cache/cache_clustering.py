"""
Cache clustering utilities.

Provides clustering and grouping capabilities for cache entries.
"""
from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional, Set
import torch

logger = logging.getLogger(__name__)


class CacheClustering:
    """
    Cache clustering manager.
    
    Groups related cache entries for better management.
    """
    
    def __init__(self, cache: Any):
        """
        Initialize cache clustering.
        
        Args:
            cache: Cache instance
        """
        self.cache = cache
        self.clusters: Dict[str, Set[int]] = {}
        self.cluster_metadata: Dict[str, Dict[str, Any]] = {}
    
    def create_cluster(
        self,
        cluster_name: str,
        positions: List[int],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Create a cluster of cache entries.
        
        Args:
            cluster_name: Name of cluster
            positions: List of positions to include
            metadata: Optional cluster metadata
        """
        self.clusters[cluster_name] = set(positions)
        self.cluster_metadata[cluster_name] = metadata or {}
        logger.info(f"Created cluster '{cluster_name}' with {len(positions)} entries")
    
    def add_to_cluster(self, cluster_name: str, position: int) -> None:
        """
        Add position to cluster.
        
        Args:
            cluster_name: Cluster name
            position: Position to add
        """
        if cluster_name not in self.clusters:
            self.clusters[cluster_name] = set()
        self.clusters[cluster_name].add(position)
    
    def remove_from_cluster(self, cluster_name: str, position: int) -> None:
        """
        Remove position from cluster.
        
        Args:
            cluster_name: Cluster name
            position: Position to remove
        """
        if cluster_name in self.clusters:
            self.clusters[cluster_name].discard(position)
    
    def get_cluster(self, cluster_name: str) -> Set[int]:
        """
        Get positions in cluster.
        
        Args:
            cluster_name: Cluster name
            
        Returns:
            Set of positions
        """
        return self.clusters.get(cluster_name, set())
    
    def evict_cluster(self, cluster_name: str) -> int:
        """
        Evict all entries in cluster.
        
        Args:
            cluster_name: Cluster name
            
        Returns:
            Number of entries evicted
        """
        if cluster_name not in self.clusters:
            return 0
        
        positions = list(self.clusters[cluster_name])
        evicted = self.cache.storage.remove(positions)
        
        # Remove cluster
        del self.clusters[cluster_name]
        if cluster_name in self.cluster_metadata:
            del self.cluster_metadata[cluster_name]
        
        logger.info(f"Evicted cluster '{cluster_name}' with {evicted} entries")
        return evicted
    
    def get_cluster_stats(self, cluster_name: str) -> Dict[str, Any]:
        """
        Get statistics for cluster.
        
        Args:
            cluster_name: Cluster name
            
        Returns:
            Dictionary with cluster statistics
        """
        if cluster_name not in self.clusters:
            return {}
        
        positions = self.clusters[cluster_name]
        storage = self.cache.storage
        
        total_memory = 0.0
        valid_entries = 0
        
        for pos in positions:
            entry = storage.get(pos)
            if entry:
                valid_entries += 1
                key, value = entry
                total_memory += (
                    key.numel() * key.element_size() +
                    value.numel() * value.element_size()
                ) / (1024 * 1024)  # MB
        
        return {
            "cluster_name": cluster_name,
            "total_positions": len(positions),
            "valid_entries": valid_entries,
            "total_memory_mb": total_memory,
            "metadata": self.cluster_metadata.get(cluster_name, {})
        }
    
    def list_clusters(self) -> List[str]:
        """
        List all clusters.
        
        Returns:
            List of cluster names
        """
        return list(self.clusters.keys())


class CacheGrouping:
    """
    Cache grouping utilities.
    
    Groups cache entries by various criteria.
    """
    
    def __init__(self, cache: Any):
        """
        Initialize cache grouping.
        
        Args:
            cache: Cache instance
        """
        self.cache = cache
    
    def group_by_size(
        self,
        size_thresholds: List[float]
    ) -> Dict[str, List[int]]:
        """
        Group entries by memory size.
        
        Args:
            size_thresholds: List of size thresholds in MB
            
        Returns:
            Dictionary mapping size category -> positions
        """
        groups: Dict[str, List[int]] = {}
        storage = self.cache.storage
        positions = storage.get_positions()
        
        for pos in positions:
            entry = storage.get(pos)
            if entry:
                key, value = entry
                size_mb = (
                    key.numel() * key.element_size() +
                    value.numel() * value.element_size()
                ) / (1024 * 1024)
                
                # Find category
                category = "large"
                for i, threshold in enumerate(size_thresholds):
                    if size_mb < threshold:
                        category = f"size_{i}"
                        break
                
                if category not in groups:
                    groups[category] = []
                groups[category].append(pos)
        
        return groups
    
    def group_by_access_frequency(
        self,
        frequency_thresholds: List[int]
    ) -> Dict[str, List[int]]:
        """
        Group entries by access frequency.
        
        Args:
            frequency_thresholds: List of frequency thresholds
            
        Returns:
            Dictionary mapping frequency category -> positions
        """
        groups: Dict[str, List[int]] = {}
        storage = self.cache.storage
        access_counts = storage.get_access_counts()
        positions = storage.get_positions()
        
        for pos in positions:
            frequency = access_counts.get(pos, 0)
            
            # Find category
            category = "very_frequent"
            for i, threshold in enumerate(frequency_thresholds):
                if frequency < threshold:
                    category = f"freq_{i}"
                    break
            
            if category not in groups:
                groups[category] = []
            groups[category].append(pos)
        
        return groups

