"""
Data Manager for Color Grading AI
===================================

Unified data management with history tracking, statistics, and persistence.
"""

import logging
from typing import Dict, Any, Optional, List, TypeVar, Generic, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import deque, defaultdict
from abc import ABC, abstractmethod
import json
from pathlib import Path

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class DataEntry(Generic[T]):
    """Data entry with timestamp."""
    data: T
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataManager(ABC, Generic[T]):
    """
    Unified data manager.
    
    Features:
    - History tracking
    - Statistics collection
    - Automatic cleanup
    - Persistence
    - Filtering and querying
    """
    
    def __init__(self, max_history: int = 10000, persist_path: Optional[Path] = None):
        """
        Initialize data manager.
        
        Args:
            max_history: Maximum history entries
            persist_path: Optional path for persistence
        """
        self.max_history = max_history
        self.persist_path = persist_path
        self._history: deque = deque(maxlen=max_history)
        self._stats: Dict[str, Any] = defaultdict(int)
        self._metadata: Dict[str, Any] = {}
    
    def add(self, data: T, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add data entry.
        
        Args:
            data: Data to add
            metadata: Optional metadata
            
        Returns:
            Entry ID
        """
        entry = DataEntry(
            data=data,
            metadata=metadata or {}
        )
        
        self._history.append(entry)
        self._update_stats(data, entry)
        
        entry_id = f"{len(self._history)}_{entry.timestamp.isoformat()}"
        logger.debug(f"Added data entry: {entry_id}")
        
        return entry_id
    
    def get_history(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None,
        filter_func: Optional[Callable[[DataEntry[T]], bool]] = None
    ) -> List[DataEntry[T]]:
        """
        Get history entries.
        
        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter
            limit: Optional limit
            filter_func: Optional filter function
            
        Returns:
            List of entries
        """
        entries = list(self._history)
        
        # Date filtering
        if start_date:
            entries = [e for e in entries if e.timestamp >= start_date]
        if end_date:
            entries = [e for e in entries if e.timestamp <= end_date]
        
        # Custom filtering
        if filter_func:
            entries = [e for e in entries if filter_func(e)]
        
        # Limit
        if limit:
            entries = entries[-limit:]
        
        return entries
    
    def get_latest(self, n: int = 1) -> List[DataEntry[T]]:
        """Get latest N entries."""
        return list(self._history)[-n:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            "total_entries": len(self._history),
            "max_history": self.max_history,
            "stats": dict(self._stats),
            "metadata": self._metadata.copy(),
        }
    
    def clear_history(self, before_date: Optional[datetime] = None):
        """
        Clear history.
        
        Args:
            before_date: Optional date to clear before
        """
        if before_date:
            self._history = deque(
                [e for e in self._history if e.timestamp >= before_date],
                maxlen=self.max_history
            )
        else:
            self._history.clear()
        
        logger.info(f"Cleared history (before: {before_date})")
    
    def reset_stats(self):
        """Reset statistics."""
        self._stats.clear()
        logger.debug("Reset statistics")
    
    def set_metadata(self, key: str, value: Any):
        """Set metadata."""
        self._metadata[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata."""
        return self._metadata.get(key, default)
    
    @abstractmethod
    def _update_stats(self, data: T, entry: DataEntry[T]):
        """Update statistics based on data entry."""
        pass
    
    def save(self, file_path: Optional[Path] = None):
        """Save data to file."""
        path = file_path or self.persist_path
        if not path:
            return
        
        try:
            data = {
                "history": [
                    {
                        "data": self._serialize_entry(e),
                        "timestamp": e.timestamp.isoformat(),
                        "metadata": e.metadata
                    }
                    for e in self._history
                ],
                "stats": dict(self._stats),
                "metadata": self._metadata,
            }
            
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Saved data to {path}")
        except Exception as e:
            logger.error(f"Error saving data: {e}")
    
    def load(self, file_path: Optional[Path] = None):
        """Load data from file."""
        path = file_path or self.persist_path
        if not path or not path.exists():
            return
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Load history
            self._history.clear()
            for entry_data in data.get("history", []):
                entry = DataEntry(
                    data=self._deserialize_entry(entry_data["data"]),
                    timestamp=datetime.fromisoformat(entry_data["timestamp"]),
                    metadata=entry_data.get("metadata", {})
                )
                self._history.append(entry)
            
            # Load stats
            self._stats = defaultdict(int, data.get("stats", {}))
            
            # Load metadata
            self._metadata = data.get("metadata", {})
            
            logger.info(f"Loaded data from {path}")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
    
    def _serialize_entry(self, data: T) -> Any:
        """Serialize entry data."""
        if hasattr(data, '__dict__'):
            return asdict(data) if hasattr(data, '__dict__') else str(data)
        return data
    
    def _deserialize_entry(self, data: Any) -> T:
        """Deserialize entry data."""
        return data


class StatisticsManager:
    """
    Statistics manager for services.
    
    Features:
    - Automatic statistics collection
    - Aggregations
    - Time-based statistics
    - Custom metrics
    """
    
    def __init__(self):
        """Initialize statistics manager."""
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._timestamps: Dict[str, datetime] = {}
        self._max_histogram_size = 1000
    
    def increment(self, metric: str, value: int = 1):
        """Increment counter metric."""
        self._counters[metric] += value
        self._timestamps[metric] = datetime.now()
    
    def set_gauge(self, metric: str, value: float):
        """Set gauge metric."""
        self._gauges[metric] = value
        self._timestamps[metric] = datetime.now()
    
    def record_histogram(self, metric: str, value: float):
        """Record histogram value."""
        self._histograms[metric].append(value)
        if len(self._histograms[metric]) > self._max_histogram_size:
            self._histograms[metric] = self._histograms[metric][-self._max_histogram_size:]
        self._timestamps[metric] = datetime.now()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get all statistics."""
        stats = {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "histograms": {
                name: {
                    "count": len(values),
                    "min": min(values) if values else 0.0,
                    "max": max(values) if values else 0.0,
                    "mean": sum(values) / len(values) if values else 0.0,
                }
                for name, values in self._histograms.items()
            },
            "timestamps": {
                name: ts.isoformat()
                for name, ts in self._timestamps.items()
            }
        }
        
        return stats
    
    def reset(self):
        """Reset all statistics."""
        self._counters.clear()
        self._gauges.clear()
        self._histograms.clear()
        self._timestamps.clear()
        logger.debug("Reset all statistics")

