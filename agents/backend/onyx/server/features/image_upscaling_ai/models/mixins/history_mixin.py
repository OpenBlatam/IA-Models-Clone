"""
History Mixin

Contains history and audit trail functionality.
"""

import logging
import json
from typing import Union, Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
from PIL import Image

logger = logging.getLogger(__name__)


class HistoryMixin:
    """
    Mixin providing history and audit trail functionality.
    
    This mixin contains:
    - Operation history
    - Audit trail
    - History search
    - History export
    - Statistics from history
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize history mixin."""
        super().__init__(*args, **kwargs)
        if not hasattr(self, '_history'):
            self._history = []
        if not hasattr(self, '_history_dir'):
            self._history_dir = Path("history")
            self._history_dir.mkdir(exist_ok=True)
    
    def add_to_history(
        self,
        operation: str,
        image_path: Optional[Union[str, Path]] = None,
        scale_factor: Optional[float] = None,
        method: Optional[str] = None,
        result_path: Optional[Union[str, Path]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        success: bool = True,
        error: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Add an operation to history.
        
        Args:
            operation: Operation name (e.g., 'upscale', 'enhance')
            image_path: Input image path
            scale_factor: Scale factor used
            method: Method used
            result_path: Output image path
            metadata: Additional metadata
            success: Whether operation succeeded
            error: Error message if failed
            
        Returns:
            Dictionary with history entry
        """
        entry = {
            "id": len(self._history),
            "operation": operation,
            "image_path": str(image_path) if image_path else None,
            "scale_factor": scale_factor,
            "method": method,
            "result_path": str(result_path) if result_path else None,
            "metadata": metadata or {},
            "success": success,
            "error": error,
            "timestamp": datetime.now().isoformat(),
        }
        
        self._history.append(entry)
        
        # Save to disk periodically
        if len(self._history) % 10 == 0:
            self._save_history()
        
        logger.debug(f"Added to history: {operation}")
        
        return entry
    
    def get_history(
        self,
        operation: Optional[str] = None,
        limit: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get history entries with optional filtering.
        
        Args:
            operation: Filter by operation name
            limit: Maximum number of entries
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
            
        Returns:
            List of history entries
        """
        history = self._history.copy()
        
        # Filter by operation
        if operation:
            history = [h for h in history if h["operation"] == operation]
        
        # Filter by date
        if start_date:
            history = [h for h in history if h["timestamp"] >= start_date]
        if end_date:
            history = [h for h in history if h["timestamp"] <= end_date]
        
        # Sort by timestamp (newest first)
        history.sort(key=lambda x: x["timestamp"], reverse=True)
        
        # Limit results
        if limit:
            history = history[:limit]
        
        return history
    
    def get_history_stats(self) -> Dict[str, Any]:
        """
        Get statistics from history.
        
        Returns:
            Dictionary with statistics
        """
        if not self._history:
            return {
                "total_operations": 0,
                "operations_by_type": {},
                "success_rate": 0.0,
                "most_used_method": None,
            }
        
        total = len(self._history)
        operations_by_type = {}
        success_count = 0
        method_counts = {}
        
        for entry in self._history:
            # Count by operation type
            op_type = entry["operation"]
            operations_by_type[op_type] = operations_by_type.get(op_type, 0) + 1
            
            # Count successes
            if entry["success"]:
                success_count += 1
            
            # Count methods
            if entry.get("method"):
                method = entry["method"]
                method_counts[method] = method_counts.get(method, 0) + 1
        
        most_used_method = max(method_counts.items(), key=lambda x: x[1])[0] if method_counts else None
        
        return {
            "total_operations": total,
            "operations_by_type": operations_by_type,
            "success_rate": success_count / total if total > 0 else 0.0,
            "most_used_method": most_used_method,
            "method_counts": method_counts,
        }
    
    def search_history(
        self,
        query: str
    ) -> List[Dict[str, Any]]:
        """
        Search history entries.
        
        Args:
            query: Search query
            
        Returns:
            List of matching history entries
        """
        query_lower = query.lower()
        results = []
        
        for entry in self._history:
            # Search in operation name
            if query_lower in entry["operation"].lower():
                results.append(entry)
                continue
            
            # Search in method
            if entry.get("method") and query_lower in entry["method"].lower():
                results.append(entry)
                continue
            
            # Search in image path
            if entry.get("image_path") and query_lower in entry["image_path"].lower():
                results.append(entry)
                continue
        
        return sorted(results, key=lambda x: x["timestamp"], reverse=True)
    
    def export_history(
        self,
        output_path: Union[str, Path],
        format: str = "json"
    ) -> bool:
        """
        Export history to file.
        
        Args:
            output_path: Output file path
            format: Export format ('json' or 'csv')
            
        Returns:
            True if successful
        """
        output_path = Path(output_path)
        
        if format == "json":
            with open(output_path, 'w') as f:
                json.dump(self._history, f, indent=2)
        elif format == "csv":
            import csv
            with open(output_path, 'w', newline='') as f:
                if self._history:
                    writer = csv.DictWriter(f, fieldnames=self._history[0].keys())
                    writer.writeheader()
                    writer.writerows(self._history)
        else:
            logger.error(f"Unsupported format: {format}")
            return False
        
        logger.info(f"History exported to {output_path}")
        return True
    
    def clear_history(self) -> int:
        """
        Clear all history.
        
        Returns:
            Number of entries cleared
        """
        count = len(self._history)
        self._history = []
        self._save_history()
        logger.info(f"History cleared: {count} entries")
        return count
    
    def _save_history(self):
        """Save history to disk."""
        history_path = self._history_dir / "history.json"
        with open(history_path, 'w') as f:
            json.dump(self._history, f, indent=2)
    
    def _load_history(self):
        """Load history from disk."""
        history_path = self._history_dir / "history.json"
        if history_path.exists():
            try:
                with open(history_path, 'r') as f:
                    self._history = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load history: {e}")
                self._history = []


