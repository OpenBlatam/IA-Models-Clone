from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import os
import json
import time
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from pathlib import Path
import structlog
from datetime import datetime
from enum import Enum
                import csv
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Change Tracking System for Key Messages ML Pipeline
Logs and manages changes to code, configurations, and models
"""


logger = structlog.get_logger(__name__)

class ChangeType(Enum):
    """Types of changes that can be tracked."""
    CONFIG_UPDATE = "config_update"
    MODEL_TRAINING = "model_training"
    MODEL_REGISTRATION = "model_registration"
    CODE_CHANGE = "code_change"
    DATA_UPDATE = "data_update"
    EXPERIMENT_RUN = "experiment_run"
    DEPLOYMENT = "deployment"
    TEST_UPDATE = "test_update"
    DOCUMENTATION_UPDATE = "documentation_update"
    DEPENDENCY_UPDATE = "dependency_update"
    INFRASTRUCTURE_CHANGE = "infrastructure_change"
    SECURITY_UPDATE = "security_update"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    BUG_FIX = "bug_fix"
    FEATURE_ADDITION = "feature_addition"

@dataclass
class ChangeEntry:
    """Represents a single change entry."""
    id: str
    change_type: ChangeType
    description: str
    timestamp: float
    author: str
    affected_files: List[str]
    metadata: Dict[str, Any]
    tags: List[str]
    severity: str = "info"  # info, warning, error, critical
    
    def __post_init__(self) -> Any:
        if not self.id or not self.description:
            raise ValueError("ID and description are required")

@dataclass
class ChangeLog:
    """Represents a change log."""
    entries: List[ChangeEntry]
    total_entries: int
    first_entry: Optional[ChangeEntry]
    last_entry: Optional[ChangeEntry]
    
    def get_entries_by_type(self, change_type: ChangeType) -> List[ChangeEntry]:
        """Get entries by change type."""
        return [entry for entry in self.entries if entry.change_type == change_type]
    
    def get_entries_by_author(self, author: str) -> List[ChangeEntry]:
        """Get entries by author."""
        return [entry for entry in self.entries if entry.author == author]
    
    def get_entries_by_severity(self, severity: str) -> List[ChangeEntry]:
        """Get entries by severity."""
        return [entry for entry in self.entries if entry.severity == severity]
    
    def get_entries_in_timerange(self, start_time: float, end_time: float) -> List[ChangeEntry]:
        """Get entries within a time range."""
        return [entry for entry in self.entries if start_time <= entry.timestamp <= end_time]

class ChangeTracker:
    """Tracks changes to the ML pipeline."""
    
    def __init__(self, log_file: str = "./change_log.json", auto_log: bool = True,
                 include_metadata: bool = True, max_entries: int = 1000):
        
    """__init__ function."""
self.log_file = Path(log_file)
        self.auto_log = auto_log
        self.include_metadata = include_metadata
        self.max_entries = max_entries
        
        # Create log directory if needed
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing log
        self._load_log()
        
        logger.info("ChangeTracker initialized", 
                   log_file=str(self.log_file),
                   auto_log=auto_log,
                   max_entries=max_entries)
    
    def _load_log(self) -> Any:
        """Load change log from file."""
        if self.log_file.exists():
            try:
                with open(self.log_file, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    data = json.load(f)
                
                self.entries = []
                for entry_data in data.get("entries", []):
                    entry = ChangeEntry(
                        id=entry_data["id"],
                        change_type=ChangeType(entry_data["change_type"]),
                        description=entry_data["description"],
                        timestamp=entry_data["timestamp"],
                        author=entry_data["author"],
                        affected_files=entry_data["affected_files"],
                        metadata=entry_data["metadata"],
                        tags=entry_data["tags"],
                        severity=entry_data.get("severity", "info")
                    )
                    self.entries.append(entry)
                
                # Sort by timestamp
                self.entries.sort(key=lambda x: x.timestamp, reverse=True)
                
            except Exception as e:
                logger.error("Failed to load change log", error=str(e))
                self.entries = []
        else:
            self.entries = []
    
    def _save_log(self) -> Any:
        """Save change log to file."""
        try:
            data = {
                "entries": [
                    {
                        "id": entry.id,
                        "change_type": entry.change_type.value,
                        "description": entry.description,
                        "timestamp": entry.timestamp,
                        "author": entry.author,
                        "affected_files": entry.affected_files,
                        "metadata": entry.metadata,
                        "tags": entry.tags,
                        "severity": entry.severity
                    }
                    for entry in self.entries
                ],
                "total_entries": len(self.entries),
                "last_updated": time.time()
            }
            
            with open(self.log_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error("Failed to save change log", error=str(e))
    
    def _generate_entry_id(self, change_type: ChangeType, description: str) -> str:
        """Generate a unique entry ID."""
        timestamp = int(time.time())
        description_hash = str(hash(description))[-8:]
        return f"{change_type.value}_{timestamp}_{description_hash}"
    
    def log_change(self, change_type: ChangeType, description: str, 
                  author: str = "ML Pipeline", affected_files: List[str] = None,
                  metadata: Dict[str, Any] = None, tags: List[str] = None,
                  severity: str = "info") -> str:
        """Log a change entry."""
        try:
            # Generate entry ID
            entry_id = self._generate_entry_id(change_type, description)
            
            # Create change entry
            entry = ChangeEntry(
                id=entry_id,
                change_type=change_type,
                description=description,
                timestamp=time.time(),
                author=author,
                affected_files=affected_files or [],
                metadata=metadata or {},
                tags=tags or [],
                severity=severity
            )
            
            # Add to entries
            self.entries.insert(0, entry)  # Add to beginning (most recent first)
            
            # Limit entries
            if len(self.entries) > self.max_entries:
                self.entries = self.entries[:self.max_entries]
            
            # Save log
            self._save_log()
            
            logger.info("Change logged", 
                       entry_id=entry_id,
                       change_type=change_type.value,
                       description=description,
                       author=author)
            
            return entry_id
            
        except Exception as e:
            logger.error("Failed to log change", 
                        change_type=change_type.value,
                        description=description,
                        error=str(e))
            return ""
    
    def log_config_update(self, old_config: Dict[str, Any], new_config: Dict[str, Any],
                         description: str = "", author: str = "ML Pipeline") -> str:
        """Log a configuration update."""
        try:
            # Compute changes
            changes = self._compute_config_changes(old_config, new_config)
            
            metadata = {
                "old_config_keys": list(old_config.keys()),
                "new_config_keys": list(new_config.keys()),
                "changes": changes,
                "change_count": len(changes)
            }
            
            return self.log_change(
                change_type=ChangeType.CONFIG_UPDATE,
                description=description or f"Configuration updated: {len(changes)} changes",
                author=author,
                affected_files=["config.yaml"],
                metadata=metadata,
                tags=["configuration"]
            )
            
        except Exception as e:
            logger.error("Failed to log config update", error=str(e))
            return ""
    
    def log_model_training(self, model_name: str, model_path: str, 
                          metrics: Dict[str, Any], training_time: str,
                          description: str = "", author: str = "ML Pipeline") -> str:
        """Log a model training event."""
        try:
            metadata = {
                "model_name": model_name,
                "model_path": model_path,
                "metrics": metrics,
                "training_time": training_time,
                "accuracy": metrics.get("accuracy", 0),
                "loss": metrics.get("loss", 0)
            }
            
            return self.log_change(
                change_type=ChangeType.MODEL_TRAINING,
                description=description or f"Model {model_name} trained with accuracy {metrics.get('accuracy', 0):.4f}",
                author=author,
                affected_files=[model_path],
                metadata=metadata,
                tags=["model", "training"]
            )
            
        except Exception as e:
            logger.error("Failed to log model training", error=str(e))
            return ""
    
    def log_model_registration(self, model_name: str, version: str, 
                              metadata: Dict[str, Any], description: str = "",
                              author: str = "ML Pipeline") -> str:
        """Log a model registration event."""
        try:
            log_metadata = {
                "model_name": model_name,
                "version": version,
                "architecture": metadata.get("architecture", ""),
                "dataset": metadata.get("dataset", ""),
                "accuracy": metadata.get("accuracy", 0)
            }
            
            return self.log_change(
                change_type=ChangeType.MODEL_REGISTRATION,
                description=description or f"Model {model_name} v{version} registered",
                author=author,
                affected_files=[f"models/{model_name}_{version}.pt"],
                metadata=log_metadata,
                tags=["model", "registration"]
            )
            
        except Exception as e:
            logger.error("Failed to log model registration", error=str(e))
            return ""
    
    def log_code_change(self, files_changed: List[str], commit_hash: str = "",
                       description: str = "", author: str = "ML Pipeline") -> str:
        """Log a code change event."""
        try:
            metadata = {
                "commit_hash": commit_hash,
                "files_count": len(files_changed),
                "file_types": list(set(Path(f).suffix for f in files_changed))
            }
            
            return self.log_change(
                change_type=ChangeType.CODE_CHANGE,
                description=description or f"Code changed: {len(files_changed)} files",
                author=author,
                affected_files=files_changed,
                metadata=metadata,
                tags=["code"]
            )
            
        except Exception as e:
            logger.error("Failed to log code change", error=str(e))
            return ""
    
    def log_experiment_run(self, experiment_name: str, metrics: Dict[str, Any],
                          description: str = "", author: str = "ML Pipeline") -> str:
        """Log an experiment run."""
        try:
            metadata = {
                "experiment_name": experiment_name,
                "metrics": metrics,
                "success": metrics.get("success", True)
            }
            
            return self.log_change(
                change_type=ChangeType.EXPERIMENT_RUN,
                description=description or f"Experiment {experiment_name} completed",
                author=author,
                affected_files=[],
                metadata=metadata,
                tags=["experiment"]
            )
            
        except Exception as e:
            logger.error("Failed to log experiment run", error=str(e))
            return ""
    
    def _compute_config_changes(self, old_config: Dict[str, Any], 
                               new_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Compute changes between two configurations."""
        changes = []
        
        # Get all keys
        all_keys = set(old_config.keys()) | set(new_config.keys())
        
        for key in all_keys:
            if key not in old_config:
                changes.append({
                    "path": key,
                    "type": "added",
                    "old_value": None,
                    "new_value": new_config[key]
                })
            elif key not in new_config:
                changes.append({
                    "path": key,
                    "type": "removed",
                    "old_value": old_config[key],
                    "new_value": None
                })
            elif old_config[key] != new_config[key]:
                changes.append({
                    "path": key,
                    "type": "modified",
                    "old_value": old_config[key],
                    "new_value": new_config[key]
                })
        
        return changes
    
    def get_changes(self, change_type: Optional[ChangeType] = None,
                   author: Optional[str] = None, severity: Optional[str] = None,
                   limit: Optional[int] = None, 
                   start_time: Optional[float] = None,
                   end_time: Optional[float] = None) -> List[ChangeEntry]:
        """Get filtered change entries."""
        try:
            filtered_entries = self.entries
            
            # Filter by change type
            if change_type:
                filtered_entries = [e for e in filtered_entries if e.change_type == change_type]
            
            # Filter by author
            if author:
                filtered_entries = [e for e in filtered_entries if e.author == author]
            
            # Filter by severity
            if severity:
                filtered_entries = [e for e in filtered_entries if e.severity == severity]
            
            # Filter by time range
            if start_time:
                filtered_entries = [e for e in filtered_entries if e.timestamp >= start_time]
            
            if end_time:
                filtered_entries = [e for e in filtered_entries if e.timestamp <= end_time]
            
            # Apply limit
            if limit:
                filtered_entries = filtered_entries[:limit]
            
            return filtered_entries
            
        except Exception as e:
            logger.error("Failed to get changes", error=str(e))
            return []
    
    def get_change_log(self, limit: Optional[int] = None) -> ChangeLog:
        """Get the complete change log."""
        try:
            entries = self.entries
            if limit:
                entries = entries[:limit]
            
            return ChangeLog(
                entries=entries,
                total_entries=len(self.entries),
                first_entry=self.entries[-1] if self.entries else None,
                last_entry=self.entries[0] if self.entries else None
            )
            
        except Exception as e:
            logger.error("Failed to get change log", error=str(e))
            return ChangeLog(entries=[], total_entries=0, first_entry=None, last_entry=None)
    
    def get_entry(self, entry_id: str) -> Optional[ChangeEntry]:
        """Get a specific change entry by ID."""
        try:
            for entry in self.entries:
                if entry.id == entry_id:
                    return entry
            return None
            
        except Exception as e:
            logger.error("Failed to get entry", entry_id=entry_id, error=str(e))
            return None
    
    def delete_entry(self, entry_id: str) -> bool:
        """Delete a specific change entry."""
        try:
            for i, entry in enumerate(self.entries):
                if entry.id == entry_id:
                    del self.entries[i]
                    self._save_log()
                    logger.info("Change entry deleted", entry_id=entry_id)
                    return True
            
            logger.warning("Entry not found for deletion", entry_id=entry_id)
            return False
            
        except Exception as e:
            logger.error("Failed to delete entry", entry_id=entry_id, error=str(e))
            return False
    
    def update_entry(self, entry_id: str, **kwargs) -> bool:
        """Update a specific change entry."""
        try:
            entry = self.get_entry(entry_id)
            if not entry:
                logger.warning("Entry not found for update", entry_id=entry_id)
                return False
            
            # Update fields
            for key, value in kwargs.items():
                if hasattr(entry, key):
                    setattr(entry, key, value)
            
            # Update timestamp
            entry.timestamp = time.time()
            
            self._save_log()
            logger.info("Change entry updated", entry_id=entry_id)
            return True
            
        except Exception as e:
            logger.error("Failed to update entry", entry_id=entry_id, error=str(e))
            return False
    
    def search_changes(self, query: str) -> List[ChangeEntry]:
        """Search for changes by description, author, or tags."""
        try:
            results = []
            query_lower = query.lower()
            
            for entry in self.entries:
                # Search in description
                if query_lower in entry.description.lower():
                    results.append(entry)
                    continue
                
                # Search in author
                if query_lower in entry.author.lower():
                    results.append(entry)
                    continue
                
                # Search in tags
                for tag in entry.tags:
                    if query_lower in tag.lower():
                        results.append(entry)
                        break
            
            logger.info("Changes searched", query=query, results=len(results))
            return results
            
        except Exception as e:
            logger.error("Failed to search changes", query=query, error=str(e))
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get change tracking statistics."""
        try:
            stats = {
                "total_entries": len(self.entries),
                "entries_by_type": {},
                "entries_by_author": {},
                "entries_by_severity": {},
                "recent_activity": {}
            }
            
            # Count by change type
            for entry in self.entries:
                change_type = entry.change_type.value
                stats["entries_by_type"][change_type] = stats["entries_by_type"].get(change_type, 0) + 1
            
            # Count by author
            for entry in self.entries:
                author = entry.author
                stats["entries_by_author"][author] = stats["entries_by_author"].get(author, 0) + 1
            
            # Count by severity
            for entry in self.entries:
                severity = entry.severity
                stats["entries_by_severity"][severity] = stats["entries_by_severity"].get(severity, 0) + 1
            
            # Recent activity (last 7 days)
            week_ago = time.time() - (7 * 24 * 60 * 60)
            recent_entries = [e for e in self.entries if e.timestamp >= week_ago]
            stats["recent_activity"]["last_7_days"] = len(recent_entries)
            
            # Recent activity (last 24 hours)
            day_ago = time.time() - (24 * 60 * 60)
            recent_entries = [e for e in self.entries if e.timestamp >= day_ago]
            stats["recent_activity"]["last_24_hours"] = len(recent_entries)
            
            return stats
            
        except Exception as e:
            logger.error("Failed to get statistics", error=str(e))
            return {}
    
    def export_log(self, export_path: str, format: str = "json") -> bool:
        """Export change log to a file."""
        try:
            if format == "json":
                data = {
                    "entries": [
                        {
                            "id": entry.id,
                            "change_type": entry.change_type.value,
                            "description": entry.description,
                            "timestamp": entry.timestamp,
                            "author": entry.author,
                            "affected_files": entry.affected_files,
                            "metadata": entry.metadata,
                            "tags": entry.tags,
                            "severity": entry.severity
                        }
                        for entry in self.entries
                    ],
                    "total_entries": len(self.entries),
                    "export_timestamp": time.time()
                }
                
                with open(export_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    json.dump(data, f, indent=2)
            
            elif format == "csv":
                
                with open(export_path, 'w', newline='') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    writer = csv.writer(f)
                    writer.writerow([
                        "ID", "Change Type", "Description", "Timestamp", 
                        "Author", "Affected Files", "Tags", "Severity"
                    ])
                    
                    for entry in self.entries:
                        writer.writerow([
                            entry.id,
                            entry.change_type.value,
                            entry.description,
                            datetime.fromtimestamp(entry.timestamp).isoformat(),
                            entry.author,
                            ";".join(entry.affected_files),
                            ";".join(entry.tags),
                            entry.severity
                        ])
            
            else:
                logger.error("Unsupported export format", format=format)
                return False
            
            logger.info("Change log exported", export_path=export_path, format=format)
            return True
            
        except Exception as e:
            logger.error("Failed to export change log", 
                        export_path=export_path,
                        format=format,
                        error=str(e))
            return False
    
    def import_log(self, import_path: str, format: str = "json") -> bool:
        """Import change log from a file."""
        try:
            if format == "json":
                with open(import_path, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    data = json.load(f)
                
                for entry_data in data.get("entries", []):
                    entry = ChangeEntry(
                        id=entry_data["id"],
                        change_type=ChangeType(entry_data["change_type"]),
                        description=entry_data["description"],
                        timestamp=entry_data["timestamp"],
                        author=entry_data["author"],
                        affected_files=entry_data["affected_files"],
                        metadata=entry_data["metadata"],
                        tags=entry_data["tags"],
                        severity=entry_data.get("severity", "info")
                    )
                    self.entries.append(entry)
            
            else:
                logger.error("Unsupported import format", format=format)
                return False
            
            # Sort by timestamp
            self.entries.sort(key=lambda x: x.timestamp, reverse=True)
            
            # Save log
            self._save_log()
            
            logger.info("Change log imported", import_path=import_path, format=format)
            return True
            
        except Exception as e:
            logger.error("Failed to import change log", 
                        import_path=import_path,
                        format=format,
                        error=str(e))
            return False
    
    def clear_log(self) -> bool:
        """Clear all change entries."""
        try:
            self.entries = []
            self._save_log()
            logger.info("Change log cleared")
            return True
            
        except Exception as e:
            logger.error("Failed to clear change log", error=str(e))
            return False 