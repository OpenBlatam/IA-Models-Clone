from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

import os
import json
import hashlib
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime
import difflib
import logging
import threading
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from typing import Any, List, Dict, Optional
import asyncio
"""
Change Tracking System
=====================

This module provides comprehensive change tracking for:
- File change monitoring
- Diff generation and visualization
- Change history with metadata
- Automatic change detection
- Integration with version control
- Change analytics and reporting
"""


# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class FileChange:
    """Represents a file change."""
    
    file_path: str
    change_type: str  # "created", "modified", "deleted", "moved"
    timestamp: str
    old_content: Optional[str] = None
    new_content: Optional[str] = None
    diff: Optional[str] = None
    file_hash: Optional[str] = None
    file_size: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FileChange':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ChangeSet:
    """Represents a set of related changes."""
    
    change_set_id: str
    timestamp: str
    description: str
    author: str
    changes: List[FileChange]
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChangeSet':
        """Create from dictionary."""
        changes = [FileChange.from_dict(c) for c in data.get('changes', [])]
        data['changes'] = changes
        return cls(**data)


class FileChangeHandler(FileSystemEventHandler):
    """File system event handler for change tracking."""
    
    def __init__(self, change_tracker: 'ChangeTracker'):
        
    """__init__ function."""
self.change_tracker = change_tracker
        self.pending_changes: Dict[str, float] = {}
        self.debounce_time = 2.0  # seconds
    
    def on_created(self, event) -> Any:
        if not event.is_directory:
            self._schedule_change(event.src_path, "created")
    
    def on_modified(self, event) -> Any:
        if not event.is_directory:
            self._schedule_change(event.src_path, "modified")
    
    def on_deleted(self, event) -> Any:
        if not event.is_directory:
            self._schedule_change(event.src_path, "deleted")
    
    def on_moved(self, event) -> Any:
        if not event.is_directory:
            self._schedule_change(event.dest_path, "moved", old_path=event.src_path)
    
    def _schedule_change(self, file_path: str, change_type: str, old_path: str = None):
        """Schedule a change for processing with debouncing."""
        current_time = time.time()
        self.pending_changes[file_path] = current_time
        
        # Schedule processing after debounce time
        threading.Timer(self.debounce_time, self._process_pending_changes).start()
    
    def _process_pending_changes(self) -> Any:
        """Process pending changes after debounce period."""
        current_time = time.time()
        to_process = []
        
        for file_path, timestamp in self.pending_changes.items():
            if current_time - timestamp >= self.debounce_time:
                to_process.append(file_path)
        
        for file_path in to_process:
            del self.pending_changes[file_path]
            self.change_tracker.track_file_change(file_path)


class ChangeTracker:
    """Main change tracking system."""
    
    def __init__(self, storage_dir: str = "change_history"):
        
    """__init__ function."""
self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Change storage
        self.changes_file = self.storage_dir / "changes.json"
        self.change_sets_file = self.storage_dir / "change_sets.json"
        
        # Load existing data
        self.changes: List[FileChange] = []
        self.change_sets: Dict[str, ChangeSet] = {}
        self._load_data()
        
        # File monitoring
        self.observer = None
        self.file_handler = None
        self.monitored_paths: Set[str] = set()
        
        # File hashes cache
        self.file_hashes: Dict[str, str] = {}
        
        logger.info(f"Change tracker initialized: {self.storage_dir}")
    
    def _load_data(self) -> Any:
        """Load existing change data."""
        # Load changes
        if self.changes_file.exists():
            try:
                with open(self.changes_file, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    data = json.load(f)
                    self.changes = [FileChange.from_dict(c) for c in data]
                logger.info(f"Loaded {len(self.changes)} file changes")
            except Exception as e:
                logger.error(f"Failed to load changes: {e}")
                self.changes = []
        
        # Load change sets
        if self.change_sets_file.exists():
            try:
                with open(self.change_sets_file, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    data = json.load(f)
                    self.change_sets = {
                        csid: ChangeSet.from_dict(cs_data)
                        for csid, cs_data in data.items()
                    }
                logger.info(f"Loaded {len(self.change_sets)} change sets")
            except Exception as e:
                logger.error(f"Failed to load change sets: {e}")
                self.change_sets = {}
    
    def _save_data(self) -> Any:
        """Save change data."""
        try:
            # Save changes
            with open(self.changes_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                json.dump([c.to_dict() for c in self.changes], f, indent=2)
            
            # Save change sets
            with open(self.change_sets_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                json.dump(
                    {csid: cs.to_dict() for csid, cs in self.change_sets.items()},
                    f, indent=2
                )
        except Exception as e:
            logger.error(f"Failed to save data: {e}")
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate hash of file content."""
        try:
            with open(file_path, 'rb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                return hashlib.sha256(f.read()).hexdigest()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        except Exception:
            return ""
    
    def _read_file_content(self, file_path: str) -> Optional[str]:
        """Read file content safely."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                return f.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        except Exception:
            return None
    
    def _generate_diff(self, old_content: str, new_content: str) -> str:
        """Generate diff between old and new content."""
        old_lines = old_content.splitlines()
        new_lines = new_content.splitlines()
        
        diff_lines = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile='old',
            tofile='new',
            lineterm=''
        )
        
        return '\n'.join(diff_lines)
    
    def track_file_change(
        self,
        file_path: str,
        change_type: str = "modified",
        old_path: str = None,
        description: str = "",
        metadata: Dict[str, Any] = None
    ) -> Optional[FileChange]:
        """Track a file change."""
        try:
            file_path = str(Path(file_path).resolve())
            timestamp = datetime.now().isoformat()
            
            # Get file info
            file_size = None
            file_hash = None
            old_content = None
            new_content = None
            
            if Path(file_path).exists():
                file_size = Path(file_path).stat().st_size
                file_hash = self._calculate_file_hash(file_path)
                new_content = self._read_file_content(file_path)
            
            # Get old content if available
            if change_type == "modified" and file_path in self.file_hashes:
                # Try to get from previous change
                for change in reversed(self.changes):
                    if change.file_path == file_path and change.new_content:
                        old_content = change.new_content
                        break
            
            # Generate diff if both contents available
            diff = None
            if old_content and new_content:
                diff = self._generate_diff(old_content, new_content)
            
            # Create change object
            change = FileChange(
                file_path=file_path,
                change_type=change_type,
                timestamp=timestamp,
                old_content=old_content,
                new_content=new_content,
                diff=diff,
                file_hash=file_hash,
                file_size=file_size,
                metadata=metadata or {}
            )
            
            # Store change
            self.changes.append(change)
            
            # Update file hash cache
            if file_hash:
                self.file_hashes[file_path] = file_hash
            
            # Save data
            self._save_data()
            
            logger.info(f"Tracked {change_type} change: {file_path}")
            return change
            
        except Exception as e:
            logger.error(f"Failed to track file change: {e}")
            return None
    
    def start_monitoring(self, paths: List[str], recursive: bool = True):
        """Start monitoring file system changes."""
        if self.observer is None:
            self.observer = Observer()
            self.file_handler = FileChangeHandler(self)
        
        for path in paths:
            path = str(Path(path).resolve())
            if path not in self.monitored_paths:
                try:
                    self.observer.schedule(
                        self.file_handler,
                        path,
                        recursive=recursive
                    )
                    self.monitored_paths.add(path)
                    logger.info(f"Started monitoring: {path}")
                except Exception as e:
                    logger.error(f"Failed to monitor {path}: {e}")
        
        if not self.observer.is_alive():
            self.observer.start()
    
    def stop_monitoring(self) -> Any:
        """Stop monitoring file system changes."""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None
            self.file_handler = None
            self.monitored_paths.clear()
            logger.info("Stopped file monitoring")
    
    def create_change_set(
        self,
        description: str,
        author: str = "system",
        tags: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Create a change set from recent changes."""
        # Get recent changes (last 5 minutes)
        cutoff_time = datetime.now().timestamp() - 300  # 5 minutes
        
        recent_changes = [
            change for change in self.changes
            if datetime.fromisoformat(change.timestamp).timestamp() > cutoff_time
        ]
        
        if not recent_changes:
            logger.warning("No recent changes to create change set")
            return ""
        
        # Generate change set ID
        change_set_id = hashlib.sha256(
            f"{description}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]
        
        # Create change set
        change_set = ChangeSet(
            change_set_id=change_set_id,
            timestamp=datetime.now().isoformat(),
            description=description,
            author=author,
            changes=recent_changes,
            tags=tags or [],
            metadata=metadata or {}
        )
        
        # Store change set
        self.change_sets[change_set_id] = change_set
        
        # Save data
        self._save_data()
        
        logger.info(f"Created change set: {change_set_id}")
        return change_set_id
    
    def get_changes(
        self,
        file_path: str = None,
        change_type: str = None,
        since: str = None,
        until: str = None,
        limit: int = None
    ) -> List[FileChange]:
        """Get filtered changes."""
        changes = self.changes.copy()
        
        # Filter by file path
        if file_path:
            file_path = str(Path(file_path).resolve())
            changes = [c for c in changes if c.file_path == file_path]
        
        # Filter by change type
        if change_type:
            changes = [c for c in changes if c.change_type == change_type]
        
        # Filter by time range
        if since:
            since_dt = datetime.fromisoformat(since)
            changes = [c for c in changes if datetime.fromisoformat(c.timestamp) >= since_dt]
        
        if until:
            until_dt = datetime.fromisoformat(until)
            changes = [c for c in changes if datetime.fromisoformat(c.timestamp) <= until_dt]
        
        # Sort by timestamp (newest first)
        changes.sort(key=lambda c: c.timestamp, reverse=True)
        
        # Apply limit
        if limit:
            changes = changes[:limit]
        
        return changes
    
    def get_change_sets(
        self,
        author: str = None,
        tags: List[str] = None,
        since: str = None,
        limit: int = None
    ) -> List[ChangeSet]:
        """Get filtered change sets."""
        change_sets = list(self.change_sets.values())
        
        # Filter by author
        if author:
            change_sets = [cs for cs in change_sets if cs.author == author]
        
        # Filter by tags
        if tags:
            change_sets = [cs for cs in change_sets if any(tag in cs.tags for tag in tags)]
        
        # Filter by time
        if since:
            since_dt = datetime.fromisoformat(since)
            change_sets = [cs for cs in change_sets if datetime.fromisoformat(cs.timestamp) >= since_dt]
        
        # Sort by timestamp (newest first)
        change_sets.sort(key=lambda cs: cs.timestamp, reverse=True)
        
        # Apply limit
        if limit:
            change_sets = change_sets[:limit]
        
        return change_sets
    
    def get_file_history(self, file_path: str, limit: int = 50) -> List[FileChange]:
        """Get complete history of a file."""
        file_path = str(Path(file_path).resolve())
        
        changes = [c for c in self.changes if c.file_path == file_path]
        changes.sort(key=lambda c: c.timestamp, reverse=True)
        
        if limit:
            changes = changes[:limit]
        
        return changes
    
    def get_change_statistics(self) -> Dict[str, Any]:
        """Get change statistics."""
        if not self.changes:
            return {
                "total_changes": 0,
                "files_changed": 0,
                "change_types": {},
                "date_range": None,
                "most_active_files": [],
                "most_active_authors": []
            }
        
        # Basic counts
        total_changes = len(self.changes)
        files_changed = len(set(c.file_path for c in self.changes))
        
        # Change types
        change_types = {}
        for change in self.changes:
            change_types[change.change_type] = change_types.get(change.change_type, 0) + 1
        
        # Date range
        timestamps = [c.timestamp for c in self.changes]
        date_range = {
            "earliest": min(timestamps),
            "latest": max(timestamps)
        }
        
        # Most active files
        file_counts = {}
        for change in self.changes:
            file_counts[change.file_path] = file_counts.get(change.file_path, 0) + 1
        
        most_active_files = sorted(
            file_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # Most active authors (from change sets)
        author_counts = {}
        for change_set in self.change_sets.values():
            author_counts[change_set.author] = author_counts.get(change_set.author, 0) + 1
        
        most_active_authors = sorted(
            author_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return {
            "total_changes": total_changes,
            "files_changed": files_changed,
            "change_types": change_types,
            "date_range": date_range,
            "most_active_files": most_active_files,
            "most_active_authors": most_active_authors
        }
    
    def export_changes(
        self,
        export_path: str,
        file_path: str = None,
        since: str = None,
        until: str = None
    ) -> bool:
        """Export changes to a file."""
        try:
            changes = self.get_changes(file_path, since=since, until=until)
            
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "filters": {
                    "file_path": file_path,
                    "since": since,
                    "until": until
                },
                "changes": [c.to_dict() for c in changes]
            }
            
            with open(export_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Exported {len(changes)} changes to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export changes: {e}")
            return False
    
    def cleanup_old_changes(self, days: int = 30):
        """Clean up changes older than specified days."""
        cutoff_time = datetime.now().timestamp() - (days * 24 * 3600)
        
        original_count = len(self.changes)
        self.changes = [
            c for c in self.changes
            if datetime.fromisoformat(c.timestamp).timestamp() > cutoff_time
        ]
        
        removed_count = original_count - len(self.changes)
        
        # Save data
        self._save_data()
        
        logger.info(f"Cleaned up {removed_count} old changes")
    
    def get_diff_summary(self, change: FileChange) -> Dict[str, Any]:
        """Get a summary of changes in a diff."""
        if not change.diff:
            return {"lines_added": 0, "lines_removed": 0, "files_changed": 0}
        
        lines_added = 0
        lines_removed = 0
        
        for line in change.diff.splitlines():
            if line.startswith('+') and not line.startswith('+++'):
                lines_added += 1
            elif line.startswith('-') and not line.startswith('---'):
                lines_removed += 1
        
        return {
            "lines_added": lines_added,
            "lines_removed": lines_removed,
            "files_changed": 1
        }


# Convenience functions
def create_change_tracker(storage_dir: str = "change_history") -> ChangeTracker:
    """Create change tracker with default settings."""
    return ChangeTracker(storage_dir)


def track_file_change(
    file_path: str,
    change_type: str = "modified",
    description: str = "",
    storage_dir: str = "change_history"
) -> Optional[FileChange]:
    """Quick function to track a file change."""
    tracker = create_change_tracker(storage_dir)
    return tracker.track_file_change(file_path, change_type, description=description)


if __name__ == "__main__":
    # Example usage
    print("🔧 Change Tracking System")
    print("=" * 40)
    
    # Create change tracker
    tracker = create_change_tracker()
    
    # Create test file
    test_file = "test_file.txt"
    with open(test_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        f.write("Initial content")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
    
    # Track initial creation
    change1 = tracker.track_file_change(test_file, "created", description="Initial file creation")
    print(f"Tracked creation: {change1.file_path}")
    
    # Modify file
    with open(test_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        f.write("Modified content")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
    
    # Track modification
    change2 = tracker.track_file_change(test_file, "modified", description="File modification")
    print(f"Tracked modification: {change2.file_path}")
    
    # Get changes
    changes = tracker.get_changes(test_file)
    print(f"Total changes for {test_file}: {len(changes)}")
    
    # Get statistics
    stats = tracker.get_change_statistics()
    print(f"Change statistics: {stats}")
    
    # Create change set
    change_set_id = tracker.create_change_set(
        "Test changes",
        author="developer",
        tags=["test", "example"]
    )
    print(f"Created change set: {change_set_id}")
    
    # Cleanup
    if os.path.exists(test_file):
        os.remove(test_file)
    
    print("✅ Change tracking example completed!") 