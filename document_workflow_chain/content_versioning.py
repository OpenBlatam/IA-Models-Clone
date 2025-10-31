"""
Content Versioning - Advanced version control system for document workflow chains
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
import hashlib
import difflib
from collections import defaultdict, deque
import copy

logger = logging.getLogger(__name__)

class VersionType(Enum):
    """Types of content versions"""
    MAJOR = "major"
    MINOR = "minor"
    PATCH = "patch"
    DRAFT = "draft"
    REVIEW = "review"
    PUBLISHED = "published"
    ARCHIVED = "archived"

class ChangeType(Enum):
    """Types of changes in content"""
    ADDITION = "addition"
    DELETION = "deletion"
    MODIFICATION = "modification"
    FORMATTING = "formatting"
    STRUCTURE = "structure"
    METADATA = "metadata"

@dataclass
class ContentChange:
    """Represents a change in content"""
    change_id: str
    change_type: ChangeType
    description: str
    old_content: Optional[str] = None
    new_content: Optional[str] = None
    location: Optional[Dict[str, Any]] = None
    author: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ContentVersion:
    """Represents a version of content"""
    version_id: str
    content_id: str
    version_number: str
    version_type: VersionType
    content: str
    content_hash: str
    changes: List[ContentChange] = field(default_factory=list)
    parent_version_id: Optional[str] = None
    branch_name: str = "main"
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    is_published: bool = False
    quality_score: Optional[float] = None

@dataclass
class VersionBranch:
    """Represents a version branch"""
    branch_name: str
    content_id: str
    head_version_id: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    description: str = ""
    is_active: bool = True

@dataclass
class VersionComparison:
    """Represents a comparison between two versions"""
    comparison_id: str
    version1_id: str
    version2_id: str
    changes: List[ContentChange] = field(default_factory=list)
    similarity_score: float = 0.0
    diff_summary: Dict[str, int] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

class ContentVersionManager:
    """Advanced content versioning system with branching, merging, and collaboration features"""
    
    def __init__(self):
        self.versions: Dict[str, ContentVersion] = {}
        self.branches: Dict[str, VersionBranch] = {}
        self.comparisons: Dict[str, VersionComparison] = {}
        self.version_history: Dict[str, List[str]] = defaultdict(list)  # content_id -> version_ids
        self.branch_history: Dict[str, List[str]] = defaultdict(list)  # content_id -> branch_names
        
        # Version numbering
        self.version_counters: Dict[str, Dict[str, int]] = defaultdict(lambda: {
            "major": 0, "minor": 0, "patch": 0
        })
        
        # Collaboration features
        self.locks: Dict[str, Dict[str, Any]] = {}  # content_id -> lock_info
        self.comments: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        logger.info("ContentVersionManager initialized")

    async def create_initial_version(
        self,
        content_id: str,
        content: str,
        version_type: VersionType = VersionType.MAJOR,
        author: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create the initial version of content"""
        version_number = self._generate_version_number(content_id, version_type)
        version_id = str(uuid.uuid4())
        content_hash = self._calculate_content_hash(content)
        
        version = ContentVersion(
            version_id=version_id,
            content_id=content_id,
            version_number=version_number,
            version_type=version_type,
            content=content,
            content_hash=content_hash,
            created_by=author,
            metadata=metadata or {}
        )
        
        self.versions[version_id] = version
        self.version_history[content_id].append(version_id)
        
        # Create main branch
        branch = VersionBranch(
            branch_name="main",
            content_id=content_id,
            head_version_id=version_id,
            created_by=author
        )
        self.branches[f"{content_id}:main"] = branch
        self.branch_history[content_id].append("main")
        
        logger.info(f"Created initial version {version_number} for content {content_id}")
        return version_id

    async def create_new_version(
        self,
        content_id: str,
        new_content: str,
        version_type: VersionType,
        author: Optional[str] = None,
        description: str = "",
        branch_name: str = "main",
        parent_version_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new version of content"""
        # Get parent version
        if parent_version_id is None:
            parent_version_id = self._get_head_version(content_id, branch_name)
            
        if parent_version_id is None:
            raise ValueError(f"No parent version found for content {content_id} in branch {branch_name}")
            
        parent_version = self.versions[parent_version_id]
        
        # Calculate changes
        changes = await self._calculate_changes(parent_version.content, new_content, description)
        
        # Generate version number
        version_number = self._generate_version_number(content_id, version_type)
        version_id = str(uuid.uuid4())
        content_hash = self._calculate_content_hash(new_content)
        
        # Create new version
        version = ContentVersion(
            version_id=version_id,
            content_id=content_id,
            version_number=version_number,
            version_type=version_type,
            content=new_content,
            content_hash=content_hash,
            changes=changes,
            parent_version_id=parent_version_id,
            branch_name=branch_name,
            created_by=author,
            metadata=metadata or {}
        )
        
        self.versions[version_id] = version
        self.version_history[content_id].append(version_id)
        
        # Update branch head
        branch_key = f"{content_id}:{branch_name}"
        if branch_key in self.branches:
            self.branches[branch_key].head_version_id = version_id
        
        logger.info(f"Created version {version_number} for content {content_id} in branch {branch_name}")
        return version_id

    async def create_branch(
        self,
        content_id: str,
        branch_name: str,
        from_version_id: Optional[str] = None,
        author: Optional[str] = None,
        description: str = ""
    ) -> str:
        """Create a new branch from a specific version"""
        if from_version_id is None:
            from_version_id = self._get_head_version(content_id, "main")
            
        if from_version_id is None:
            raise ValueError(f"No version found to create branch from for content {content_id}")
            
        branch_key = f"{content_id}:{branch_name}"
        if branch_key in self.branches:
            raise ValueError(f"Branch {branch_name} already exists for content {content_id}")
            
        branch = VersionBranch(
            branch_name=branch_name,
            content_id=content_id,
            head_version_id=from_version_id,
            created_by=author,
            description=description
        )
        
        self.branches[branch_key] = branch
        self.branch_history[content_id].append(branch_name)
        
        logger.info(f"Created branch {branch_name} for content {content_id}")
        return branch_name

    async def merge_branch(
        self,
        content_id: str,
        source_branch: str,
        target_branch: str = "main",
        author: Optional[str] = None,
        merge_strategy: str = "auto"
    ) -> str:
        """Merge changes from one branch to another"""
        source_branch_key = f"{content_id}:{source_branch}"
        target_branch_key = f"{content_id}:{target_branch}"
        
        if source_branch_key not in self.branches:
            raise ValueError(f"Source branch {source_branch} not found")
        if target_branch_key not in self.branches:
            raise ValueError(f"Target branch {target_branch} not found")
            
        source_head = self.branches[source_branch_key].head_version_id
        target_head = self.branches[target_branch_key].head_version_id
        
        source_version = self.versions[source_head]
        target_version = self.versions[target_head]
        
        # Perform merge based on strategy
        if merge_strategy == "auto":
            merged_content = await self._auto_merge(source_version.content, target_version.content)
        elif merge_strategy == "source":
            merged_content = source_version.content
        elif merge_strategy == "target":
            merged_content = target_version.content
        else:
            raise ValueError(f"Unknown merge strategy: {merge_strategy}")
            
        # Create merge version
        version_id = await self.create_new_version(
            content_id=content_id,
            new_content=merged_content,
            version_type=VersionType.MINOR,
            author=author,
            description=f"Merged {source_branch} into {target_branch}",
            branch_name=target_branch,
            parent_version_id=target_head
        )
        
        logger.info(f"Merged branch {source_branch} into {target_branch} for content {content_id}")
        return version_id

    async def compare_versions(
        self,
        version1_id: str,
        version2_id: str,
        include_metadata: bool = False
    ) -> VersionComparison:
        """Compare two versions of content"""
        if version1_id not in self.versions or version2_id not in self.versions:
            raise ValueError("One or both versions not found")
            
        version1 = self.versions[version1_id]
        version2 = self.versions[version2_id]
        
        # Calculate changes
        changes = await self._calculate_changes(version1.content, version2.content, "Version comparison")
        
        # Calculate similarity
        similarity_score = self._calculate_similarity(version1.content, version2.content)
        
        # Generate diff summary
        diff_summary = self._generate_diff_summary(changes)
        
        comparison = VersionComparison(
            comparison_id=str(uuid.uuid4()),
            version1_id=version1_id,
            version2_id=version2_id,
            changes=changes,
            similarity_score=similarity_score,
            diff_summary=diff_summary
        )
        
        self.comparisons[comparison.comparison_id] = comparison
        
        logger.info(f"Compared versions {version1.version_number} and {version2.version_number}")
        return comparison

    async def get_version_history(
        self,
        content_id: str,
        branch_name: Optional[str] = None,
        limit: int = 100
    ) -> List[ContentVersion]:
        """Get version history for content"""
        if branch_name:
            # Get versions for specific branch
            versions = [
                v for v in self.versions.values()
                if v.content_id == content_id and v.branch_name == branch_name
            ]
        else:
            # Get all versions for content
            versions = [
                v for v in self.versions.values()
                if v.content_id == content_id
            ]
            
        # Sort by creation time
        versions.sort(key=lambda v: v.created_at, reverse=True)
        
        return versions[:limit]

    async def get_branches(self, content_id: str) -> List[VersionBranch]:
        """Get all branches for content"""
        branches = [
            branch for branch in self.branches.values()
            if branch.content_id == content_id
        ]
        return branches

    async def get_version(self, version_id: str) -> Optional[ContentVersion]:
        """Get a specific version"""
        return self.versions.get(version_id)

    async def get_latest_version(
        self,
        content_id: str,
        branch_name: str = "main"
    ) -> Optional[ContentVersion]:
        """Get the latest version for a branch"""
        branch_key = f"{content_id}:{branch_name}"
        if branch_key not in self.branches:
            return None
            
        head_version_id = self.branches[branch_key].head_version_id
        return self.versions.get(head_version_id)

    async def tag_version(
        self,
        version_id: str,
        tag: str,
        author: Optional[str] = None
    ) -> bool:
        """Add a tag to a version"""
        if version_id not in self.versions:
            return False
            
        version = self.versions[version_id]
        if tag not in version.tags:
            version.tags.append(tag)
            logger.info(f"Added tag '{tag}' to version {version.version_number}")
            return True
            
        return False

    async def publish_version(
        self,
        version_id: str,
        author: Optional[str] = None
    ) -> bool:
        """Mark a version as published"""
        if version_id not in self.versions:
            return False
            
        version = self.versions[version_id]
        version.is_published = True
        version.version_type = VersionType.PUBLISHED
        
        logger.info(f"Published version {version.version_number}")
        return True

    async def archive_version(
        self,
        version_id: str,
        author: Optional[str] = None
    ) -> bool:
        """Archive a version"""
        if version_id not in self.versions:
            return False
            
        version = self.versions[version_id]
        version.version_type = VersionType.ARCHIVED
        
        logger.info(f"Archived version {version.version_number}")
        return True

    async def lock_content(
        self,
        content_id: str,
        user_id: str,
        branch_name: str = "main",
        lock_type: str = "edit"
    ) -> bool:
        """Lock content for editing"""
        lock_key = f"{content_id}:{branch_name}"
        
        if lock_key in self.locks:
            existing_lock = self.locks[lock_key]
            if existing_lock["user_id"] != user_id:
                return False  # Already locked by another user
                
        self.locks[lock_key] = {
            "user_id": user_id,
            "branch_name": branch_name,
            "lock_type": lock_type,
            "locked_at": datetime.utcnow(),
            "expires_at": datetime.utcnow() + timedelta(hours=1)
        }
        
        logger.info(f"Locked content {content_id} in branch {branch_name} for user {user_id}")
        return True

    async def unlock_content(
        self,
        content_id: str,
        user_id: str,
        branch_name: str = "main"
    ) -> bool:
        """Unlock content"""
        lock_key = f"{content_id}:{branch_name}"
        
        if lock_key in self.locks:
            lock_info = self.locks[lock_key]
            if lock_info["user_id"] == user_id:
                del self.locks[lock_key]
                logger.info(f"Unlocked content {content_id} in branch {branch_name}")
                return True
                
        return False

    async def add_comment(
        self,
        version_id: str,
        comment: str,
        author: str,
        location: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a comment to a version"""
        comment_id = str(uuid.uuid4())
        comment_data = {
            "comment_id": comment_id,
            "version_id": version_id,
            "comment": comment,
            "author": author,
            "location": location,
            "created_at": datetime.utcnow()
        }
        
        self.comments[version_id].append(comment_data)
        
        logger.info(f"Added comment to version {version_id}")
        return comment_id

    async def get_comments(self, version_id: str) -> List[Dict[str, Any]]:
        """Get comments for a version"""
        return self.comments.get(version_id, [])

    async def rollback_to_version(
        self,
        content_id: str,
        target_version_id: str,
        author: Optional[str] = None,
        branch_name: str = "main"
    ) -> str:
        """Rollback to a previous version"""
        if target_version_id not in self.versions:
            raise ValueError("Target version not found")
            
        target_version = self.versions[target_version_id]
        
        # Create new version with the old content
        version_id = await self.create_new_version(
            content_id=content_id,
            new_content=target_version.content,
            version_type=VersionType.PATCH,
            author=author,
            description=f"Rollback to version {target_version.version_number}",
            branch_name=branch_name
        )
        
        logger.info(f"Rolled back content {content_id} to version {target_version.version_number}")
        return version_id

    def _generate_version_number(self, content_id: str, version_type: VersionType) -> str:
        """Generate version number based on type"""
        counter = self.version_counters[content_id]
        
        if version_type == VersionType.MAJOR:
            counter["major"] += 1
            counter["minor"] = 0
            counter["patch"] = 0
            return f"{counter['major']}.0.0"
        elif version_type == VersionType.MINOR:
            counter["minor"] += 1
            counter["patch"] = 0
            return f"{counter['major']}.{counter['minor']}.0"
        elif version_type == VersionType.PATCH:
            counter["patch"] += 1
            return f"{counter['major']}.{counter['minor']}.{counter['patch']}"
        else:
            # For draft, review, etc., use timestamp
            timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
            return f"{counter['major']}.{counter['minor']}.{counter['patch']}-{version_type.value}-{timestamp}"

    def _calculate_content_hash(self, content: str) -> str:
        """Calculate hash of content"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def _get_head_version(self, content_id: str, branch_name: str) -> Optional[str]:
        """Get the head version for a branch"""
        branch_key = f"{content_id}:{branch_name}"
        if branch_key in self.branches:
            return self.branches[branch_key].head_version_id
        return None

    async def _calculate_changes(
        self,
        old_content: str,
        new_content: str,
        description: str
    ) -> List[ContentChange]:
        """Calculate changes between two content versions"""
        changes = []
        
        # Use difflib to find differences
        differ = difflib.unified_diff(
            old_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile='old',
            tofile='new',
            lineterm=''
        )
        
        diff_lines = list(differ)
        
        if not diff_lines:
            return changes
            
        # Parse diff to create change objects
        current_change = None
        line_number = 0
        
        for line in diff_lines:
            if line.startswith('@@'):
                # New hunk
                if current_change:
                    changes.append(current_change)
                    
                # Parse line numbers
                parts = line.split()
                if len(parts) >= 3:
                    old_range = parts[1]
                    new_range = parts[2]
                    
                    current_change = ContentChange(
                        change_id=str(uuid.uuid4()),
                        change_type=ChangeType.MODIFICATION,
                        description=description,
                        location={
                            "old_range": old_range,
                            "new_range": new_range
                        }
                    )
            elif line.startswith('-'):
                # Deletion
                if current_change:
                    current_change.change_type = ChangeType.DELETION
                    current_change.old_content = line[1:]
            elif line.startswith('+'):
                # Addition
                if current_change:
                    if current_change.change_type == ChangeType.DELETION:
                        current_change.change_type = ChangeType.MODIFICATION
                    else:
                        current_change.change_type = ChangeType.ADDITION
                    current_change.new_content = line[1:]
                    
        if current_change:
            changes.append(current_change)
            
        return changes

    def _calculate_similarity(self, content1: str, content2: str) -> float:
        """Calculate similarity between two content versions"""
        return difflib.SequenceMatcher(None, content1, content2).ratio()

    def _generate_diff_summary(self, changes: List[ContentChange]) -> Dict[str, int]:
        """Generate summary of changes"""
        summary = {
            "additions": 0,
            "deletions": 0,
            "modifications": 0,
            "total_changes": len(changes)
        }
        
        for change in changes:
            if change.change_type == ChangeType.ADDITION:
                summary["additions"] += 1
            elif change.change_type == ChangeType.DELETION:
                summary["deletions"] += 1
            elif change.change_type == ChangeType.MODIFICATION:
                summary["modifications"] += 1
                
        return summary

    async def _auto_merge(self, source_content: str, target_content: str) -> str:
        """Automatically merge two content versions"""
        # Simple merge strategy - can be enhanced with more sophisticated algorithms
        if source_content == target_content:
            return source_content
            
        # For now, prefer source content if there are conflicts
        # In a real implementation, this would use more sophisticated merge algorithms
        return source_content

    async def get_version_statistics(self, content_id: str) -> Dict[str, Any]:
        """Get statistics for content versions"""
        versions = [v for v in self.versions.values() if v.content_id == content_id]
        branches = [b for b in self.branches.values() if b.content_id == content_id]
        
        if not versions:
            return {"message": "No versions found"}
            
        # Calculate statistics
        version_types = {}
        for version in versions:
            version_type = version.version_type.value
            version_types[version_type] = version_types.get(version_type, 0) + 1
            
        # Calculate average changes per version
        total_changes = sum(len(v.changes) for v in versions)
        avg_changes = total_changes / len(versions) if versions else 0
        
        # Find most active branch
        branch_activity = {}
        for version in versions:
            branch = version.branch_name
            branch_activity[branch] = branch_activity.get(branch, 0) + 1
            
        most_active_branch = max(branch_activity.items(), key=lambda x: x[1])[0] if branch_activity else None
        
        return {
            "total_versions": len(versions),
            "total_branches": len(branches),
            "version_types": version_types,
            "average_changes_per_version": avg_changes,
            "most_active_branch": most_active_branch,
            "branch_activity": branch_activity,
            "first_version": min(versions, key=lambda v: v.created_at).created_at,
            "last_version": max(versions, key=lambda v: v.created_at).created_at
        }

# Global version manager instance
content_version_manager = ContentVersionManager()

# Convenience functions
async def create_content_version(
    content_id: str,
    content: str,
    version_type: VersionType = VersionType.MAJOR,
    author: Optional[str] = None
) -> str:
    """Create a new content version"""
    return await content_version_manager.create_new_version(
        content_id, content, version_type, author
    )

async def get_content_history(content_id: str, limit: int = 50) -> List[ContentVersion]:
    """Get content version history"""
    return await content_version_manager.get_version_history(content_id, limit=limit)

async def compare_content_versions(version1_id: str, version2_id: str) -> VersionComparison:
    """Compare two content versions"""
    return await content_version_manager.compare_versions(version1_id, version2_id)
