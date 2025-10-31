"""
Version Control Service
======================

Advanced version control and document history management.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, asdict
from uuid import uuid4
import hashlib
import difflib
from pathlib import Path

logger = logging.getLogger(__name__)


class VersionType(str, Enum):
    """Version type."""
    MAJOR = "major"
    MINOR = "minor"
    PATCH = "patch"
    DRAFT = "draft"
    REVIEW = "review"
    APPROVED = "approved"


class ChangeType(str, Enum):
    """Change type."""
    CREATED = "created"
    UPDATED = "updated"
    DELETED = "deleted"
    RENAMED = "renamed"
    MOVED = "moved"
    MERGED = "merged"
    RESTORED = "restored"


@dataclass
class DocumentVersion:
    """Document version."""
    version_id: str
    document_id: str
    version_number: str
    version_type: VersionType
    title: str
    content: str
    metadata: Dict[str, Any]
    created_by: str
    created_at: datetime
    change_summary: str
    file_size: int
    content_hash: str
    parent_version_id: Optional[str] = None
    tags: List[str] = None
    is_current: bool = False
    is_published: bool = False


@dataclass
class VersionChange:
    """Version change."""
    change_id: str
    version_id: str
    change_type: ChangeType
    field_name: str
    old_value: Any
    new_value: Any
    timestamp: datetime
    user_id: str
    description: str


@dataclass
class VersionBranch:
    """Version branch."""
    branch_id: str
    document_id: str
    branch_name: str
    base_version_id: str
    current_version_id: str
    created_by: str
    created_at: datetime
    description: str
    is_merged: bool = False
    merged_at: Optional[datetime] = None


class VersionControlService:
    """Version control service for documents."""
    
    def __init__(self, storage_path: str = "document_versions"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # In-memory storage (in production, use database)
        self.versions: Dict[str, List[DocumentVersion]] = {}  # document_id -> versions
        self.changes: Dict[str, List[VersionChange]] = {}  # version_id -> changes
        self.branches: Dict[str, List[VersionBranch]] = {}  # document_id -> branches
        self.version_graph: Dict[str, Dict[str, List[str]]] = {}  # document_id -> version_id -> children
        
    async def create_initial_version(
        self,
        document_id: str,
        title: str,
        content: str,
        metadata: Dict[str, Any],
        created_by: str,
        version_type: VersionType = VersionType.DRAFT
    ) -> DocumentVersion:
        """Create initial version of a document."""
        
        try:
            version_id = str(uuid4())
            content_hash = self._calculate_content_hash(content)
            
            version = DocumentVersion(
                version_id=version_id,
                document_id=document_id,
                version_number="1.0.0",
                version_type=version_type,
                title=title,
                content=content,
                metadata=metadata,
                created_by=created_by,
                created_at=datetime.now(),
                change_summary="Initial version",
                file_size=len(content.encode('utf-8')),
                content_hash=content_hash,
                is_current=True
            )
            
            # Store version
            if document_id not in self.versions:
                self.versions[document_id] = []
                self.changes[document_id] = []
                self.branches[document_id] = []
                self.version_graph[document_id] = {}
            
            self.versions[document_id].append(version)
            self.version_graph[document_id][version_id] = []
            
            # Save to file
            await self._save_version_to_file(version)
            
            logger.info(f"Created initial version {version.version_number} for document {document_id}")
            
            return version
            
        except Exception as e:
            logger.error(f"Error creating initial version: {str(e)}")
            raise
    
    async def create_new_version(
        self,
        document_id: str,
        title: str,
        content: str,
        metadata: Dict[str, Any],
        created_by: str,
        change_summary: str,
        version_type: VersionType = VersionType.MINOR,
        parent_version_id: Optional[str] = None,
        tags: List[str] = None
    ) -> DocumentVersion:
        """Create new version of a document."""
        
        try:
            if document_id not in self.versions:
                raise ValueError(f"Document {document_id} not found")
            
            # Get current version
            current_version = self._get_current_version(document_id)
            if not current_version:
                raise ValueError(f"No current version found for document {document_id}")
            
            # Calculate new version number
            new_version_number = self._calculate_next_version_number(
                current_version.version_number,
                version_type
            )
            
            # Create new version
            version_id = str(uuid4())
            content_hash = self._calculate_content_hash(content)
            
            new_version = DocumentVersion(
                version_id=version_id,
                document_id=document_id,
                version_number=new_version_number,
                version_type=version_type,
                title=title,
                content=content,
                metadata=metadata,
                created_by=created_by,
                created_at=datetime.now(),
                change_summary=change_summary,
                file_size=len(content.encode('utf-8')),
                content_hash=content_hash,
                parent_version_id=parent_version_id or current_version.version_id,
                tags=tags or [],
                is_current=True
            )
            
            # Mark previous version as not current
            current_version.is_current = False
            
            # Store new version
            self.versions[document_id].append(new_version)
            
            # Update version graph
            parent_id = parent_version_id or current_version.version_id
            if parent_id in self.version_graph[document_id]:
                self.version_graph[document_id][parent_id].append(version_id)
            self.version_graph[document_id][version_id] = []
            
            # Track changes
            await self._track_version_changes(current_version, new_version, created_by)
            
            # Save to file
            await self._save_version_to_file(new_version)
            await self._save_version_to_file(current_version)
            
            logger.info(f"Created new version {new_version.version_number} for document {document_id}")
            
            return new_version
            
        except Exception as e:
            logger.error(f"Error creating new version: {str(e)}")
            raise
    
    async def get_document_versions(
        self,
        document_id: str,
        limit: int = 50,
        offset: int = 0,
        version_type: Optional[VersionType] = None
    ) -> List[DocumentVersion]:
        """Get versions of a document."""
        
        if document_id not in self.versions:
            return []
        
        versions = self.versions[document_id]
        
        # Filter by version type if specified
        if version_type:
            versions = [v for v in versions if v.version_type == version_type]
        
        # Sort by creation date (newest first)
        versions.sort(key=lambda x: x.created_at, reverse=True)
        
        # Apply pagination
        return versions[offset:offset + limit]
    
    async def get_version_by_id(self, document_id: str, version_id: str) -> Optional[DocumentVersion]:
        """Get specific version by ID."""
        
        if document_id not in self.versions:
            return None
        
        for version in self.versions[document_id]:
            if version.version_id == version_id:
                return version
        
        return None
    
    async def get_current_version(self, document_id: str) -> Optional[DocumentVersion]:
        """Get current version of a document."""
        
        return self._get_current_version(document_id)
    
    async def compare_versions(
        self,
        document_id: str,
        version1_id: str,
        version2_id: str
    ) -> Dict[str, Any]:
        """Compare two versions of a document."""
        
        try:
            version1 = await self.get_version_by_id(document_id, version1_id)
            version2 = await self.get_version_by_id(document_id, version2_id)
            
            if not version1 or not version2:
                raise ValueError("One or both versions not found")
            
            # Calculate differences
            content_diff = self._calculate_content_diff(version1.content, version2.content)
            metadata_diff = self._calculate_metadata_diff(version1.metadata, version2.metadata)
            
            # Calculate statistics
            stats = {
                "lines_added": len([line for line in content_diff if line.startswith('+')]),
                "lines_removed": len([line for line in content_diff if line.startswith('-')]),
                "lines_unchanged": len([line for line in content_diff if line.startswith(' ')]),
                "content_similarity": self._calculate_similarity(version1.content, version2.content),
                "size_change": version2.file_size - version1.file_size,
                "size_change_percent": ((version2.file_size - version1.file_size) / version1.file_size * 100) if version1.file_size > 0 else 0
            }
            
            return {
                "version1": {
                    "version_id": version1.version_id,
                    "version_number": version1.version_number,
                    "created_at": version1.created_at.isoformat(),
                    "created_by": version1.created_by
                },
                "version2": {
                    "version_id": version2.version_id,
                    "version_number": version2.version_number,
                    "created_at": version2.created_at.isoformat(),
                    "created_by": version2.created_by
                },
                "content_diff": content_diff,
                "metadata_diff": metadata_diff,
                "statistics": stats,
                "comparison_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error comparing versions: {str(e)}")
            raise
    
    async def restore_version(
        self,
        document_id: str,
        version_id: str,
        restored_by: str,
        reason: str
    ) -> DocumentVersion:
        """Restore a previous version."""
        
        try:
            # Get version to restore
            version_to_restore = await self.get_version_by_id(document_id, version_id)
            if not version_to_restore:
                raise ValueError(f"Version {version_id} not found")
            
            # Get current version
            current_version = self._get_current_version(document_id)
            
            # Create new version based on restored version
            new_version = await self.create_new_version(
                document_id=document_id,
                title=version_to_restore.title,
                content=version_to_restore.content,
                metadata=version_to_restore.metadata,
                created_by=restored_by,
                change_summary=f"Restored from version {version_to_restore.version_number}. Reason: {reason}",
                version_type=VersionType.PATCH,
                parent_version_id=current_version.version_id if current_version else None,
                tags=["restored"]
            )
            
            # Track restoration
            if document_id not in self.changes:
                self.changes[document_id] = []
            
            restoration_change = VersionChange(
                change_id=str(uuid4()),
                version_id=new_version.version_id,
                change_type=ChangeType.RESTORED,
                field_name="content",
                old_value=current_version.content if current_version else "",
                new_value=version_to_restore.content,
                timestamp=datetime.now(),
                user_id=restored_by,
                description=f"Restored from version {version_to_restore.version_number}"
            )
            
            self.changes[document_id].append(restoration_change)
            
            logger.info(f"Restored version {version_to_restore.version_number} for document {document_id}")
            
            return new_version
            
        except Exception as e:
            logger.error(f"Error restoring version: {str(e)}")
            raise
    
    async def create_branch(
        self,
        document_id: str,
        branch_name: str,
        base_version_id: str,
        created_by: str,
        description: str
    ) -> VersionBranch:
        """Create a new branch from a version."""
        
        try:
            # Verify base version exists
            base_version = await self.get_version_by_id(document_id, base_version_id)
            if not base_version:
                raise ValueError(f"Base version {base_version_id} not found")
            
            # Check if branch name already exists
            existing_branches = self.branches.get(document_id, [])
            if any(branch.branch_name == branch_name for branch in existing_branches):
                raise ValueError(f"Branch {branch_name} already exists")
            
            # Create branch
            branch = VersionBranch(
                branch_id=str(uuid4()),
                document_id=document_id,
                branch_name=branch_name,
                base_version_id=base_version_id,
                current_version_id=base_version_id,
                created_by=created_by,
                created_at=datetime.now(),
                description=description
            )
            
            # Store branch
            if document_id not in self.branches:
                self.branches[document_id] = []
            self.branches[document_id].append(branch)
            
            logger.info(f"Created branch {branch_name} for document {document_id}")
            
            return branch
            
        except Exception as e:
            logger.error(f"Error creating branch: {str(e)}")
            raise
    
    async def merge_branch(
        self,
        document_id: str,
        branch_id: str,
        target_version_id: str,
        merged_by: str,
        merge_message: str
    ) -> DocumentVersion:
        """Merge a branch into a target version."""
        
        try:
            # Get branch
            branch = None
            for b in self.branches.get(document_id, []):
                if b.branch_id == branch_id:
                    branch = b
                    break
            
            if not branch:
                raise ValueError(f"Branch {branch_id} not found")
            
            if branch.is_merged:
                raise ValueError(f"Branch {branch.branch_name} is already merged")
            
            # Get branch's current version
            branch_version = await self.get_version_by_id(document_id, branch.current_version_id)
            if not branch_version:
                raise ValueError(f"Branch version {branch.current_version_id} not found")
            
            # Get target version
            target_version = await self.get_version_by_id(document_id, target_version_id)
            if not target_version:
                raise ValueError(f"Target version {target_version_id} not found")
            
            # Create merged version
            merged_version = await self.create_new_version(
                document_id=document_id,
                title=branch_version.title,
                content=branch_version.content,
                metadata=branch_version.metadata,
                created_by=merged_by,
                change_summary=f"Merged branch {branch.branch_name}: {merge_message}",
                version_type=VersionType.MINOR,
                parent_version_id=target_version_id,
                tags=["merged", f"branch:{branch.branch_name}"]
            )
            
            # Mark branch as merged
            branch.is_merged = True
            branch.merged_at = datetime.now()
            
            logger.info(f"Merged branch {branch.branch_name} into document {document_id}")
            
            return merged_version
            
        except Exception as e:
            logger.error(f"Error merging branch: {str(e)}")
            raise
    
    async def get_version_history(
        self,
        document_id: str,
        version_id: str
    ) -> List[DocumentVersion]:
        """Get version history (ancestors) for a version."""
        
        try:
            history = []
            current_version_id = version_id
            
            while current_version_id:
                version = await self.get_version_by_id(document_id, current_version_id)
                if not version:
                    break
                
                history.append(version)
                current_version_id = version.parent_version_id
            
            return history
            
        except Exception as e:
            logger.error(f"Error getting version history: {str(e)}")
            return []
    
    async def get_version_analytics(self, document_id: str) -> Dict[str, Any]:
        """Get version analytics for a document."""
        
        if document_id not in self.versions:
            return {
                "total_versions": 0,
                "version_types": {},
                "creation_timeline": [],
                "collaborators": {},
                "average_version_size": 0,
                "version_frequency": 0
            }
        
        versions = self.versions[document_id]
        
        # Count version types
        version_types = {}
        for version in versions:
            version_types[version.version_type.value] = version_types.get(version.version_type.value, 0) + 1
        
        # Creation timeline
        timeline = []
        for version in versions:
            timeline.append({
                "version_number": version.version_number,
                "created_at": version.created_at.isoformat(),
                "created_by": version.created_by,
                "version_type": version.version_type.value,
                "file_size": version.file_size
            })
        
        # Collaborators
        collaborators = {}
        for version in versions:
            collaborators[version.created_by] = collaborators.get(version.created_by, 0) + 1
        
        # Calculate averages
        total_size = sum(v.file_size for v in versions)
        average_size = total_size / len(versions) if versions else 0
        
        # Version frequency (versions per month)
        if len(versions) > 1:
            time_span = (versions[0].created_at - versions[-1].created_at).days
            frequency = len(versions) / (time_span / 30) if time_span > 0 else 0
        else:
            frequency = 0
        
        return {
            "total_versions": len(versions),
            "version_types": version_types,
            "creation_timeline": sorted(timeline, key=lambda x: x["created_at"]),
            "collaborators": collaborators,
            "average_version_size": average_size,
            "version_frequency": frequency,
            "current_version": versions[0].version_number if versions else None,
            "oldest_version": versions[-1].version_number if versions else None
        }
    
    def _get_current_version(self, document_id: str) -> Optional[DocumentVersion]:
        """Get current version of a document."""
        
        if document_id not in self.versions:
            return None
        
        for version in self.versions[document_id]:
            if version.is_current:
                return version
        
        return None
    
    def _calculate_next_version_number(
        self,
        current_version: str,
        version_type: VersionType
    ) -> str:
        """Calculate next version number."""
        
        try:
            major, minor, patch = map(int, current_version.split('.'))
            
            if version_type == VersionType.MAJOR:
                return f"{major + 1}.0.0"
            elif version_type == VersionType.MINOR:
                return f"{major}.{minor + 1}.0"
            elif version_type == VersionType.PATCH:
                return f"{major}.{minor}.{patch + 1}"
            else:
                return f"{major}.{minor}.{patch + 1}-{version_type.value}"
                
        except ValueError:
            # If version format is invalid, start from 1.0.0
            if version_type == VersionType.MAJOR:
                return "2.0.0"
            else:
                return "1.0.1"
    
    def _calculate_content_hash(self, content: str) -> str:
        """Calculate content hash."""
        
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def _calculate_content_diff(self, content1: str, content2: str) -> List[str]:
        """Calculate content differences."""
        
        lines1 = content1.splitlines(keepends=True)
        lines2 = content2.splitlines(keepends=True)
        
        diff = list(difflib.unified_diff(
            lines1, lines2,
            fromfile="version1",
            tofile="version2",
            lineterm=""
        ))
        
        return diff
    
    def _calculate_metadata_diff(self, metadata1: Dict[str, Any], metadata2: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate metadata differences."""
        
        diff = {}
        
        # Find added keys
        for key in metadata2:
            if key not in metadata1:
                diff[f"+{key}"] = metadata2[key]
        
        # Find removed keys
        for key in metadata1:
            if key not in metadata2:
                diff[f"-{key}"] = metadata1[key]
        
        # Find changed values
        for key in metadata1:
            if key in metadata2 and metadata1[key] != metadata2[key]:
                diff[f"~{key}"] = {
                    "old": metadata1[key],
                    "new": metadata2[key]
                }
        
        return diff
    
    def _calculate_similarity(self, content1: str, content2: str) -> float:
        """Calculate content similarity percentage."""
        
        if not content1 and not content2:
            return 100.0
        
        if not content1 or not content2:
            return 0.0
        
        # Use difflib to calculate similarity
        matcher = difflib.SequenceMatcher(None, content1, content2)
        return matcher.ratio() * 100
    
    async def _track_version_changes(
        self,
        old_version: DocumentVersion,
        new_version: DocumentVersion,
        user_id: str
    ):
        """Track changes between versions."""
        
        if new_version.document_id not in self.changes:
            self.changes[new_version.document_id] = []
        
        changes = []
        
        # Track title changes
        if old_version.title != new_version.title:
            changes.append(VersionChange(
                change_id=str(uuid4()),
                version_id=new_version.version_id,
                change_type=ChangeType.UPDATED,
                field_name="title",
                old_value=old_version.title,
                new_value=new_version.title,
                timestamp=datetime.now(),
                user_id=user_id,
                description="Title updated"
            ))
        
        # Track content changes
        if old_version.content != new_version.content:
            changes.append(VersionChange(
                change_id=str(uuid4()),
                version_id=new_version.version_id,
                change_type=ChangeType.UPDATED,
                field_name="content",
                old_value=old_version.content,
                new_value=new_version.content,
                timestamp=datetime.now(),
                user_id=user_id,
                description="Content updated"
            ))
        
        # Track metadata changes
        for key in set(old_version.metadata.keys()) | set(new_version.metadata.keys()):
            old_value = old_version.metadata.get(key)
            new_value = new_version.metadata.get(key)
            
            if old_value != new_value:
                changes.append(VersionChange(
                    change_id=str(uuid4()),
                    version_id=new_version.version_id,
                    change_type=ChangeType.UPDATED,
                    field_name=f"metadata.{key}",
                    old_value=old_value,
                    new_value=new_value,
                    timestamp=datetime.now(),
                    user_id=user_id,
                    description=f"Metadata field '{key}' updated"
                ))
        
        self.changes[new_version.document_id].extend(changes)
    
    async def _save_version_to_file(self, version: DocumentVersion):
        """Save version to file."""
        
        try:
            version_dir = self.storage_path / version.document_id
            version_dir.mkdir(exist_ok=True)
            
            version_file = version_dir / f"{version.version_id}.json"
            
            version_data = asdict(version)
            version_data["created_at"] = version.created_at.isoformat()
            
            with open(version_file, 'w', encoding='utf-8') as f:
                json.dump(version_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Error saving version to file: {str(e)}")
    
    async def _load_version_from_file(self, document_id: str, version_id: str) -> Optional[DocumentVersion]:
        """Load version from file."""
        
        try:
            version_file = self.storage_path / document_id / f"{version_id}.json"
            
            if not version_file.exists():
                return None
            
            with open(version_file, 'r', encoding='utf-8') as f:
                version_data = json.load(f)
            
            version_data["created_at"] = datetime.fromisoformat(version_data["created_at"])
            version_data["version_type"] = VersionType(version_data["version_type"])
            
            return DocumentVersion(**version_data)
            
        except Exception as e:
            logger.error(f"Error loading version from file: {str(e)}")
            return None



























