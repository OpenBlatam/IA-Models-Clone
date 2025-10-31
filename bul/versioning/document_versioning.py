"""
BUL Document Versioning System
==============================

Document versioning and history tracking system for comprehensive document lifecycle management.
"""

import asyncio
import json
import time
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
import uuid
import difflib
from pathlib import Path
import sqlite3
import threading

from ..utils import get_logger, get_cache_manager
from ..config import get_config
from ..core import BusinessArea, DocumentType

logger = get_logger(__name__)

class VersionStatus(str, Enum):
    """Document version status"""
    DRAFT = "draft"
    REVIEW = "review"
    APPROVED = "approved"
    PUBLISHED = "published"
    ARCHIVED = "archived"
    DELETED = "deleted"

class ChangeType(str, Enum):
    """Types of changes"""
    CREATED = "created"
    UPDATED = "updated"
    DELETED = "deleted"
    RESTORED = "restored"
    PUBLISHED = "published"
    ARCHIVED = "archived"
    STATUS_CHANGED = "status_changed"
    COMMENTED = "commented"
    APPROVED = "approved"
    REJECTED = "rejected"

class DiffType(str, Enum):
    """Types of differences"""
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    UNCHANGED = "unchanged"

@dataclass
class DocumentVersion:
    """Document version"""
    version_id: str
    document_id: str
    version_number: str  # e.g., "1.0", "1.1", "2.0"
    title: str
    content: str
    content_hash: str
    status: VersionStatus
    created_by: str
    created_at: datetime
    updated_at: datetime
    change_summary: str
    change_type: ChangeType
    parent_version_id: Optional[str] = None
    is_major_version: bool = False
    metadata: Dict[str, Any] = None
    tags: List[str] = None
    comments: List[str] = None

@dataclass
class VersionDiff:
    """Version difference"""
    diff_id: str
    from_version_id: str
    to_version_id: str
    diff_type: DiffType
    changes: List[Dict[str, Any]]
    added_lines: int
    removed_lines: int
    modified_lines: int
    similarity_score: float
    created_at: datetime

@dataclass
class VersionComment:
    """Version comment"""
    comment_id: str
    version_id: str
    author: str
    content: str
    created_at: datetime
    is_resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None

@dataclass
class VersionBranch:
    """Version branch"""
    branch_id: str
    document_id: str
    branch_name: str
    base_version_id: str
    current_version_id: str
    created_by: str
    created_at: datetime
    is_active: bool = True
    description: str = ""

class DocumentVersioningSystem:
    """Document Versioning System"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.cache_manager = get_cache_manager()
        self.config = get_config()
        
        # Data storage
        self.document_versions: Dict[str, DocumentVersion] = {}
        self.version_diffs: Dict[str, VersionDiff] = {}
        self.version_comments: Dict[str, VersionComment] = {}
        self.version_branches: Dict[str, VersionBranch] = {}
        
        # Database
        self.db_path = Path("data/document_versions.db")
        self.db_path.parent.mkdir(exist_ok=True)
        self.db_lock = threading.Lock()
        
        # Initialize system
        self._initialize_database()
        self._load_data_from_database()
    
    def _initialize_database(self):
        """Initialize SQLite database for versioning"""
        try:
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Create tables
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS document_versions (
                        version_id TEXT PRIMARY KEY,
                        document_id TEXT NOT NULL,
                        version_number TEXT NOT NULL,
                        title TEXT NOT NULL,
                        content TEXT NOT NULL,
                        content_hash TEXT NOT NULL,
                        status TEXT NOT NULL,
                        created_by TEXT NOT NULL,
                        created_at TIMESTAMP NOT NULL,
                        updated_at TIMESTAMP NOT NULL,
                        change_summary TEXT,
                        change_type TEXT NOT NULL,
                        parent_version_id TEXT,
                        is_major_version BOOLEAN DEFAULT FALSE,
                        metadata TEXT,
                        tags TEXT,
                        comments TEXT
                    )
                ''')
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS version_diffs (
                        diff_id TEXT PRIMARY KEY,
                        from_version_id TEXT NOT NULL,
                        to_version_id TEXT NOT NULL,
                        diff_type TEXT NOT NULL,
                        changes TEXT NOT NULL,
                        added_lines INTEGER DEFAULT 0,
                        removed_lines INTEGER DEFAULT 0,
                        modified_lines INTEGER DEFAULT 0,
                        similarity_score REAL DEFAULT 0.0,
                        created_at TIMESTAMP NOT NULL
                    )
                ''')
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS version_comments (
                        comment_id TEXT PRIMARY KEY,
                        version_id TEXT NOT NULL,
                        author TEXT NOT NULL,
                        content TEXT NOT NULL,
                        created_at TIMESTAMP NOT NULL,
                        is_resolved BOOLEAN DEFAULT FALSE,
                        resolved_at TIMESTAMP,
                        resolved_by TEXT
                    )
                ''')
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS version_branches (
                        branch_id TEXT PRIMARY KEY,
                        document_id TEXT NOT NULL,
                        branch_name TEXT NOT NULL,
                        base_version_id TEXT NOT NULL,
                        current_version_id TEXT NOT NULL,
                        created_by TEXT NOT NULL,
                        created_at TIMESTAMP NOT NULL,
                        is_active BOOLEAN DEFAULT TRUE,
                        description TEXT
                    )
                ''')
                
                # Create indexes
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_document_versions_document_id ON document_versions(document_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_document_versions_version_number ON document_versions(version_number)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_document_versions_created_at ON document_versions(created_at)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_version_diffs_from_version ON version_diffs(from_version_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_version_diffs_to_version ON version_diffs(to_version_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_version_comments_version_id ON version_comments(version_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_version_branches_document_id ON version_branches(document_id)')
                
                conn.commit()
                conn.close()
                
                self.logger.info("Document versioning database initialized successfully")
        
        except Exception as e:
            self.logger.error(f"Failed to initialize versioning database: {e}")
    
    def _load_data_from_database(self):
        """Load data from database into memory"""
        try:
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Load document versions
                cursor.execute('SELECT * FROM document_versions')
                for row in cursor.fetchall():
                    version = DocumentVersion(
                        version_id=row[0],
                        document_id=row[1],
                        version_number=row[2],
                        title=row[3],
                        content=row[4],
                        content_hash=row[5],
                        status=VersionStatus(row[6]),
                        created_by=row[7],
                        created_at=datetime.fromisoformat(row[8]),
                        updated_at=datetime.fromisoformat(row[9]),
                        change_summary=row[10] or "",
                        change_type=ChangeType(row[11]),
                        parent_version_id=row[12],
                        is_major_version=bool(row[13]),
                        metadata=json.loads(row[14]) if row[14] else {},
                        tags=json.loads(row[15]) if row[15] else [],
                        comments=json.loads(row[16]) if row[16] else []
                    )
                    self.document_versions[version.version_id] = version
                
                # Load version diffs
                cursor.execute('SELECT * FROM version_diffs')
                for row in cursor.fetchall():
                    diff = VersionDiff(
                        diff_id=row[0],
                        from_version_id=row[1],
                        to_version_id=row[2],
                        diff_type=DiffType(row[3]),
                        changes=json.loads(row[4]),
                        added_lines=row[5],
                        removed_lines=row[6],
                        modified_lines=row[7],
                        similarity_score=row[8],
                        created_at=datetime.fromisoformat(row[9])
                    )
                    self.version_diffs[diff.diff_id] = diff
                
                # Load version comments
                cursor.execute('SELECT * FROM version_comments')
                for row in cursor.fetchall():
                    comment = VersionComment(
                        comment_id=row[0],
                        version_id=row[1],
                        author=row[2],
                        content=row[3],
                        created_at=datetime.fromisoformat(row[4]),
                        is_resolved=bool(row[5]),
                        resolved_at=datetime.fromisoformat(row[6]) if row[6] else None,
                        resolved_by=row[7]
                    )
                    self.version_comments[comment.comment_id] = comment
                
                # Load version branches
                cursor.execute('SELECT * FROM version_branches')
                for row in cursor.fetchall():
                    branch = VersionBranch(
                        branch_id=row[0],
                        document_id=row[1],
                        branch_name=row[2],
                        base_version_id=row[3],
                        current_version_id=row[4],
                        created_by=row[5],
                        created_at=datetime.fromisoformat(row[6]),
                        is_active=bool(row[7]),
                        description=row[8] or ""
                    )
                    self.version_branches[branch.branch_id] = branch
                
                conn.close()
                
                self.logger.info(f"Loaded {len(self.document_versions)} versions, {len(self.version_diffs)} diffs, {len(self.version_comments)} comments, {len(self.version_branches)} branches")
        
        except Exception as e:
            self.logger.error(f"Error loading data from database: {e}")
    
    def _save_version_to_database(self, version: DocumentVersion):
        """Save document version to database"""
        try:
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO document_versions 
                    (version_id, document_id, version_number, title, content, content_hash, 
                     status, created_by, created_at, updated_at, change_summary, change_type,
                     parent_version_id, is_major_version, metadata, tags, comments)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    version.version_id, version.document_id, version.version_number,
                    version.title, version.content, version.content_hash,
                    version.status.value, version.created_by,
                    version.created_at.isoformat(), version.updated_at.isoformat(),
                    version.change_summary, version.change_type.value,
                    version.parent_version_id, version.is_major_version,
                    json.dumps(version.metadata or {}),
                    json.dumps(version.tags or []),
                    json.dumps(version.comments or [])
                ))
                
                conn.commit()
                conn.close()
        
        except Exception as e:
            self.logger.error(f"Error saving version to database: {e}")
    
    def _save_diff_to_database(self, diff: VersionDiff):
        """Save version diff to database"""
        try:
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO version_diffs 
                    (diff_id, from_version_id, to_version_id, diff_type, changes,
                     added_lines, removed_lines, modified_lines, similarity_score, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    diff.diff_id, diff.from_version_id, diff.to_version_id,
                    diff.diff_type.value, json.dumps(diff.changes),
                    diff.added_lines, diff.removed_lines, diff.modified_lines,
                    diff.similarity_score, diff.created_at.isoformat()
                ))
                
                conn.commit()
                conn.close()
        
        except Exception as e:
            self.logger.error(f"Error saving diff to database: {e}")
    
    def _save_comment_to_database(self, comment: VersionComment):
        """Save version comment to database"""
        try:
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO version_comments 
                    (comment_id, version_id, author, content, created_at,
                     is_resolved, resolved_at, resolved_by)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    comment.comment_id, comment.version_id, comment.author,
                    comment.content, comment.created_at.isoformat(),
                    comment.is_resolved,
                    comment.resolved_at.isoformat() if comment.resolved_at else None,
                    comment.resolved_by
                ))
                
                conn.commit()
                conn.close()
        
        except Exception as e:
            self.logger.error(f"Error saving comment to database: {e}")
    
    def _save_branch_to_database(self, branch: VersionBranch):
        """Save version branch to database"""
        try:
            with self.db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO version_branches 
                    (branch_id, document_id, branch_name, base_version_id, current_version_id,
                     created_by, created_at, is_active, description)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    branch.branch_id, branch.document_id, branch.branch_name,
                    branch.base_version_id, branch.current_version_id,
                    branch.created_by, branch.created_at.isoformat(),
                    branch.is_active, branch.description
                ))
                
                conn.commit()
                conn.close()
        
        except Exception as e:
            self.logger.error(f"Error saving branch to database: {e}")
    
    def _calculate_content_hash(self, content: str) -> str:
        """Calculate content hash"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def _generate_version_number(self, document_id: str, is_major: bool = False) -> str:
        """Generate next version number"""
        try:
            # Get existing versions for this document
            existing_versions = [
                v for v in self.document_versions.values() 
                if v.document_id == document_id
            ]
            
            if not existing_versions:
                return "1.0"
            
            # Parse existing version numbers
            version_numbers = []
            for version in existing_versions:
                try:
                    parts = version.version_number.split('.')
                    major = int(parts[0])
                    minor = int(parts[1]) if len(parts) > 1 else 0
                    version_numbers.append((major, minor))
                except (ValueError, IndexError):
                    continue
            
            if not version_numbers:
                return "1.0"
            
            # Get latest version
            latest_major, latest_minor = max(version_numbers)
            
            if is_major:
                return f"{latest_major + 1}.0"
            else:
                return f"{latest_major}.{latest_minor + 1}"
        
        except Exception as e:
            self.logger.error(f"Error generating version number: {e}")
            return "1.0"
    
    async def create_document_version(
        self,
        document_id: str,
        title: str,
        content: str,
        created_by: str,
        change_summary: str = "",
        change_type: ChangeType = ChangeType.CREATED,
        is_major_version: bool = False,
        parent_version_id: Optional[str] = None,
        status: VersionStatus = VersionStatus.DRAFT,
        tags: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> DocumentVersion:
        """Create a new document version"""
        try:
            # Generate version ID and number
            version_id = str(uuid.uuid4())
            version_number = self._generate_version_number(document_id, is_major_version)
            
            # Calculate content hash
            content_hash = self._calculate_content_hash(content)
            
            # Create version
            version = DocumentVersion(
                version_id=version_id,
                document_id=document_id,
                version_number=version_number,
                title=title,
                content=content,
                content_hash=content_hash,
                status=status,
                created_by=created_by,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                change_summary=change_summary,
                change_type=change_type,
                parent_version_id=parent_version_id,
                is_major_version=is_major_version,
                metadata=metadata or {},
                tags=tags or [],
                comments=[]
            )
            
            # Save to memory and database
            self.document_versions[version_id] = version
            self._save_version_to_database(version)
            
            # Create diff if there's a parent version
            if parent_version_id and parent_version_id in self.document_versions:
                await self._create_version_diff(parent_version_id, version_id)
            
            self.logger.info(f"Created document version {version_number} for document {document_id}")
            return version
        
        except Exception as e:
            self.logger.error(f"Error creating document version: {e}")
            raise
    
    async def _create_version_diff(self, from_version_id: str, to_version_id: str) -> VersionDiff:
        """Create diff between two versions"""
        try:
            from_version = self.document_versions[from_version_id]
            to_version = self.document_versions[to_version_id]
            
            # Calculate differences
            from_lines = from_version.content.splitlines()
            to_lines = to_version.content.splitlines()
            
            # Use difflib to calculate differences
            differ = difflib.unified_diff(
                from_lines, to_lines,
                fromfile=f"v{from_version.version_number}",
                tofile=f"v{to_version.version_number}",
                lineterm=""
            )
            
            changes = list(differ)
            
            # Count changes
            added_lines = sum(1 for line in changes if line.startswith('+') and not line.startswith('+++'))
            removed_lines = sum(1 for line in changes if line.startswith('-') and not line.startswith('---'))
            modified_lines = min(added_lines, removed_lines)
            
            # Calculate similarity score
            similarity_score = difflib.SequenceMatcher(
                None, from_version.content, to_version.content
            ).ratio()
            
            # Determine diff type
            if added_lines > 0 and removed_lines == 0:
                diff_type = DiffType.ADDED
            elif added_lines == 0 and removed_lines > 0:
                diff_type = DiffType.REMOVED
            elif added_lines > 0 or removed_lines > 0:
                diff_type = DiffType.MODIFIED
            else:
                diff_type = DiffType.UNCHANGED
            
            # Create diff
            diff = VersionDiff(
                diff_id=str(uuid.uuid4()),
                from_version_id=from_version_id,
                to_version_id=to_version_id,
                diff_type=diff_type,
                changes=changes,
                added_lines=added_lines,
                removed_lines=removed_lines,
                modified_lines=modified_lines,
                similarity_score=similarity_score,
                created_at=datetime.now()
            )
            
            # Save diff
            self.version_diffs[diff.diff_id] = diff
            self._save_diff_to_database(diff)
            
            return diff
        
        except Exception as e:
            self.logger.error(f"Error creating version diff: {e}")
            raise
    
    async def get_document_versions(self, document_id: str) -> List[DocumentVersion]:
        """Get all versions of a document"""
        try:
            versions = [
                v for v in self.document_versions.values() 
                if v.document_id == document_id
            ]
            
            # Sort by version number
            def version_key(version):
                try:
                    parts = version.version_number.split('.')
                    major = int(parts[0])
                    minor = int(parts[1]) if len(parts) > 1 else 0
                    return (major, minor)
                except (ValueError, IndexError):
                    return (0, 0)
            
            versions.sort(key=version_key)
            return versions
        
        except Exception as e:
            self.logger.error(f"Error getting document versions: {e}")
            return []
    
    async def get_version_diff(self, from_version_id: str, to_version_id: str) -> Optional[VersionDiff]:
        """Get diff between two versions"""
        try:
            # Look for existing diff
            for diff in self.version_diffs.values():
                if (diff.from_version_id == from_version_id and 
                    diff.to_version_id == to_version_id):
                    return diff
            
            # Create new diff if not found
            if (from_version_id in self.document_versions and 
                to_version_id in self.document_versions):
                return await self._create_version_diff(from_version_id, to_version_id)
            
            return None
        
        except Exception as e:
            self.logger.error(f"Error getting version diff: {e}")
            return None
    
    async def add_version_comment(
        self,
        version_id: str,
        author: str,
        content: str
    ) -> VersionComment:
        """Add comment to a version"""
        try:
            if version_id not in self.document_versions:
                raise ValueError(f"Version {version_id} not found")
            
            comment = VersionComment(
                comment_id=str(uuid.uuid4()),
                version_id=version_id,
                author=author,
                content=content,
                created_at=datetime.now()
            )
            
            # Save comment
            self.version_comments[comment.comment_id] = comment
            self._save_comment_to_database(comment)
            
            # Add to version comments list
            version = self.document_versions[version_id]
            version.comments.append(comment.comment_id)
            self._save_version_to_database(version)
            
            self.logger.info(f"Added comment to version {version_id}")
            return comment
        
        except Exception as e:
            self.logger.error(f"Error adding version comment: {e}")
            raise
    
    async def create_version_branch(
        self,
        document_id: str,
        branch_name: str,
        base_version_id: str,
        created_by: str,
        description: str = ""
    ) -> VersionBranch:
        """Create a new version branch"""
        try:
            if base_version_id not in self.document_versions:
                raise ValueError(f"Base version {base_version_id} not found")
            
            # Check if branch name already exists for this document
            existing_branches = [
                b for b in self.version_branches.values()
                if b.document_id == document_id and b.branch_name == branch_name
            ]
            
            if existing_branches:
                raise ValueError(f"Branch '{branch_name}' already exists for document {document_id}")
            
            branch = VersionBranch(
                branch_id=str(uuid.uuid4()),
                document_id=document_id,
                branch_name=branch_name,
                base_version_id=base_version_id,
                current_version_id=base_version_id,
                created_by=created_by,
                created_at=datetime.now(),
                description=description
            )
            
            # Save branch
            self.version_branches[branch.branch_id] = branch
            self._save_branch_to_database(branch)
            
            self.logger.info(f"Created branch '{branch_name}' for document {document_id}")
            return branch
        
        except Exception as e:
            self.logger.error(f"Error creating version branch: {e}")
            raise
    
    async def get_document_branches(self, document_id: str) -> List[VersionBranch]:
        """Get all branches for a document"""
        try:
            branches = [
                b for b in self.version_branches.values()
                if b.document_id == document_id and b.is_active
            ]
            
            # Sort by creation date
            branches.sort(key=lambda b: b.created_at)
            return branches
        
        except Exception as e:
            self.logger.error(f"Error getting document branches: {e}")
            return []
    
    async def get_version_history(self, document_id: str) -> List[Dict[str, Any]]:
        """Get complete version history for a document"""
        try:
            versions = await self.get_document_versions(document_id)
            history = []
            
            for version in versions:
                # Get comments for this version
                version_comments = [
                    c for c in self.version_comments.values()
                    if c.version_id == version.version_id
                ]
                
                # Get diffs from this version
                version_diffs = [
                    d for d in self.version_diffs.values()
                    if d.from_version_id == version.version_id or d.to_version_id == version.version_id
                ]
                
                history_item = {
                    "version": asdict(version),
                    "comments": [asdict(c) for c in version_comments],
                    "diffs": [asdict(d) for d in version_diffs],
                    "comment_count": len(version_comments),
                    "diff_count": len(version_diffs)
                }
                
                history.append(history_item)
            
            return history
        
        except Exception as e:
            self.logger.error(f"Error getting version history: {e}")
            return []
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get document versioning system status"""
        try:
            total_versions = len(self.document_versions)
            total_diffs = len(self.version_diffs)
            total_comments = len(self.version_comments)
            total_branches = len(self.version_branches)
            active_branches = len([b for b in self.version_branches.values() if b.is_active])
            
            # Count by status
            status_counts = {}
            for version in self.document_versions.values():
                status = version.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
            
            # Count by change type
            change_type_counts = {}
            for version in self.document_versions.values():
                change_type = version.change_type.value
                change_type_counts[change_type] = change_type_counts.get(change_type, 0) + 1
            
            return {
                'total_versions': total_versions,
                'total_diffs': total_diffs,
                'total_comments': total_comments,
                'total_branches': total_branches,
                'active_branches': active_branches,
                'status_distribution': status_counts,
                'change_type_distribution': change_type_counts,
                'database_path': str(self.db_path),
                'system_health': 'active'
            }
        
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {}

# Global document versioning system
_document_versioning_system: Optional[DocumentVersioningSystem] = None

def get_document_versioning_system() -> DocumentVersioningSystem:
    """Get the global document versioning system"""
    global _document_versioning_system
    if _document_versioning_system is None:
        _document_versioning_system = DocumentVersioningSystem()
    return _document_versioning_system

# Document versioning router
versioning_router = APIRouter(prefix="/versioning", tags=["Document Versioning"])

@versioning_router.post("/create-version")
async def create_document_version_endpoint(
    document_id: str = Field(..., description="Document ID"),
    title: str = Field(..., description="Document title"),
    content: str = Field(..., description="Document content"),
    created_by: str = Field(..., description="Creator user ID"),
    change_summary: str = Field("", description="Summary of changes"),
    change_type: ChangeType = Field(ChangeType.CREATED, description="Type of change"),
    is_major_version: bool = Field(False, description="Is this a major version"),
    parent_version_id: str = Field(None, description="Parent version ID"),
    status: VersionStatus = Field(VersionStatus.DRAFT, description="Version status"),
    tags: List[str] = Field([], description="Version tags"),
    metadata: Dict[str, Any] = Field({}, description="Version metadata")
):
    """Create a new document version"""
    try:
        system = get_document_versioning_system()
        version = await system.create_document_version(
            document_id, title, content, created_by, change_summary,
            change_type, is_major_version, parent_version_id, status, tags, metadata
        )
        return {"version": asdict(version), "success": True}
    
    except Exception as e:
        logger.error(f"Error creating document version: {e}")
        raise HTTPException(status_code=500, detail="Failed to create document version")

@versioning_router.get("/document/{document_id}/versions")
async def get_document_versions_endpoint(document_id: str):
    """Get all versions of a document"""
    try:
        system = get_document_versioning_system()
        versions = await system.get_document_versions(document_id)
        return {"versions": [asdict(v) for v in versions], "count": len(versions)}
    
    except Exception as e:
        logger.error(f"Error getting document versions: {e}")
        raise HTTPException(status_code=500, detail="Failed to get document versions")

@versioning_router.get("/diff/{from_version_id}/{to_version_id}")
async def get_version_diff_endpoint(from_version_id: str, to_version_id: str):
    """Get diff between two versions"""
    try:
        system = get_document_versioning_system()
        diff = await system.get_version_diff(from_version_id, to_version_id)
        if diff:
            return {"diff": asdict(diff), "success": True}
        else:
            return {"diff": None, "success": False}
    
    except Exception as e:
        logger.error(f"Error getting version diff: {e}")
        raise HTTPException(status_code=500, detail="Failed to get version diff")

@versioning_router.post("/version/{version_id}/comment")
async def add_version_comment_endpoint(
    version_id: str,
    author: str = Field(..., description="Comment author"),
    content: str = Field(..., description="Comment content")
):
    """Add comment to a version"""
    try:
        system = get_document_versioning_system()
        comment = await system.add_version_comment(version_id, author, content)
        return {"comment": asdict(comment), "success": True}
    
    except Exception as e:
        logger.error(f"Error adding version comment: {e}")
        raise HTTPException(status_code=500, detail="Failed to add version comment")

@versioning_router.post("/branch")
async def create_version_branch_endpoint(
    document_id: str = Field(..., description="Document ID"),
    branch_name: str = Field(..., description="Branch name"),
    base_version_id: str = Field(..., description="Base version ID"),
    created_by: str = Field(..., description="Creator user ID"),
    description: str = Field("", description="Branch description")
):
    """Create a new version branch"""
    try:
        system = get_document_versioning_system()
        branch = await system.create_version_branch(
            document_id, branch_name, base_version_id, created_by, description
        )
        return {"branch": asdict(branch), "success": True}
    
    except Exception as e:
        logger.error(f"Error creating version branch: {e}")
        raise HTTPException(status_code=500, detail="Failed to create version branch")

@versioning_router.get("/document/{document_id}/branches")
async def get_document_branches_endpoint(document_id: str):
    """Get all branches for a document"""
    try:
        system = get_document_versioning_system()
        branches = await system.get_document_branches(document_id)
        return {"branches": [asdict(b) for b in branches], "count": len(branches)}
    
    except Exception as e:
        logger.error(f"Error getting document branches: {e}")
        raise HTTPException(status_code=500, detail="Failed to get document branches")

@versioning_router.get("/document/{document_id}/history")
async def get_version_history_endpoint(document_id: str):
    """Get complete version history for a document"""
    try:
        system = get_document_versioning_system()
        history = await system.get_version_history(document_id)
        return {"history": history, "count": len(history)}
    
    except Exception as e:
        logger.error(f"Error getting version history: {e}")
        raise HTTPException(status_code=500, detail="Failed to get version history")

@versioning_router.get("/status")
async def get_versioning_system_status_endpoint():
    """Get document versioning system status"""
    try:
        system = get_document_versioning_system()
        status = await system.get_system_status()
        return {"status": status}
    
    except Exception as e:
        logger.error(f"Error getting versioning system status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get versioning system status")

