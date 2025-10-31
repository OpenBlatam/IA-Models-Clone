"""
Domain Entities - Core Business Objects
=====================================

Domain entities representing the core business concepts.
"""

import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

from .value_objects import DocumentId, ProcessingStatus, DocumentType, UserId


class DocumentStatus(str, Enum):
    """Document status enumeration."""
    DRAFT = "draft"
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    ARCHIVED = "archived"


@dataclass
class Document:
    """Document domain entity."""
    id: DocumentId
    title: str
    content: str
    document_type: DocumentType
    status: DocumentStatus
    created_by: UserId
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    version: int = 1
    
    def __post_init__(self):
        """Validate entity after initialization."""
        if not self.title or not self.title.strip():
            raise ValueError("Document title cannot be empty")
        
        if not self.content or not self.content.strip():
            raise ValueError("Document content cannot be empty")
    
    def update_content(self, new_content: str, updated_by: UserId) -> None:
        """Update document content."""
        if not new_content or not new_content.strip():
            raise ValueError("Document content cannot be empty")
        
        self.content = new_content
        self.updated_at = datetime.utcnow()
        self.version += 1
        self._add_audit_trail("content_updated", updated_by)
    
    def update_title(self, new_title: str, updated_by: UserId) -> None:
        """Update document title."""
        if not new_title or not new_title.strip():
            raise ValueError("Document title cannot be empty")
        
        self.title = new_title
        self.updated_at = datetime.utcnow()
        self.version += 1
        self._add_audit_trail("title_updated", updated_by)
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the document."""
        if tag and tag not in self.tags:
            self.tags.append(tag)
            self.updated_at = datetime.utcnow()
    
    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the document."""
        if tag in self.tags:
            self.tags.remove(tag)
            self.updated_at = datetime.utcnow()
    
    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata value."""
        self.metadata[key] = value
        self.updated_at = datetime.utcnow()
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value."""
        return self.metadata.get(key, default)
    
    def change_status(self, new_status: DocumentStatus) -> None:
        """Change document status."""
        if self.status == new_status:
            return
        
        # Validate status transition
        valid_transitions = {
            DocumentStatus.DRAFT: [DocumentStatus.PENDING, DocumentStatus.ARCHIVED],
            DocumentStatus.PENDING: [DocumentStatus.PROCESSING, DocumentStatus.ARCHIVED],
            DocumentStatus.PROCESSING: [DocumentStatus.COMPLETED, DocumentStatus.FAILED],
            DocumentStatus.COMPLETED: [DocumentStatus.ARCHIVED],
            DocumentStatus.FAILED: [DocumentStatus.PENDING, DocumentStatus.ARCHIVED],
            DocumentStatus.ARCHIVED: [DocumentStatus.DRAFT]
        }
        
        if new_status not in valid_transitions.get(self.status, []):
            raise ValueError(f"Invalid status transition from {self.status} to {new_status}")
        
        self.status = new_status
        self.updated_at = datetime.utcnow()
        self._add_audit_trail("status_changed", None, {"old_status": self.status, "new_status": new_status})
    
    def _add_audit_trail(self, action: str, user_id: Optional[UserId], details: Optional[Dict[str, Any]] = None) -> None:
        """Add audit trail entry."""
        if "audit_trail" not in self.metadata:
            self.metadata["audit_trail"] = []
        
        audit_entry = {
            "action": action,
            "user_id": str(user_id) if user_id else None,
            "timestamp": datetime.utcnow().isoformat(),
            "details": details or {}
        }
        
        self.metadata["audit_trail"].append(audit_entry)
    
    def is_editable(self) -> bool:
        """Check if document is editable."""
        return self.status in [DocumentStatus.DRAFT, DocumentStatus.FAILED]
    
    def is_processable(self) -> bool:
        """Check if document can be processed."""
        return self.status in [DocumentStatus.PENDING, DocumentStatus.FAILED]
    
    def get_word_count(self) -> int:
        """Get word count of document content."""
        return len(self.content.split())
    
    def get_character_count(self) -> int:
        """Get character count of document content."""
        return len(self.content)


@dataclass
class ProcessingResult:
    """Processing result domain entity."""
    id: str
    document_id: DocumentId
    status: ProcessingStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    extracted_text: Optional[str] = None
    classified_type: Optional[str] = None
    transformed_content: Optional[str] = None
    error_message: Optional[str] = None
    processing_time_seconds: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate entity after initialization."""
        if not self.id:
            raise ValueError("Processing result ID cannot be empty")
    
    def mark_completed(self, extracted_text: str, classified_type: str, transformed_content: str) -> None:
        """Mark processing as completed."""
        if self.status != ProcessingStatus.IN_PROGRESS:
            raise ValueError("Only in-progress processing can be marked as completed")
        
        self.status = ProcessingStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.extracted_text = extracted_text
        self.classified_type = classified_type
        self.transformed_content = transformed_content
        
        if self.started_at:
            self.processing_time_seconds = (self.completed_at - self.started_at).total_seconds()
    
    def mark_failed(self, error_message: str) -> None:
        """Mark processing as failed."""
        if self.status != ProcessingStatus.IN_PROGRESS:
            raise ValueError("Only in-progress processing can be marked as failed")
        
        self.status = ProcessingStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.error_message = error_message
        
        if self.started_at:
            self.processing_time_seconds = (self.completed_at - self.started_at).total_seconds()
    
    def is_completed(self) -> bool:
        """Check if processing is completed."""
        return self.status == ProcessingStatus.COMPLETED
    
    def is_failed(self) -> bool:
        """Check if processing failed."""
        return self.status == ProcessingStatus.FAILED
    
    def is_in_progress(self) -> bool:
        """Check if processing is in progress."""
        return self.status == ProcessingStatus.IN_PROGRESS
    
    def get_processing_duration(self) -> Optional[float]:
        """Get processing duration in seconds."""
        if self.completed_at and self.started_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


@dataclass
class User:
    """User domain entity."""
    id: UserId
    username: str
    email: str
    full_name: str
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    roles: List[str] = field(default_factory=lambda: ["user"])
    preferences: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate entity after initialization."""
        if not self.username or not self.username.strip():
            raise ValueError("Username cannot be empty")
        
        if not self.email or not self.email.strip():
            raise ValueError("Email cannot be empty")
        
        if not self.full_name or not self.full_name.strip():
            raise ValueError("Full name cannot be empty")
    
    def update_profile(self, full_name: str, email: str) -> None:
        """Update user profile."""
        if not full_name or not full_name.strip():
            raise ValueError("Full name cannot be empty")
        
        if not email or not email.strip():
            raise ValueError("Email cannot be empty")
        
        self.full_name = full_name
        self.email = email
        self.updated_at = datetime.utcnow()
    
    def add_role(self, role: str) -> None:
        """Add role to user."""
        if role and role not in self.roles:
            self.roles.append(role)
            self.updated_at = datetime.utcnow()
    
    def remove_role(self, role: str) -> None:
        """Remove role from user."""
        if role in self.roles:
            self.roles.remove(role)
            self.updated_at = datetime.utcnow()
    
    def has_role(self, role: str) -> bool:
        """Check if user has specific role."""
        return role in self.roles
    
    def is_admin(self) -> bool:
        """Check if user is admin."""
        return "admin" in self.roles
    
    def activate(self) -> None:
        """Activate user account."""
        self.is_active = True
        self.updated_at = datetime.utcnow()
    
    def deactivate(self) -> None:
        """Deactivate user account."""
        self.is_active = False
        self.updated_at = datetime.utcnow()
    
    def set_preference(self, key: str, value: Any) -> None:
        """Set user preference."""
        self.preferences[key] = value
        self.updated_at = datetime.utcnow()
    
    def get_preference(self, key: str, default: Any = None) -> Any:
        """Get user preference."""
        return self.preferences.get(key, default)


@dataclass
class Organization:
    """Organization domain entity."""
    id: str
    name: str
    description: Optional[str] = None
    created_by: UserId = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True
    settings: Dict[str, Any] = field(default_factory=dict)
    members: List[UserId] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate entity after initialization."""
        if not self.name or not self.name.strip():
            raise ValueError("Organization name cannot be empty")
    
    def add_member(self, user_id: UserId) -> None:
        """Add member to organization."""
        if user_id not in self.members:
            self.members.append(user_id)
            self.updated_at = datetime.utcnow()
    
    def remove_member(self, user_id: UserId) -> None:
        """Remove member from organization."""
        if user_id in self.members:
            self.members.remove(user_id)
            self.updated_at = datetime.utcnow()
    
    def is_member(self, user_id: UserId) -> bool:
        """Check if user is member of organization."""
        return user_id in self.members
    
    def update_settings(self, settings: Dict[str, Any]) -> None:
        """Update organization settings."""
        self.settings.update(settings)
        self.updated_at = datetime.utcnow()
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get organization setting."""
        return self.settings.get(key, default)
    
    def activate(self) -> None:
        """Activate organization."""
        self.is_active = True
        self.updated_at = datetime.utcnow()
    
    def deactivate(self) -> None:
        """Deactivate organization."""
        self.is_active = False
        self.updated_at = datetime.utcnow()

















