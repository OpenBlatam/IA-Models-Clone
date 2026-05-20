"""
Tests for CollaborationSystem utility
"""

import pytest
from pathlib import Path

from ..utils.collaboration_system import CollaborationSystem


class TestCollaborationSystem:
    """Test suite for CollaborationSystem"""

    def test_init(self, temp_dir):
        """Test CollaborationSystem initialization"""
        collab = CollaborationSystem(data_dir=temp_dir / "collaboration")
        assert collab.data_dir == temp_dir / "collaboration"
        assert collab.data_dir.exists()
        assert collab.project_collaborators == {}
        assert collab.project_permissions == {}
        assert collab.comments == {}

    def test_add_collaborator(self, temp_dir):
        """Test adding a collaborator"""
        collab = CollaborationSystem(data_dir=temp_dir / "collaboration")
        
        result = collab.add_collaborator("project-123", "user-456", role="editor")
        
        assert result is True
        assert "user-456" in collab.project_collaborators["project-123"]
        assert "user-456" in collab.project_permissions["project-123"]["editor"]

    def test_add_collaborator_duplicate(self, temp_dir):
        """Test adding duplicate collaborator"""
        collab = CollaborationSystem(data_dir=temp_dir / "collaboration")
        
        collab.add_collaborator("project-123", "user-456", role="editor")
        result = collab.add_collaborator("project-123", "user-456", role="viewer")
        
        assert result is False  # Should not add duplicate

    def test_remove_collaborator(self, temp_dir):
        """Test removing a collaborator"""
        collab = CollaborationSystem(data_dir=temp_dir / "collaboration")
        
        collab.add_collaborator("project-123", "user-456", role="editor")
        result = collab.remove_collaborator("project-123", "user-456")
        
        assert result is True
        assert "user-456" not in collab.project_collaborators["project-123"]

    def test_remove_collaborator_not_found(self, temp_dir):
        """Test removing non-existent collaborator"""
        collab = CollaborationSystem(data_dir=temp_dir / "collaboration")
        
        result = collab.remove_collaborator("project-123", "user-999")
        
        assert result is False

    def test_get_collaborators(self, temp_dir):
        """Test getting collaborators"""
        collab = CollaborationSystem(data_dir=temp_dir / "collaboration")
        
        collab.add_collaborator("project-123", "user-1", role="owner")
        collab.add_collaborator("project-123", "user-2", role="editor")
        collab.add_collaborator("project-123", "user-3", role="viewer")
        
        collaborators = collab.get_collaborators("project-123")
        
        assert len(collaborators) == 3
        roles = [c["role"] for c in collaborators]
        assert "owner" in roles
        assert "editor" in roles
        assert "viewer" in roles

    def test_add_comment(self, temp_dir):
        """Test adding a comment"""
        collab = CollaborationSystem(data_dir=temp_dir / "collaboration")
        
        comment = collab.add_comment("project-123", "user-456", "Great project!")
        
        assert comment is not None
        assert comment["comment"] == "Great project!"
        assert comment["user_id"] == "user-456"
        assert len(collab.comments["project-123"]) == 1

    def test_add_comment_reply(self, temp_dir):
        """Test adding a reply to a comment"""
        collab = CollaborationSystem(data_dir=temp_dir / "collaboration")
        
        parent = collab.add_comment("project-123", "user-1", "Original comment")
        reply = collab.add_comment("project-123", "user-2", "Reply", parent_comment_id=parent["id"])
        
        assert reply is not None
        assert reply["parent_comment_id"] == parent["id"]

    def test_get_comments(self, temp_dir):
        """Test getting comments"""
        collab = CollaborationSystem(data_dir=temp_dir / "collaboration")
        
        collab.add_comment("project-123", "user-1", "Comment 1")
        collab.add_comment("project-123", "user-2", "Comment 2")
        
        comments = collab.get_comments("project-123")
        
        assert len(comments) == 2

