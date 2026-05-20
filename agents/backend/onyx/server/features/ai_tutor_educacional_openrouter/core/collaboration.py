"""
Collaboration features for group learning.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class CollaborationType(Enum):
    """Types of collaboration."""
    STUDY_GROUP = "study_group"
    PEER_REVIEW = "peer_review"
    GROUP_PROJECT = "group_project"
    DISCUSSION = "discussion"


class CollaborationManager:
    """
    Manages collaborative learning features.
    """
    
    def __init__(self):
        self.groups: Dict[str, Dict[str, Any]] = {}
        self.sessions: Dict[str, Dict[str, Any]] = {}
    
    def create_group(
        self,
        group_name: str,
        creator_id: str,
        collaboration_type: CollaborationType,
        subject: str,
        max_members: int = 10,
        settings: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a new collaboration group.
        
        Args:
            group_name: Name of the group
            creator_id: ID of the creator
            collaboration_type: Type of collaboration
            subject: Subject area
            max_members: Maximum number of members
            settings: Additional settings
        
        Returns:
            Group information
        """
        group_id = f"group_{datetime.now().timestamp()}"
        
        group = {
            "group_id": group_id,
            "group_name": group_name,
            "creator_id": creator_id,
            "collaboration_type": collaboration_type.value,
            "subject": subject,
            "members": [creator_id],
            "max_members": max_members,
            "settings": settings or {},
            "created_at": datetime.now().isoformat(),
            "status": "active"
        }
        
        self.groups[group_id] = group
        logger.info(f"Created group {group_name} ({group_id})")
        
        return group
    
    def join_group(
        self,
        group_id: str,
        student_id: str
    ) -> Dict[str, Any]:
        """
        Add student to a group.
        
        Args:
            group_id: Group identifier
            student_id: Student identifier
        
        Returns:
            Updated group information
        """
        if group_id not in self.groups:
            return {"error": "Group not found"}
        
        group = self.groups[group_id]
        
        if student_id in group["members"]:
            return {"error": "Already a member"}
        
        if len(group["members"]) >= group["max_members"]:
            return {"error": "Group is full"}
        
        group["members"].append(student_id)
        logger.info(f"Student {student_id} joined group {group_id}")
        
        return group
    
    def leave_group(
        self,
        group_id: str,
        student_id: str
    ) -> Dict[str, Any]:
        """
        Remove student from a group.
        
        Args:
            group_id: Group identifier
            student_id: Student identifier
        
        Returns:
            Updated group information
        """
        if group_id not in self.groups:
            return {"error": "Group not found"}
        
        group = self.groups[group_id]
        
        if student_id not in group["members"]:
            return {"error": "Not a member"}
        
        group["members"].remove(student_id)
        
        # If creator leaves and group is empty, archive it
        if group["creator_id"] == student_id and len(group["members"]) == 0:
            group["status"] = "archived"
        
        logger.info(f"Student {student_id} left group {group_id}")
        
        return group
    
    def create_session(
        self,
        group_id: str,
        session_name: str,
        session_type: str,
        content: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a collaboration session.
        
        Args:
            group_id: Group identifier
            session_name: Name of the session
            session_type: Type of session
            content: Session content
        
        Returns:
            Session information
        """
        if group_id not in self.groups:
            return {"error": "Group not found"}
        
        session_id = f"session_{datetime.now().timestamp()}"
        
        session = {
            "session_id": session_id,
            "group_id": group_id,
            "session_name": session_name,
            "session_type": session_type,
            "content": content or {},
            "participants": self.groups[group_id]["members"].copy(),
            "created_at": datetime.now().isoformat(),
            "status": "active"
        }
        
        self.sessions[session_id] = session
        logger.info(f"Created session {session_name} ({session_id})")
        
        return session
    
    def add_contribution(
        self,
        session_id: str,
        student_id: str,
        contribution: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Add contribution to a session.
        
        Args:
            session_id: Session identifier
            student_id: Student identifier
            contribution: Contribution data
        
        Returns:
            Updated session
        """
        if session_id not in self.sessions:
            return {"error": "Session not found"}
        
        session = self.sessions[session_id]
        
        if student_id not in session["participants"]:
            return {"error": "Not a participant"}
        
        contribution_entry = {
            "student_id": student_id,
            "contribution": contribution,
            "timestamp": datetime.now().isoformat()
        }
        
        if "contributions" not in session:
            session["contributions"] = []
        
        session["contributions"].append(contribution_entry)
        logger.info(f"Added contribution from {student_id} to session {session_id}")
        
        return session
    
    def get_group_stats(self, group_id: str) -> Dict[str, Any]:
        """Get statistics for a group."""
        if group_id not in self.groups:
            return {"error": "Group not found"}
        
        group = self.groups[group_id]
        sessions = [s for s in self.sessions.values() if s["group_id"] == group_id]
        
        return {
            "group_id": group_id,
            "group_name": group["group_name"],
            "member_count": len(group["members"]),
            "max_members": group["max_members"],
            "total_sessions": len(sessions),
            "active_sessions": len([s for s in sessions if s["status"] == "active"]),
            "created_at": group["created_at"]
        }
    
    def search_groups(
        self,
        subject: Optional[str] = None,
        collaboration_type: Optional[CollaborationType] = None,
        has_space: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Search for available groups.
        
        Args:
            subject: Filter by subject
            collaboration_type: Filter by type
            has_space: Only show groups with available space
        
        Returns:
            List of matching groups
        """
        results = []
        
        for group in self.groups.values():
            if group["status"] != "active":
                continue
            
            if subject and group["subject"] != subject:
                continue
            
            if collaboration_type and group["collaboration_type"] != collaboration_type.value:
                continue
            
            if has_space and len(group["members"]) >= group["max_members"]:
                continue
            
            results.append({
                "group_id": group["group_id"],
                "group_name": group["group_name"],
                "collaboration_type": group["collaboration_type"],
                "subject": group["subject"],
                "member_count": len(group["members"]),
                "max_members": group["max_members"]
            })
        
        return results




