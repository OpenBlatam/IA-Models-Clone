"""
Workflow Status Value Object
===========================

Value object representing workflow status with business rules.
"""

from __future__ import annotations
from enum import Enum
from typing import List


class WorkflowStatus(Enum):
    """
    Workflow Status Enum
    
    Represents the possible states of a workflow with business rules.
    """
    
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"
    DELETED = "deleted"
    
    @classmethod
    def get_active_statuses(cls) -> List[WorkflowStatus]:
        """Get statuses that represent active workflows"""
        return [cls.ACTIVE, cls.PAUSED]
    
    @classmethod
    def get_final_statuses(cls) -> List[WorkflowStatus]:
        """Get statuses that represent final states"""
        return [cls.COMPLETED, cls.CANCELLED, cls.DELETED]
    
    @classmethod
    def get_editable_statuses(cls) -> List[WorkflowStatus]:
        """Get statuses that allow editing"""
        return [cls.DRAFT, cls.ACTIVE, cls.PAUSED]
    
    def is_active(self) -> bool:
        """Check if status represents an active workflow"""
        return self in self.get_active_statuses()
    
    def is_final(self) -> bool:
        """Check if status represents a final state"""
        return self in self.get_final_statuses()
    
    def is_editable(self) -> bool:
        """Check if status allows editing"""
        return self in self.get_editable_statuses()
    
    def can_transition_to(self, target_status: WorkflowStatus) -> bool:
        """Check if transition to target status is valid"""
        valid_transitions = {
            self.DRAFT: [self.ACTIVE, self.DELETED],
            self.ACTIVE: [self.PAUSED, self.COMPLETED, self.CANCELLED, self.DELETED],
            self.PAUSED: [self.ACTIVE, self.CANCELLED, self.DELETED],
            self.COMPLETED: [self.ACTIVE, self.DELETED],
            self.ERROR: [self.DRAFT, self.ACTIVE, self.DELETED],
            self.CANCELLED: [self.DELETED],
            self.DELETED: []  # Cannot transition from deleted
        }
        
        return target_status in valid_transitions.get(self, [])
    
    def get_valid_transitions(self) -> List[WorkflowStatus]:
        """Get list of valid transitions from current status"""
        valid_transitions = {
            self.DRAFT: [self.ACTIVE, self.DELETED],
            self.ACTIVE: [self.PAUSED, self.COMPLETED, self.CANCELLED, self.DELETED],
            self.PAUSED: [self.ACTIVE, self.CANCELLED, self.DELETED],
            self.COMPLETED: [self.ACTIVE, self.DELETED],
            self.ERROR: [self.DRAFT, self.ACTIVE, self.DELETED],
            self.CANCELLED: [self.DELETED],
            self.DELETED: []
        }
        
        return valid_transitions.get(self, [])




