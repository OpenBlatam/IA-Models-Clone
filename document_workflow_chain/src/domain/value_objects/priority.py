"""
Priority Value Object
====================

Value object representing priority levels with business rules.
"""

from __future__ import annotations
from enum import IntEnum
from typing import List


class Priority(IntEnum):
    """
    Priority Enum
    
    Represents priority levels with numeric values for comparison.
    Higher numbers indicate higher priority.
    """
    
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5
    
    @classmethod
    def get_high_priorities(cls) -> List[Priority]:
        """Get high priority levels"""
        return [cls.HIGH, cls.URGENT, cls.CRITICAL]
    
    @classmethod
    def get_low_priorities(cls) -> List[Priority]:
        """Get low priority levels"""
        return [cls.LOW, cls.NORMAL]
    
    def is_high_priority(self) -> bool:
        """Check if this is a high priority"""
        return self in self.get_high_priorities()
    
    def is_low_priority(self) -> bool:
        """Check if this is a low priority"""
        return self in self.get_low_priorities()
    
    def is_critical(self) -> bool:
        """Check if this is critical priority"""
        return self == self.CRITICAL
    
    def is_urgent(self) -> bool:
        """Check if this is urgent priority"""
        return self == self.URGENT
    
    def get_display_name(self) -> str:
        """Get human-readable display name"""
        display_names = {
            self.LOW: "Low",
            self.NORMAL: "Normal",
            self.HIGH: "High",
            self.URGENT: "Urgent",
            self.CRITICAL: "Critical"
        }
        return display_names.get(self, "Unknown")
    
    def get_color_code(self) -> str:
        """Get color code for UI display"""
        color_codes = {
            self.LOW: "#28a745",      # Green
            self.NORMAL: "#17a2b8",   # Blue
            self.HIGH: "#ffc107",     # Yellow
            self.URGENT: "#fd7e14",   # Orange
            self.CRITICAL: "#dc3545"  # Red
        }
        return color_codes.get(self, "#6c757d")  # Gray
    
    def get_processing_time_multiplier(self) -> float:
        """Get processing time multiplier based on priority"""
        multipliers = {
            self.LOW: 1.0,
            self.NORMAL: 1.0,
            self.HIGH: 0.8,
            self.URGENT: 0.6,
            self.CRITICAL: 0.4
        }
        return multipliers.get(self, 1.0)




