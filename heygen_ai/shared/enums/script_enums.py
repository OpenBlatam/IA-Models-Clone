"""
Script Enums
============

Enumerations for script generation.
"""

from enum import Enum


class ScriptStyle(str, Enum):
    """Script style enumeration."""
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    EDUCATIONAL = "educational"
    MARKETING = "marketing"
    STORYTELLING = "storytelling"



