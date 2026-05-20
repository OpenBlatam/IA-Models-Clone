"""
Micro-frontend architecture for Export IA.
"""

from .components import (
    ExportWidget,
    QualityWidget,
    TaskWidget,
    DashboardWidget,
    AnalyticsWidget
)
from .shell import MicroFrontendShell
from .registry import ComponentRegistry
from .communication import InterComponentCommunication
from .routing import MicroFrontendRouter

__all__ = [
    "ExportWidget",
    "QualityWidget",
    "TaskWidget",
    "DashboardWidget",
    "AnalyticsWidget",
    "MicroFrontendShell",
    "ComponentRegistry",
    "InterComponentCommunication",
    "MicroFrontendRouter"
]




