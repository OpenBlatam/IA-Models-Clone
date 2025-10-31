"""
Plugin ecosystem for Export IA - Extensible architecture.
"""

from .base import BasePlugin, PluginManager, PluginRegistry
from .export_plugins import (
    ExportPlugin,
    PDFExportPlugin,
    DOCXExportPlugin,
    HTMLExportPlugin
)
from .quality_plugins import (
    QualityPlugin,
    GrammarCheckPlugin,
    StyleCheckPlugin,
    AccessibilityPlugin
)
from .ai_plugins import (
    AIPlugin,
    ContentGenerationPlugin,
    TranslationPlugin,
    SummarizationPlugin
)
from .workflow_plugins import (
    WorkflowPlugin,
    NotificationPlugin,
    StoragePlugin,
    AnalyticsPlugin
)

__all__ = [
    "BasePlugin",
    "PluginManager",
    "PluginRegistry",
    "ExportPlugin",
    "PDFExportPlugin",
    "DOCXExportPlugin",
    "HTMLExportPlugin",
    "QualityPlugin",
    "GrammarCheckPlugin",
    "StyleCheckPlugin",
    "AccessibilityPlugin",
    "AIPlugin",
    "ContentGenerationPlugin",
    "TranslationPlugin",
    "SummarizationPlugin",
    "WorkflowPlugin",
    "NotificationPlugin",
    "StoragePlugin",
    "AnalyticsPlugin"
]