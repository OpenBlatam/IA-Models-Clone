"""
Visualization domain services
"""

from services.domains import register_service

try:
    from services.visualization_service import VisualizationService
    from services.advanced_visual_progress_service import AdvancedVisualProgressService
    
    def register_services():
        register_service("visualization", "visualization", VisualizationService)
        register_service("visualization", "advanced_progress", AdvancedVisualProgressService)
except ImportError:
    pass



