"""
Dashboard Manager
================

Dashboard management.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class DashboardWidget:
    """Dashboard widget."""
    id: str
    type: str  # chart, metric, table, etc.
    title: str
    config: Dict[str, Any]
    position: Dict[str, int] = None  # x, y, width, height
    
    def __post_init__(self):
        if self.position is None:
            self.position = {"x": 0, "y": 0, "width": 4, "height": 3}


@dataclass
class Dashboard:
    """Dashboard definition."""
    id: str
    name: str
    widgets: List[DashboardWidget]
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()


class DashboardManager:
    """Dashboard manager."""
    
    def __init__(self):
        self._dashboards: Dict[str, Dashboard] = {}
    
    def create_dashboard(self, dashboard_id: str, name: str) -> Dashboard:
        """Create dashboard."""
        dashboard = Dashboard(
            id=dashboard_id,
            name=name,
            widgets=[]
        )
        
        self._dashboards[dashboard_id] = dashboard
        logger.info(f"Created dashboard: {dashboard_id}")
        return dashboard
    
    def add_widget(
        self,
        dashboard_id: str,
        widget: DashboardWidget
    ):
        """Add widget to dashboard."""
        if dashboard_id not in self._dashboards:
            raise ValueError(f"Dashboard {dashboard_id} not found")
        
        dashboard = self._dashboards[dashboard_id]
        dashboard.widgets.append(widget)
        dashboard.updated_at = datetime.now()
        
        logger.info(f"Added widget {widget.id} to dashboard {dashboard_id}")
    
    def remove_widget(self, dashboard_id: str, widget_id: str):
        """Remove widget from dashboard."""
        if dashboard_id not in self._dashboards:
            return False
        
        dashboard = self._dashboards[dashboard_id]
        dashboard.widgets = [w for w in dashboard.widgets if w.id != widget_id]
        dashboard.updated_at = datetime.now()
        
        logger.info(f"Removed widget {widget_id} from dashboard {dashboard_id}")
        return True
    
    def get_dashboard(self, dashboard_id: str) -> Optional[Dashboard]:
        """Get dashboard by ID."""
        return self._dashboards.get(dashboard_id)
    
    def list_dashboards(self) -> List[Dashboard]:
        """List all dashboards."""
        return list(self._dashboards.values())
    
    def get_dashboard_stats(self) -> Dict[str, Any]:
        """Get dashboard statistics."""
        return {
            "total_dashboards": len(self._dashboards),
            "total_widgets": sum(len(d.widgets) for d in self._dashboards.values()),
            "dashboards": {
                dashboard_id: {
                    "name": dashboard.name,
                    "widgets": len(dashboard.widgets)
                }
                for dashboard_id, dashboard in self._dashboards.items()
            }
        }















