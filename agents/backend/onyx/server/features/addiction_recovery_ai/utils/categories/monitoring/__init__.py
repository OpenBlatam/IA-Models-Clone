"""
Monitoring and observability utilities
"""

from utils.categories import register_utility

try:
    from utils.metrics import Metrics
    from utils.metrics_dashboard import MetricsDashboard
    from utils.monitoring_dashboard import MonitoringDashboard
    from utils.analytics import Analytics
    from utils.profiler import Profiler
    
    def register_utilities():
        register_utility("monitoring", "metrics", Metrics)
        register_utility("monitoring", "metrics_dashboard", MetricsDashboard)
        register_utility("monitoring", "monitoring_dashboard", MonitoringDashboard)
        register_utility("monitoring", "analytics", Analytics)
        register_utility("monitoring", "profiler", Profiler)
except ImportError:
    pass



