"""
Manufacturing AI - Sistema de IA para Optimización de Manufactura
==================================================================

Sistema completo de IA para optimización de procesos de manufactura incluyendo:
- Optimización de capacidad
- Predicción de demanda
- Optimización de inventario
- Mantenimiento predictivo
- Optimización de procesos
- Control de calidad
- Análisis de producción
- Planificación de producción
- Optimización de rutas de producción
"""

__version__ = "1.0.0"
__author__ = "Blatam Academy"
__license__ = "Proprietary"
__description__ = "Complete AI system for manufacturing process optimization including capacity, demand forecasting, inventory, and quality control"

# Core services - import directly from modules with error handling
try:
    from .core.capacity_optimizer import CapacityOptimizer
    from .core.demand_forecasting import DemandForecasting
    from .core.inventory_optimizer import InventoryOptimizer
    from .core.predictive_maintenance import PredictiveMaintenance
    from .core.process_optimizer import ProcessOptimizer
    from .core.production_analyzer import ProductionAnalyzer
    from .core.production_planner import ProductionPlanner
    from .core.production_route_optimizer import ProductionRouteOptimizer
    from .core.quality_control import QualityControl
except ImportError:
    CapacityOptimizer = None
    DemandForecasting = None
    InventoryOptimizer = None
    PredictiveMaintenance = None
    ProcessOptimizer = None
    ProductionAnalyzer = None
    ProductionPlanner = None
    ProductionRouteOptimizer = None
    QualityControl = None

__all__ = [
    "CapacityOptimizer",
    "DemandForecasting",
    "InventoryOptimizer",
    "PredictiveMaintenance",
    "ProcessOptimizer",
    "ProductionAnalyzer",
    "ProductionPlanner",
    "ProductionRouteOptimizer",
    "QualityControl",
]
