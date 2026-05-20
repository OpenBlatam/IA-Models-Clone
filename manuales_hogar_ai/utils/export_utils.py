"""
Utilidades de Exportación (Legacy)
===================================

Este archivo mantiene compatibilidad hacia atrás.
Nuevo código debe usar utils.export.manual_exporter.ManualExporter
"""

from .export.manual_exporter import ManualExporter

__all__ = ["ManualExporter"]
