"""
Summary Generator
================

Generador de resúmenes del proyecto.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class ProjectSummaryGenerator:
    """
    Generador de resumen del proyecto.
    
    Genera un resumen completo del estado del proyecto.
    """
    
    def __init__(self, project_root: str = "."):
        """
        Inicializar generador.
        
        Args:
            project_root: Raíz del proyecto
        """
        self.project_root = Path(project_root)
    
    def generate_summary(self) -> Dict[str, Any]:
        """
        Generar resumen completo del proyecto.
        
        Returns:
            Resumen del proyecto
        """
        summary = {
            "project_name": "Robot Movement AI",
            "version": "1.0.0",
            "generated_at": datetime.now().isoformat(),
            "statistics": self._get_statistics(),
            "modules": self._get_modules_summary(),
            "features": self._get_features_summary(),
            "apis": self._get_apis_summary(),
            "algorithms": self._get_algorithms_summary(),
            "systems": self._get_systems_summary()
        }
        
        return summary
    
    def _get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas del proyecto."""
        core_dir = self.project_root / "core"
        api_dir = self.project_root / "api"
        
        core_files = list(core_dir.glob("*.py")) if core_dir.exists() else []
        api_files = list(api_dir.glob("*.py")) if api_dir.exists() else []
        
        total_lines = 0
        for file in core_files + api_files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    total_lines += len(f.readlines())
            except Exception:
                pass
        
        return {
            "total_core_modules": len(core_files),
            "total_api_modules": len(api_files),
            "total_python_files": len(core_files) + len(api_files),
            "estimated_lines_of_code": total_lines
        }
    
    def _get_modules_summary(self) -> List[str]:
        """Obtener resumen de módulos."""
        return [
            "Trajectory Optimizer",
            "Movement Engine",
            "Inverse Kinematics",
            "Visual Processor",
            "Real-time Feedback",
            "Chat Controller",
            "Metrics System",
            "Event System",
            "Cache System",
            "Health Check System",
            "Monitoring System",
            "Analytics Engine",
            "Auto Optimizer",
            "Continuous Learning",
            "Benchmarking",
            "Task Scheduler",
            "Workflow System",
            "Rate Limiter",
            "Validation Engine",
            "Notification System",
            "API Client",
            "Data Pipeline",
            "State Machine",
            "Documentation Generator",
            "Code Quality Analyzer",
            "Performance Monitor",
            "Error Tracker"
        ]
    
    def _get_features_summary(self) -> List[str]:
        """Obtener resumen de características."""
        return [
            "Reinforcement Learning (PPO, DQN)",
            "Pathfinding Algorithms (A*, RRT)",
            "Multi-objective Optimization",
            "Real-time Feedback (up to 2000 Hz)",
            "Chat-based Robot Control",
            "ROS Integration",
            "Multi-robot Support",
            "Advanced Caching",
            "Event-driven Architecture",
            "Comprehensive Monitoring",
            "Auto-optimization",
            "Continuous Learning",
            "Workflow Management",
            "State Machine Management",
            "Data Pipeline Processing",
            "Error Tracking",
            "Performance Monitoring",
            "Quality Assurance",
            "Version Management",
            "Backup and Recovery",
            "Dynamic Configuration",
            "Rate Limiting",
            "Validation Engine",
            "Notification System",
            "API Client",
            "Documentation Generation",
            "Code Quality Analysis"
        ]
    
    def _get_apis_summary(self) -> List[str]:
        """Obtener resumen de APIs."""
        return [
            "Robot Control API",
            "Metrics API",
            "Resources API",
            "System API",
            "Analytics API",
            "Tasks API",
            "Notifications API",
            "Monitoring API"
        ]
    
    def _get_algorithms_summary(self) -> List[str]:
        """Obtener resumen de algoritmos."""
        return [
            "PPO (Proximal Policy Optimization)",
            "DQN (Deep Q-Network)",
            "A* Pathfinding",
            "RRT (Rapidly-exploring Random Tree)",
            "Heuristic Optimization",
            "Multi-objective Optimization"
        ]
    
    def _get_systems_summary(self) -> List[str]:
        """Obtener resumen de sistemas."""
        return [
            "Initialization System",
            "Metrics System",
            "Event System",
            "Cache System",
            "Health Check System",
            "Monitoring System",
            "Analytics System",
            "Auto Optimization System",
            "Continuous Learning System",
            "Benchmarking System",
            "Task Scheduler",
            "Workflow System",
            "Rate Limiter",
            "Validation Engine",
            "Notification System",
            "API Client",
            "Data Pipeline System",
            "State Machine System",
            "Documentation Generator",
            "Code Quality System",
            "Performance Monitor",
            "Error Tracker",
            "Version Management",
            "Backup System",
            "Dynamic Configuration",
            "Resource Manager",
            "Quality Assurance"
        ]
    
    def export_summary(self, output_file: str, format: str = "json") -> None:
        """
        Exportar resumen.
        
        Args:
            output_file: Archivo de salida
            format: Formato ("json" o "markdown")
        """
        summary = self.generate_summary()
        
        if format == "json":
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
        elif format == "markdown":
            lines = [
                f"# {summary['project_name']} - Project Summary",
                "",
                f"**Version:** {summary['version']}",
                f"**Generated:** {summary['generated_at']}",
                "",
                "## Statistics",
                "",
                f"- Total Core Modules: {summary['statistics']['total_core_modules']}",
                f"- Total API Modules: {summary['statistics']['total_api_modules']}",
                f"- Total Python Files: {summary['statistics']['total_python_files']}",
                f"- Estimated Lines of Code: {summary['statistics']['estimated_lines_of_code']}",
                "",
                "## Modules",
                ""
            ]
            
            for module in summary['modules']:
                lines.append(f"- {module}")
            
            lines.extend([
                "",
                "## Features",
                ""
            ])
            
            for feature in summary['features']:
                lines.append(f"- {feature}")
            
            lines.extend([
                "",
                "## APIs",
                ""
            ])
            
            for api in summary['apis']:
                lines.append(f"- {api}")
            
            lines.extend([
                "",
                "## Algorithms",
                ""
            ])
            
            for algorithm in summary['algorithms']:
                lines.append(f"- {algorithm}")
            
            lines.extend([
                "",
                "## Systems",
                ""
            ])
            
            for system in summary['systems']:
                lines.append(f"- {system}")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
        
        logger.info(f"Summary exported to {output_file}")


# Instancia global
_summary_generator: Optional[ProjectSummaryGenerator] = None


def get_summary_generator(project_root: str = ".") -> ProjectSummaryGenerator:
    """Obtener instancia global del generador de resumen."""
    global _summary_generator
    if _summary_generator is None:
        _summary_generator = ProjectSummaryGenerator(project_root=project_root)
    return _summary_generator

