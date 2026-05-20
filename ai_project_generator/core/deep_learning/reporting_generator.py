"""
Reporting Generator - Generador de utilidades de reportes
==========================================================

Genera utilidades para generar reportes automáticos:
- Training reports
- Model evaluation reports
- Experiment reports
- HTML/PDF export
"""

import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ReportingGenerator:
    """Generador de utilidades de reportes"""
    
    def __init__(self):
        """Inicializa el generador de reportes"""
        pass
    
    def generate(
        self,
        utils_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """
        Genera utilidades de reportes.
        
        Args:
            utils_dir: Directorio donde generar las utilidades
            keywords: Keywords extraídos
            project_info: Información del proyecto
        """
        utils_dir.mkdir(parents=True, exist_ok=True)
        
        reporting_dir = utils_dir / "reporting"
        reporting_dir.mkdir(parents=True, exist_ok=True)
        
        self._generate_report_generator(reporting_dir, keywords, project_info)
        self._generate_reporting_init(reporting_dir, keywords)
    
    def _generate_reporting_init(
        self,
        reporting_dir: Path,
        keywords: Dict[str, Any],
    ) -> None:
        """Genera __init__.py del módulo de reportes"""
        
        init_content = '''"""
Reporting Utilities Module
============================

Utilidades para generar reportes automáticos.
"""

from .report_generator import (
    ReportGenerator,
    generate_training_report,
    generate_evaluation_report,
    generate_experiment_report,
)

__all__ = [
    "ReportGenerator",
    "generate_training_report",
    "generate_evaluation_report",
    "generate_experiment_report",
]
'''
        
        (reporting_dir / "__init__.py").write_text(init_content, encoding="utf-8")
    
    def _generate_report_generator(
        self,
        reporting_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera generador de reportes"""
        
        report_content = '''"""
Report Generator - Generador de reportes
==========================================

Utilidades para generar reportes automáticos en HTML y texto.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generador de reportes automáticos.
    
    Genera reportes de entrenamiento, evaluación y experimentos.
    """
    
    def __init__(self, reports_dir: Path):
        """
        Inicializa el generador.
        
        Args:
            reports_dir: Directorio donde guardar reportes
        """
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_training_report(
        self,
        training_metrics: Dict[str, Any],
        model_info: Optional[Dict[str, Any]] = None,
        output_format: str = "html",
    ) -> Path:
        """
        Genera reporte de entrenamiento.
        
        Args:
            training_metrics: Métricas de entrenamiento
            model_info: Información del modelo (opcional)
            output_format: Formato de salida (html, text)
        
        Returns:
            Ruta al reporte generado
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.reports_dir / f"training_report_{timestamp}.{output_format}"
        
        if output_format == "html":
            content = self._generate_html_training_report(training_metrics, model_info)
        else:
            content = self._generate_text_training_report(training_metrics, model_info)
        
        report_path.write_text(content, encoding="utf-8")
        logger.info(f"Reporte de entrenamiento generado: {report_path}")
        return report_path
    
    def generate_evaluation_report(
        self,
        evaluation_metrics: Dict[str, Any],
        predictions_info: Optional[Dict[str, Any]] = None,
        output_format: str = "html",
    ) -> Path:
        """
        Genera reporte de evaluación.
        
        Args:
            evaluation_metrics: Métricas de evaluación
            predictions_info: Información de predicciones (opcional)
            output_format: Formato de salida (html, text)
        
        Returns:
            Ruta al reporte generado
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.reports_dir / f"evaluation_report_{timestamp}.{output_format}"
        
        if output_format == "html":
            content = self._generate_html_evaluation_report(evaluation_metrics, predictions_info)
        else:
            content = self._generate_text_evaluation_report(evaluation_metrics, predictions_info)
        
        report_path.write_text(content, encoding="utf-8")
        logger.info(f"Reporte de evaluación generado: {report_path}")
        return report_path
    
    def generate_experiment_report(
        self,
        experiment_info: Dict[str, Any],
        results: Dict[str, Any],
        output_format: str = "html",
    ) -> Path:
        """
        Genera reporte de experimento.
        
        Args:
            experiment_info: Información del experimento
            results: Resultados del experimento
            output_format: Formato de salida (html, text)
        
        Returns:
            Ruta al reporte generado
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.reports_dir / f"experiment_report_{timestamp}.{output_format}"
        
        if output_format == "html":
            content = self._generate_html_experiment_report(experiment_info, results)
        else:
            content = self._generate_text_experiment_report(experiment_info, results)
        
        report_path.write_text(content, encoding="utf-8")
        logger.info(f"Reporte de experimento generado: {report_path}")
        return report_path
    
    def _generate_html_training_report(
        self,
        metrics: Dict[str, Any],
        model_info: Optional[Dict[str, Any]],
    ) -> str:
        """Genera reporte HTML de entrenamiento"""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Training Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .metric {{ font-weight: bold; color: #2196F3; }}
    </style>
</head>
<body>
    <h1>Training Report</h1>
    <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    
    <h2>Training Metrics</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
"""
        
        for key, value in metrics.items():
            html += f"        <tr><td class='metric'>{key}</td><td>{value}</td></tr>\\n"
        
        html += "    </table>\\n"
        
        if model_info:
            html += """
    <h2>Model Information</h2>
    <table>
        <tr><th>Property</th><th>Value</th></tr>
"""
            for key, value in model_info.items():
                html += f"        <tr><td>{key}</td><td>{value}</td></tr>\\n"
            
            html += "    </table>\\n"
        
        html += """
</body>
</html>
"""
        return html
    
    def _generate_text_training_report(
        self,
        metrics: Dict[str, Any],
        model_info: Optional[Dict[str, Any]],
    ) -> str:
        """Genera reporte de texto de entrenamiento"""
        text = f"""Training Report
===============
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Training Metrics:
"""
        for key, value in metrics.items():
            text += f"  {key}: {value}\\n"
        
        if model_info:
            text += "\\nModel Information:\\n"
            for key, value in model_info.items():
                text += f"  {key}: {value}\\n"
        
        return text
    
    def _generate_html_evaluation_report(
        self,
        metrics: Dict[str, Any],
        predictions_info: Optional[Dict[str, Any]],
    ) -> str:
        """Genera reporte HTML de evaluación"""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Evaluation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #2196F3; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>Evaluation Report</h1>
    <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    
    <h2>Evaluation Metrics</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
"""
        
        for key, value in metrics.items():
            html += f"        <tr><td>{key}</td><td>{value}</td></tr>\\n"
        
        html += "    </table>\\n"
        
        if predictions_info:
            html += """
    <h2>Predictions Information</h2>
    <table>
        <tr><th>Property</th><th>Value</th></tr>
"""
            for key, value in predictions_info.items():
                html += f"        <tr><td>{key}</td><td>{value}</td></tr>\\n"
            
            html += "    </table>\\n"
        
        html += """
</body>
</html>
"""
        return html
    
    def _generate_text_evaluation_report(
        self,
        metrics: Dict[str, Any],
        predictions_info: Optional[Dict[str, Any]],
    ) -> str:
        """Genera reporte de texto de evaluación"""
        text = f"""Evaluation Report
================
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Evaluation Metrics:
"""
        for key, value in metrics.items():
            text += f"  {key}: {value}\\n"
        
        if predictions_info:
            text += "\\nPredictions Information:\\n"
            for key, value in predictions_info.items():
                text += f"  {key}: {value}\\n"
        
        return text
    
    def _generate_html_experiment_report(
        self,
        experiment_info: Dict[str, Any],
        results: Dict[str, Any],
    ) -> str:
        """Genera reporte HTML de experimento"""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Experiment Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #FF9800; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>Experiment Report</h1>
    <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    
    <h2>Experiment Information</h2>
    <table>
        <tr><th>Property</th><th>Value</th></tr>
"""
        
        for key, value in experiment_info.items():
            html += f"        <tr><td>{key}</td><td>{value}</td></tr>\\n"
        
        html += """    </table>
    
    <h2>Results</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
"""
        
        for key, value in results.items():
            html += f"        <tr><td>{key}</td><td>{value}</td></tr>\\n"
        
        html += """
    </table>
</body>
</html>
"""
        return html
    
    def _generate_text_experiment_report(
        self,
        experiment_info: Dict[str, Any],
        results: Dict[str, Any],
    ) -> str:
        """Genera reporte de texto de experimento"""
        text = f"""Experiment Report
=================
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Experiment Information:
"""
        for key, value in experiment_info.items():
            text += f"  {key}: {value}\\n"
        
        text += "\\nResults:\\n"
        for key, value in results.items():
            text += f"  {key}: {value}\\n"
        
        return text


def generate_training_report(
    reports_dir: Path,
    training_metrics: Dict[str, Any],
    **kwargs,
) -> Path:
    """
    Función helper para generar reporte de entrenamiento.
    
    Args:
        reports_dir: Directorio de reportes
        training_metrics: Métricas de entrenamiento
        **kwargs: Argumentos adicionales
    
    Returns:
        Ruta al reporte
    """
    generator = ReportGenerator(reports_dir)
    return generator.generate_training_report(training_metrics, **kwargs)


def generate_evaluation_report(
    reports_dir: Path,
    evaluation_metrics: Dict[str, Any],
    **kwargs,
) -> Path:
    """
    Función helper para generar reporte de evaluación.
    
    Args:
        reports_dir: Directorio de reportes
        evaluation_metrics: Métricas de evaluación
        **kwargs: Argumentos adicionales
    
    Returns:
        Ruta al reporte
    """
    generator = ReportGenerator(reports_dir)
    return generator.generate_evaluation_report(evaluation_metrics, **kwargs)


def generate_experiment_report(
    reports_dir: Path,
    experiment_info: Dict[str, Any],
    results: Dict[str, Any],
    **kwargs,
) -> Path:
    """
    Función helper para generar reporte de experimento.
    
    Args:
        reports_dir: Directorio de reportes
        experiment_info: Información del experimento
        results: Resultados
        **kwargs: Argumentos adicionales
    
    Returns:
        Ruta al reporte
    """
    generator = ReportGenerator(reports_dir)
    return generator.generate_experiment_report(experiment_info, results, **kwargs)
'''
        
        (reporting_dir / "report_generator.py").write_text(report_content, encoding="utf-8")

