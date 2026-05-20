"""
Report generator for student progress and analytics.
"""

import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generates comprehensive reports for students, teachers, and administrators.
    """
    
    def __init__(self, learning_analyzer, metrics_collector):
        self.learning_analyzer = learning_analyzer
        self.metrics_collector = metrics_collector
    
    def generate_student_report(
        self,
        student_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive student progress report.
        
        Args:
            student_id: Student identifier
            start_date: Start date for report period
            end_date: End date for report period
        
        Returns:
            Complete student report
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now()
        
        profile = self.learning_analyzer.student_profiles.get(student_id, {})
        strengths_weaknesses = self.learning_analyzer.get_strengths_and_weaknesses(student_id)
        
        report = {
            "student_id": student_id,
            "report_period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "generated_at": datetime.now().isoformat(),
            "overview": {
                "total_subjects_studied": len(profile.get("subjects", {})),
                "overall_performance": profile.get("overall_performance", 0.0),
                "learning_style": profile.get("learning_style", "balanced")
            },
            "strengths": strengths_weaknesses.get("strengths", {}),
            "weaknesses": strengths_weaknesses.get("weaknesses", {}),
            "subject_breakdown": {},
            "recommendations": []
        }
        
        # Subject breakdown
        for subject, topics in profile.get("subjects", {}).items():
            subject_performance = []
            for topic, data in topics.items():
                subject_performance.append({
                    "topic": topic,
                    "performance": data.get("performance", 0.0),
                    "difficulty": data.get("difficulty", "intermedio"),
                    "last_practiced": data.get("last_practiced", "")
                })
            
            avg_performance = sum(p["performance"] for p in subject_performance) / len(subject_performance) if subject_performance else 0.0
            
            report["subject_breakdown"][subject] = {
                "average_performance": avg_performance,
                "topics": subject_performance,
                "total_topics": len(subject_performance)
            }
        
        # Generate recommendations
        report["recommendations"] = self._generate_recommendations(student_id, strengths_weaknesses)
        
        return report
    
    def generate_class_report(
        self,
        student_ids: List[str],
        subject: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate report for a class or group of students.
        
        Args:
            student_ids: List of student identifiers
            subject: Optional subject filter
        
        Returns:
            Class report with aggregated statistics
        """
        reports = []
        for student_id in student_ids:
            report = self.generate_student_report(student_id)
            if subject is None or subject in report["subject_breakdown"]:
                reports.append(report)
        
        # Aggregate statistics
        total_students = len(reports)
        if total_students == 0:
            return {"error": "No students found"}
        
        avg_performance = sum(r["overview"]["overall_performance"] for r in reports) / total_students
        
        subject_stats = {}
        for report in reports:
            for subject, data in report["subject_breakdown"].items():
                if subject not in subject_stats:
                    subject_stats[subject] = {
                        "students_count": 0,
                        "total_performance": 0.0,
                        "topics_covered": set()
                    }
                subject_stats[subject]["students_count"] += 1
                subject_stats[subject]["total_performance"] += data["average_performance"]
                subject_stats[subject]["topics_covered"].update(
                    [t["topic"] for t in data["topics"]]
                )
        
        # Calculate averages
        for subject in subject_stats:
            stats = subject_stats[subject]
            stats["average_performance"] = stats["total_performance"] / stats["students_count"]
            stats["topics_covered"] = list(stats["topics_covered"])
            del stats["total_performance"]
        
        return {
            "class_id": "class_001",
            "generated_at": datetime.now().isoformat(),
            "total_students": total_students,
            "average_performance": avg_performance,
            "subject_statistics": subject_stats,
            "individual_reports": reports
        }
    
    def export_report(
        self,
        report: Dict[str, Any],
        format: str = "json",
        output_path: Optional[str] = None
    ) -> str:
        """
        Export report to file.
        
        Args:
            report: Report data
            format: Export format (json, markdown, html)
            output_path: Optional output path
        
        Returns:
            Path to exported file
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            student_id = report.get("student_id", "unknown")
            output_path = f"reports/{student_id}_{timestamp}.{format}"
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        elif format == "markdown":
            content = self._report_to_markdown(report)
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(content)
        elif format == "html":
            content = self._report_to_html(report)
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(content)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return str(output_file)
    
    def _generate_recommendations(
        self,
        student_id: str,
        strengths_weaknesses: Dict
    ) -> List[str]:
        """Generate personalized recommendations."""
        recommendations = []
        
        weaknesses = strengths_weaknesses.get("weaknesses", {})
        if weaknesses:
            for subject, topics in weaknesses.items():
                recommendations.append(
                    f"Enfócate en practicar {', '.join(topics[:3])} en {subject}"
                )
        
        strengths = strengths_weaknesses.get("strengths", {})
        if strengths:
            for subject, topics in strengths.items():
                recommendations.append(
                    f"Considera temas más avanzados en {subject} basado en tu dominio de {', '.join(topics[:2])}"
                )
        
        if not recommendations:
            recommendations.append("Continúa practicando regularmente para mantener tu progreso")
        
        return recommendations
    
    def _report_to_markdown(self, report: Dict[str, Any]) -> str:
        """Convert report to markdown format."""
        md = f"# Reporte de Progreso Estudiantil\n\n"
        md += f"**Estudiante:** {report.get('student_id', 'N/A')}\n"
        md += f"**Período:** {report.get('report_period', {}).get('start', 'N/A')} - {report.get('report_period', {}).get('end', 'N/A')}\n"
        md += f"**Generado:** {report.get('generated_at', 'N/A')}\n\n"
        
        overview = report.get("overview", {})
        md += f"## Resumen\n\n"
        md += f"- **Rendimiento General:** {overview.get('overall_performance', 0.0):.1%}\n"
        md += f"- **Materias Estudiadas:** {overview.get('total_subjects_studied', 0)}\n"
        md += f"- **Estilo de Aprendizaje:** {overview.get('learning_style', 'N/A')}\n\n"
        
        md += f"## Fortalezas\n\n"
        for subject, topics in report.get("strengths", {}).items():
            md += f"### {subject}\n"
            for topic in topics:
                md += f"- {topic}\n"
            md += "\n"
        
        md += f"## Áreas de Mejora\n\n"
        for subject, topics in report.get("weaknesses", {}).items():
            md += f"### {subject}\n"
            for topic in topics:
                md += f"- {topic}\n"
            md += "\n"
        
        md += f"## Recomendaciones\n\n"
        for rec in report.get("recommendations", []):
            md += f"- {rec}\n"
        
        return md
    
    def _report_to_html(self, report: Dict[str, Any]) -> str:
        """Convert report to HTML format."""
        html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Reporte de Progreso</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #2c3e50; }
        h2 { color: #34495e; margin-top: 30px; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #3498db; color: white; }
        .strength { color: #27ae60; }
        .weakness { color: #e74c3c; }
    </style>
</head>
<body>
"""
        html += f"<h1>Reporte de Progreso Estudiantil</h1>"
        html += f"<p><strong>Estudiante:</strong> {report.get('student_id', 'N/A')}</p>"
        html += f"<p><strong>Período:</strong> {report.get('report_period', {}).get('start', 'N/A')} - {report.get('report_period', {}).get('end', 'N/A')}</p>"
        
        overview = report.get("overview", {})
        html += f"<h2>Resumen</h2>"
        html += f"<p>Rendimiento General: <strong>{overview.get('overall_performance', 0.0):.1%}</strong></p>"
        html += f"<p>Materias Estudiadas: {overview.get('total_subjects_studied', 0)}</p>"
        
        html += f"<h2>Fortalezas</h2>"
        for subject, topics in report.get("strengths", {}).items():
            html += f"<h3>{subject}</h3><ul>"
            for topic in topics:
                html += f"<li class='strength'>{topic}</li>"
            html += "</ul>"
        
        html += f"<h2>Áreas de Mejora</h2>"
        for subject, topics in report.get("weaknesses", {}).items():
            html += f"<h3>{subject}</h3><ul>"
            for topic in topics:
                html += f"<li class='weakness'>{topic}</li>"
            html += "</ul>"
        
        html += f"<h2>Recomendaciones</h2><ul>"
        for rec in report.get("recommendations", []):
            html += f"<li>{rec}</li>"
        html += "</ul>"
        
        html += "</body></html>"
        return html






