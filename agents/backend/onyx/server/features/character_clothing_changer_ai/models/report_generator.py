"""
Report Generator for Flux2 Clothing Changer
============================================

Automatic report generation and scheduling.
"""

import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class ReportConfig:
    """Report configuration."""
    report_type: str
    schedule: str  # "daily", "weekly", "monthly", "custom"
    format: str  # "json", "markdown", "html", "csv"
    output_path: Path
    metrics: List[str]
    include_charts: bool = False


class ReportGenerator:
    """Automatic report generation system."""
    
    def __init__(
        self,
        reports_dir: Path = Path("reports"),
    ):
        """
        Initialize report generator.
        
        Args:
            reports_dir: Directory for reports
        """
        self.reports_dir = reports_dir
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        self.report_configs: Dict[str, ReportConfig] = {}
        self.generated_reports: List[Dict[str, Any]] = []
    
    def register_report(self, config: ReportConfig) -> None:
        """Register a report configuration."""
        self.report_configs[config.report_type] = config
        logger.info(f"Registered report: {config.report_type}")
    
    def generate_report(
        self,
        report_type: str,
        data: Dict[str, Any],
        time_range: Optional[timedelta] = None,
    ) -> Path:
        """
        Generate a report.
        
        Args:
            report_type: Type of report
            data: Data for report
            time_range: Optional time range
            
        Returns:
            Path to generated report
        """
        if report_type not in self.report_configs:
            raise ValueError(f"Unknown report type: {report_type}")
        
        config = self.report_configs[report_type]
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{report_type}_{timestamp}.{config.format}"
        file_path = config.output_path / filename
        
        # Generate based on format
        if config.format == "json":
            self._generate_json_report(data, file_path, config)
        elif config.format == "markdown":
            self._generate_markdown_report(data, file_path, config)
        elif config.format == "html":
            self._generate_html_report(data, file_path, config)
        elif config.format == "csv":
            self._generate_csv_report(data, file_path, config)
        else:
            raise ValueError(f"Unsupported format: {config.format}")
        
        # Record generation
        self.generated_reports.append({
            "report_type": report_type,
            "file_path": str(file_path),
            "timestamp": datetime.now().isoformat(),
            "time_range": str(time_range) if time_range else None,
        })
        
        logger.info(f"Generated report: {file_path}")
        return file_path
    
    def _generate_json_report(
        self,
        data: Dict[str, Any],
        file_path: Path,
        config: ReportConfig,
    ) -> None:
        """Generate JSON report."""
        report_data = {
            "report_type": config.report_type,
            "generated_at": datetime.now().isoformat(),
            "metrics": {metric: data.get(metric) for metric in config.metrics if metric in data},
            "data": data,
        }
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, default=str)
    
    def _generate_markdown_report(
        self,
        data: Dict[str, Any],
        file_path: Path,
        config: ReportConfig,
    ) -> None:
        """Generate Markdown report."""
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"# {config.report_type.replace('_', ' ').title()} Report\n\n")
            f.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
            
            # Metrics section
            f.write("## Metrics\n\n")
            for metric in config.metrics:
                if metric in data:
                    value = data[metric]
                    f.write(f"- **{metric.replace('_', ' ').title()}**: {value}\n")
            
            f.write("\n## Details\n\n")
            f.write("```json\n")
            f.write(json.dumps(data, indent=2, default=str))
            f.write("\n```\n")
    
    def _generate_html_report(
        self,
        data: Dict[str, Any],
        file_path: Path,
        config: ReportConfig,
    ) -> None:
        """Generate HTML report."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{config.report_type.replace('_', ' ').title()} Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .metric {{ margin: 10px 0; padding: 10px; background: #f5f5f5; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
    </style>
</head>
<body>
    <h1>{config.report_type.replace('_', ' ').title()} Report</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <h2>Metrics</h2>
    <div class="metrics">
"""
        
        for metric in config.metrics:
            if metric in data:
                value = data[metric]
                html += f'        <div class="metric"><strong>{metric.replace("_", " ").title()}:</strong> {value}</div>\n'
        
        html += """
    </div>
    
    <h2>Data</h2>
    <pre>""" + json.dumps(data, indent=2, default=str) + """</pre>
</body>
</html>
"""
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(html)
    
    def _generate_csv_report(
        self,
        data: Dict[str, Any],
        file_path: Path,
        config: ReportConfig,
    ) -> None:
        """Generate CSV report."""
        import csv
        
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow(["Metric", "Value"])
            
            # Metrics
            for metric in config.metrics:
                if metric in data:
                    writer.writerow([metric, data[metric]])
    
    def get_report_history(
        self,
        report_type: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get report generation history.
        
        Args:
            report_type: Filter by report type
            limit: Maximum number of reports
            
        Returns:
            List of report records
        """
        history = self.generated_reports
        
        if report_type:
            history = [r for r in history if r["report_type"] == report_type]
        
        return history[-limit:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get report generation statistics."""
        by_type = {}
        for report in self.generated_reports:
            report_type = report["report_type"]
            by_type[report_type] = by_type.get(report_type, 0) + 1
        
        return {
            "total_reports": len(self.generated_reports),
            "registered_configs": len(self.report_configs),
            "reports_by_type": by_type,
        }


