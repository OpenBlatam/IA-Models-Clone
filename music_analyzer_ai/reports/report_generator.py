"""
Advanced Report Generation
Generate comprehensive reports for music analysis
"""

from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
from pathlib import Path
import json

logger = logging.getLogger(__name__)

try:
    from jinja2 import Template
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False
    logger.warning("Jinja2 not available, HTML reports disabled")


class ReportGenerator:
    """
    Generate comprehensive reports for music analysis
    """
    
    def __init__(self, output_dir: str = "./reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_analysis_report(
        self,
        analysis_data: Dict[str, Any],
        track_info: Dict[str, Any],
        format: str = "json"  # "json", "html", "markdown"
    ) -> str:
        """Generate analysis report"""
        report = {
            "track_info": track_info,
            "analysis": analysis_data,
            "generated_at": datetime.now().isoformat(),
            "version": "2.8.0"
        }
        
        if format == "json":
            return self._generate_json_report(report)
        elif format == "html":
            return self._generate_html_report(report)
        elif format == "markdown":
            return self._generate_markdown_report(report)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def _generate_json_report(self, report: Dict[str, Any]) -> str:
        """Generate JSON report"""
        filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Generated JSON report: {filepath}")
        return str(filepath)
    
    def _generate_html_report(self, report: Dict[str, Any]) -> str:
        """Generate HTML report"""
        if not JINJA2_AVAILABLE:
            logger.warning("Jinja2 not available, falling back to JSON")
            return self._generate_json_report(report)
        
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Music Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #333; }
                h2 { color: #666; margin-top: 30px; }
                .section { margin: 20px 0; padding: 15px; background: #f5f5f5; border-radius: 5px; }
                .metric { display: inline-block; margin: 10px; padding: 10px; background: white; border-radius: 3px; }
                .metric-label { font-weight: bold; color: #666; }
                .metric-value { font-size: 24px; color: #333; }
            </style>
        </head>
        <body>
            <h1>Music Analysis Report</h1>
            <div class="section">
                <h2>Track Information</h2>
                <p><strong>Name:</strong> {{ track_info.name }}</p>
                <p><strong>Artists:</strong> {{ track_info.artists|join(', ') }}</p>
                <p><strong>Generated At:</strong> {{ generated_at }}</p>
            </div>
            <div class="section">
                <h2>Analysis Results</h2>
                <div class="metric">
                    <div class="metric-label">Genre</div>
                    <div class="metric-value">{{ analysis.genre|default('N/A') }}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Mood</div>
                    <div class="metric-value">{{ analysis.mood|default('N/A') }}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Energy</div>
                    <div class="metric-value">{{ analysis.energy|default('N/A') }}</div>
                </div>
            </div>
        </body>
        </html>
        """
        
        template = Template(html_template)
        html_content = template.render(**report)
        
        filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        filepath = self.output_dir / filename
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        logger.info(f"Generated HTML report: {filepath}")
        return str(filepath)
    
    def _generate_markdown_report(self, report: Dict[str, Any]) -> str:
        """Generate Markdown report"""
        markdown = f"""# Music Analysis Report

## Track Information

- **Name**: {report['track_info'].get('name', 'N/A')}
- **Artists**: {', '.join(report['track_info'].get('artists', []))}
- **Generated At**: {report['generated_at']}

## Analysis Results

### Genre
{report['analysis'].get('genre', 'N/A')}

### Mood
{report['analysis'].get('mood', 'N/A')}

### Energy
{report['analysis'].get('energy', 'N/A')}

### Additional Metrics
"""
        
        # Add more metrics
        for key, value in report['analysis'].items():
            if key not in ['genre', 'mood', 'energy']:
                markdown += f"- **{key}**: {value}\n"
        
        filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        filepath = self.output_dir / filename
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(markdown)
        
        logger.info(f"Generated Markdown report: {filepath}")
        return str(filepath)
    
    def generate_training_report(
        self,
        training_history: Dict[str, List[float]],
        model_info: Dict[str, Any],
        format: str = "html"
    ) -> str:
        """Generate training report"""
        report = {
            "model_info": model_info,
            "training_history": training_history,
            "generated_at": datetime.now().isoformat()
        }
        
        return self.generate_analysis_report(report, {"name": "Training Report"}, format)

