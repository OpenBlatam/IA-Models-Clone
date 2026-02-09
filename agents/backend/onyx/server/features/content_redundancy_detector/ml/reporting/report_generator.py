"""
Report Generator
Generate comprehensive reports
"""

from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generate comprehensive reports
    """
    
    def __init__(self, output_dir: Path = Path("reports")):
        """
        Initialize report generator
        
        Args:
            output_dir: Output directory for reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_training_report(
        self,
        history: Dict[str, Any],
        config: Dict[str, Any],
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Generate training report
        
        Args:
            history: Training history
            config: Training configuration
            output_path: Output file path (optional)
            
        Returns:
            Path to generated report
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"training_report_{timestamp}.md"
        
        report_lines = [
            "# Training Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Configuration",
            "",
        ]
        
        # Add config
        for section, values in config.items():
            report_lines.append(f"### {section.title()}")
            for key, value in values.items():
                report_lines.append(f"- **{key}**: {value}")
            report_lines.append("")
        
        # Add history
        report_lines.append("## Training History")
        report_lines.append("")
        
        if 'best_metrics' in history:
            report_lines.append("### Best Metrics")
            for metric, value in history['best_metrics'].items():
                report_lines.append(f"- **{metric}**: {value:.4f}")
            report_lines.append("")
        
        if 'final_metrics' in history:
            report_lines.append("### Final Metrics")
            for metric, value in history['final_metrics'].items():
                report_lines.append(f"- **{metric}**: {value:.4f}")
            report_lines.append("")
        
        # Save report
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(report_lines))
        
        logger.info(f"Generated training report: {output_path}")
        return output_path
    
    def generate_model_report(
        self,
        model_info: Dict[str, Any],
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Generate model report
        
        Args:
            model_info: Model information
            output_path: Output file path (optional)
            
        Returns:
            Path to generated report
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"model_report_{timestamp}.md"
        
        report_lines = [
            "# Model Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Model Information",
            "",
        ]
        
        for key, value in model_info.items():
            report_lines.append(f"- **{key}**: {value}")
        
        # Save report
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(report_lines))
        
        logger.info(f"Generated model report: {output_path}")
        return output_path



