"""
Summary Generator
Generate summaries of experiments and results
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class SummaryGenerator:
    """
    Generate summaries
    """
    
    @staticmethod
    def generate_experiment_summary(
        experiments: List[Dict[str, Any]],
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Generate experiment summary
        
        Args:
            experiments: List of experiment dictionaries
            output_path: Output file path (optional)
            
        Returns:
            Path to generated summary
        """
        if output_path is None:
            output_path = Path("experiment_summary.md")
        
        summary_lines = [
            "# Experiment Summary",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Experiments: {len(experiments)}",
            "",
        ]
        
        for i, exp in enumerate(experiments, 1):
            summary_lines.append(f"## Experiment {i}")
            summary_lines.append("")
            
            for key, value in exp.items():
                if isinstance(value, dict):
                    summary_lines.append(f"### {key.title()}")
                    for k, v in value.items():
                        summary_lines.append(f"- **{k}**: {v}")
                else:
                    summary_lines.append(f"- **{key}**: {value}")
            
            summary_lines.append("")
        
        # Save summary
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(summary_lines))
        
        logger.info(f"Generated experiment summary: {output_path}")
        return output_path
    
    @staticmethod
    def generate_metrics_summary(
        metrics: Dict[str, List[float]],
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Generate metrics summary
        
        Args:
            metrics: Dictionary of metric_name -> values
            output_path: Output file path (optional)
            
        Returns:
            Path to generated summary
        """
        if output_path is None:
            output_path = Path("metrics_summary.md")
        
        summary_lines = [
            "# Metrics Summary",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]
        
        for metric_name, values in metrics.items():
            if values:
                summary_lines.append(f"## {metric_name}")
                summary_lines.append("")
                summary_lines.append(f"- **Count**: {len(values)}")
                summary_lines.append(f"- **Mean**: {sum(values) / len(values):.4f}")
                summary_lines.append(f"- **Min**: {min(values):.4f}")
                summary_lines.append(f"- **Max**: {max(values):.4f}")
                summary_lines.append(f"- **Latest**: {values[-1]:.4f}")
                summary_lines.append("")
        
        # Save summary
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(summary_lines))
        
        logger.info(f"Generated metrics summary: {output_path}")
        return output_path



