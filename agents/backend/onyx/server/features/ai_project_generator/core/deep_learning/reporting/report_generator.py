"""
Report Generator
================

Advanced report generation utilities.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import markdown
try:
    import markdown
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False


def generate_training_report(
    training_history: Dict[str, List[float]],
    config: Dict[str, Any],
    output_path: Path,
    format: str = 'markdown'
) -> Path:
    """
    Generate training report.
    
    Args:
        training_history: Training history metrics
        config: Training configuration
        output_path: Output file path
        format: Report format ('markdown', 'json', 'html')
        
    Returns:
        Path to generated report
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'markdown':
        report = []
        report.append("# Training Report\n\n")
        report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Configuration
        report.append("## Configuration\n\n")
        for key, value in config.items():
            report.append(f"- **{key}:** {value}\n")
        report.append("\n")
        
        # Metrics Summary
        report.append("## Metrics Summary\n\n")
        for metric_name, values in training_history.items():
            if values:
                report.append(f"### {metric_name}\n\n")
                report.append(f"- **Final Value:** {values[-1]:.4f}\n")
                report.append(f"- **Best Value:** {min(values) if 'loss' in metric_name.lower() else max(values):.4f}\n")
                report.append(f"- **Average:** {sum(values) / len(values):.4f}\n\n")
        
        content = ''.join(report)
        output_path.write_text(content)
        
    elif format == 'json':
        report = {
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'metrics': training_history,
            'summary': {
                metric: {
                    'final': values[-1] if values else None,
                    'best': min(values) if values and 'loss' in metric.lower() else max(values) if values else None,
                    'average': sum(values) / len(values) if values else None
                }
                for metric, values in training_history.items()
            }
        }
        output_path.write_text(json.dumps(report, indent=2))
        
    else:
        raise ValueError(f"Unknown format: {format}")
    
    logger.info(f"Training report generated: {output_path}")
    return output_path


def generate_model_report(
    model_info: Dict[str, Any],
    output_path: Path,
    format: str = 'markdown'
) -> Path:
    """
    Generate model report.
    
    Args:
        model_info: Model information
        output_path: Output file path
        format: Report format
        
    Returns:
        Path to generated report
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'markdown':
        report = []
        report.append("# Model Report\n\n")
        report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Model Information
        report.append("## Model Information\n\n")
        for key, value in model_info.items():
            if isinstance(value, (int, float)):
                report.append(f"- **{key}:** {value:,}\n" if isinstance(value, int) else f"- **{key}:** {value:.4f}\n")
            else:
                report.append(f"- **{key}:** {value}\n")
        report.append("\n")
        
        content = ''.join(report)
        output_path.write_text(content)
        
    elif format == 'json':
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_info': model_info
        }
        output_path.write_text(json.dumps(report, indent=2))
        
    else:
        raise ValueError(f"Unknown format: {format}")
    
    logger.info(f"Model report generated: {output_path}")
    return output_path


def generate_experiment_report(
    experiments: List[Dict[str, Any]],
    output_path: Path,
    format: str = 'markdown'
) -> Path:
    """
    Generate experiment comparison report.
    
    Args:
        experiments: List of experiment results
        output_path: Output file path
        format: Report format
        
    Returns:
        Path to generated report
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'markdown':
        report = []
        report.append("# Experiment Report\n\n")
        report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        report.append(f"**Total Experiments:** {len(experiments)}\n\n")
        
        # Experiments
        report.append("## Experiments\n\n")
        for idx, exp in enumerate(experiments, 1):
            report.append(f"### Experiment {idx}: {exp.get('name', f'Experiment {idx}')}\n\n")
            for key, value in exp.items():
                if key != 'name':
                    if isinstance(value, (int, float)):
                        report.append(f"- **{key}:** {value:,}\n" if isinstance(value, int) else f"- **{key}:** {value:.4f}\n")
                    else:
                        report.append(f"- **{key}:** {value}\n")
            report.append("\n")
        
        content = ''.join(report)
        output_path.write_text(content)
        
    elif format == 'json':
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_experiments': len(experiments),
            'experiments': experiments
        }
        output_path.write_text(json.dumps(report, indent=2))
        
    else:
        raise ValueError(f"Unknown format: {format}")
    
    logger.info(f"Experiment report generated: {output_path}")
    return output_path


class ReportGenerator:
    """
    Comprehensive report generator.
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize report generator.
        
        Args:
            output_dir: Output directory
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_all_reports(
        self,
        training_history: Optional[Dict[str, List[float]]] = None,
        model_info: Optional[Dict[str, Any]] = None,
        experiments: Optional[List[Dict[str, Any]]] = None,
        format: str = 'markdown'
    ) -> Dict[str, Path]:
        """
        Generate all reports.
        
        Args:
            training_history: Training history
            model_info: Model information
            experiments: Experiment results
            format: Report format
            
        Returns:
            Dictionary with report paths
        """
        reports = {}
        
        if training_history:
            reports['training'] = generate_training_report(
                training_history, {}, self.output_dir / 'training_report.md', format
            )
        
        if model_info:
            reports['model'] = generate_model_report(
                model_info, self.output_dir / 'model_report.md', format
            )
        
        if experiments:
            reports['experiments'] = generate_experiment_report(
                experiments, self.output_dir / 'experiment_report.md', format
            )
        
        return reports



