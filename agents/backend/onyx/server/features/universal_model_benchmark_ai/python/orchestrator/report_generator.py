"""
Report Generator for Orchestrator - Enhanced reporting integration.
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from benchmarks.types import BenchmarkResult
from core.reporting import ReportGenerator, ReportConfig
from core.visualization import VisualizationGenerator, prepare_dashboard_data

logger = logging.getLogger(__name__)


class OrchestratorReportGenerator:
    """Enhanced report generator for orchestrator results."""
    
    def __init__(self, output_dir: str = "reports"):
        """
        Initialize orchestrator report generator.
        
        Args:
            output_dir: Output directory for reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.report_generator = ReportGenerator(
            config=ReportConfig(
                include_detailed_results=True,
                include_percentiles=True,
                include_memory_stats=True,
            )
        )
        
        self.viz_generator = VisualizationGenerator()
    
    def generate_execution_report(
        self,
        execution_results: List[Any],  # ExecutionResult from orchestrator
        output_format: str = "json",
    ) -> Path:
        """
        Generate comprehensive execution report.
        
        Args:
            execution_results: List of ExecutionResult objects
            output_format: Output format (json, markdown, html)
            
        Returns:
            Path to generated report
        """
        # Convert execution results to report format
        reports = []
        for result in execution_results:
            if result.success and result.result:
                report = self.report_generator.generate_report(
                    result.result,
                    result.model_name,
                    result.benchmark_name,
                    metadata={
                        "execution_time": result.execution_time,
                        "success": result.success,
                    }
                )
                reports.append(report)
        
        # Generate comparison if multiple models
        if len(reports) > 1:
            comparison = self.report_generator.generate_comparison_report(
                [r.get("result") for r in reports if r.get("result")],
                [r.get("model_name") for r in reports],
                reports[0].get("benchmark_name", "unknown") if reports else "unknown",
            )
            
            # Export comparison
            comparison_path = self.output_dir / f"comparison_{Path(output_format).suffix or '.json'}"
            self.report_generator.config.format = output_format
            self.report_generator.export_report(comparison, comparison_path)
        
        # Export individual reports
        for report in reports:
            report_path = self.output_dir / f"{report['report_id']}.{output_format}"
            self.report_generator.export_report(report, report_path)
        
        # Generate visualization data
        dashboard_data = prepare_dashboard_data([
            {"model": r.get("model_name"), "result": r.get("result")}
            for r in reports if r.get("result")
        ])
        
        viz_path = self.output_dir / "dashboard_data.json"
        import json
        with open(viz_path, 'w') as f:
            json.dump(dashboard_data, f, indent=2, default=str)
        
        logger.info(f"Reports generated in {self.output_dir}")
        return self.output_dir

