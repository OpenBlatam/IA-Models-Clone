"""
Generate Report Use Case

Use case for generating inspection reports.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from ..dto import InspectionResponse, QualityMetricsDTO

logger = logging.getLogger(__name__)


class GenerateReportUseCase:
    """
    Use case for generating inspection reports.
    
    This use case:
    1. Aggregates inspection data
    2. Calculates metrics
    3. Generates report in requested format
    4. Returns report data
    """
    
    def __init__(
        self,
        report_generator=None,  # Will be injected from infrastructure
    ):
        """
        Initialize use case.
        
        Args:
            report_generator: Infrastructure adapter for report generation
        """
        self.report_generator = report_generator
    
    def execute(
        self,
        inspections: List[InspectionResponse],
        report_format: str = "json",
        include_visualizations: bool = False,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Execute generate report use case.
        
        Args:
            inspections: List of inspection responses
            report_format: Format of report ('json', 'html', 'csv', 'pdf')
            include_visualizations: Whether to include visualizations
            period_start: Start of reporting period
            period_end: End of reporting period
        
        Returns:
            Dictionary with report data
        
        Raises:
            ValueError: If report format is invalid
        """
        try:
            # Validate format
            valid_formats = ['json', 'html', 'csv', 'pdf']
            if report_format not in valid_formats:
                raise ValueError(
                    f"Invalid report format: {report_format}. "
                    f"Must be one of: {valid_formats}"
                )
            
            # Calculate metrics
            quality_metrics = self._calculate_metrics(
                inspections, period_start, period_end
            )
            
            # Generate report
            if self.report_generator:
                report_data = self.report_generator.generate(
                    format=report_format,
                    inspections=inspections,
                    quality_metrics=quality_metrics,
                    include_visualizations=include_visualizations,
                    period_start=period_start,
                    period_end=period_end,
                )
            else:
                # Fallback: generate simple JSON report
                report_data = self._generate_simple_report(
                    inspections, quality_metrics, report_format
                )
            
            logger.info(
                f"Report generated: format={report_format}, "
                f"inspections={len(inspections)}"
            )
            
            return report_data
        
        except Exception as e:
            logger.error(f"Report generation failed: {str(e)}", exc_info=True)
            raise
    
    def _calculate_metrics(
        self,
        inspections: List[InspectionResponse],
        period_start: Optional[datetime],
        period_end: Optional[datetime],
    ) -> QualityMetricsDTO:
        """
        Calculate quality metrics from inspections.
        
        Args:
            inspections: List of inspections
            period_start: Start of period
            period_end: End of period
        
        Returns:
            Quality metrics DTO
        """
        if not inspections:
            return QualityMetricsDTO(
                total_inspections=0,
                average_quality_score=0.0,
                period_start=period_start,
                period_end=period_end,
            )
        
        total_inspections = len(inspections)
        average_quality_score = sum(i.quality_score for i in inspections) / total_inspections
        
        total_defects = sum(len(i.defects) for i in inspections)
        total_anomalies = sum(len(i.anomalies) for i in inspections)
        
        rejected_count = sum(1 for i in inspections if not i.is_acceptable)
        rejection_rate = (rejected_count / total_inspections) * 100.0
        
        return QualityMetricsDTO(
            total_inspections=total_inspections,
            average_quality_score=average_quality_score,
            defects_count=total_defects,
            anomalies_count=total_anomalies,
            rejection_rate=rejection_rate,
            acceptance_rate=100.0 - rejection_rate,
            defects_per_inspection=total_defects / total_inspections,
            anomalies_per_inspection=total_anomalies / total_inspections,
            period_start=period_start,
            period_end=period_end,
        )
    
    def _generate_simple_report(
        self,
        inspections: List[InspectionResponse],
        quality_metrics: QualityMetricsDTO,
        format: str,
    ) -> Dict[str, Any]:
        """
        Generate a simple report without external generator.
        
        Args:
            inspections: List of inspections
            quality_metrics: Quality metrics
            format: Report format
        
        Returns:
            Report data dictionary
        """
        return {
            "format": format,
            "generated_at": datetime.utcnow().isoformat(),
            "quality_metrics": quality_metrics.to_dict(),
            "inspections": [i.to_dict() for i in inspections],
            "summary": {
                "total_inspections": quality_metrics.total_inspections,
                "average_quality_score": quality_metrics.average_quality_score,
                "rejection_rate": quality_metrics.rejection_rate,
            },
        }



