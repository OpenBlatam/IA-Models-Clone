"""
Quality Routes - API endpoints for quality assurance and testing
Following FastAPI best practices: functional programming, RORO pattern, async operations
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field, validator

from ..core.quality_assurance_engine import (
    assess_code_quality,
    assess_content_quality,
    run_quality_tests,
    generate_quality_report,
    get_quality_metrics,
    get_quality_reports,
    get_test_results,
    get_quality_engine_health,
    QualityMetric,
    QualityReport,
    TestResult,
    CodeQuality,
    ContentQuality
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/quality", tags=["Quality Assurance"])


# Request/Response Models
class CodeQualityAssessmentRequest(BaseModel):
    """Request model for code quality assessment"""
    file_path: str = Field(..., description="Path to the code file to assess")
    
    @validator('file_path')
    def validate_file_path(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('File path cannot be empty')
        return v


class ContentQualityAssessmentRequest(BaseModel):
    """Request model for content quality assessment"""
    content: str = Field(..., description="Content to assess for quality")
    content_id: str = Field("", description="Unique identifier for the content")
    
    @validator('content')
    def validate_content(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Content cannot be empty')
        return v


class QualityTestRequest(BaseModel):
    """Request model for quality tests"""
    test_type: str = Field("all", description="Type of tests to run")
    
    @validator('test_type')
    def validate_test_type(cls, v):
        if v not in ['all', 'unit', 'integration', 'performance', 'security', 'quality']:
            raise ValueError('Test type must be one of: all, unit, integration, performance, security, quality')
        return v


class QualityReportRequest(BaseModel):
    """Request model for quality report generation"""
    report_type: str = Field("comprehensive", description="Type of quality report to generate")
    
    @validator('report_type')
    def validate_report_type(cls, v):
        if v not in ['comprehensive', 'code_quality', 'content_quality', 'performance_quality']:
            raise ValueError('Report type must be one of: comprehensive, code_quality, content_quality, performance_quality')
        return v


class CodeQualityAssessmentResponse(BaseModel):
    """Response model for code quality assessment"""
    success: bool
    data: Optional[CodeQuality] = None
    error: Optional[str] = None
    processing_time: float
    timestamp: datetime


class ContentQualityAssessmentResponse(BaseModel):
    """Response model for content quality assessment"""
    success: bool
    data: Optional[ContentQuality] = None
    error: Optional[str] = None
    processing_time: float
    timestamp: datetime


class QualityTestResponse(BaseModel):
    """Response model for quality tests"""
    success: bool
    data: Optional[List[TestResult]] = None
    error: Optional[str] = None
    processing_time: float
    timestamp: datetime


class QualityReportResponse(BaseModel):
    """Response model for quality report"""
    success: bool
    data: Optional[QualityReport] = None
    error: Optional[str] = None
    processing_time: float
    timestamp: datetime


class QualityMetricsResponse(BaseModel):
    """Response model for quality metrics"""
    success: bool
    data: Optional[List[QualityMetric]] = None
    error: Optional[str] = None
    processing_time: float
    timestamp: datetime


class QualityReportsResponse(BaseModel):
    """Response model for quality reports"""
    success: bool
    data: Optional[List[QualityReport]] = None
    error: Optional[str] = None
    processing_time: float
    timestamp: datetime


class TestResultsResponse(BaseModel):
    """Response model for test results"""
    success: bool
    data: Optional[List[TestResult]] = None
    error: Optional[str] = None
    processing_time: float
    timestamp: datetime


class HealthResponse(BaseModel):
    """Response model for health check"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: datetime


# Route Handlers
@router.post("/assess-code", response_model=CodeQualityAssessmentResponse)
async def assess_code_quality_endpoint(
    request: CodeQualityAssessmentRequest,
    background_tasks: BackgroundTasks
) -> CodeQualityAssessmentResponse:
    """Assess code quality for a specific file"""
    start_time = datetime.now()
    
    try:
        logger.info(f"Assessing code quality for file: {request.file_path}")
        
        # Assess code quality
        code_quality = await assess_code_quality(request.file_path)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Log assessment
        background_tasks.add_task(
            log_code_quality_assessment,
            request.file_path,
            code_quality.quality_score,
            code_quality.cyclomatic_complexity
        )
        
        return CodeQualityAssessmentResponse(
            success=True,
            data=code_quality,
            processing_time=processing_time,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"Code quality assessment failed: {e}")
        
        return CodeQualityAssessmentResponse(
            success=False,
            error=str(e),
            processing_time=processing_time,
            timestamp=datetime.now()
        )


@router.post("/assess-content", response_model=ContentQualityAssessmentResponse)
async def assess_content_quality_endpoint(
    request: ContentQualityAssessmentRequest,
    background_tasks: BackgroundTasks
) -> ContentQualityAssessmentResponse:
    """Assess content quality"""
    start_time = datetime.now()
    
    try:
        logger.info(f"Assessing content quality for content: {request.content_id or 'unknown'}")
        
        # Assess content quality
        content_quality = await assess_content_quality(
            content=request.content,
            content_id=request.content_id
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Log assessment
        background_tasks.add_task(
            log_content_quality_assessment,
            request.content_id or "unknown",
            content_quality.overall_quality_score,
            content_quality.readability_score
        )
        
        return ContentQualityAssessmentResponse(
            success=True,
            data=content_quality,
            processing_time=processing_time,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"Content quality assessment failed: {e}")
        
        return ContentQualityAssessmentResponse(
            success=False,
            error=str(e),
            processing_time=processing_time,
            timestamp=datetime.now()
        )


@router.post("/run-tests", response_model=QualityTestResponse)
async def run_quality_tests_endpoint(
    request: QualityTestRequest,
    background_tasks: BackgroundTasks
) -> QualityTestResponse:
    """Run quality tests"""
    start_time = datetime.now()
    
    try:
        logger.info(f"Running quality tests: {request.test_type}")
        
        # Run quality tests
        test_results = await run_quality_tests(request.test_type)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Log test execution
        background_tasks.add_task(
            log_quality_tests,
            request.test_type,
            len(test_results),
            sum(1 for r in test_results if r.status == "passed")
        )
        
        return QualityTestResponse(
            success=True,
            data=test_results,
            processing_time=processing_time,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"Quality tests failed: {e}")
        
        return QualityTestResponse(
            success=False,
            error=str(e),
            processing_time=processing_time,
            timestamp=datetime.now()
        )


@router.post("/generate-report", response_model=QualityReportResponse)
async def generate_quality_report_endpoint(
    request: QualityReportRequest,
    background_tasks: BackgroundTasks
) -> QualityReportResponse:
    """Generate quality report"""
    start_time = datetime.now()
    
    try:
        logger.info(f"Generating quality report: {request.report_type}")
        
        # Generate quality report
        quality_report = await generate_quality_report(request.report_type)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Log report generation
        background_tasks.add_task(
            log_quality_report,
            request.report_type,
            quality_report.overall_score,
            quality_report.quality_level
        )
        
        return QualityReportResponse(
            success=True,
            data=quality_report,
            processing_time=processing_time,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"Quality report generation failed: {e}")
        
        return QualityReportResponse(
            success=False,
            error=str(e),
            processing_time=processing_time,
            timestamp=datetime.now()
        )


@router.get("/metrics", response_model=QualityMetricsResponse)
async def get_quality_metrics_endpoint(
    limit: int = 100,
    background_tasks: BackgroundTasks = None
) -> QualityMetricsResponse:
    """Get quality metrics"""
    start_time = datetime.now()
    
    try:
        logger.info(f"Getting quality metrics (limit: {limit})")
        
        # Get quality metrics
        metrics = await get_quality_metrics(limit)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Log metrics retrieval
        if background_tasks:
            background_tasks.add_task(
                log_metrics_retrieval,
                limit,
                len(metrics)
            )
        
        return QualityMetricsResponse(
            success=True,
            data=metrics,
            processing_time=processing_time,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"Failed to get quality metrics: {e}")
        
        return QualityMetricsResponse(
            success=False,
            error=str(e),
            processing_time=processing_time,
            timestamp=datetime.now()
        )


@router.get("/reports", response_model=QualityReportsResponse)
async def get_quality_reports_endpoint(
    limit: int = 50,
    background_tasks: BackgroundTasks = None
) -> QualityReportsResponse:
    """Get quality reports"""
    start_time = datetime.now()
    
    try:
        logger.info(f"Getting quality reports (limit: {limit})")
        
        # Get quality reports
        reports = await get_quality_reports(limit)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Log reports retrieval
        if background_tasks:
            background_tasks.add_task(
                log_reports_retrieval,
                limit,
                len(reports)
            )
        
        return QualityReportsResponse(
            success=True,
            data=reports,
            processing_time=processing_time,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"Failed to get quality reports: {e}")
        
        return QualityReportsResponse(
            success=False,
            error=str(e),
            processing_time=processing_time,
            timestamp=datetime.now()
        )


@router.get("/test-results", response_model=TestResultsResponse)
async def get_test_results_endpoint(
    limit: int = 100,
    background_tasks: BackgroundTasks = None
) -> TestResultsResponse:
    """Get test results"""
    start_time = datetime.now()
    
    try:
        logger.info(f"Getting test results (limit: {limit})")
        
        # Get test results
        results = await get_test_results(limit)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Log results retrieval
        if background_tasks:
            background_tasks.add_task(
                log_test_results_retrieval,
                limit,
                len(results)
            )
        
        return TestResultsResponse(
            success=True,
            data=results,
            processing_time=processing_time,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"Failed to get test results: {e}")
        
        return TestResultsResponse(
            success=False,
            error=str(e),
            processing_time=processing_time,
            timestamp=datetime.now()
        )


@router.get("/health", response_model=HealthResponse)
async def get_quality_engine_health_endpoint(
    background_tasks: BackgroundTasks = None
) -> HealthResponse:
    """Get quality engine health status"""
    try:
        logger.info("Checking quality engine health")
        
        # Get health status
        health_data = await get_quality_engine_health()
        
        # Log health check
        if background_tasks:
            background_tasks.add_task(
                log_health_check,
                health_data.get("status", "unknown")
            )
        
        return HealthResponse(
            success=True,
            data=health_data,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        
        return HealthResponse(
            success=False,
            error=str(e),
            timestamp=datetime.now()
        )


@router.get("/standards")
async def get_quality_standards() -> Dict[str, Any]:
    """Get quality standards and thresholds"""
    return {
        "quality_standards": {
            "code_quality": {
                "cyclomatic_complexity": {
                    "excellent": "≤ 5",
                    "good": "≤ 10",
                    "fair": "≤ 15",
                    "poor": "> 15"
                },
                "maintainability_index": {
                    "excellent": "≥ 80",
                    "good": "≥ 70",
                    "fair": "≥ 60",
                    "poor": "< 60"
                },
                "code_coverage": {
                    "excellent": "≥ 90%",
                    "good": "≥ 80%",
                    "fair": "≥ 70%",
                    "poor": "< 70%"
                },
                "test_coverage": {
                    "excellent": "≥ 95%",
                    "good": "≥ 85%",
                    "fair": "≥ 75%",
                    "poor": "< 75%"
                },
                "code_duplication": {
                    "excellent": "≤ 5%",
                    "good": "≤ 10%",
                    "fair": "≤ 15%",
                    "poor": "> 15%"
                },
                "security_issues": {
                    "excellent": "0",
                    "good": "≤ 1",
                    "fair": "≤ 3",
                    "poor": "> 3"
                },
                "style_violations": {
                    "excellent": "0",
                    "good": "≤ 5",
                    "fair": "≤ 10",
                    "poor": "> 10"
                }
            },
            "content_quality": {
                "readability_score": {
                    "excellent": "≥ 80",
                    "good": "≥ 70",
                    "fair": "≥ 60",
                    "poor": "< 60"
                },
                "grammar_score": {
                    "excellent": "≥ 95%",
                    "good": "≥ 90%",
                    "fair": "≥ 80%",
                    "poor": "< 80%"
                },
                "style_score": {
                    "excellent": "≥ 85",
                    "good": "≥ 75",
                    "fair": "≥ 65",
                    "poor": "< 65"
                },
                "originality_score": {
                    "excellent": "≥ 90%",
                    "good": "≥ 80%",
                    "fair": "≥ 70%",
                    "poor": "< 70%"
                },
                "accuracy_score": {
                    "excellent": "≥ 95%",
                    "good": "≥ 90%",
                    "fair": "≥ 80%",
                    "poor": "< 80%"
                },
                "completeness_score": {
                    "excellent": "≥ 90%",
                    "good": "≥ 80%",
                    "fair": "≥ 70%",
                    "poor": "< 70%"
                },
                "consistency_score": {
                    "excellent": "≥ 90%",
                    "good": "≥ 80%",
                    "fair": "≥ 70%",
                    "poor": "< 70%"
                },
                "engagement_score": {
                    "excellent": "≥ 85",
                    "good": "≥ 75",
                    "fair": "≥ 65",
                    "poor": "< 65"
                },
                "seo_score": {
                    "excellent": "≥ 90",
                    "good": "≥ 80",
                    "fair": "≥ 70",
                    "poor": "< 70"
                }
            },
            "performance_quality": {
                "response_time": {
                    "excellent": "≤ 100ms",
                    "good": "≤ 500ms",
                    "fair": "≤ 1000ms",
                    "poor": "> 1000ms"
                },
                "throughput": {
                    "excellent": "≥ 1000 ops/sec",
                    "good": "≥ 500 ops/sec",
                    "fair": "≥ 100 ops/sec",
                    "poor": "< 100 ops/sec"
                },
                "memory_usage": {
                    "excellent": "≤ 100MB",
                    "good": "≤ 500MB",
                    "fair": "≤ 1000MB",
                    "poor": "> 1000MB"
                },
                "cpu_usage": {
                    "excellent": "≤ 50%",
                    "good": "≤ 70%",
                    "fair": "≤ 85%",
                    "poor": "> 85%"
                },
                "error_rate": {
                    "excellent": "≤ 0.1%",
                    "good": "≤ 1%",
                    "fair": "≤ 5%",
                    "poor": "> 5%"
                }
            }
        },
        "quality_levels": {
            "excellent": "Outstanding quality with no issues",
            "good": "High quality with minor issues",
            "fair": "Acceptable quality with some issues",
            "poor": "Low quality with significant issues"
        },
        "test_types": {
            "unit": "Tests individual functions and methods",
            "integration": "Tests component interactions",
            "performance": "Tests performance and load",
            "security": "Tests security vulnerabilities",
            "quality": "Tests code quality and style"
        }
    }


@router.get("/capabilities")
async def get_quality_capabilities() -> Dict[str, Any]:
    """Get quality assurance capabilities"""
    return {
        "quality_capabilities": {
            "code_quality_assessment": {
                "cyclomatic_complexity": "Measures code complexity",
                "maintainability_index": "Measures code maintainability",
                "code_coverage": "Measures test coverage",
                "code_duplication": "Detects duplicate code",
                "security_analysis": "Identifies security issues",
                "style_analysis": "Checks coding standards"
            },
            "content_quality_assessment": {
                "readability_analysis": "Measures content readability",
                "grammar_checking": "Checks grammar and syntax",
                "style_analysis": "Analyzes writing style",
                "originality_checking": "Detects plagiarism",
                "accuracy_verification": "Verifies factual accuracy",
                "completeness_checking": "Checks content completeness",
                "consistency_analysis": "Analyzes consistency",
                "engagement_analysis": "Measures engagement potential",
                "seo_analysis": "Analyzes SEO optimization"
            },
            "testing_framework": {
                "unit_testing": "Automated unit test execution",
                "integration_testing": "Component integration tests",
                "performance_testing": "Performance and load tests",
                "security_testing": "Security vulnerability tests",
                "quality_testing": "Code quality and style tests"
            },
            "reporting": {
                "comprehensive_reports": "Detailed quality reports",
                "metric_tracking": "Quality metric monitoring",
                "trend_analysis": "Quality trend analysis",
                "recommendations": "Quality improvement recommendations"
            }
        },
        "quality_tools": {
            "static_analysis": "Code analysis without execution",
            "dynamic_analysis": "Runtime code analysis",
            "coverage_analysis": "Test coverage measurement",
            "security_scanning": "Security vulnerability scanning",
            "style_checking": "Code style and format checking",
            "performance_profiling": "Performance analysis and profiling"
        },
        "quality_metrics": {
            "quantitative": "Measurable quality metrics",
            "qualitative": "Subjective quality assessments",
            "comparative": "Quality comparisons and benchmarks",
            "predictive": "Quality prediction and forecasting"
        }
    }


# Background Tasks
async def log_code_quality_assessment(file_path: str, quality_score: float, complexity: float) -> None:
    """Log code quality assessment"""
    try:
        logger.info(f"Code quality assessed - File: {file_path}, Score: {quality_score:.2f}, Complexity: {complexity:.2f}")
    except Exception as e:
        logger.warning(f"Failed to log code quality assessment: {e}")


async def log_content_quality_assessment(content_id: str, overall_score: float, readability_score: float) -> None:
    """Log content quality assessment"""
    try:
        logger.info(f"Content quality assessed - ID: {content_id}, Overall: {overall_score:.2f}, Readability: {readability_score:.2f}")
    except Exception as e:
        logger.warning(f"Failed to log content quality assessment: {e}")


async def log_quality_tests(test_type: str, total_tests: int, passed_tests: int) -> None:
    """Log quality tests"""
    try:
        logger.info(f"Quality tests completed - Type: {test_type}, Total: {total_tests}, Passed: {passed_tests}")
    except Exception as e:
        logger.warning(f"Failed to log quality tests: {e}")


async def log_quality_report(report_type: str, overall_score: float, quality_level: str) -> None:
    """Log quality report"""
    try:
        logger.info(f"Quality report generated - Type: {report_type}, Score: {overall_score:.2f}, Level: {quality_level}")
    except Exception as e:
        logger.warning(f"Failed to log quality report: {e}")


async def log_metrics_retrieval(limit: int, metrics_count: int) -> None:
    """Log metrics retrieval"""
    try:
        logger.info(f"Quality metrics retrieved - Limit: {limit}, Count: {metrics_count}")
    except Exception as e:
        logger.warning(f"Failed to log metrics retrieval: {e}")


async def log_reports_retrieval(limit: int, reports_count: int) -> None:
    """Log reports retrieval"""
    try:
        logger.info(f"Quality reports retrieved - Limit: {limit}, Count: {reports_count}")
    except Exception as e:
        logger.warning(f"Failed to log reports retrieval: {e}")


async def log_test_results_retrieval(limit: int, results_count: int) -> None:
    """Log test results retrieval"""
    try:
        logger.info(f"Test results retrieved - Limit: {limit}, Count: {results_count}")
    except Exception as e:
        logger.warning(f"Failed to log test results retrieval: {e}")


async def log_health_check(status: str) -> None:
    """Log health check"""
    try:
        logger.info(f"Quality engine health check - Status: {status}")
    except Exception as e:
        logger.warning(f"Failed to log health check: {e}")


