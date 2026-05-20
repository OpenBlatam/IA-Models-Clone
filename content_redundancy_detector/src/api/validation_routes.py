"""
Validation Routes - API endpoints for advanced validation and testing
Following FastAPI best practices: functional programming, RORO pattern, async operations
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field, validator

from ..core.advanced_validation_engine import (
    validate_data,
    run_tests,
    generate_validation_report,
    get_validation_results,
    get_test_results,
    get_quality_metrics,
    get_validation_engine_health,
    ValidationResult,
    TestResult,
    QualityMetric,
    ValidationReport
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/validation", tags=["Advanced Validation"])


# Request/Response Models
class DataValidationRequest(BaseModel):
    """Request model for data validation"""
    data: Any = Field(..., description="Data to validate")
    validation_type: str = Field(..., description="Type of validation to perform")
    schema_name: Optional[str] = Field(None, description="Schema name for validation")
    
    @validator('validation_type')
    def validate_validation_type(cls, v):
        if v not in ['code_quality', 'content_quality', 'security', 'performance', 'api_request', 'user_data']:
            raise ValueError('Validation type must be one of: code_quality, content_quality, security, performance, api_request, user_data')
        return v


class TestExecutionRequest(BaseModel):
    """Request model for test execution"""
    test_type: str = Field("all", description="Type of tests to run")
    
    @validator('test_type')
    def validate_test_type(cls, v):
        if v not in ['all', 'unit', 'integration', 'performance', 'security', 'validation']:
            raise ValueError('Test type must be one of: all, unit, integration, performance, security, validation')
        return v


class ValidationReportRequest(BaseModel):
    """Request model for validation report generation"""
    report_type: str = Field("comprehensive", description="Type of validation report to generate")
    
    @validator('report_type')
    def validate_report_type(cls, v):
        if v not in ['comprehensive', 'code_quality', 'content_quality', 'security', 'performance']:
            raise ValueError('Report type must be one of: comprehensive, code_quality, content_quality, security, performance')
        return v


class DataValidationResponse(BaseModel):
    """Response model for data validation"""
    success: bool
    data: Optional[List[ValidationResult]] = None
    error: Optional[str] = None
    processing_time: float
    timestamp: datetime


class TestExecutionResponse(BaseModel):
    """Response model for test execution"""
    success: bool
    data: Optional[List[TestResult]] = None
    error: Optional[str] = None
    processing_time: float
    timestamp: datetime


class ValidationReportResponse(BaseModel):
    """Response model for validation report"""
    success: bool
    data: Optional[ValidationReport] = None
    error: Optional[str] = None
    processing_time: float
    timestamp: datetime


class ValidationResultsResponse(BaseModel):
    """Response model for validation results"""
    success: bool
    data: Optional[List[ValidationResult]] = None
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


class QualityMetricsResponse(BaseModel):
    """Response model for quality metrics"""
    success: bool
    data: Optional[List[QualityMetric]] = None
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
@router.post("/validate-data", response_model=DataValidationResponse)
async def validate_data_endpoint(
    request: DataValidationRequest,
    background_tasks: BackgroundTasks
) -> DataValidationResponse:
    """Validate data against rules and schemas"""
    start_time = datetime.now()
    
    try:
        logger.info(f"Validating data with type: {request.validation_type}")
        
        # Validate data
        validation_results = await validate_data(
            data=request.data,
            validation_type=request.validation_type,
            schema_name=request.schema_name
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Log validation
        background_tasks.add_task(
            log_data_validation,
            request.validation_type,
            len(validation_results),
            len([r for r in validation_results if r.status == "passed"])
        )
        
        return DataValidationResponse(
            success=True,
            data=validation_results,
            processing_time=processing_time,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"Data validation failed: {e}")
        
        return DataValidationResponse(
            success=False,
            error=str(e),
            processing_time=processing_time,
            timestamp=datetime.now()
        )


@router.post("/run-tests", response_model=TestExecutionResponse)
async def run_tests_endpoint(
    request: TestExecutionRequest,
    background_tasks: BackgroundTasks
) -> TestExecutionResponse:
    """Run validation tests"""
    start_time = datetime.now()
    
    try:
        logger.info(f"Running tests: {request.test_type}")
        
        # Run tests
        test_results = await run_tests(request.test_type)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Log test execution
        background_tasks.add_task(
            log_test_execution,
            request.test_type,
            len(test_results),
            len([r for r in test_results if r.status == "passed"])
        )
        
        return TestExecutionResponse(
            success=True,
            data=test_results,
            processing_time=processing_time,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"Test execution failed: {e}")
        
        return TestExecutionResponse(
            success=False,
            error=str(e),
            processing_time=processing_time,
            timestamp=datetime.now()
        )


@router.post("/generate-report", response_model=ValidationReportResponse)
async def generate_validation_report_endpoint(
    request: ValidationReportRequest,
    background_tasks: BackgroundTasks
) -> ValidationReportResponse:
    """Generate validation report"""
    start_time = datetime.now()
    
    try:
        logger.info(f"Generating validation report: {request.report_type}")
        
        # Generate validation report
        validation_report = await generate_validation_report(request.report_type)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Log report generation
        background_tasks.add_task(
            log_validation_report,
            request.report_type,
            validation_report.validation_score,
            validation_report.total_validations
        )
        
        return ValidationReportResponse(
            success=True,
            data=validation_report,
            processing_time=processing_time,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"Validation report generation failed: {e}")
        
        return ValidationReportResponse(
            success=False,
            error=str(e),
            processing_time=processing_time,
            timestamp=datetime.now()
        )


@router.get("/results", response_model=ValidationResultsResponse)
async def get_validation_results_endpoint(
    limit: int = 100,
    background_tasks: BackgroundTasks = None
) -> ValidationResultsResponse:
    """Get validation results"""
    start_time = datetime.now()
    
    try:
        logger.info(f"Getting validation results (limit: {limit})")
        
        # Get validation results
        results = await get_validation_results(limit)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Log results retrieval
        if background_tasks:
            background_tasks.add_task(
                log_validation_results_retrieval,
                limit,
                len(results)
            )
        
        return ValidationResultsResponse(
            success=True,
            data=results,
            processing_time=processing_time,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"Failed to get validation results: {e}")
        
        return ValidationResultsResponse(
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


@router.get("/quality-metrics", response_model=QualityMetricsResponse)
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
                log_quality_metrics_retrieval,
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


@router.get("/health", response_model=HealthResponse)
async def get_validation_engine_health_endpoint(
    background_tasks: BackgroundTasks = None
) -> HealthResponse:
    """Get validation engine health status"""
    try:
        logger.info("Checking validation engine health")
        
        # Get health status
        health_data = await get_validation_engine_health()
        
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


@router.get("/validation-types")
async def get_validation_types() -> Dict[str, Any]:
    """Get available validation types"""
    return {
        "validation_types": {
            "code_quality": {
                "description": "Code quality validation including complexity, maintainability, and style",
                "rules": [
                    "Function length should be under 50 lines",
                    "Class length should be under 200 lines",
                    "Functions should have less than 5 parameters",
                    "Use snake_case for variables and functions",
                    "Functions should have docstrings"
                ]
            },
            "content_quality": {
                "description": "Content quality validation including readability, grammar, and style",
                "rules": [
                    "Content should be readable by target audience",
                    "Content should be grammatically correct",
                    "Content should have appropriate length",
                    "Sentences should not be too long"
                ]
            },
            "security": {
                "description": "Security validation including injection protection and authentication",
                "rules": [
                    "Prevent SQL injection vulnerabilities",
                    "Prevent XSS vulnerabilities",
                    "Passwords should be strong"
                ]
            },
            "performance": {
                "description": "Performance validation including response time and resource usage",
                "rules": [
                    "API responses should be under 1 second",
                    "Memory usage should be under 500MB",
                    "CPU usage should be under 80%"
                ]
            },
            "api_request": {
                "description": "API request validation using JSON Schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "minLength": 1},
                        "content_id": {"type": "string"},
                        "options": {"type": "object"}
                    },
                    "required": ["content"]
                }
            },
            "user_data": {
                "description": "User data validation using Pydantic models",
                "schema": {
                    "type": "object",
                    "properties": {
                        "username": {"type": "string", "minLength": 3, "maxLength": 50},
                        "email": {"type": "string", "format": "email"},
                        "password": {"type": "string", "minLength": 8}
                    },
                    "required": ["username", "email", "password"]
                }
            }
        },
        "test_types": {
            "unit": "Unit tests for individual functions and methods",
            "integration": "Integration tests for component interactions",
            "performance": "Performance and load tests",
            "security": "Security vulnerability tests",
            "validation": "Validation rule tests"
        },
        "custom_validators": {
            "email": "Email address validation",
            "phone": "Phone number validation",
            "url": "URL validation",
            "date": "Date string validation",
            "json": "JSON string validation",
            "xml": "XML string validation",
            "csv": "CSV string validation",
            "password": "Password strength validation",
            "credit_card": "Credit card number validation using Luhn algorithm",
            "uuid": "UUID string validation"
        }
    }


@router.get("/capabilities")
async def get_validation_capabilities() -> Dict[str, Any]:
    """Get validation engine capabilities"""
    return {
        "validation_capabilities": {
            "data_validation": {
                "json_schema": "JSON Schema validation for structured data",
                "pydantic_models": "Pydantic model validation with type checking",
                "custom_rules": "Custom validation rules with expressions",
                "custom_validators": "Specialized validators for different data types"
            },
            "testing_framework": {
                "unit_testing": "Automated unit test execution",
                "integration_testing": "Component integration tests",
                "performance_testing": "Performance and load tests",
                "security_testing": "Security vulnerability tests",
                "validation_testing": "Validation rule tests"
            },
            "quality_monitoring": {
                "real_time_metrics": "Real-time quality metrics monitoring",
                "trend_analysis": "Quality trend analysis over time",
                "alerting": "Automatic alerts for quality issues",
                "reporting": "Comprehensive quality reports"
            },
            "validation_types": {
                "code_quality": "Code quality validation with complexity analysis",
                "content_quality": "Content quality validation with readability analysis",
                "security": "Security validation with vulnerability detection",
                "performance": "Performance validation with resource monitoring"
            }
        },
        "validation_features": {
            "rule_engine": "Configurable validation rules with expressions",
            "schema_validation": "Multiple schema validation formats",
            "custom_validators": "Extensible custom validation functions",
            "test_automation": "Automated test execution and reporting",
            "quality_metrics": "Comprehensive quality metrics collection",
            "real_time_monitoring": "Real-time validation and quality monitoring"
        },
        "supported_formats": {
            "json": "JSON data validation",
            "xml": "XML data validation",
            "csv": "CSV data validation",
            "yaml": "YAML data validation",
            "python_objects": "Python object validation",
            "api_requests": "API request validation",
            "user_data": "User data validation"
        }
    }


# Background Tasks
async def log_data_validation(validation_type: str, total_results: int, passed_results: int) -> None:
    """Log data validation"""
    try:
        logger.info(f"Data validation completed - Type: {validation_type}, Total: {total_results}, Passed: {passed_results}")
    except Exception as e:
        logger.warning(f"Failed to log data validation: {e}")


async def log_test_execution(test_type: str, total_tests: int, passed_tests: int) -> None:
    """Log test execution"""
    try:
        logger.info(f"Test execution completed - Type: {test_type}, Total: {total_tests}, Passed: {passed_tests}")
    except Exception as e:
        logger.warning(f"Failed to log test execution: {e}")


async def log_validation_report(report_type: str, validation_score: float, total_validations: int) -> None:
    """Log validation report"""
    try:
        logger.info(f"Validation report generated - Type: {report_type}, Score: {validation_score:.2f}, Total: {total_validations}")
    except Exception as e:
        logger.warning(f"Failed to log validation report: {e}")


async def log_validation_results_retrieval(limit: int, results_count: int) -> None:
    """Log validation results retrieval"""
    try:
        logger.info(f"Validation results retrieved - Limit: {limit}, Count: {results_count}")
    except Exception as e:
        logger.warning(f"Failed to log validation results retrieval: {e}")


async def log_test_results_retrieval(limit: int, results_count: int) -> None:
    """Log test results retrieval"""
    try:
        logger.info(f"Test results retrieved - Limit: {limit}, Count: {results_count}")
    except Exception as e:
        logger.warning(f"Failed to log test results retrieval: {e}")


async def log_quality_metrics_retrieval(limit: int, metrics_count: int) -> None:
    """Log quality metrics retrieval"""
    try:
        logger.info(f"Quality metrics retrieved - Limit: {limit}, Count: {metrics_count}")
    except Exception as e:
        logger.warning(f"Failed to log quality metrics retrieval: {e}")


async def log_health_check(status: str) -> None:
    """Log health check"""
    try:
        logger.info(f"Validation engine health check - Status: {status}")
    except Exception as e:
        logger.warning(f"Failed to log health check: {e}")

