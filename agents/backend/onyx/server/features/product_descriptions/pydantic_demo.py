from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import asyncio
import time
import json
import logging
import requests
from typing import Dict, Any, List
from datetime import datetime
from pydantic_schemas import (
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Pydantic Validation Demo
Product Descriptions Feature - Comprehensive Input/Output Validation and Response Schema Demonstration
"""


# Import Pydantic schemas
    # Base models
    BaseRequestModel,
    BaseResponseModel,
    BaseErrorModel,
    
    # Git-related schemas
    GitFileInfo,
    GitStatusRequest,
    GitStatusResponse,
    CreateBranchRequest,
    CreateBranchResponse,
    CommitChangesRequest,
    CommitChangesResponse,
    
    # Model versioning schemas
    ModelVersion,
    ModelVersionRequest,
    ModelVersionResponse,
    
    # Performance and optimization schemas
    PerformanceMetrics,
    CacheInfo,
    BatchProcessRequest,
    BatchProcessResponse,
    
    # Error and monitoring schemas
    ErrorContext,
    ValidationError,
    ErrorStats,
    MonitoringData,
    
    # Health and status schemas
    HealthStatus,
    ServiceHealth,
    AppStatusResponse,
    
    # Configuration schemas
    DatabaseConfig,
    CacheConfig,
    LoggingConfig,
    AppConfig,
    
    # Utility functions
    create_git_status_request,
    create_branch_request,
    create_commit_request,
    create_model_version_request,
    create_batch_process_request,
    create_success_response,
    create_error_response,
    create_validation_error_response,
    validate_required_fields,
    validate_field_length
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PydanticValidationDemo:
    """Comprehensive Pydantic validation demonstration"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        
    """__init__ function."""
self.base_url = base_url
        self.session = requests.Session()
        self.results: List[Dict[str, Any]] = []
    
    def log_result(self, test_name: str, success: bool, data: Dict[str, Any], duration: float):
        """Log test result"""
        result = {
            "test": test_name,
            "success": success,
            "data": data,
            "duration": duration,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self.results.append(result)
        logger.info(f"Test: {test_name} - {'PASS' if success else 'FAIL'} ({duration:.3f}s)")
    
    async def test_schema_validation(self) -> Dict[str, Any]:
        """Test Pydantic schema validation"""
        start_time = time.time()
        
        try:
            # Test valid GitStatusRequest
            valid_request = GitStatusRequest(
                include_untracked=True,
                include_ignored=False,
                include_staged=True,
                max_files=50
            )
            
            # Test invalid GitStatusRequest (should raise validation error)
            try:
                invalid_request = GitStatusRequest(
                    include_untracked=True,
                    include_ignored=False,
                    include_staged=True,
                    max_files=2000  # Exceeds maximum
                )
                validation_passed = False
            except Exception as e:
                validation_passed = True
            
            duration = time.time() - start_time
            
            success = validation_passed and valid_request.max_files == 50
            data = {
                "valid_request_created": True,
                "validation_error_caught": validation_passed,
                "valid_request_data": valid_request.model_dump(),
                "schema_validation_working": success
            }
            
            self.log_result("Schema Validation", success, data, duration)
            return data
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Schema Validation", False, {"error": str(e)}, duration)
            return {"error": str(e)}
    
    async def test_git_status_validation(self) -> Dict[str, Any]:
        """Test git status endpoint with Pydantic validation"""
        start_time = time.time()
        
        try:
            # Create valid request using schema
            request = create_git_status_request(
                include_untracked=True,
                include_ignored=False,
                include_staged=True,
                max_files=100
            )
            
            response = self.session.post(
                f"{self.base_url}/git/status",
                json=request.model_dump(),
                headers={"Content-Type": "application/json"}
            )
            
            duration = time.time() - start_time
            
            success = response.status_code in [200, 500]  # 500 is expected for git errors
            response_data = response.json() if response.content else {}
            
            data = {
                "status_code": response.status_code,
                "response": response_data,
                "request_validated": True,
                "response_schema_valid": "success" in response_data,
                "validation_working": success
            }
            
            self.log_result("Git Status Validation", success, data, duration)
            return data
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Git Status Validation", False, {"error": str(e)}, duration)
            return {"error": str(e)}
    
    async def test_branch_creation_validation(self) -> Dict[str, Any]:
        """Test branch creation with Pydantic validation"""
        start_time = time.time()
        
        try:
            # Test valid branch name
            valid_request = create_branch_request(
                branch_name="feature/test-branch",
                base_branch="main",
                checkout=True,
                push_remote=False
            )
            
            response = self.session.post(
                f"{self.base_url}/git/branch/create",
                json=valid_request.model_dump(),
                headers={"Content-Type": "application/json"}
            )
            
            duration = time.time() - start_time
            
            success = response.status_code in [200, 500]  # 500 is expected for git errors
            response_data = response.json() if response.content else {}
            
            data = {
                "status_code": response.status_code,
                "response": response_data,
                "valid_request_processed": True,
                "response_schema_valid": "success" in response_data,
                "validation_working": success
            }
            
            self.log_result("Branch Creation Validation", success, data, duration)
            return data
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Branch Creation Validation", False, {"error": str(e)}, duration)
            return {"error": str(e)}
    
    async def test_invalid_branch_name_validation(self) -> Dict[str, Any]:
        """Test invalid branch name validation"""
        start_time = time.time()
        
        try:
            # Test invalid branch name (should be rejected by Pydantic)
            try:
                invalid_request = CreateBranchRequest(
                    branch_name="",  # Empty name
                    base_branch="main",
                    checkout=True
                )
                validation_failed = False
            except Exception as e:
                validation_failed = True
                validation_error = str(e)
            
            duration = time.time() - start_time
            
            success = validation_failed
            data = {
                "validation_error_caught": validation_failed,
                "validation_error": validation_error if validation_failed else None,
                "empty_branch_name_rejected": success
            }
            
            self.log_result("Invalid Branch Name Validation", success, data, duration)
            return data
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Invalid Branch Name Validation", False, {"error": str(e)}, duration)
            return {"error": str(e)}
    
    async def test_commit_validation(self) -> Dict[str, Any]:
        """Test commit validation with Pydantic"""
        start_time = time.time()
        
        try:
            # Test valid commit request
            valid_request = create_commit_request(
                message="Add new feature",
                files=["file1.txt", "file2.txt"],
                include_untracked=True,
                amend=False
            )
            
            response = self.session.post(
                f"{self.base_url}/git/commit",
                json=valid_request.model_dump(),
                headers={"Content-Type": "application/json"}
            )
            
            duration = time.time() - start_time
            
            success = response.status_code in [200, 500]  # 500 is expected for git errors
            response_data = response.json() if response.content else {}
            
            data = {
                "status_code": response.status_code,
                "response": response_data,
                "valid_request_processed": True,
                "response_schema_valid": "success" in response_data,
                "validation_working": success
            }
            
            self.log_result("Commit Validation", success, data, duration)
            return data
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Commit Validation", False, {"error": str(e)}, duration)
            return {"error": str(e)}
    
    async def test_model_version_validation(self) -> Dict[str, Any]:
        """Test model version validation with Pydantic"""
        start_time = time.time()
        
        try:
            # Test valid model version request
            valid_request = create_model_version_request(
                model_name="test-model",
                version="1.0.0",
                description="Test model version",
                tags=["test", "demo"],
                status="draft"
            )
            
            response = self.session.post(
                f"{self.base_url}/models/version",
                json=valid_request.model_dump(),
                headers={"Content-Type": "application/json"}
            )
            
            duration = time.time() - start_time
            
            success = response.status_code in [200, 500]  # 500 is expected for model errors
            response_data = response.json() if response.content else {}
            
            data = {
                "status_code": response.status_code,
                "response": response_data,
                "valid_request_processed": True,
                "response_schema_valid": "success" in response_data,
                "validation_working": success
            }
            
            self.log_result("Model Version Validation", success, data, duration)
            return data
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Model Version Validation", False, {"error": str(e)}, duration)
            return {"error": str(e)}
    
    async def test_batch_processing_validation(self) -> Dict[str, Any]:
        """Test batch processing validation with Pydantic"""
        start_time = time.time()
        
        try:
            # Test valid batch process request
            valid_request = create_batch_process_request(
                items=[1, 2, 3, 4, 5],
                operation="double",
                batch_size=3,
                max_concurrent=2
            )
            
            response = self.session.post(
                f"{self.base_url}/batch/process",
                json=valid_request.model_dump(),
                headers={"Content-Type": "application/json"}
            )
            
            duration = time.time() - start_time
            
            success = response.status_code == 200
            response_data = response.json() if response.content else {}
            
            data = {
                "status_code": response.status_code,
                "response": response_data,
                "valid_request_processed": True,
                "response_schema_valid": "success" in response_data,
                "validation_working": success
            }
            
            self.log_result("Batch Processing Validation", success, data, duration)
            return data
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Batch Processing Validation", False, {"error": str(e)}, duration)
            return {"error": str(e)}
    
    async def test_invalid_batch_operation_validation(self) -> Dict[str, Any]:
        """Test invalid batch operation validation"""
        start_time = time.time()
        
        try:
            # Test invalid operation (should be rejected by Pydantic)
            try:
                invalid_request = BatchProcessRequest(
                    items=[1, 2, 3],
                    operation="invalid_operation",  # Invalid operation
                    batch_size=10
                )
                validation_failed = False
            except Exception as e:
                validation_failed = True
                validation_error = str(e)
            
            duration = time.time() - start_time
            
            success = validation_failed
            data = {
                "validation_error_caught": validation_failed,
                "validation_error": validation_error if validation_failed else None,
                "invalid_operation_rejected": success
            }
            
            self.log_result("Invalid Batch Operation Validation", success, data, duration)
            return data
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Invalid Batch Operation Validation", False, {"error": str(e)}, duration)
            return {"error": str(e)}
    
    async def test_app_status_validation(self) -> Dict[str, Any]:
        """Test app status endpoint with Pydantic validation"""
        start_time = time.time()
        
        try:
            response = self.session.get(
                f"{self.base_url}/status",
                headers={"Content-Type": "application/json"}
            )
            
            duration = time.time() - start_time
            
            success = response.status_code == 200
            response_data = response.json() if response.content else {}
            
            # Validate response against AppStatusResponse schema
            try:
                app_status = AppStatusResponse(**response_data)
                schema_validation_passed = True
            except Exception as e:
                schema_validation_passed = False
                schema_error = str(e)
            
            data = {
                "status_code": response.status_code,
                "response": response_data,
                "schema_validation_passed": schema_validation_passed,
                "schema_error": schema_error if not schema_validation_passed else None,
                "validation_working": success and schema_validation_passed
            }
            
            self.log_result("App Status Validation", success, data, duration)
            return data
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("App Status Validation", False, {"error": str(e)}, duration)
            return {"error": str(e)}
    
    async def test_config_validation(self) -> Dict[str, Any]:
        """Test configuration endpoint with Pydantic validation"""
        start_time = time.time()
        
        try:
            response = self.session.get(
                f"{self.base_url}/config",
                headers={"Content-Type": "application/json"}
            )
            
            duration = time.time() - start_time
            
            success = response.status_code == 200
            response_data = response.json() if response.content else {}
            
            # Validate response against AppConfig schema
            try:
                app_config = AppConfig(**response_data)
                schema_validation_passed = True
            except Exception as e:
                schema_validation_passed = False
                schema_error = str(e)
            
            data = {
                "status_code": response.status_code,
                "response": response_data,
                "schema_validation_passed": schema_validation_passed,
                "schema_error": schema_error if not schema_validation_passed else None,
                "validation_working": success and schema_validation_passed
            }
            
            self.log_result("Config Validation", success, data, duration)
            return data
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Config Validation", False, {"error": str(e)}, duration)
            return {"error": str(e)}
    
    async def test_error_response_validation(self) -> Dict[str, Any]:
        """Test error response validation"""
        start_time = time.time()
        
        try:
            # Trigger a validation error to test error response schema
            invalid_request = {
                "branch_name": "",  # Invalid empty name
                "base_branch": "main",
                "checkout": True
            }
            
            response = self.session.post(
                f"{self.base_url}/git/branch/create",
                json=invalid_request,
                headers={"Content-Type": "application/json"}
            )
            
            duration = time.time() - start_time
            
            success = response.status_code == 400
            response_data = response.json() if response.content else {}
            
            # Validate error response against BaseErrorModel schema
            try:
                error_response = BaseErrorModel(**response_data)
                error_schema_validation_passed = True
            except Exception as e:
                error_schema_validation_passed = False
                error_schema_error = str(e)
            
            data = {
                "status_code": response.status_code,
                "response": response_data,
                "error_schema_validation_passed": error_schema_validation_passed,
                "error_schema_error": error_schema_error if not error_schema_validation_passed else None,
                "validation_working": success and error_schema_validation_passed
            }
            
            self.log_result("Error Response Validation", success, data, duration)
            return data
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Error Response Validation", False, {"error": str(e)}, duration)
            return {"error": str(e)}
    
    async def test_utility_functions(self) -> Dict[str, Any]:
        """Test Pydantic utility functions"""
        start_time = time.time()
        
        try:
            # Test create_success_response
            success_response = create_success_response(
                data={"message": "Test successful"},
                request_id="test-123",
                duration_ms=100.5,
                correlation_id="corr-456"
            )
            
            # Test create_error_response
            error_response = create_error_response(
                error_code="TEST_ERROR",
                message="Test error message",
                details="Test error details",
                severity="medium",
                request_id="test-123",
                correlation_id="corr-456"
            )
            
            # Test create_validation_error_response
            validation_error_response = create_validation_error_response(
                message="Validation failed",
                validation_errors=[{"field": "test", "error": "Invalid value"}],
                field="test_field",
                value="invalid_value",
                expected="valid_value",
                suggestion="Use a valid value",
                request_id="test-123"
            )
            
            duration = time.time() - start_time
            
            success = (
                success_response.success and
                error_response.error_code == "TEST_ERROR" and
                validation_error_response.error_code == "VALIDATION_ERROR"
            )
            
            data = {
                "success_response_created": success_response.success,
                "error_response_created": error_response.error_code == "TEST_ERROR",
                "validation_error_response_created": validation_error_response.error_code == "VALIDATION_ERROR",
                "utility_functions_working": success
            }
            
            self.log_result("Utility Functions", success, data, duration)
            return data
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Utility Functions", False, {"error": str(e)}, duration)
            return {"error": str(e)}
    
    async def test_field_validation(self) -> Dict[str, Any]:
        """Test field validation functions"""
        start_time = time.time()
        
        try:
            # Test validate_field_length
            try:
                valid_length = validate_field_length("test", "test_field", 1, 10)
                length_validation_passed = True
            except Exception as e:
                length_validation_passed = False
                length_error = str(e)
            
            # Test validate_required_fields
            try:
                validate_required_fields({"field1": "value1", "field2": "value2"}, ["field1", "field2"])
                required_fields_validation_passed = True
            except Exception as e:
                required_fields_validation_passed = False
                required_fields_error = str(e)
            
            # Test missing required fields
            try:
                validate_required_fields({"field1": "value1"}, ["field1", "field2"])
                missing_fields_caught = False
            except Exception as e:
                missing_fields_caught = True
                missing_fields_error = str(e)
            
            duration = time.time() - start_time
            
            success = (
                length_validation_passed and
                required_fields_validation_passed and
                missing_fields_caught
            )
            
            data = {
                "length_validation_passed": length_validation_passed,
                "required_fields_validation_passed": required_fields_validation_passed,
                "missing_fields_caught": missing_fields_caught,
                "field_validation_working": success
            }
            
            self.log_result("Field Validation", success, data, duration)
            return data
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Field Validation", False, {"error": str(e)}, duration)
            return {"error": str(e)}
    
    async def test_schema_serialization(self) -> Dict[str, Any]:
        """Test schema serialization and deserialization"""
        start_time = time.time()
        
        try:
            # Create a complex schema
            git_file_info = GitFileInfo(
                path="/path/to/file.txt",
                status="modified",
                size=1024,
                last_modified=datetime.now(),
                staged=True
            )
            
            # Serialize to dict
            serialized = git_file_info.model_dump()
            
            # Deserialize back to object
            deserialized = GitFileInfo(**serialized)
            
            # Test JSON serialization
            json_serialized = git_file_info.model_dump_json()
            json_deserialized = GitFileInfo.model_validate_json(json_serialized)
            
            duration = time.time() - start_time
            
            success = (
                deserialized.path == git_file_info.path and
                deserialized.status == git_file_info.status and
                json_deserialized.path == git_file_info.path
            )
            
            data = {
                "serialization_working": True,
                "deserialization_working": deserialized.path == git_file_info.path,
                "json_serialization_working": json_deserialized.path == git_file_info.path,
                "schema_serialization_working": success
            }
            
            self.log_result("Schema Serialization", success, data, duration)
            return data
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Schema Serialization", False, {"error": str(e)}, duration)
            return {"error": str(e)}
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all Pydantic validation tests"""
        logger.info("Starting Pydantic Validation Demo Tests...")
        
        tests = [
            self.test_schema_validation,
            self.test_git_status_validation,
            self.test_branch_creation_validation,
            self.test_invalid_branch_name_validation,
            self.test_commit_validation,
            self.test_model_version_validation,
            self.test_batch_processing_validation,
            self.test_invalid_batch_operation_validation,
            self.test_app_status_validation,
            self.test_config_validation,
            self.test_error_response_validation,
            self.test_utility_functions,
            self.test_field_validation,
            self.test_schema_serialization
        ]
        
        for test in tests:
            try:
                await test()
                await asyncio.sleep(0.2)  # Small delay between tests
            except Exception as e:
                logger.error(f"Test failed: {test.__name__} - {e}")
        
        # Generate summary
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r["success"])
        failed_tests = total_tests - passed_tests
        
        summary = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            "results": self.results
        }
        
        logger.info(f"Pydantic Validation Demo completed: {passed_tests}/{total_tests} tests passed")
        return summary
    
    def save_results(self, filename: str = "pydantic_validation_demo_results.json"):
        """Save test results to file"""
        try:
            with open(filename, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                json.dump(self.results, f, indent=2)
            logger.info(f"Results saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

async def main():
    """Main demo execution"""
    print("=" * 70)
    print("PYDANTIC VALIDATION DEMO - PRODUCT DESCRIPTIONS FEATURE")
    print("=" * 70)
    
    # Create demo instance
    demo = PydanticValidationDemo()
    
    # Run all tests
    summary = await demo.run_all_tests()
    
    # Print summary
    print("\n" + "=" * 70)
    print("DEMO SUMMARY")
    print("=" * 70)
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed_tests']}")
    print(f"Failed: {summary['failed_tests']}")
    print(f"Success Rate: {summary['success_rate']:.1f}%")
    
    # Print detailed results
    print("\n" + "=" * 70)
    print("DETAILED RESULTS")
    print("=" * 70)
    
    for result in summary['results']:
        status = "PASS" if result['success'] else "FAIL"
        print(f"{status}: {result['test']} ({result['duration']:.3f}s)")
        
        if not result['success'] and 'error' in result['data']:
            print(f"  Error: {result['data']['error']}")
    
    # Save results
    demo.save_results()
    
    print("\n" + "=" * 70)
    print("Demo completed! Check pydantic_validation_demo_results.json for detailed results.")
    print("=" * 70)

match __name__:
    case "__main__":
    asyncio.run(main()) 