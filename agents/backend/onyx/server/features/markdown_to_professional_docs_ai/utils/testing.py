"""Testing and quality validation utilities"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """Test case definition"""
    name: str
    input: Dict[str, Any]
    expected_output: Dict[str, Any]
    description: Optional[str] = None


@dataclass
class TestResult:
    """Test result"""
    test_name: str
    passed: bool
    error: Optional[str] = None
    execution_time: float = 0.0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class TestRunner:
    """Test runner for quality validation"""
    
    def __init__(self):
        self.test_cases: List[TestCase] = []
        self.results: List[TestResult] = []
        self._load_default_tests()
    
    def _load_default_tests(self):
        """Load default test cases"""
        # Basic conversion test
        self.test_cases.append(TestCase(
            name="basic_conversion",
            input={
                "markdown_content": "# Test\n\nThis is a test.",
                "output_format": "pdf"
            },
            expected_output={
                "success": True,
                "output_path": None  # Will be checked if exists
            },
            description="Basic Markdown to PDF conversion"
        ))
        
        # Table conversion test
        self.test_cases.append(TestCase(
            name="table_conversion",
            input={
                "markdown_content": "| Col1 | Col2 |\n|------|------|\n| A    | 1    |",
                "output_format": "excel"
            },
            expected_output={
                "success": True,
                "has_tables": True
            },
            description="Table conversion to Excel"
        ))
    
    def add_test_case(self, test_case: TestCase):
        """Add test case"""
        self.test_cases.append(test_case)
    
    async def run_test(self, test_case: TestCase) -> TestResult:
        """
        Run a single test
        
        Args:
            test_case: Test case to run
            
        Returns:
            Test result
        """
        import time
        start_time = time.time()
        
        try:
            from services.converter_service import ConverterService
            from services.markdown_parser import MarkdownParser
            from utils.security import get_security_sanitizer
            
            # Parse markdown
            sanitizer = get_security_sanitizer()
            sanitized = sanitizer.sanitize_markdown(test_case.input["markdown_content"])
            parser = MarkdownParser()
            parsed_content = parser.parse(sanitized)
            
            # Convert
            converter = ConverterService()
            output_path = await converter.convert(
                parsed_content=parsed_content,
                output_format=test_case.input["output_format"]
            )
            
            # Validate result
            from pathlib import Path
            success = Path(output_path).exists() if output_path else False
            
            # Check expected output
            passed = success == test_case.expected_output.get("success", True)
            
            execution_time = time.time() - start_time
            
            result = TestResult(
                test_name=test_case.name,
                passed=passed,
                execution_time=execution_time
            )
            
            if not passed:
                result.error = f"Expected success={test_case.expected_output.get('success')}, got {success}"
            
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name=test_case.name,
                passed=False,
                error=str(e),
                execution_time=execution_time
            )
    
    async def run_all_tests(self) -> List[TestResult]:
        """
        Run all test cases
        
        Returns:
            List of test results
        """
        results = []
        
        for test_case in self.test_cases:
            result = await self.run_test(test_case)
            results.append(result)
            self.results.append(result)
        
        return results
    
    def get_test_summary(self) -> Dict[str, Any]:
        """Get test summary"""
        if not self.results:
            return {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "pass_rate": 0.0
            }
        
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        pass_rate = (passed / total) * 100 if total > 0 else 0.0
        
        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": pass_rate,
            "average_execution_time": sum(r.execution_time for r in self.results) / total if total > 0 else 0.0
        }


# Global test runner
_test_runner: Optional[TestRunner] = None


def get_test_runner() -> TestRunner:
    """Get global test runner"""
    global _test_runner
    if _test_runner is None:
        _test_runner = TestRunner()
    return _test_runner

