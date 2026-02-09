"""
Test Runner
===========

Advanced test runner with reporting and coverage.
"""

import sys
import subprocess
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class TestRunner:
    """Advanced test runner."""
    
    def __init__(self, test_dir: Optional[Path] = None):
        """
        Initialize test runner.
        
        Args:
            test_dir: Test directory path
        """
        self.test_dir = test_dir or Path(__file__).parent
        self.results: List[Dict[str, Any]] = []
    
    def run_tests(
        self,
        pattern: Optional[str] = None,
        verbose: bool = True,
        coverage: bool = False,
        parallel: bool = False
    ) -> Dict[str, Any]:
        """
        Run tests.
        
        Args:
            pattern: Optional test pattern
            verbose: Verbose output
            coverage: Enable coverage reporting
            parallel: Run tests in parallel
            
        Returns:
            Test results dictionary
        """
        cmd = ["pytest"]
        
        if pattern:
            cmd.append(pattern)
        else:
            cmd.append(str(self.test_dir))
        
        if verbose:
            cmd.append("-v")
        
        if coverage:
            cmd.extend(["--cov", ".", "--cov-report", "html", "--cov-report", "term"])
        
        if parallel:
            cmd.extend(["-n", "auto"])
        
        logger.info(f"Running tests: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.test_dir.parent
            )
            
            return {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error running tests: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def run_specific_test(self, test_file: str, test_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Run specific test.
        
        Args:
            test_file: Test file path
            test_name: Optional specific test name
            
        Returns:
            Test results dictionary
        """
        test_path = self.test_dir / test_file
        
        if not test_path.exists():
            return {
                "success": False,
                "error": f"Test file not found: {test_path}",
                "timestamp": datetime.now().isoformat()
            }
        
        cmd = ["pytest", str(test_path), "-v"]
        
        if test_name:
            cmd.append(f"::{test_name}")
        
        logger.info(f"Running specific test: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.test_dir.parent
            )
            
            return {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error running test: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_test_summary(self) -> Dict[str, Any]:
        """Get test summary."""
        return {
            "total_runs": len(self.results),
            "successful_runs": sum(1 for r in self.results if r.get("success")),
            "failed_runs": sum(1 for r in self.results if not r.get("success")),
            "last_run": self.results[-1] if self.results else None
        }




