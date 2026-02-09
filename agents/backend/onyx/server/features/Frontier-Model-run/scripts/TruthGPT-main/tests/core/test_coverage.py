"""
Test Coverage Utilities
Tools for measuring and reporting test coverage
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    import coverage
    COVERAGE_AVAILABLE = True
except ImportError:
    COVERAGE_AVAILABLE = False

def get_coverage_report():
    """Get current test coverage report"""
    if not COVERAGE_AVAILABLE:
        return "Coverage.py not installed. Install with: pip install coverage"
    
    try:
        cov = coverage.Coverage()
        cov.load()
        return cov.report()
    except Exception as e:
        return f"Error generating coverage report: {e}"

def generate_html_coverage():
    """Generate HTML coverage report"""
    if not COVERAGE_AVAILABLE:
        return False, "Coverage.py not installed"
    
    try:
        cov = coverage.Coverage()
        cov.load()
        cov.html_report()
        return True, "HTML coverage report generated in htmlcov/"
    except Exception as e:
        return False, f"Error: {e}"
