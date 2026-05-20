import os
import sys
import pytest
import time
from datetime import datetime

def run_tests():
    """Run all tests and display results."""
    print(f"\n=== Running API Tests at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    
    # Change to the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(project_root)
    
    # Add the project root to Python path
    sys.path.insert(0, project_root)
    
    start_time = time.time()
    
    # Run tests with detailed output
    result = pytest.main([
        "-v",  # verbose output
        "-s",  # show print statements
        "--tb=short",  # shorter traceback
        os.path.join(os.path.dirname(__file__), "test_api.py")
    ])
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n=== Test Results ===")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Exit Code: {result}")
    
    if result == 0:
        print("\n✅ All tests passed successfully!")
    else:
        print("\n❌ Some tests failed. Please check the output above for details.")
    
    return result

if __name__ == "__main__":
    sys.exit(run_tests()) 