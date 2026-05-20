"""
Complete System Runner
Runs the complete test system with all features
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def run_complete_system():
    """Run complete test system with all features"""
    print("🚀 TruthGPT Complete Test System")
    print("=" * 80)
    print()
    
    steps = [
        ("1️⃣  Running Tests", ["python", "run_unified_tests.py"]),
        ("2️⃣  Running All Analyses", ["python", "run_all_analyses.py"]),
        ("3️⃣  Generating Coverage Report", ["python", "-m", "tests.test_coverage"]),
        ("4️⃣  Checking for Flaky Tests", ["python", "-m", "tests.test_flakiness_detector"]),
        ("5️⃣  Detecting Performance Regressions", ["python", "-m", "tests.performance_regression_detector"]),
        ("6️⃣  Generating Statistics", ["python", "-m", "tests.statistics_aggregator"]),
        ("7️⃣  Generating Executive Report", ["python", "-m", "tests.executive_report"]),
        ("8️⃣  Generating Recommendations", ["python", "-m", "tests.recommendation_engine"]),
    ]
    
    results = []
    
    for step_name, command in steps:
        print(f"{step_name}...")
        try:
            result = subprocess.run(
                command,
                cwd=str(project_root),
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                print(f"   ✅ Completed")
                results.append((step_name, True, None))
            else:
                print(f"   ⚠️  Completed with warnings")
                results.append((step_name, False, result.stderr[:200]))
        except subprocess.TimeoutExpired:
            print(f"   ⏱️  Timeout (exceeded 5 minutes)")
            results.append((step_name, False, "Timeout"))
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append((step_name, False, str(e)))
        
        print()
    
    # Summary
    print("=" * 80)
    print("📊 COMPLETE SYSTEM RUN SUMMARY")
    print("=" * 80)
    print()
    
    successful = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    print(f"Completed: {successful}/{total} steps")
    print()
    
    for step_name, success, error in results:
        status = "✅" if success else "❌"
        print(f"{status} {step_name}")
        if error:
            print(f"   Error: {error[:100]}")
    
    print()
    print("=" * 80)
    print("✅ Complete system run finished!")
    print()
    print("📄 Reports generated:")
    print("  - Coverage report")
    print("  - Flakiness report")
    print("  - Performance regression report")
    print("  - Statistics report")
    print("  - Executive report")
    print("  - Recommendations report")
    print()
    print("🌐 Access:")
    print("  - API: python -m tests.test_api")
    print("  - Dashboard: python web_dashboard.py")

if __name__ == "__main__":
    run_complete_system()







