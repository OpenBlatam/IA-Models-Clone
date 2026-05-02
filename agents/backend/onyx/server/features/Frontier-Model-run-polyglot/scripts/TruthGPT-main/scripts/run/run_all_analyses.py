"""
Run All Test Analyses
Runs all analysis tools and generates comprehensive report
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def run_all_analyses():
    """Run all test analysis tools"""
    print("🔍 Running All Test Analyses")
    print("=" * 80)
    print()
    
    analyses = []
    
    # 1. Coverage Analysis
    print("1️⃣  Running Coverage Analysis...")
    try:
        from tests.test_coverage import CoverageAnalyzer
        analyzer = CoverageAnalyzer(project_root)
        coverage = analyzer.analyze_test_coverage()
        analyses.append(('Coverage', coverage))
        print(f"   ✅ Coverage: {coverage['coverage_percentage']:.1f}%")
    except Exception as e:
        print(f"   ⚠️  Error: {e}")
    
    print()
    
    # 2. Flakiness Detection
    print("2️⃣  Running Flakiness Detection...")
    try:
        from tests.test_flakiness_detector import FlakinessDetector
        detector = FlakinessDetector(project_root)
        flakiness = detector.analyze_flakiness(min_runs=5)
        if 'error' not in flakiness:
            analyses.append(('Flakiness', flakiness))
            print(f"   ✅ Found {len(flakiness.get('flaky_tests', []))} flaky tests")
        else:
            print(f"   ⚠️  {flakiness['error']}")
    except Exception as e:
        print(f"   ⚠️  Error: {e}")
    
    print()
    
    # 3. Dependency Analysis
    print("3️⃣  Running Dependency Analysis...")
    try:
        from tests.test_dependency_analyzer import TestDependencyAnalyzer
        analyzer = TestDependencyAnalyzer(project_root)
        dependencies = analyzer.analyze_test_dependencies()
        analyses.append(('Dependencies', dependencies))
        print(f"   ✅ Analyzed {len(dependencies.get('test_methods', {}))} test methods")
    except Exception as e:
        print(f"   ⚠️  Error: {e}")
    
    print()
    
    # 4. Performance Regression Detection
    print("4️⃣  Running Performance Regression Detection...")
    try:
        from tests.performance_regression_detector import PerformanceRegressionDetector
        detector = PerformanceRegressionDetector(project_root)
        regressions = detector.detect_regressions(lookback=10)
        if 'error' not in regressions:
            analyses.append(('Performance', regressions))
            if regressions.get('overall_regression'):
                print(f"   ⚠️  Regression detected: {regressions['overall_regression']['regression_percent']:.1f}%")
            else:
                print("   ✅ No performance regression detected")
        else:
            print(f"   ⚠️  {regressions['error']}")
    except Exception as e:
        print(f"   ⚠️  Error: {e}")
    
    print()
    
    # 5. Test History
    print("5️⃣  Loading Test History...")
    try:
        from tests.test_history import TestHistory
        history = TestHistory()
        stats = history.get_statistics()
        analyses.append(('History', stats))
        print(f"   ✅ {stats.get('total_runs', 0)} test runs recorded")
    except Exception as e:
        print(f"   ⚠️  Error: {e}")
    
    print()
    print("=" * 80)
    print("📊 ANALYSIS SUMMARY")
    print("=" * 80)
    print()
    
    # Generate summary
    for name, data in analyses:
        print(f"{name}:")
        if isinstance(data, dict):
            if 'coverage_percentage' in data:
                print(f"  Coverage: {data['coverage_percentage']:.1f}%")
            elif 'total_runs' in data:
                print(f"  Total Runs: {data['total_runs']}")
                print(f"  Avg Success Rate: {data.get('average_success_rate', 0):.1f}%")
            elif 'flaky_tests' in data:
                print(f"  Flaky Tests: {len(data['flaky_tests'])}")
            elif 'test_methods' in data:
                print(f"  Test Methods: {len(data['test_methods'])}")
            elif 'overall_regression' in data:
                if data['overall_regression']:
                    print(f"  Regression: {data['overall_regression']['regression_percent']:.1f}%")
                else:
                    print("  No regression detected")
        print()
    
    print("=" * 80)
    print("✅ All analyses complete!")
    print()
    print("📄 Individual reports saved:")
    print("  - coverage_report.txt")
    print("  - flakiness_report.txt")
    print("  - dependency_analysis_report.txt")
    print("  - performance_regression_report.txt")
    print("  - test_history_report.txt")

if __name__ == "__main__":
    run_all_analyses()







