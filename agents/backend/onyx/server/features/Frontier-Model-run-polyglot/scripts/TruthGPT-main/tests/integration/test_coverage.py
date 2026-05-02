"""
Test Coverage Analyzer
Analyzes test coverage and generates coverage reports
"""

import sys
import subprocess
from pathlib import Path
from typing import Dict, Any, List

def check_coverage_installed() -> bool:
    """Check if coverage.py is installed"""
    try:
        import coverage
        return True
    except ImportError:
        return False

def install_coverage():
    """Install coverage.py"""
    print("📦 Installing coverage.py...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "coverage"])
        print("✅ coverage.py installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install coverage.py")
        return False

def run_coverage():
    """Run tests with coverage"""
    if not check_coverage_installed():
        print("coverage.py not found. Installing...")
        if not install_coverage():
            return False
    
    print("🔍 Running tests with coverage...")
    
    # Run coverage
    try:
        subprocess.check_call([
            sys.executable, "-m", "coverage", "run",
            "--source=core",
            "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"
        ])
        
        # Generate report
        print("\n📊 Generating coverage report...")
        subprocess.check_call([sys.executable, "-m", "coverage", "report"])
        
        # Generate HTML report
        print("\n📄 Generating HTML coverage report...")
        subprocess.check_call([sys.executable, "-m", "coverage", "html"])
        print("✅ HTML report generated in htmlcov/index.html")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running coverage: {e}")
        return False

def analyze_coverage():
    """Analyze coverage and provide recommendations"""
    if not Path("htmlcov").exists():
        print("❌ No coverage data found. Run 'python test_coverage.py run' first")
        return
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "coverage", "report"],
            capture_output=True,
            text=True
        )
        
        print("📊 Coverage Analysis")
        print("=" * 60)
        print(result.stdout)
        
        # Parse coverage percentage
        lines = result.stdout.split('\n')
        for line in lines:
            if 'TOTAL' in line:
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        coverage_pct = float(parts[-1].rstrip('%'))
                        print(f"\n🎯 Overall Coverage: {coverage_pct:.1f}%")
                        
                        if coverage_pct < 50:
                            print("⚠️  Coverage is low. Consider adding more tests.")
                        elif coverage_pct < 80:
                            print("✅ Coverage is good, but could be improved.")
                        else:
                            print("🎉 Excellent coverage!")
                    except ValueError:
                        pass
                break
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error analyzing coverage: {e}")

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python test_coverage.py run      # Run tests with coverage")
        print("  python test_coverage.py analyze  # Analyze coverage results")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == 'run':
        success = run_coverage()
        sys.exit(0 if success else 1)
    
    elif command == 'analyze':
        analyze_coverage()
    
    else:
        print(f"❌ Unknown command: {command}")
        sys.exit(1)

if __name__ == "__main__":
    main()
