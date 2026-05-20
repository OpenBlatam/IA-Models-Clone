#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Blaze AI System Improvement Script.
Integrates diagnostics, cleanup, and testing for comprehensive system improvement.
"""

import sys
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Any

class SystemImprover:
    """Main system improvement class."""
    
    def __init__(self):
        self.improvements = []
        self.issues_found = []
        self.actions_taken = []
    
    def run_diagnostic(self) -> bool:
        """Run system diagnostic."""
        print("🔍 Running system diagnostic...")
        
        try:
            result = subprocess.run([
                sys.executable, "diagnose_system.py"
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                print("  ✅ Diagnostic completed successfully")
                return True
            else:
                print(f"  ❌ Diagnostic failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"  ❌ Diagnostic error: {e}")
            return False
    
    def run_cleanup(self, aggressive: bool = False) -> bool:
        """Run system cleanup."""
        print("🧹 Running system cleanup...")
        
        try:
            cmd = [sys.executable, "cleanup_system.py"]
            if aggressive:
                cmd.append("--aggressive")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            
            if result.returncode == 0:
                print("  ✅ Cleanup completed successfully")
                return True
            else:
                print(f"  ❌ Cleanup failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"  ❌ Cleanup error: {e}")
            return False
    
    def run_tests(self) -> bool:
        """Run system tests."""
        print("🧪 Running system tests...")
        
        try:
            result = subprocess.run([
                sys.executable, "simple_test.py"
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                print("  ✅ Tests completed successfully")
                return True
            else:
                print(f"  ❌ Tests failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"  ❌ Test error: {e}")
            return False
    
    def check_disk_space(self) -> Dict[str, Any]:
        """Check current disk space."""
        try:
            import shutil
            current_dir = Path.cwd()
            total, used, free = shutil.disk_usage(current_dir)
            
            return {
                "total_gb": total / (1024**3),
                "used_gb": used / (1024**3),
                "free_gb": free / (1024**3),
                "free_percent": (free / total) * 100
            }
        except Exception as e:
            return {"error": str(e)}
    
    def suggest_improvements(self, disk_info: Dict[str, Any]) -> List[str]:
        """Suggest improvements based on current system state."""
        suggestions = []
        
        if "error" not in disk_info:
            free_gb = disk_info["free_gb"]
            free_percent = disk_info["free_percent"]
            
            if free_gb < 1.0:
                suggestions.append("🚨 CRITICAL: Very low disk space. Run aggressive cleanup immediately.")
            elif free_gb < 5.0:
                suggestions.append("⚠️  WARNING: Low disk space. Consider running cleanup.")
            elif free_percent < 20:
                suggestions.append("💡 SUGGESTION: Disk space below 20%. Consider cleanup.")
            
            if free_gb < 10.0:
                suggestions.append("💡 SUGGESTION: Run cleanup to free up more space")
        
        # General suggestions
        suggestions.extend([
            "💡 SUGGESTION: Run tests regularly to catch issues early",
            "💡 SUGGESTION: Keep Python cache clean for better performance",
            "💡 SUGGESTION: Monitor log files and clean old ones periodically"
        ])
        
        return suggestions
    
    def generate_improvement_report(self, disk_before: Dict[str, Any], 
                                  disk_after: Dict[str, Any]) -> None:
        """Generate improvement report."""
        print("\n" + "=" * 70)
        print("📊 IMPROVEMENT REPORT")
        print("=" * 70)
        
        print("💾 Disk Space Analysis:")
        if "error" not in disk_before and "error" not in disk_after:
            freed_gb = disk_before["free_gb"] - disk_after["free_gb"]
            if freed_gb > 0:
                print(f"  ✅ Freed {freed_gb:.2f} GB of disk space")
            else:
                print(f"  ⚠️  No additional space was freed")
            
            print(f"  📊 Before: {disk_before['free_gb']:.2f} GB free")
            print(f"  📊 After: {disk_after['free_gb']:.2f} GB free")
            print(f"  📊 Total: {disk_after['total_gb']:.2f} GB")
        else:
            print("  ❌ Could not analyze disk space changes")
        
        print(f"\n🔧 Actions Taken:")
        for action in self.actions_taken:
            print(f"  - {action}")
        
        if self.issues_found:
            print(f"\n❌ Issues Found:")
            for issue in self.issues_found:
                print(f"  - {issue}")
        
        if self.improvements:
            print(f"\n✅ Improvements Made:")
            for improvement in self.improvements:
                print(f"  - {improvement}")
    
    def run_full_improvement(self, aggressive_cleanup: bool = False) -> bool:
        """Run full system improvement process."""
        print("🚀 Blaze AI System Improvement")
        print("=" * 70)
        
        # Initial disk space check
        print("📊 Initial system state...")
        disk_before = self.check_disk_space()
        if "error" not in disk_before:
            print(f"  💾 Available disk space: {disk_before['free_gb']:.2f} GB")
        else:
            print(f"  ❌ Could not check disk space: {disk_before['error']}")
        
        # Run diagnostic
        print(f"\n🔍 Step 1: System Diagnostic")
        diagnostic_success = self.run_diagnostic()
        if diagnostic_success:
            self.actions_taken.append("System diagnostic completed")
        else:
            self.issues_found.append("System diagnostic failed")
        
        # Run cleanup
        print(f"\n🧹 Step 2: System Cleanup")
        cleanup_success = self.run_cleanup(aggressive=aggressive_cleanup)
        if cleanup_success:
            self.actions_taken.append("System cleanup completed")
            if aggressive_cleanup:
                self.improvements.append("Aggressive cleanup performed")
            else:
                self.improvements.append("Standard cleanup performed")
        else:
            self.issues_found.append("System cleanup failed")
        
        # Wait a moment for cleanup to complete
        time.sleep(2)
        
        # Check disk space after cleanup
        print(f"\n📊 Checking disk space after cleanup...")
        disk_after = self.check_disk_space()
        if "error" not in disk_after:
            print(f"  💾 Available disk space: {disk_after['free_gb']:.2f} GB")
        else:
            print(f"  ❌ Could not check disk space: {disk_after['error']}")
        
        # Run tests
        print(f"\n🧪 Step 3: System Testing")
        test_success = self.run_tests()
        if test_success:
            self.actions_taken.append("System tests completed")
            self.improvements.append("System tests passed")
        else:
            self.issues_found.append("System tests failed")
        
        # Generate suggestions
        print(f"\n💡 Generating improvement suggestions...")
        suggestions = self.suggest_improvements(disk_after)
        for suggestion in suggestions:
            print(f"  {suggestion}")
        
        # Generate final report
        self.generate_improvement_report(disk_before, disk_after)
        
        # Overall success
        overall_success = diagnostic_success and cleanup_success and test_success
        return overall_success

def main():
    """Main improvement function."""
    try:
        import argparse
        
        parser = argparse.ArgumentParser(description="Blaze AI System Improvement")
        parser.add_argument("--aggressive", action="store_true",
                          help="Perform aggressive cleanup")
        parser.add_argument("--diagnostic-only", action="store_true",
                          help="Run only diagnostic")
        parser.add_argument("--cleanup-only", action="store_true",
                          help="Run only cleanup")
        parser.add_argument("--test-only", action="store_true",
                          help="Run only tests")
        
        args = parser.parse_args()
        
        improver = SystemImprover()
        
        if args.diagnostic_only:
            print("🔍 Running diagnostic only...")
            success = improver.run_diagnostic()
        elif args.cleanup_only:
            print("🧹 Running cleanup only...")
            success = improver.run_cleanup(aggressive=args.aggressive)
        elif args.test_only:
            print("🧪 Running tests only...")
            success = improver.run_tests()
        else:
            print("🚀 Running full system improvement...")
            success = improver.run_full_improvement(aggressive=args.aggressive)
        
        if success:
            print("\n🎉 System improvement completed successfully!")
            sys.exit(0)
        else:
            print("\n⚠️  System improvement completed with issues.")
            sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n\n⏹️  System improvement interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error during system improvement: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
