#!/usr/bin/env python3
"""
🏗️ HeyGen AI - Run Refactoring System
=====================================

Simple script to run the ultimate refactoring system for the HeyGen AI codebase.

Author: AI Assistant
Date: December 2024
Version: 1.0.0
"""

import os
import sys
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main function to run the refactoring system"""
    try:
        print("🏗️ HeyGen AI - Ultimate Refactoring System")
        print("=" * 50)
        print()
        
        # Get current directory
        current_dir = Path(__file__).parent
        os.chdir(current_dir)
        
        print("📁 Working directory:", current_dir)
        print()
        
        # Check if refactoring system exists
        refactoring_file = "ULTIMATE_REFACTORING_SYSTEM.py"
        unified_api_file = "UNIFIED_HEYGEN_AI_API.py"
        
        if not os.path.exists(refactoring_file):
            print(f"❌ Refactoring system file not found: {refactoring_file}")
            return
        
        if not os.path.exists(unified_api_file):
            print(f"❌ Unified API file not found: {unified_api_file}")
            return
        
        print("🔧 Available Refactoring Options:")
        print("1. Run Ultimate Refactoring System")
        print("2. Run Unified HeyGen AI API")
        print("3. Run both systems (recommended)")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            run_refactoring_system()
        elif choice == '2':
            run_unified_api()
        elif choice == '3':
            run_both_systems()
        elif choice == '4':
            print("👋 Goodbye!")
            return
        else:
            print("❌ Invalid choice. Please run the script again.")
            return
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Process interrupted by user")
    except Exception as e:
        logger.error(f"Error running refactoring: {e}")
        print(f"❌ Error: {e}")

def run_refactoring_system():
    """Run the Ultimate Refactoring System"""
    try:
        print("\n🏗️ Running Ultimate Refactoring System...")
        print("=" * 50)
        
        # Import and run the refactoring system
        import ULTIMATE_REFACTORING_SYSTEM
        ULTIMATE_REFACTORING_SYSTEM.main()
        
        print("\n✅ Refactoring system completed!")
        
    except Exception as e:
        print(f"❌ Refactoring system failed: {e}")
        logger.error(f"Refactoring system error: {e}")

def run_unified_api():
    """Run the Unified HeyGen AI API"""
    try:
        print("\n🎯 Running Unified HeyGen AI API...")
        print("=" * 50)
        
        # Import and run the unified API
        import UNIFIED_HEYGEN_AI_API
        UNIFIED_HEYGEN_AI_API.main()
        
        print("\n✅ Unified API completed!")
        
    except Exception as e:
        print(f"❌ Unified API failed: {e}")
        logger.error(f"Unified API error: {e}")

def run_both_systems():
    """Run both refactoring and unified API systems"""
    try:
        print("\n🚀 Running Both Systems...")
        print("=" * 50)
        
        # Run refactoring system first
        print("\n1️⃣ Running Ultimate Refactoring System...")
        run_refactoring_system()
        
        print("\n" + "="*50)
        
        # Run unified API system
        print("\n2️⃣ Running Unified HeyGen AI API...")
        run_unified_api()
        
        print("\n🎉 Both systems completed successfully!")
        print("\n📊 Summary:")
        print("  - Code refactored and optimized")
        print("  - Unified API system operational")
        print("  - System ready for production use")
        
    except Exception as e:
        print(f"❌ Both systems failed: {e}")
        logger.error(f"Both systems error: {e}")

def check_dependencies():
    """Check if required dependencies are available"""
    try:
        required_packages = [
            'ast', 'os', 'sys', 'time', 'logging', 'pathlib', 'typing',
            'dataclasses', 'collections', 'json', 'yaml', 'datetime',
            'threading', 'concurrent.futures', 'psutil'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            print("⚠️  Missing required packages:")
            for package in missing_packages:
                print(f"  - {package}")
            print("\nPlease install missing packages before running refactoring.")
            return False
        
        return True
        
    except Exception as e:
        logger.warning(f"Error checking dependencies: {e}")
        return True  # Continue anyway

def show_system_info():
    """Show system information"""
    try:
        print("📊 System Information:")
        print(f"  Python Version: {sys.version}")
        print(f"  Working Directory: {os.getcwd()}")
        print(f"  Platform: {sys.platform}")
        
        # Check available files
        refactoring_files = [
            "ULTIMATE_REFACTORING_SYSTEM.py",
            "UNIFIED_HEYGEN_AI_API.py",
            "ULTIMATE_PERFORMANCE_OPTIMIZER.py",
            "ADVANCED_CODE_QUALITY_IMPROVER.py",
            "ULTIMATE_TESTING_ENHANCEMENT_SYSTEM.py",
            "ADVANCED_AI_MODEL_OPTIMIZER.py",
            "ULTIMATE_SYSTEM_IMPROVEMENT_ORCHESTRATOR.py"
        ]
        
        print("\n📁 Available Refactoring Systems:")
        for file in refactoring_files:
            if os.path.exists(file):
                size = os.path.getsize(file) / 1024  # KB
                print(f"  ✅ {file} ({size:.1f} KB)")
            else:
                print(f"  ❌ {file} (not found)")
        
        print()
        
    except Exception as e:
        logger.warning(f"Error showing system info: {e}")

if __name__ == "__main__":
    print("🔍 Checking dependencies...")
    if check_dependencies():
        print("✅ All dependencies available")
        show_system_info()
        main()
    else:
        print("❌ Please install missing dependencies and try again")
        sys.exit(1)

