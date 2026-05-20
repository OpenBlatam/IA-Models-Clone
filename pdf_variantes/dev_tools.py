#!/usr/bin/env python3
"""
Development Tools
================
Collection of development and debugging tools.
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_api_debug():
    """Run API with debugging."""
    print("🚀 Starting API with debugging...")
    subprocess.run([sys.executable, "run_api_debug.py"])


def run_debug_tool():
    """Run interactive debug tool."""
    print("🐛 Starting debug tool...")
    subprocess.run([sys.executable, "debug_api.py"])


def run_monitor():
    """Run API monitor."""
    print("📊 Starting API monitor...")
    subprocess.run([sys.executable, "api_monitor.py"])


def run_profiler():
    """Run API profiler."""
    print("📈 Starting API profiler...")
    subprocess.run([sys.executable, "api_profiler.py"])


def run_tests():
    """Run tests."""
    print("🧪 Running tests...")
    subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"])


def run_tests_debug():
    """Run tests with debugging."""
    print("🧪 Running tests with debugging...")
    subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v", "-s", "--pdb"])


def run_linter():
    """Run linter."""
    print("🔍 Running linter...")
    try:
        subprocess.run([sys.executable, "-m", "flake8", "."])
    except:
        try:
            subprocess.run([sys.executable, "-m", "pylint", "."])
        except:
            print("⚠️ No linter found. Install flake8 or pylint.")


def show_menu():
    """Show development tools menu."""
    print("=" * 60)
    print("🛠️  Development Tools")
    print("=" * 60)
    print("1. Run API with debugging")
    print("2. Run debug tool (interactive)")
    print("3. Run API monitor")
    print("4. Run API profiler")
    print("5. Run tests")
    print("6. Run tests with debugging")
    print("7. Run linter")
    print("8. Exit")
    print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Development tools")
    parser.add_argument("tool", nargs="?", help="Tool to run (api, debug, monitor, profiler, tests, lint)")
    
    args = parser.parse_args()
    
    if args.tool:
        # Direct tool execution
        tools = {
            "api": run_api_debug,
            "debug": run_debug_tool,
            "monitor": run_monitor,
            "profiler": run_profiler,
            "tests": run_tests,
            "test-debug": run_tests_debug,
            "lint": run_linter
        }
        
        if args.tool in tools:
            tools[args.tool]()
        else:
            print(f"❌ Unknown tool: {args.tool}")
            print("Available tools: api, debug, monitor, profiler, tests, test-debug, lint")
    else:
        # Interactive menu
        while True:
            show_menu()
            choice = input("Select option (1-8): ").strip()
            
            if choice == "1":
                run_api_debug()
            elif choice == "2":
                run_debug_tool()
            elif choice == "3":
                run_monitor()
            elif choice == "4":
                run_profiler()
            elif choice == "5":
                run_tests()
            elif choice == "6":
                run_tests_debug()
            elif choice == "7":
                run_linter()
            elif choice == "8":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid option")
            
            print()


if __name__ == "__main__":
    main()



