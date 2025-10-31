#!/usr/bin/env python3
"""
Debug Launcher for Video-OpusClip

Quick access to debugging tools and common debugging scenarios.
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

def print_debug_banner():
    """Print debug launcher banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                    🐛 Video-OpusClip Debug Launcher         ║
    ║                                                              ║
    ║              Comprehensive Debugging and Troubleshooting     ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def show_debug_options():
    """Show available debug options."""
    options = {
        "1": {
            "name": "🔍 Quick System Check",
            "description": "Run basic system diagnostics",
            "function": quick_system_check
        },
        "2": {
            "name": "📊 Performance Analysis",
            "description": "Analyze system performance",
            "function": performance_analysis
        },
        "3": {
            "name": "💾 Memory Analysis",
            "description": "Check memory usage and leaks",
            "function": memory_analysis
        },
        "4": {
            "name": "🚨 Error Analysis",
            "description": "Analyze error patterns",
            "function": error_analysis
        },
        "5": {
            "name": "🔧 Interactive Debugger",
            "description": "Start interactive debugging session",
            "function": interactive_debugger
        },
        "6": {
            "name": "📈 Comprehensive Debug Report",
            "description": "Generate full debug report",
            "function": comprehensive_debug_report
        },
        "7": {
            "name": "🧪 Debug Test Suite",
            "description": "Run debug tools test suite",
            "function": debug_test_suite
        },
        "8": {
            "name": "📋 Debug Status",
            "description": "Show current debug status",
            "function": debug_status
        },
        "0": {
            "name": "🚪 Exit",
            "description": "Exit debug launcher",
            "function": None
        }
    }
    
    print("\n🎯 Available Debug Options:")
    print("-" * 50)
    
    for key, option in options.items():
        print(f"{key}. {option['name']}")
        print(f"   {option['description']}")
        print()
    
    return options

def get_user_choice(options):
    """Get user choice for debug option."""
    while True:
        try:
            choice = input("Enter your choice (0-8): ").strip()
            if choice in options:
                return choice
            else:
                print("❌ Invalid choice. Please enter a number between 0-8.")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            sys.exit(0)

def quick_system_check():
    """Run quick system check."""
    print("\n🔍 Running Quick System Check...")
    
    try:
        from debug_tools import SystemDiagnostics
        
        diagnostics = SystemDiagnostics()
        health_report = diagnostics.run_full_diagnostics()
        
        print(f"✅ System Health Score: {health_report['health_score']}/100")
        
        # Check critical metrics
        resource_usage = health_report['resource_usage']
        print(f"📊 CPU Usage: {resource_usage['cpu_usage']:.1f}%")
        print(f"💾 Memory Usage: {resource_usage['memory_usage']:.1f}%")
        print(f"💿 Disk Usage: {resource_usage['disk_usage']:.1f}%")
        
        # Check network
        network_status = health_report['network_status']
        print(f"🌐 Network Status: {network_status['status']}")
        
        # Show recommendations
        if health_report['recommendations']:
            print("\n📋 Recommendations:")
            for rec in health_report['recommendations']:
                print(f"  • {rec}")
        
        return health_report
        
    except Exception as e:
        print(f"❌ System check failed: {e}")
        return None

def performance_analysis():
    """Run performance analysis."""
    print("\n📊 Running Performance Analysis...")
    
    try:
        from debug_tools import PerformanceProfiler
        
        profiler = PerformanceProfiler()
        
        # Run some test operations
        @profiler.profile_function("test_operation")
        def test_operation():
            time.sleep(0.1)  # Simulate work
            return "test_result"
        
        # Run multiple operations
        for i in range(5):
            test_operation()
        
        # Get performance report
        report = profiler.get_profile_report()
        
        print(f"✅ Performance analysis completed")
        print(f"📈 Total profiles: {report['total_profiles']}")
        
        if report['summary']:
            summary = report['summary']
            print(f"📊 Total calls: {summary['total_calls']}")
            print(f"⏱️ Average duration: {summary['average_duration']:.3f}s")
            print(f"🎯 Success rate: {summary['success_rate']:.2%}")
        
        return report
        
    except Exception as e:
        print(f"❌ Performance analysis failed: {e}")
        return None

def memory_analysis():
    """Run memory analysis."""
    print("\n💾 Running Memory Analysis...")
    
    try:
        from debug_tools import MemoryAnalyzer
        
        analyzer = MemoryAnalyzer()
        
        # Take initial snapshot
        analyzer.take_snapshot("analysis_start")
        
        # Simulate some memory usage
        test_data = [i for i in range(10000)]
        analyzer.take_snapshot("after_data_creation")
        
        # Clear data
        del test_data
        analyzer.take_snapshot("after_data_clear")
        
        # Force garbage collection
        collected = analyzer.force_garbage_collection()
        analyzer.take_snapshot("after_gc")
        
        # Analyze memory usage
        analysis = analyzer.analyze_memory_usage()
        
        print(f"✅ Memory analysis completed")
        print(f"📊 Current memory: {analysis['rss']['current'] / (1024**2):.2f}MB")
        print(f"📈 Memory growth: {analysis['memory_growth']['total_growth'] / (1024**2):.2f}MB")
        print(f"🗑️ Objects collected: {collected}")
        
        # Check for leaks
        if analysis['leak_detection']['potential_leak']:
            print("⚠️ Potential memory leak detected!")
            print(f"📈 Growth rate: {analysis['leak_detection']['growth_percentage']:.2f}%")
        else:
            print("✅ No memory leaks detected")
        
        # Show recommendations
        if analysis['recommendations']:
            print("\n📋 Memory Recommendations:")
            for rec in analysis['recommendations']:
                print(f"  • {rec}")
        
        return analysis
        
    except Exception as e:
        print(f"❌ Memory analysis failed: {e}")
        return None

def error_analysis():
    """Run error analysis."""
    print("\n🚨 Running Error Analysis...")
    
    try:
        from debug_tools import ErrorAnalyzer
        
        analyzer = ErrorAnalyzer()
        
        # Simulate some errors for testing
        try:
            raise ValueError("Test error for analysis")
        except Exception as e:
            analyzer.record_error(e, "test_context", "Test stack trace")
        
        try:
            raise RuntimeError("Another test error")
        except Exception as e:
            analyzer.record_error(e, "test_context", "Another test stack trace")
        
        # Analyze errors
        analysis = analyzer.analyze_errors()
        
        print(f"✅ Error analysis completed")
        print(f"📊 Total errors: {analysis['total_errors']}")
        
        if analysis['error_types']['most_common']:
            print(f"🎯 Most common error: {analysis['error_types']['most_common']}")
        
        if analysis['error_patterns']['patterns']:
            patterns = analysis['error_patterns']['patterns']
            print(f"📈 Error patterns found: {len(patterns)}")
            
            if patterns:
                most_frequent = patterns[0]
                print(f"🔍 Most frequent pattern: {most_frequent['pattern']}")
                print(f"📊 Occurrences: {most_frequent['count']}")
        
        # Show recommendations
        if analysis['recommendations']:
            print("\n📋 Error Recommendations:")
            for rec in analysis['recommendations']:
                print(f"  • {rec}")
        
        return analysis
        
    except Exception as e:
        print(f"❌ Error analysis failed: {e}")
        return None

def interactive_debugger():
    """Start interactive debugger."""
    print("\n🔧 Starting Interactive Debugger...")
    
    try:
        from debug_tools import VideoOpusClipDebugger
        
        debugger = VideoOpusClipDebugger()
        
        # Set up some example breakpoints
        debugger.set_breakpoint("test_function")
        
        # Add watch variables
        debugger.add_watch_variable("test_var", "test_var")
        
        @debugger.debug_function
        def test_function(input_data):
            test_var = input_data * 2
            print(f"Processing: {input_data} -> {test_var}")
            return test_var
        
        print("✅ Interactive debugger started")
        print("💡 Use the test_function to trigger breakpoints")
        print("💡 Type 'help' at breakpoints for available commands")
        
        # Test the debugger
        try:
            result = test_function(5)
            print(f"✅ Test function completed: {result}")
        except Exception as e:
            print(f"❌ Test function failed: {e}")
        
        return debugger.get_debug_report()
        
    except Exception as e:
        print(f"❌ Interactive debugger failed: {e}")
        return None

def comprehensive_debug_report():
    """Generate comprehensive debug report."""
    print("\n📈 Generating Comprehensive Debug Report...")
    
    try:
        from debug_tools import DebugManager
        
        debug_manager = DebugManager()
        debug_manager.enable_debugging()
        
        # Run comprehensive analysis
        report = debug_manager.run_comprehensive_debug()
        
        print("✅ Comprehensive debug report generated")
        
        # Show summary
        summary = report['summary']
        print(f"🏥 System Health: {summary['system_health']}/100")
        print(f"🚨 Total Errors: {summary['total_errors']}")
        print(f"💾 Memory Usage: {summary['memory_usage'] / (1024**2):.2f}MB")
        
        # Show recommendations
        if summary['recommendations']:
            print("\n📋 Recommendations:")
            for rec in summary['recommendations']:
                print(f"  • {rec}")
        
        # Save report to file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"debug_report_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📄 Report saved to: {filename}")
        
        return report
        
    except Exception as e:
        print(f"❌ Comprehensive debug report failed: {e}")
        return None

def debug_test_suite():
    """Run debug tools test suite."""
    print("\n🧪 Running Debug Test Suite...")
    
    try:
        # Import and run test script
        from test_error_handling import main as run_error_tests
        
        print("Running error handling tests...")
        run_error_tests()
        
        print("✅ Debug test suite completed")
        
        return {"status": "success", "message": "All tests passed"}
        
    except Exception as e:
        print(f"❌ Debug test suite failed: {e}")
        return {"status": "error", "message": str(e)}

def debug_status():
    """Show current debug status."""
    print("\n📋 Current Debug Status...")
    
    try:
        from debug_tools import DebugManager
        
        debug_manager = DebugManager()
        status = debug_manager.get_debug_status()
        
        print("✅ Debug Status:")
        print(f"  🎛️ Debug enabled: {status['debug_enabled']}")
        print(f"  🔍 Debugger active: {status['debugger_active']}")
        print(f"  📊 Profiling enabled: {status['profiling_enabled']}")
        print(f"  💾 Memory analysis enabled: {status['memory_analysis_enabled']}")
        print(f"  🚨 Error analysis enabled: {status['error_analysis_enabled']}")
        print(f"  📈 Active profiles: {len(status['active_profiles'])}")
        print(f"  📸 Memory snapshots: {status['memory_snapshots']}")
        print(f"  🚨 Recorded errors: {status['recorded_errors']}")
        
        return status
        
    except Exception as e:
        print(f"❌ Debug status check failed: {e}")
        return None

def show_debug_help():
    """Show debug help information."""
    help_text = """
    🐛 Video-OpusClip Debug Launcher - Help Guide
    
    Debug Options:
    • Quick System Check: Basic system diagnostics
    • Performance Analysis: System performance profiling
    • Memory Analysis: Memory usage and leak detection
    • Error Analysis: Error pattern analysis
    • Interactive Debugger: Interactive debugging session
    • Comprehensive Debug Report: Full system analysis
    • Debug Test Suite: Test all debug tools
    • Debug Status: Current debug system status
    
    Usage Examples:
    • python debug_launcher.py --quick-check
    • python debug_launcher.py --performance
    • python debug_launcher.py --memory
    • python debug_launcher.py --comprehensive
    • python debug_launcher.py --interactive
    
    Tips:
    • Start with Quick System Check for basic diagnostics
    • Use Performance Analysis to identify bottlenecks
    • Run Memory Analysis if you suspect memory issues
    • Use Error Analysis to understand error patterns
    • Interactive Debugger is best for step-by-step debugging
    • Comprehensive Report provides full system overview
    
    For more information, see the debugging documentation.
    """
    print(help_text)

def main():
    """Main debug launcher function."""
    
    print_debug_banner()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Video-OpusClip Debug Launcher")
    parser.add_argument("--quick-check", action="store_true", help="Run quick system check")
    parser.add_argument("--performance", action="store_true", help="Run performance analysis")
    parser.add_argument("--memory", action="store_true", help="Run memory analysis")
    parser.add_argument("--errors", action="store_true", help="Run error analysis")
    parser.add_argument("--interactive", action="store_true", help="Start interactive debugger")
    parser.add_argument("--comprehensive", action="store_true", help="Generate comprehensive report")
    parser.add_argument("--test-suite", action="store_true", help="Run debug test suite")
    parser.add_argument("--status", action="store_true", help="Show debug status")
    parser.add_argument("--help-debug", action="store_true", help="Show debug help")
    
    args = parser.parse_args()
    
    # Show help if requested
    if args.help_debug:
        show_debug_help()
        return
    
    # Handle command line options
    if args.quick_check:
        quick_system_check()
        return
    elif args.performance:
        performance_analysis()
        return
    elif args.memory:
        memory_analysis()
        return
    elif args.errors:
        error_analysis()
        return
    elif args.interactive:
        interactive_debugger()
        return
    elif args.comprehensive:
        comprehensive_debug_report()
        return
    elif args.test_suite:
        debug_test_suite()
        return
    elif args.status:
        debug_status()
        return
    
    # Interactive mode
    print("🎯 Welcome to the Video-OpusClip Debug Launcher!")
    print("💡 Use --help-debug for detailed information")
    
    # Main loop
    while True:
        try:
            # Show options
            options = show_debug_options()
            
            # Get user choice
            choice = get_user_choice(options)
            
            if choice == "0":
                print("👋 Goodbye!")
                break
            
            # Execute selected option
            option = options[choice]
            if option["function"]:
                result = option["function"]()
                
                if result:
                    print(f"\n✅ {option['name']} completed successfully")
                else:
                    print(f"\n❌ {option['name']} failed")
            
            # Ask if user wants to continue
            print("\n" + "="*50)
            continue_choice = input("Would you like to run another debug option? (y/n): ").strip().lower()
            if continue_choice not in ['y', 'yes']:
                print("👋 Thanks for using the debug launcher!")
                break
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            print("Please try again or contact support.")

if __name__ == "__main__":
    main() 