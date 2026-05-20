#!/usr/bin/env python3
"""
Start API and Debug - Enhanced Version
======================================
Start the API server and open debugging tools with enhanced features.
"""

import os
import sys
import subprocess
import time
import signal
import requests
from pathlib import Path
from threading import Thread
from typing import Optional
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Add tools to path for imports
tools_path = project_root / "tools"
if tools_path.exists():
    sys.path.insert(0, str(tools_path.parent))


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_colored(text: str, color: str = Colors.RESET):
    """Print colored text."""
    print(f"{color}{text}{Colors.RESET}")


def print_header(text: str):
    """Print header."""
    print_colored("\n" + "=" * 70, Colors.CYAN)
    print_colored(text, Colors.BOLD + Colors.CYAN)
    print_colored("=" * 70, Colors.CYAN)


def print_success(text: str):
    """Print success message."""
    print_colored(f"✅ {text}", Colors.GREEN)


def print_error(text: str):
    """Print error message."""
    print_colored(f"❌ {text}", Colors.RED)


def print_warning(text: str):
    """Print warning message."""
    print_colored(f"⚠️  {text}", Colors.YELLOW)


def print_info(text: str):
    """Print info message."""
    print_colored(f"ℹ️  {text}", Colors.BLUE)


def check_port_available(port: int = 8000) -> bool:
    """Check if port is available."""
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            result = s.connect_ex(('localhost', port))
            return result != 0
    except:
        return True


def check_api_health(base_url: str = "http://localhost:8000", max_retries: int = 30) -> bool:
    """Wait for API to be healthy with progress indicator."""
    print_info("Waiting for API to start...")
    
    for i in range(max_retries):
        try:
            response = requests.get(f"{base_url}/health", timeout=2)
            if response.status_code == 200:
                data = response.json()
                print_success(f"API is healthy and ready! (Status: {data.get('status', 'ok')})")
                return True
        except requests.exceptions.ConnectionError:
            pass
        except Exception as e:
            print_warning(f"Health check error: {e}")
        
        # Progress indicator
        if i % 5 == 0 and i > 0:
            elapsed = i
            remaining = max_retries - i
            print_info(f"Still waiting... ({elapsed}s elapsed, ~{remaining}s remaining)")
        
        time.sleep(1)
    
    print_error("API did not become healthy in time")
    return False


def start_api_server(port: int = 8000) -> Optional[subprocess.Popen]:
    """Start the API server with enhanced error handling."""
    # Check if port is available
    if not check_port_available(port):
        print_error(f"Port {port} is already in use!")
        print_info("Options:")
        print_info("  1. Stop the existing server")
        print_info("  2. Use a different port (set PORT environment variable)")
        return None
    
    print_header("🚀 Starting API Server with Debugging")
    
    # Set debugging environment
    env = os.environ.copy()
    env.update({
        "DEBUG": "true",
        "LOG_LEVEL": "debug",
        "DETAILED_ERRORS": "true",
        "LOG_REQUESTS": "true",
        "LOG_RESPONSES": "true",
        "ENABLE_METRICS": "true",
        "ENABLE_PROFILING": "true",
        "PORT": str(port)
    })
    
    print_info("Debugging features enabled:")
    print_info("  • Debug mode: ON")
    print_info("  • Detailed errors: ON")
    print_info("  • Request logging: ON")
    print_info("  • Response logging: ON")
    print_info("  • Metrics: ON")
    print_info("  • Profiling: ON")
    
    try:
        # Start server
        process = subprocess.Popen(
            [sys.executable, "run_api_debug.py"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        print_success(f"API server process started (PID: {process.pid})")
        return process
    
    except Exception as e:
        print_error(f"Failed to start API server: {e}")
        return None


def print_api_output(process: subprocess.Popen, show_output: bool = True):
    """Print API output in real-time with filtering."""
    if not show_output:
        return
    
    for line in iter(process.stdout.readline, ''):
        if line:
            line = line.rstrip()
            # Filter and colorize output
            if "ERROR" in line or "error" in line.lower():
                print_colored(f"[API] {line}", Colors.RED)
            elif "WARNING" in line or "warning" in line.lower():
                print_colored(f"[API] {line}", Colors.YELLOW)
            elif "INFO" in line or "info" in line.lower():
                print_colored(f"[API] {line}", Colors.BLUE)
            else:
                print(f"[API] {line}")


def open_debug_tool():
    """Open debug tool with enhanced interface."""
    print_header("🐛 Opening Debug Tool")
    print_info("Use 'help' in the debug tool for available commands")
    print()
    
    subprocess.run([sys.executable, "debug_api.py"])


def open_monitor():
    """Open monitor with enhanced interface."""
    print_header("📊 Opening API Monitor")
    print_info("Use 'help' in the monitor for available commands")
    print()
    
    subprocess.run([sys.executable, "api_monitor.py"])


def open_profiler():
    """Open profiler."""
    print_header("📈 Opening API Profiler")
    print()
    
    subprocess.run([sys.executable, "api_profiler.py"])


def open_dashboard():
    """Open dashboard."""
    print_header("📊 Opening API Dashboard")
    print_info("Live dashboard will open. Press Ctrl+C to stop.")
    print()
    
    subprocess.run([sys.executable, "api_dashboard.py"])


def run_test_suite():
    """Run test suite."""
    print_header("🧪 Running API Test Suite")
    print()
    
    subprocess.run([sys.executable, "api_test_suite.py"])


def run_health_checker():
    """Run health checker."""
    print_header("🔍 Running Health Checker")
    print()
    
    subprocess.run([sys.executable, "api_health_checker.py"])


def run_api_logger():
    """Run API logger."""
    print_header("📝 Running API Logger")
    print_info("Logger will capture requests and responses")
    print()
    
    subprocess.run([sys.executable, "api_logger.py"])


def run_benchmark():
    """Run benchmark."""
    print_header("🔥 Running API Benchmark")
    print_info("Benchmark will test endpoint performance")
    print()
    
    subprocess.run([sys.executable, "api_benchmark.py"])


def run_comparator():
    """Run comparator."""
    print_header("🔄 Running API Comparator")
    print_info("Comparator will compare two API endpoints")
    print()
    
    subprocess.run([sys.executable, "api_comparator.py", "--help"])


def run_reporter():
    """Run reporter."""
    print_header("📄 Running API Reporter")
    print_info("Reporter will generate comprehensive reports")
    print()
    
    subprocess.run([sys.executable, "api_reporter.py", "--help"])


def run_analyzer():
    """Run analyzer."""
    print_header("📊 Running API Analyzer")
    print_info("Analyzer will perform advanced analysis")
    print()
    
    subprocess.run([sys.executable, "api_analyzer.py", "--help"])


def run_automated_pipeline():
    """Run automated pipeline."""
    print_header("🤖 Running Automated Testing Pipeline")
    print_info("Pipeline will run full testing suite")
    print()
    
    subprocess.run([sys.executable, "automated_testing.py", "--help"])


def run_config_manager():
    """Run config manager."""
    print_header("⚙️  Running Configuration Manager")
    print()
    
    subprocess.run([sys.executable, "api_config.py", "--list"])


def run_alerts():
    """Run alert system."""
    print_header("🔔 Running API Alert System")
    print_info("Alert system will monitor API and send alerts")
    print()
    
    subprocess.run([sys.executable, "api_alerts.py", "--help"])


def run_visualizer():
    """Run visualizer."""
    print_header("📈 Running API Visualizer")
    print_info("Visualizer will generate charts and dashboards")
    print()
    
    subprocess.run([sys.executable, "api_visualizer.py", "--help"])


def run_utils():
    """Run utilities."""
    print_header("🛠️  Running API Utilities")
    print()
    
    subprocess.run([sys.executable, "api_utils.py", "--help"])


def run_tool_manager(port: int = 8000):
    """Run refactored tool manager."""
    print_header("🔧 Running Tool Manager (Refactored)")
    print_info("Tool Manager provides unified access to all refactored tools")
    print()
    
    base_url = f"http://localhost:{port}"
    subprocess.run([sys.executable, "-m", "tools.manager", "--list", "--url", base_url])


def run_tool_chain(port: int = 8000):
    """Run tool chain."""
    print_header("⛓️  Running Tool Chain")
    print_info("Execute multiple tools in sequence")
    print()
    
    # Example chain: health -> benchmark
    from tools.chain import create_chain
    from tools.config import ToolConfig
    
    config = ToolConfig(base_url=f"http://localhost:{port}")
    
    chain = create_chain()
    chain.add_tool("health", endpoints=["/health", "/"])
    chain.add_tool("benchmark", endpoint="/health", iterations=10)
    
    results = chain.execute()
    chain.print_summary()


def run_tool_executor(port: int = 8000):
    """Run tool executor."""
    print_header("⚡ Running Tool Executor")
    print_info("Advanced executor with plugins and parallel execution")
    print()
    
    from tools.executor import ToolExecutor
    
    executor = ToolExecutor()
    
    # Example: execute health check
    result = executor.execute("health", endpoints=["/health", "/"])
    
    if result.success:
        print_success(result.message)
    else:
        print_error(result.message)
    
    # Show metrics
    metrics = executor.get_metrics()
    if metrics:
        print_info(f"Execution metrics: {len(metrics)} tool(s) executed")


def show_api_info(port: int = 8000):
    """Show API information."""
    base_url = f"http://localhost:{port}"
    
    print_header("✅ API Server Information")
    print_colored(f"📍 URL: {base_url}", Colors.CYAN)
    print_colored(f"📚 API Docs: {base_url}/docs", Colors.CYAN)
    print_colored(f"🔍 ReDoc: {base_url}/redoc", Colors.CYAN)
    print_colored(f"💚 Health: {base_url}/health", Colors.CYAN)
    print_colored(f"📊 OpenAPI JSON: {base_url}/openapi.json", Colors.CYAN)


def show_menu(port: int = 8000):
    """Show enhanced menu for debugging tools."""
    print_header("🛠️  Debugging Tools Menu")
    print_colored(f"API is running at http://localhost:{port}", Colors.GREEN)
    print()
    print_colored("Available tools:", Colors.BOLD)
    print("  1. 🐛 Debug Tool (Interactive API testing)")
    print("  2. 📊 API Monitor (Real-time monitoring)")
    print("  3. 📈 API Profiler (Performance profiling)")
    print("  4. 📊 API Dashboard (Live dashboard)")
    print("  5. 🧪 API Test Suite (Automated tests)")
    print("  6. 🔍 Health Checker (Comprehensive health check)")
    print("  7. 📝 API Logger (Request/response logging)")
    print("  8. 🔥 API Benchmark (Performance benchmarking)")
    print("  9. 🔄 API Comparator (Compare APIs)")
    print(" 10. 📄 API Reporter (Generate reports)")
    print(" 11. 📊 API Analyzer (Advanced analysis)")
    print(" 12. 🤖 Automated Pipeline (Full testing pipeline)")
    print(" 13. ⚙️  Config Manager (Manage configurations)")
    print(" 14. 🔔 API Alerts (Alert system)")
    print(" 15. 📈 API Visualizer (Data visualization)")
    print(" 16. 🛠️  API Utils (Utility functions)")
    print(" 17. 🔧 Tool Manager (Refactored tools system)")
    print(" 18. ⛓️  Tool Chain (Execute tool chains)")
    print(" 19. ⚡ Tool Executor (Advanced executor)")
    print(" 20. 🌐 Open API Docs (Browser)")
    print(" 21. 📖 Open ReDoc (Browser)")
    print(" 22. 💚 Quick Health Check")
    print(" 23. 📋 Show API Info")
    print(" 24. 🛑 Exit (Stop API)")
    print()


def open_browser(url: str):
    """Open URL in browser with error handling."""
    try:
        import webbrowser
        webbrowser.open(url)
        print_success(f"Opened {url} in browser")
    except Exception as e:
        print_error(f"Failed to open browser: {e}")
        print_info(f"Please open manually: {url}")


def check_health_status(port: int = 8000):
    """Check and display health status."""
    base_url = f"http://localhost:{port}"
    
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print_success("API Health Check")
            print(f"  Status: {data.get('status', 'unknown')}")
            print(f"  Version: {data.get('version', 'N/A')}")
            print(f"  Response Time: {response.elapsed.total_seconds() * 1000:.2f}ms")
        else:
            print_warning(f"API returned status {response.status_code}")
    except requests.exceptions.ConnectionError:
        print_error("Cannot connect to API. Is it running?")
    except Exception as e:
        print_error(f"Health check failed: {e}")


def save_session_info(port: int = 8000):
    """Save session information to file."""
    session_info = {
        "started_at": datetime.now().isoformat(),
        "port": port,
        "base_url": f"http://localhost:{port}",
        "debugging_enabled": True,
        "features": {
            "debug_mode": True,
            "detailed_errors": True,
            "request_logging": True,
            "response_logging": True,
            "metrics": True,
            "profiling": True
        }
    }
    
    session_file = project_root / "debug_session.json"
    with open(session_file, "w") as f:
        json.dump(session_info, f, indent=2)
    
    print_info(f"Session info saved to {session_file}")


def main():
    """Main entry point with enhanced features."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Start API and debugging tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start_api_and_debug.py              # Interactive menu
  python start_api_and_debug.py --tool debug # Open debug tool directly
  python start_api_and_debug.py --port 8001  # Use different port
  python start_api_and_debug.py --quiet      # Minimal output
        """
    )
    parser.add_argument("--tool", choices=["debug", "monitor", "profiler", "dashboard", "test", "health", "logger", "benchmark", "comparator", "reporter", "analyzer", "pipeline", "config", "alerts", "visualizer", "utils"], 
                       help="Open specific tool directly")
    parser.add_argument("--no-wait", action="store_true",
                       help="Don't wait for API to be ready")
    parser.add_argument("--port", type=int, default=8000,
                       help="Port to run API on (default: 8000)")
    parser.add_argument("--quiet", action="store_true",
                       help="Minimal output (hide API logs)")
    parser.add_argument("--no-menu", action="store_true",
                       help="Skip menu and exit after starting API")
    
    args = parser.parse_args()
    
    # Start API server
    api_process = start_api_server(port=args.port)
    
    if not api_process:
        print_error("Failed to start API server. Exiting.")
        sys.exit(1)
    
    # Handle cleanup on exit
    def cleanup():
        print_header("🛑 Stopping API Server")
        print_info("Sending termination signal...")
        api_process.terminate()
        
        try:
            api_process.wait(timeout=5)
            print_success("API server stopped gracefully")
        except subprocess.TimeoutExpired:
            print_warning("API server didn't stop, forcing kill...")
            api_process.kill()
            api_process.wait()
            print_success("API server killed")
        
        sys.exit(0)
    
    signal.signal(signal.SIGINT, lambda s, f: cleanup())
    signal.signal(signal.SIGTERM, lambda s, f: cleanup())
    
    # Print API output in background
    output_thread = Thread(
        target=print_api_output, 
        args=(api_process, not args.quiet), 
        daemon=True
    )
    output_thread.start()
    
    # Wait for API to be ready
    if not args.no_wait:
        if not check_api_health(base_url=f"http://localhost:{args.port}"):
            print_warning("Continuing anyway...")
        time.sleep(1)  # Give it a moment
    
    # Save session info
    save_session_info(port=args.port)
    
    # Show API info
    show_api_info(port=args.port)
    
    # If no-menu, just wait
    if args.no_menu:
        print_info("API is running. Press Ctrl+C to stop.")
        try:
            api_process.wait()
        except KeyboardInterrupt:
            cleanup()
        return
    
    # Open specific tool or show menu
    if args.tool:
        if args.tool == "debug":
            open_debug_tool()
        elif args.tool == "monitor":
            open_monitor()
        elif args.tool == "profiler":
            open_profiler()
        elif args.tool == "dashboard":
            open_dashboard()
        elif args.tool == "test":
            run_test_suite()
        elif args.tool == "health":
            run_health_checker()
        elif args.tool == "logger":
            run_api_logger()
        elif args.tool == "benchmark":
            run_benchmark()
        elif args.tool == "comparator":
            run_comparator()
        elif args.tool == "reporter":
            run_reporter()
        elif args.tool == "analyzer":
            run_analyzer()
        elif args.tool == "pipeline":
            run_automated_pipeline()
        elif args.tool == "config":
            run_config_manager()
        elif args.tool == "alerts":
            run_alerts()
        elif args.tool == "visualizer":
            run_visualizer()
        elif args.tool == "utils":
            run_utils()
        elif args.tool == "tool-manager":
            run_tool_manager(port=args.port)
        elif args.tool == "tool-chain":
            run_tool_chain(port=args.port)
        elif args.tool == "tool-executor":
            run_tool_executor(port=args.port)
    else:
        # Interactive menu
        while True:
            try:
                show_menu(port=args.port)
                choice = input("Select option (1-24): ").strip()
                
                if choice == "1":
                    open_debug_tool()
                elif choice == "2":
                    open_monitor()
                elif choice == "3":
                    open_profiler()
                elif choice == "4":
                    open_dashboard()
                elif choice == "5":
                    run_test_suite()
                elif choice == "6":
                    run_health_checker()
                elif choice == "7":
                    run_api_logger()
                elif choice == "8":
                    run_benchmark()
                elif choice == "9":
                    run_comparator()
                elif choice == "10":
                    run_reporter()
                elif choice == "11":
                    run_analyzer()
                elif choice == "12":
                    run_automated_pipeline()
                elif choice == "13":
                    run_config_manager()
                elif choice == "14":
                    run_alerts()
                elif choice == "15":
                    run_visualizer()
                elif choice == "16":
                    run_utils()
                elif choice == "17":
                    run_tool_manager(port=args.port)
                elif choice == "18":
                    run_tool_chain(port=args.port)
                elif choice == "19":
                    run_tool_executor(port=args.port)
                elif choice == "20":
                    open_browser(f"http://localhost:{args.port}/docs")
                elif choice == "21":
                    open_browser(f"http://localhost:{args.port}/redoc")
                elif choice == "22":
                    check_health_status(port=args.port)
                elif choice == "23":
                    show_api_info(port=args.port)
                elif choice == "24":
                    cleanup()
                    break
                else:
                    print_error("Invalid option. Please select 1-24.")
                
                if choice != "8":
                    input("\nPress Enter to continue...")
            
            except KeyboardInterrupt:
                cleanup()
                break
            except Exception as e:
                print_error(f"Error: {e}")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()
