from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

import os
import sys
import argparse
import subprocess
import threading
import time
import webbrowser
from typing import List, Dict, Optional
import logging
import sys
import os
from {demo['file'].replace('.py', '')} import main
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Comprehensive Demo Launcher
==========================

This module provides a unified launcher for all interactive demos:
- Main interactive demos
- Real-time inference demo
- Radio integration demo
- Individual component demos
"""


# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DemoLauncher:
    """Comprehensive demo launcher for all interactive demos"""
    
    def __init__(self) -> Any:
        self.demos = {
            'main': {
                'name': 'Main Interactive Demos',
                'file': 'gradio_interactive_demos.py',
                'port': 7860,
                'description': 'Comprehensive demos for text, image, audio, training, and radio'
            },
            'realtime': {
                'name': 'Real-time Inference Demo',
                'file': 'realtime_inference_demo.py',
                'port': 7861,
                'description': 'Live model inference with real-time performance monitoring'
            },
            'radio': {
                'name': 'Radio Integration Demo',
                'file': 'radio_integration_demo.py',
                'port': 7862,
                'description': 'Radio streaming and audio processing demos'
            },
            'user-friendly': {
                'name': 'User-Friendly Interfaces',
                'file': 'user_friendly_interfaces.py',
                'port': 7863,
                'description': 'Beautiful, intuitive interfaces with modern UX/UI design'
            },
            'accessibility': {
                'name': 'Accessibility Interfaces',
                'file': 'accessibility_interfaces.py',
                'port': 7864,
                'description': 'Inclusive interfaces for users with different accessibility needs'
            },
            'error-handled': {
                'name': 'Error-Handled Interface',
                'file': 'error_handling_gradio.py',
                'port': 7865,
                'description': 'Comprehensive error handling and input validation demo'
            },
            'enhanced': {
                'name': 'Enhanced Gradio Demos',
                'file': 'enhanced_gradio_demos.py',
                'port': 7866,
                'description': 'Enhanced demos with integrated error handling and validation'
            },
            'debugging': {
                'name': 'Advanced Debugging System',
                'file': 'advanced_debugging_system.py',
                'port': 7867,
                'description': 'Comprehensive debugging and error analysis tools'
            },
            'troubleshooting': {
                'name': 'Troubleshooting System',
                'file': 'troubleshooting_system.py',
                'port': 7868,
                'description': 'System diagnostics and health monitoring'
            },
            'robust-error-handling': {
                'name': 'Robust Error Handling',
                'file': 'robust_error_handling.py',
                'port': 7869,
                'description': 'Comprehensive try-except blocks for data loading and model inference'
            },
            'training-logging': {
                'name': 'Training Logging System',
                'file': 'training_logging_system.py',
                'port': 7870,
                'description': 'Comprehensive logging for training progress and errors'
            },
            'pytorch-debugging': {
                'name': 'PyTorch Debugging Tools',
                'file': 'pytorch_debugging_tools.py',
                'port': 7871,
                'description': 'Comprehensive debugging tools with autograd.detect_anomaly() and more'
            },
            'performance-optimization': {
                'name': 'Advanced Performance Optimization',
                'file': 'advanced_performance_optimization.py',
                'port': 7872,
                'description': 'Comprehensive performance optimization tools for AI systems'
            }
        }
        
        self.running_demos = {}
        logger.info("Demo Launcher initialized")
    
    def list_demos(self) -> None:
        """List all available demos"""
        print("\n🎯 Available Interactive Demos:")
        print("=" * 60)
        
        for key, demo in self.demos.items():
            print(f"\n📌 {demo['name']}")
            print(f"   Key: {key}")
            print(f"   File: {demo['file']}")
            print(f"   Port: {demo['port']}")
            print(f"   Description: {demo['description']}")
        
        print(f"\n🚀 Usage:")
        print(f"   python demo_launcher.py --demo main")
        print(f"   python demo_launcher.py --demo realtime")
        print(f"   python demo_launcher.py --demo radio")
        print(f"   python demo_launcher.py --all")
    
    def check_dependencies(self) -> bool:
        """Check if all required dependencies are installed"""
        print("🔍 Checking dependencies...")
        
        required_packages = [
            'gradio', 'torch', 'numpy', 'matplotlib', 'seaborn', 
            'plotly', 'Pillow', 'librosa', 'psutil'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                print(f"   ✅ {package}")
            except ImportError:
                print(f"   ❌ {package} (missing)")
                missing_packages.append(package)
        
        if missing_packages:
            print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
            print("Install them with: pip install -r requirements_gradio_demos.txt")
            return False
        
        print("✅ All dependencies are installed!")
        return True
    
    def launch_demo(self, demo_key: str, port: Optional[int] = None, share: bool = False) -> bool:
        """Launch a specific demo"""
        if demo_key not in self.demos:
            print(f"❌ Demo '{demo_key}' not found")
            return False
        
        demo = self.demos[demo_key]
        demo_port = port or demo['port']
        
        print(f"🚀 Launching {demo['name']}...")
        print(f"   File: {demo['file']}")
        print(f"   Port: {demo_port}")
        
        try:
            # Launch demo in a separate process
            cmd = [
                sys.executable, demo['file'],
                '--port', str(demo_port)
            ]
            
            if share:
                cmd.append('--share')
            
            process = subprocess.Popen(
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Store process info
            self.running_demos[demo_key] = {
                'process': process,
                'port': demo_port,
                'start_time': time.time()
            }
            
            # Wait a moment for the server to start
            time.sleep(3)
            
            # Check if process is still running
            if process.poll() is None:
                print(f"✅ {demo['name']} launched successfully!")
                print(f"   URL: http://localhost:{demo_port}")
                
                # Open browser
                try:
                    webbrowser.open(f"http://localhost:{demo_port}")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                except:
                    pass
                
                return True
            else:
                stdout, stderr = process.communicate()
                print(f"❌ Failed to launch {demo['name']}")
                print(f"   Error: {stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error launching {demo['name']}: {e}")
            return False
    
    def launch_all_demos(self, share: bool = False) -> bool:
        """Launch all demos"""
        print("🚀 Launching all interactive demos...")
        
        success_count = 0
        total_demos = len(self.demos)
        
        for demo_key in self.demos:
            if self.launch_demo(demo_key, share=share):
                success_count += 1
                # Wait between launches to avoid port conflicts
                time.sleep(2)
        
        print(f"\n📊 Demo Launch Summary:")
        print(f"   Successful: {success_count}/{total_demos}")
        print(f"   Failed: {total_demos - success_count}/{total_demos}")
        
        if success_count > 0:
            print(f"\n🌐 Demo URLs:")
            for demo_key, info in self.running_demos.items():
                demo_name = self.demos[demo_key]['name']
                print(f"   {demo_name}: http://localhost:{info['port']}")
        
        return success_count == total_demos
    
    def stop_demo(self, demo_key: str) -> bool:
        """Stop a specific demo"""
        if demo_key not in self.running_demos:
            print(f"❌ Demo '{demo_key}' is not running")
            return False
        
        demo_info = self.running_demos[demo_key]
        process = demo_info['process']
        
        print(f"🛑 Stopping {self.demos[demo_key]['name']}...")
        
        try:
            process.terminate()
            process.wait(timeout=5)
            del self.running_demos[demo_key]
            print(f"✅ {self.demos[demo_key]['name']} stopped")
            return True
        except subprocess.TimeoutExpired:
            process.kill()
            del self.running_demos[demo_key]
            print(f"⚠️  {self.demos[demo_key]['name']} force stopped")
            return True
        except Exception as e:
            print(f"❌ Error stopping {self.demos[demo_key]['name']}: {e}")
            return False
    
    def stop_all_demos(self) -> bool:
        """Stop all running demos"""
        if not self.running_demos:
            print("ℹ️  No demos are currently running")
            return True
        
        print("🛑 Stopping all running demos...")
        
        success_count = 0
        total_demos = len(self.running_demos)
        
        for demo_key in list(self.running_demos.keys()):
            if self.stop_demo(demo_key):
                success_count += 1
        
        print(f"📊 Demo Stop Summary:")
        print(f"   Stopped: {success_count}/{total_demos}")
        
        return success_count == total_demos
    
    def show_status(self) -> None:
        """Show status of all demos"""
        print("\n📊 Demo Status:")
        print("=" * 40)
        
        for demo_key, demo in self.demos.items():
            status = "🟢 Running" if demo_key in self.running_demos else "🔴 Stopped"
            port = self.running_demos[demo_key]['port'] if demo_key in self.running_demos else demo['port']
            
            print(f"\n{demo['name']}")
            print(f"   Status: {status}")
            print(f"   Port: {port}")
            
            if demo_key in self.running_demos:
                runtime = time.time() - self.running_demos[demo_key]['start_time']
                print(f"   Runtime: {runtime:.1f} seconds")
                print(f"   URL: http://localhost:{port}")
    
    def monitor_demos(self) -> None:
        """Monitor running demos"""
        print("👀 Monitoring running demos...")
        print("Press Ctrl+C to stop monitoring")
        
        try:
            while True:
                os.system('clear' if os.name == 'posix' else 'cls')
                self.show_status()
                time.sleep(5)
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped")
    
    def create_demo_script(self, demo_key: str) -> str:
        """Create a standalone script for a demo"""
        if demo_key not in self.demos:
            return f"Demo '{demo_key}' not found"
        
        demo = self.demos[demo_key]
        script_content = f'''#!/usr/bin/env python3
"""
Standalone Demo Script: {demo['name']}
====================================

{demo['description']}

Usage: python {demo['file']}
"""


# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import and run the demo

if __name__ == "__main__":
    main()
'''
        
        script_file = f"run_{demo_key}_demo.py"
        with open(script_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write(script_content)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        
        print(f"✅ Created standalone script: {script_file}")
        return script_file


def main():
    """Main function for the demo launcher"""
    parser = argparse.ArgumentParser(description="Interactive Demo Launcher")
    parser.add_argument('--demo', type=str, help='Demo to launch (main, realtime, radio)')
    parser.add_argument('--all', action='store_true', help='Launch all demos')
    parser.add_argument('--list', action='store_true', help='List all available demos')
    parser.add_argument('--check', action='store_true', help='Check dependencies')
    parser.add_argument('--stop', type=str, help='Stop a specific demo')
    parser.add_argument('--stop-all', action='store_true', help='Stop all running demos')
    parser.add_argument('--status', action='store_true', help='Show demo status')
    parser.add_argument('--monitor', action='store_true', help='Monitor running demos')
    parser.add_argument('--port', type=int, help='Custom port for demo')
    parser.add_argument('--share', action='store_true', help='Enable public sharing')
    parser.add_argument('--create-script', type=str, help='Create standalone script for demo')
    
    args = parser.parse_args()
    
    launcher = DemoLauncher()
    
    if args.list:
        launcher.list_demos()
    
    elif args.check:
        launcher.check_dependencies()
    
    elif args.demo:
        if launcher.check_dependencies():
            launcher.launch_demo(args.demo, port=args.port, share=args.share)
    
    elif args.all:
        if launcher.check_dependencies():
            launcher.launch_all_demos(share=args.share)
    
    elif args.stop:
        launcher.stop_demo(args.stop)
    
    elif args.stop_all:
        launcher.stop_all_demos()
    
    elif args.status:
        launcher.show_status()
    
    elif args.monitor:
        launcher.monitor_demos()
    
    elif args.create_script:
        launcher.create_demo_script(args.create_script)
    
    else:
        print("🎯 Interactive Demo Launcher")
        print("=" * 40)
        print("Use --help for available options")
        print("\nQuick start:")
        print("  python demo_launcher.py --list")
        print("  python demo_launcher.py --demo main")
        print("  python demo_launcher.py --all")


match __name__:
    case "__main__":
    main() 