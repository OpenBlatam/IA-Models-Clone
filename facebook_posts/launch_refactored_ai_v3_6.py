#!/usr/bin/env python3
"""
Launch Script for Refactored Unified AI Interface v3.6
Enhanced launch mechanism with improved error handling and validation
"""
import os
import sys
import time
import json
import argparse
from datetime import datetime
from pathlib import Path

def print_banner():
    """Print system banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    REFACTORED UNIFIED AI INTERFACE v3.6                     ║
║                                                                              ║
║  🚀 Enhanced Architecture • 🔧 Intelligent Optimization • 🔮 Predictive AI  ║
║  📊 Real-time Monitoring • 🧠 Machine Learning • ⚡ Performance Boost      ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def print_system_info():
    """Print system information"""
    print("🔍 System Information:")
    print(f"   • Python Version: {sys.version}")
    print(f"   • Platform: {sys.platform}")
    print(f"   • Working Directory: {os.getcwd()}")
    print(f"   • Launch Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

def check_dependencies():
    """Check if required dependencies are available"""
    print("📦 Checking Dependencies...")
    
    required_modules = [
        'numpy',
        'pandas', 
        'plotly',
        'psutil',
        'torch'
    ]
    
    missing_modules = []
    available_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
            available_modules.append(module)
            print(f"   ✅ {module}")
        except ImportError:
            missing_modules.append(module)
            print(f"   ❌ {module} (missing)")
    
    print()
    
    if missing_modules:
        print("⚠️  Missing required modules. Some features may not work properly.")
        print("   Install missing modules with: pip install " + " ".join(missing_modules))
        print()
    
    return len(missing_modules) == 0

def check_enhanced_components():
    """Check if enhanced system components are available"""
    print("🔧 Checking Enhanced Components...")
    
    components = [
        ('Enhanced System Integrator', 'enhanced_system_integrator'),
        ('Advanced Performance Monitor', 'advanced_performance_monitor'),
        ('Intelligent Optimization Engine', 'intelligent_optimization_engine'),
        ('Predictive Analytics System', 'predictive_analytics_system')
    ]
    
    available_components = []
    missing_components = []
    
    for name, module in components:
        try:
            __import__(module)
            available_components.append(name)
            print(f"   ✅ {name}")
        except ImportError:
            missing_components.append(name)
            print(f"   ❌ {name} (missing)")
    
    print()
    
    if missing_components:
        print("⚠️  Some enhanced components are missing. System will run in limited mode.")
        print()
    
    return len(available_components) > 0

def validate_configuration():
    """Validate system configuration"""
    print("⚙️  Validating Configuration...")
    
    # Check if config file exists
    config_file = Path("ai_config_v3_6.json")
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            print(f"   ✅ Configuration file loaded: {config_file}")
            return config
        except Exception as e:
            print(f"   ❌ Error loading config: {e}")
    else:
        print(f"   ℹ️  No configuration file found, using defaults")
    
    return None

def create_default_config():
    """Create default configuration file"""
    default_config = {
        "system": {
            "auto_start": True,
            "health_check_interval": 30.0,
            "max_retries": 3,
            "timeout": 60.0
        },
        "monitoring": {
            "enabled": True,
            "interval": 2.0,
            "history_size": 1000,
            "alert_thresholds": {
                "cpu": 80.0,
                "memory": 85.0,
                "disk": 90.0,
                "gpu": 95.0
            }
        },
        "optimization": {
            "enabled": True,
            "auto_optimize": True,
            "interval": 30.0,
            "threshold": 0.7
        },
        "analytics": {
            "enabled": True,
            "prediction_horizon": 300,
            "confidence_threshold": 0.75
        },
        "ui": {
            "theme": "dark",
            "auto_refresh": True,
            "refresh_interval": 5.0
        }
    }
    
    config_file = Path("ai_config_v3_6.json")
    try:
        with open(config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
        print(f"   ✅ Default configuration created: {config_file}")
        return default_config
    except Exception as e:
        print(f"   ❌ Error creating config: {e}")
        return None

def launch_ai_interface(config=None):
    """Launch the refactored AI interface"""
    print("🚀 Launching Refactored AI Interface...")
    
    try:
        # Import the refactored interface
        from refactored_unified_ai_interface_v3_6 import RefactoredUnifiedAIInterface
        
        # Create interface instance
        ai_interface = RefactoredUnifiedAIInterface(config)
        
        # Get initial status
        status = ai_interface.get_system_status()
        print(f"   ✅ Interface Status: {status['interface']['state']}")
        
        # Check if system is running
        if status['interface']['state'] == 'running':
            print("   🎉 AI Interface is running successfully!")
        else:
            print(f"   ⚠️  Interface state: {status['interface']['state']}")
        
        return ai_interface
        
    except ImportError as e:
        print(f"   ❌ Error importing AI interface: {e}")
        return None
    except Exception as e:
        print(f"   ❌ Error launching AI interface: {e}")
        return None

def run_interactive_mode(ai_interface):
    """Run interactive mode for testing and monitoring"""
    print("\n🎮 Interactive Mode - Type 'help' for commands")
    print("   Type 'exit' to quit")
    
    commands = {
        'help': 'Show available commands',
        'status': 'Show system status',
        'health': 'Show health report',
        'performance': 'Show performance metrics',
        'optimization': 'Show optimization status',
        'analytics': 'Show analytics insights',
        'export': 'Export system data',
        'optimize': 'Trigger manual optimization',
        'exit': 'Exit interactive mode'
    }
    
    while True:
        try:
            command = input("\n🤖 AI> ").strip().lower()
            
            if command == 'exit':
                print("👋 Goodbye!")
                break
            elif command == 'help':
                print("\n📚 Available Commands:")
                for cmd, desc in commands.items():
                    print(f"   • {cmd}: {desc}")
            elif command == 'status':
                status = ai_interface.get_system_status()
                print(f"\n📊 System Status: {json.dumps(status, indent=2)}")
            elif command == 'health':
                health = ai_interface.get_health_report()
                print(f"\n🏥 Health Report: {json.dumps(health, indent=2)}")
            elif command == 'performance':
                perf = ai_interface.get_performance_metrics()
                print(f"\n⚡ Performance: {json.dumps(perf, indent=2)}")
            elif command == 'optimization':
                opt = ai_interface.get_optimization_status()
                print(f"\n🔧 Optimization: {json.dumps(opt, indent=2)}")
            elif command == 'analytics':
                analytics = ai_interface.get_analytics_insights()
                print(f"\n🔮 Analytics: {json.dumps(analytics, indent=2)}")
            elif command == 'export':
                data = ai_interface.export_system_data('json')
                print(f"\n📤 Exported {len(data)} characters of system data")
            elif command == 'optimize':
                success = ai_interface.trigger_optimization()
                print(f"\n🔧 Manual optimization: {'Success' if success else 'Failed'}")
            else:
                print(f"❓ Unknown command: {command}")
                print("   Type 'help' for available commands")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error executing command: {e}")

def main():
    """Main launch function"""
    parser = argparse.ArgumentParser(description='Launch Refactored Unified AI Interface v3.6')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    parser.add_argument('--create-config', action='store_true', help='Create default configuration file')
    parser.add_argument('--validate-only', action='store_true', help='Only validate system, do not launch')
    
    args = parser.parse_args()
    
    try:
        # Print banner and system info
        print_banner()
        print_system_info()
        
        # Check dependencies
        deps_ok = check_dependencies()
        
        # Check enhanced components
        enhanced_ok = check_enhanced_components()
        
        # Handle configuration
        if args.create_config:
            config = create_default_config()
            if config:
                print("✅ Configuration file created successfully!")
            return
        
        # Load or create configuration
        if args.config:
            config_file = Path(args.config)
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                    print(f"✅ Configuration loaded from: {config_file}")
                except Exception as e:
                    print(f"❌ Error loading config: {e}")
                    config = None
            else:
                print(f"❌ Configuration file not found: {config_file}")
                config = None
        else:
            config = validate_configuration()
            if not config:
                config = create_default_config()
        
        if args.validate_only:
            print("✅ System validation completed!")
            return
        
        # Launch AI interface
        ai_interface = launch_ai_interface(config)
        
        if ai_interface:
            if args.interactive:
                run_interactive_mode(ai_interface)
            else:
                print("\n🔄 AI Interface is running in background mode...")
                print("   Press Ctrl+C to stop")
                
                try:
                    # Keep running until interrupted
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("\n\n🛑 Stopping AI Interface...")
                finally:
                    ai_interface.cleanup()
                    print("✅ AI Interface stopped")
        else:
            print("❌ Failed to launch AI Interface")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
