#!/usr/bin/env python3
"""
Quick Start UI for Video-OpusClip

Easy-to-use script for launching user-friendly interfaces
with minimal setup and configuration.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def print_welcome_banner():
    """Print welcome banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                    🎬 Video-OpusClip                        ║
    ║                                                              ║
    ║              Quick Start - User-Friendly Interface           ║
    ╚══════════════════════════════════════════════════════════════╝
    
    🚀 Ready to create amazing videos with AI!
    """
    print(banner)

def check_requirements():
    """Check if requirements are installed."""
    print("🔍 Checking requirements...")
    
    try:
        import gradio
        import torch
        import numpy
        print("✅ All required packages are installed!")
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("\n📦 Installing requirements...")
        
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "-r", "requirements_optimized.txt"
            ])
            print("✅ Requirements installed successfully!")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install requirements")
            print("\n💡 Try installing manually:")
            print("pip install -r requirements_optimized.txt")
            return False

def show_interface_options():
    """Show available interface options."""
    options = {
        "1": {
            "name": "🚀 User-Friendly Interface",
            "description": "Modern, intuitive interface with guided workflows",
            "command": "python interface_launcher.py --interface user_friendly"
        },
        "2": {
            "name": "🎬 Demo Interface",
            "description": "Interactive demonstrations of all capabilities",
            "command": "python interface_launcher.py --interface demo"
        },
        "3": {
            "name": "🔧 Simple Interface",
            "description": "Basic interface for quick testing",
            "command": "python interface_launcher.py --interface simple"
        },
        "4": {
            "name": "📚 Show Features",
            "description": "Explore interface capabilities",
            "command": "python interface_launcher.py --features"
        },
        "5": {
            "name": "📖 Show Tutorials",
            "description": "View available learning resources",
            "command": "python interface_launcher.py --tutorials"
        },
        "6": {
            "name": "⚡ Performance Test",
            "description": "Run interface performance benchmark",
            "command": "python interface_launcher.py --benchmark"
        },
        "7": {
            "name": "❓ Help",
            "description": "Show detailed help information",
            "command": "python interface_launcher.py --help-interface"
        },
        "0": {
            "name": "🚪 Exit",
            "description": "Exit the quick start script",
            "command": "exit"
        }
    }
    
    print("\n🎯 Choose an option:")
    print("-" * 50)
    
    for key, option in options.items():
        print(f"{key}. {option['name']}")
        print(f"   {option['description']}")
        print()
    
    return options

def get_user_choice(options):
    """Get user choice for interface."""
    while True:
        try:
            choice = input("Enter your choice (0-7): ").strip()
            if choice in options:
                return choice
            else:
                print("❌ Invalid choice. Please enter a number between 0-7.")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            sys.exit(0)

def launch_interface(choice, options):
    """Launch the selected interface."""
    option = options[choice]
    
    if choice == "0":
        print("👋 Goodbye!")
        return
    
    print(f"\n🚀 Launching {option['name']}...")
    print(f"📝 {option['description']}")
    
    if choice in ["4", "5", "6", "7"]:
        # These are info commands, not interface launches
        try:
            subprocess.run(option['command'].split(), check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Error: {e}")
        return
    
    # Launch interface
    try:
        print("\n⏳ Starting interface...")
        print("💡 Tip: Press Ctrl+C to stop the interface")
        print("-" * 50)
        
        # Launch the interface
        subprocess.run(option['command'].split(), check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching interface: {e}")
        print("\n💡 Try running manually:")
        print(option['command'])
    except KeyboardInterrupt:
        print("\n👋 Interface stopped by user")

def show_quick_tips():
    """Show quick tips for getting started."""
    tips = [
        "🎯 Start with the User-Friendly Interface for the best experience",
        "📚 Check out the tutorials to learn advanced features",
        "⚡ Use the performance test to optimize your setup",
        "🔧 Adjust settings based on your hardware capabilities",
        "💾 Save your work regularly using project management",
        "📱 The interface works great on mobile devices too!",
        "🆘 Use the help system for detailed guidance"
    ]
    
    print("\n💡 Quick Tips:")
    print("-" * 30)
    for tip in tips:
        print(f"• {tip}")

def show_system_info():
    """Show system information."""
    import platform
    import psutil
    
    print("\n💻 System Information:")
    print("-" * 30)
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")
    print(f"CPU: {psutil.cpu_count()} cores")
    print(f"Memory: {psutil.virtual_memory().total // (1024**3)} GB")
    
    # Check for GPU
    try:
        import torch
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA: {torch.version.cuda}")
        else:
            print("GPU: Not available (CPU only)")
    except:
        print("GPU: Unable to detect")

def main():
    """Main quick start function."""
    print_welcome_banner()
    
    # Show system info
    show_system_info()
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Please install requirements and try again.")
        sys.exit(1)
    
    # Show quick tips
    show_quick_tips()
    
    # Main loop
    while True:
        try:
            # Show options
            options = show_interface_options()
            
            # Get user choice
            choice = get_user_choice(options)
            
            # Launch interface
            launch_interface(choice, options)
            
            # Ask if user wants to continue
            if choice != "0":
                print("\n" + "="*50)
                continue_choice = input("Would you like to try another option? (y/n): ").strip().lower()
                if continue_choice not in ['y', 'yes']:
                    print("👋 Thanks for using Video-OpusClip!")
                    break
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            print("Please try again or contact support.")

if __name__ == "__main__":
    main() 