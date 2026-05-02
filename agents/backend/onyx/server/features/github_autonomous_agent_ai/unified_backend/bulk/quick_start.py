"""
BUL Quick Start Script
======================

Quick start script to get the BUL system running immediately.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

def setup_environment():
    """Setup basic environment for BUL."""
    print("🔧 Setting up BUL environment...")
    
    # Create necessary directories
    directories = [
        "generated_documents",
        "generated_documents/marketing",
        "generated_documents/sales", 
        "generated_documents/operations",
        "generated_documents/hr",
        "generated_documents/finance",
        "generated_documents/legal",
        "generated_documents/technical",
        "generated_documents/content",
        "generated_documents/strategy",
        "generated_documents/customer_service"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ Created directory: {directory}")
    
    # Check for environment file
    env_file = Path(".env")
    if not env_file.exists():
        print("   ⚠️  .env file not found. Please create one from env_example.txt")
        print("   📝 Copy env_example.txt to .env and add your OpenRouter API key")
    else:
        print("   ✅ Environment file found")
    
    print("   🎯 Environment setup complete!")

def check_dependencies():
    """Check if required dependencies are installed."""
    print("\n📦 Checking dependencies...")
    
    required_packages = [
        "fastapi",
        "uvicorn", 
        "langchain",
        "langchain-openai",
        "pydantic"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n   🔧 Install missing packages:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    print("   🎯 All dependencies are installed!")
    return True

async def quick_demo():
    """Run a quick demonstration of BUL."""
    print("\n🚀 Running BUL Quick Demo...")
    
    try:
        from .config.bul_config import BULConfig
        from .core.bul_engine import BULEngine
        
        # Initialize system
        config = BULConfig()
        engine = BULEngine(config)
        
        print("   ✅ BUL Engine initialized")
        
        # Submit a test query
        test_query = "Create a simple marketing strategy for a small business"
        print(f"   📝 Submitting test query: {test_query}")
        
        task_id = await engine.submit_query(test_query, priority=1)
        print(f"   ✅ Query submitted with task ID: {task_id}")
        
        # Get system stats
        stats = engine.get_processing_stats()
        print(f"   📊 System Stats:")
        print(f"      - Total Tasks: {stats['total_tasks']}")
        print(f"      - Enabled Areas: {len(stats['enabled_areas'])}")
        print(f"      - Processing: {stats['is_processing']}")
        
        print("   🎯 Quick demo completed successfully!")
        
    except Exception as e:
        print(f"   ❌ Demo failed: {e}")
        print("   💡 Make sure you have set up your OpenRouter API key")

def show_usage_instructions():
    """Show usage instructions."""
    print("\n📖 Usage Instructions:")
    print("=" * 50)
    
    print("\n1. 🚀 Start the full system:")
    print("   python main.py --mode full")
    
    print("\n2. 🌐 Start only the API server:")
    print("   python main.py --mode api --port 8000")
    
    print("\n3. ⚙️  Start only the processor:")
    print("   python main.py --mode processor")
    
    print("\n4. 🧪 Run the demo:")
    print("   python demo.py")
    
    print("\n5. 📚 API Documentation:")
    print("   http://localhost:8000/docs")
    
    print("\n6. 🔍 Health Check:")
    print("   http://localhost:8000/health")
    
    print("\n7. 📊 System Stats:")
    print("   http://localhost:8000/stats")

def main():
    """Main quick start function."""
    print("🎯 BUL - Business Unlimited Quick Start")
    print("=" * 50)
    
    # Setup environment
    setup_environment()
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies and try again.")
        return
    
    # Run quick demo
    try:
        asyncio.run(quick_demo())
    except Exception as e:
        print(f"\n⚠️  Demo failed: {e}")
        print("   This is normal if OpenRouter API key is not configured.")
    
    # Show usage instructions
    show_usage_instructions()
    
    print("\n🎉 BUL Quick Start completed!")
    print("   The system is ready to use. Check the README.md for detailed instructions.")

if __name__ == "__main__":
    main()

