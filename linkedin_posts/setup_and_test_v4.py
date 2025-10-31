#!/usr/bin/env python3
"""
🚀 ENHANCED LINKEDIN OPTIMIZER v4.0 - SETUP & TESTING SCRIPT
============================================================

This script will:
1. Install all required dependencies
2. Download necessary AI models
3. Run comprehensive system tests
4. Provide a demo of all v4.0 features
5. Generate a system health report

Usage: python setup_and_test_v4.py
"""

import os
import sys
import subprocess
import asyncio
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_command(command: str, description: str) -> bool:
    """Run a shell command and return success status."""
    print(f"\n🔧 {description}...")
    print(f"   Command: {command}")
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            check=True
        )
        print(f"   ✅ Success: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Error: {e.stderr.strip()}")
        return False

def check_python_version() -> bool:
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    
    if sys.version_info < (3, 8):
        print("   ❌ Python 3.8+ required. Current version:", sys.version)
        return False
    
    print(f"   ✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def install_dependencies() -> bool:
    """Install all required dependencies."""
    print("\n📦 Installing dependencies...")
    
    # Check if requirements file exists
    requirements_file = "requirements_v4.txt"
    if not os.path.exists(requirements_file):
        print(f"   ❌ Requirements file {requirements_file} not found")
        return False
    
    # Install dependencies
    success = run_command(
        f"pip install -r {requirements_file}",
        "Installing Python packages"
    )
    
    if not success:
        print("   ⚠️  Some packages may have failed to install. Continuing...")
    
    return True

def download_ai_models() -> bool:
    """Download necessary AI models."""
    print("\n🤖 Downloading AI models...")
    
    models_to_download = [
        ("spacy", "python -m spacy download en_core_web_sm"),
        ("sentence-transformers", "python -c \"from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')\""),
        ("transformers", "python -c \"from transformers import pipeline; pipeline('zero-shot-classification')\""),
    ]
    
    success_count = 0
    for model_name, command in models_to_download:
        if run_command(command, f"Downloading {model_name} model"):
            success_count += 1
    
    print(f"   📊 Downloaded {success_count}/{len(models_to_download)} models successfully")
    return success_count > 0

def run_system_tests() -> bool:
    """Run comprehensive system tests."""
    print("\n🧪 Running system tests...")
    
    # Test imports
    test_imports = [
        ("ai_content_intelligence_v4", "AI Content Intelligence"),
        ("real_time_analytics_v4", "Real-Time Analytics"),
        ("security_compliance_v4", "Security & Compliance"),
        ("enhanced_system_integration_v4", "System Integration"),
    ]
    
    success_count = 0
    for module_name, description in test_imports:
        try:
            __import__(module_name)
            print(f"   ✅ {description} module imported successfully")
            success_count += 1
        except ImportError as e:
            print(f"   ❌ {description} module import failed: {e}")
    
    return success_count == len(test_imports)

async def run_demo() -> bool:
    """Run a comprehensive demo of all v4.0 features."""
    print("\n🎯 Running v4.0 system demo...")
    
    try:
        # Import the enhanced system
        from enhanced_system_integration_v4 import EnhancedLinkedInOptimizer
        
        # Initialize the system
        print("   🔄 Initializing Enhanced LinkedIn Optimizer v4.0...")
        optimizer = EnhancedLinkedInOptimizer()
        
        # Test content optimization
        print("   📝 Testing content optimization...")
        sample_content = """
        Excited to share our latest breakthrough in AI-powered content optimization! 
        Our new system leverages advanced machine learning to analyze engagement patterns 
        and predict content performance with unprecedented accuracy.
        
        Key features:
        • Real-time sentiment analysis
        • Predictive engagement forecasting
        • Advanced security & compliance
        • Multi-platform integration
        
        #AI #ContentOptimization #MachineLearning #Innovation
        """
        
        # Run optimization
        start_time = time.time()
        result = await optimizer.optimize_content(
            content=sample_content,
            platform="linkedin",
            target_audience="tech_professionals",
            optimization_goals=["engagement", "reach", "professional_branding"]
        )
        processing_time = time.time() - start_time
        
        print(f"   ⚡ Content optimized in {processing_time:.2f} seconds")
        print(f"   📊 Optimization score: {result.get('optimization_score', 'N/A')}")
        print(f"   🎯 Sentiment: {result.get('sentiment_analysis', {}).get('overall_sentiment', 'N/A')}")
        print(f"   📈 Predicted engagement: {result.get('engagement_prediction', {}).get('predicted_level', 'N/A')}")
        
        # Test batch processing
        print("   🔄 Testing batch processing...")
        batch_results = await optimizer.batch_optimize([
            {"content": "Sample post 1", "platform": "linkedin"},
            {"content": "Sample post 2", "platform": "linkedin"},
        ])
        
        print(f"   📦 Batch processed {len(batch_results)} posts successfully")
        
        # Test system health
        print("   💓 Testing system health monitoring...")
        health_status = await optimizer.get_system_health()
        print(f"   🟢 System status: {health_status.get('status', 'N/A')}")
        print(f"   💾 Memory usage: {health_status.get('memory_usage_mb', 'N/A')} MB")
        print(f"   🔥 CPU usage: {health_status.get('cpu_usage_percent', 'N/A')}%")
        
        # Graceful shutdown
        print("   🛑 Performing graceful shutdown...")
        await optimizer.shutdown()
        
        print("   ✅ Demo completed successfully!")
        return True
        
    except Exception as e:
        print(f"   ❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_system_report() -> Dict[str, Any]:
    """Generate a comprehensive system report."""
    print("\n📋 Generating system report...")
    
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "system_version": "v4.0",
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": sys.platform,
        "working_directory": os.getcwd(),
        "files_present": {},
        "dependencies": {},
        "ai_models": {},
        "test_results": {},
        "recommendations": []
    }
    
    # Check file presence
    v4_files = [
        "ai_content_intelligence_v4.py",
        "real_time_analytics_v4.py", 
        "security_compliance_v4.py",
        "enhanced_system_integration_v4.py",
        "requirements_v4.txt",
        "README_v4_ENHANCEMENTS.md"
    ]
    
    for file in v4_files:
        report["files_present"][file] = os.path.exists(file)
    
    # Check dependencies
    try:
        import torch
        report["dependencies"]["torch"] = torch.__version__
    except ImportError:
        report["dependencies"]["torch"] = "Not installed"
    
    try:
        import transformers
        report["dependencies"]["transformers"] = transformers.__version__
    except ImportError:
        report["dependencies"]["transformers"] = "Not installed"
    
    try:
        import spacy
        report["dependencies"]["spacy"] = spacy.__version__
    except ImportError:
        report["dependencies"]["spacy"] = "Not installed"
    
    # Generate recommendations
    missing_files = [f for f, present in report["files_present"].items() if not present]
    if missing_files:
        report["recommendations"].append(f"Missing files: {', '.join(missing_files)}")
    
    missing_deps = [d for d, version in report["dependencies"].items() if version == "Not installed"]
    if missing_deps:
        report["recommendations"].append(f"Install missing dependencies: {', '.join(missing_deps)}")
    
    if not report["recommendations"]:
        report["recommendations"].append("System is ready for production use!")
    
    return report

def save_report(report: Dict[str, Any]) -> None:
    """Save the system report to a file."""
    report_file = "v4_system_report.json"
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"   💾 System report saved to {report_file}")

def print_summary(report: Dict[str, Any]) -> None:
    """Print a summary of the system status."""
    print("\n" + "="*60)
    print("🎯 ENHANCED LINKEDIN OPTIMIZER v4.0 - SETUP SUMMARY")
    print("="*60)
    
    print(f"📅 Setup completed: {report['timestamp']}")
    print(f"🐍 Python version: {report['python_version']}")
    print(f"🖥️  Platform: {report['platform']}")
    
    print(f"\n📁 Files status:")
    for file, present in report["files_present"].items():
        status = "✅" if present else "❌"
        print(f"   {status} {file}")
    
    print(f"\n📦 Dependencies status:")
    for dep, version in report["dependencies"].items():
        status = "✅" if version != "Not installed" else "❌"
        print(f"   {status} {dep}: {version}")
    
    print(f"\n💡 Recommendations:")
    for rec in report["recommendations"]:
        print(f"   • {rec}")
    
    print("\n" + "="*60)

async def main():
    """Main setup and testing function."""
    print("🚀 ENHANCED LINKEDIN OPTIMIZER v4.0 - SETUP & TESTING")
    print("=" * 60)
    
    # Step 1: Check Python version
    if not check_python_version():
        print("\n❌ Setup failed: Incompatible Python version")
        return False
    
    # Step 2: Install dependencies
    if not install_dependencies():
        print("\n⚠️  Dependencies installation had issues, but continuing...")
    
    # Step 3: Download AI models
    if not download_ai_models():
        print("\n⚠️  AI model download had issues, but continuing...")
    
    # Step 4: Run system tests
    if not run_system_tests():
        print("\n❌ System tests failed")
        return False
    
    # Step 5: Run demo
    if not await run_demo():
        print("\n❌ Demo failed")
        return False
    
    # Step 6: Generate and save report
    report = generate_system_report()
    save_report(report)
    
    # Step 7: Print summary
    print_summary(report)
    
    print("\n🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("Your Enhanced LinkedIn Optimizer v4.0 is ready to use!")
    
    return True

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Setup failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
