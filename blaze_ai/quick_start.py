"""
Quick start guide and examples for Blaze AI module.
"""
from __future__ import annotations

import asyncio
import yaml
from pathlib import Path
from typing import Dict, Any

from .core.interfaces import CoreConfig
from .gradio import (
    create_blaze_ai_interface,
    create_text_generation_demo,
    create_image_generation_demo,
    create_model_comparison_demo,
    create_training_visualization_demo,
    create_performance_analysis_demo,
    create_error_analysis_demo,
    GradioLauncher
)

async def quick_demo():
    """Quick demonstration of Blaze AI capabilities."""
    print("🚀 Blaze AI Quick Demo")
    print("=" * 50)
    
    # Initialize with basic configuration
    config = CoreConfig(
        system_mode="development",
        log_level="INFO",
        enable_async=True,
        max_concurrent_requests=10
    )
    
    # Create Blaze AI instance
    from . import create_modular_ai
    ai = create_modular_ai(config)
    
    # Demo 1: Health Status
    print("\n1. 📊 System Health Check")
    health = await ai.get_health_status()
    print(f"   Status: {health['status']}")
    print(f"   Components: {len(health['components'])}")
    
    # Demo 2: Text Generation
    print("\n2. 📝 Text Generation")
    try:
        result = await ai.generate_text(
            prompt="The future of artificial intelligence",
            max_length=100,
            temperature=0.7
        )
        print(f"   Generated: {result['text'][:100]}...")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Demo 3: SEO Analysis
    print("\n3. 🔍 SEO Analysis")
    try:
        seo_result = await ai.analyze_seo(
            content="This is a sample content for SEO analysis.",
            keywords=["AI", "technology", "future"]
        )
        print(f"   Score: {seo_result.get('score', 'N/A')}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Demo 4: Brand Voice Application
    print("\n4. 🎨 Brand Voice Application")
    try:
        brand_result = await ai.apply_brand_voice(
            brand_name="TechCorp",
            action="apply",
            content="We are a technology company."
        )
        print(f"   Result: {brand_result['content'][:100]}...")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Demo 5: Content Generation
    print("\n5. ✍️ Content Generation")
    try:
        content_result = await ai.generate_content(
            content_type="blog_post",
            topic="Machine Learning Trends",
            length="medium"
        )
        print(f"   Generated: {content_result['content'][:100]}...")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n✅ Quick demo completed!")

def show_gradio_demos():
    """Show how to use Gradio demos."""
    print("\n🎨 Gradio Demo Examples")
    print("=" * 50)
    
    # Example 1: Main Interface
    print("\n1. 🏠 Main Blaze AI Interface")
    print("   ```python")
    print("   from blaze_ai.gradio import create_blaze_ai_interface")
    print("   ")
    print("   interface = create_blaze_ai_interface()")
    print("   interface.launch(port=7860)")
    print("   ```")
    
    # Example 2: Text Generation Demo
    print("\n2. 📝 Text Generation Demo")
    print("   ```python")
    print("   from blaze_ai.gradio import create_text_generation_demo")
    print("   ")
    print("   demo = create_text_generation_demo()")
    print("   demo.launch(port=7861)")
    print("   ```")
    
    # Example 3: Image Generation Demo
    print("\n3. 🎨 Image Generation Demo")
    print("   ```python")
    print("   from blaze_ai.gradio import create_image_generation_demo")
    print("   ")
    print("   demo = create_image_generation_demo()")
    print("   demo.launch(port=7862)")
    print("   ```")
    
    # Example 4: Model Comparison Demo
    print("\n4. 🔍 Model Comparison Demo")
    print("   ```python")
    print("   from blaze_ai.gradio import create_model_comparison_demo")
    print("   ")
    print("   demo = create_model_comparison_demo()")
    print("   demo.launch(port=7863)")
    print("   ```")
    
    # Example 5: Training Visualization Demo
    print("\n5. 📊 Training Visualization Demo")
    print("   ```python")
    print("   from blaze_ai.gradio import create_training_visualization_demo")
    print("   ")
    print("   demo = create_training_visualization_demo()")
    print("   demo.launch(port=7864)")
    print("   ```")
    
    # Example 6: Performance Analysis Demo
    print("\n6. ⚡ Performance Analysis Demo")
    print("   ```python")
    print("   from blaze_ai.gradio import create_performance_analysis_demo")
    print("   ")
    print("   demo = create_performance_analysis_demo()")
    print("   demo.launch(port=7865)")
    print("   ```")
    
    # Example 7: Error Analysis Demo
    print("\n7. 🐛 Error Analysis Demo")
    print("   ```python")
    print("   from blaze_ai.gradio import create_error_analysis_demo")
    print("   ")
    print("   demo = create_error_analysis_demo()")
    print("   demo.launch(port=7866)")
    print("   ```")
    
    # Example 8: Launch All Demos
    print("\n8. 🚀 Launch All Demos")
    print("   ```python")
    print("   from blaze_ai.gradio import GradioLauncher")
    print("   ")
    print("   launcher = GradioLauncher()")
    print("   launcher.launch_all_demos(base_port=7860)")
    print("   ```")
    
    # Example 9: Command Line Usage
    print("\n9. 💻 Command Line Usage")
    print("   ```bash")
    print("   # Launch main interface")
    print("   python -m blaze_ai.gradio.launcher --demo main --port 7860")
    print("   ")
    print("   # Launch text generation demo")
    print("   python -m blaze_ai.gradio.launcher --demo text --port 7861")
    print("   ")
    print("   # Launch all demos")
    print("   python -m blaze_ai.gradio.launcher --demo all --port 7860")
    print("   ")
    print("   # Share demos publicly")
    print("   python -m blaze_ai.gradio.launcher --demo all --port 7860 --share")
    print("   ```")

def show_api_endpoints():
    """Show available API endpoints."""
    print("\n🔌 API Endpoints")
    print("=" * 50)
    
    endpoints = [
        ("POST", "/generate-text", "Generate text with AI"),
        ("POST", "/generate-image", "Generate images with diffusion models"),
        ("POST", "/analyze-seo", "Analyze content for SEO"),
        ("POST", "/apply-brand-voice", "Apply brand voice to content"),
        ("POST", "/generate-content", "Generate various content types"),
        ("GET", "/health", "Get system health status"),
        ("GET", "/metrics", "Get system metrics"),
        ("POST", "/train-model", "Train or fine-tune models"),
        ("GET", "/models", "List available models"),
        ("POST", "/evaluate", "Evaluate model performance")
    ]
    
    for method, endpoint, description in endpoints:
        print(f"   {method:<6} {endpoint:<20} - {description}")

def show_configuration():
    """Show configuration options."""
    print("\n⚙️ Configuration Options")
    print("=" * 50)
    
    config_example = {
        "system_mode": "development",  # or "production"
        "log_level": "INFO",          # DEBUG, INFO, WARNING, ERROR
        "enable_async": True,
        "max_concurrent_requests": 10,
        "enable_circuit_breaker": True,
        "enable_rate_limiting": True,
        "enable_caching": True,
        "models": {
            "llm": {
                "model_name": "gpt2",
                "device": "auto",
                "precision": "float16",
                "mixed_precision": True
            },
            "diffusion": {
                "model_id": "runwayml/stable-diffusion-v1-5",
                "device": "auto",
                "precision": "float16",
                "enable_xformers": True
            }
        }
    }
    
    print("   Example configuration:")
    print("   ```yaml")
    for key, value in config_example.items():
        if isinstance(value, dict):
            print(f"   {key}:")
            for sub_key, sub_value in value.items():
                print(f"     {sub_key}: {sub_value}")
        else:
            print(f"   {key}: {value}")
    print("   ```")

def main():
    """Main function to run the quick start guide."""
    print("🚀 Blaze AI - Quick Start Guide")
    print("=" * 60)
    
    # Run quick demo
    asyncio.run(quick_demo())
    
    # Show examples
    show_gradio_demos()
    show_api_endpoints()
    show_configuration()
    
    print("\n🎉 Quick start guide completed!")
    print("\nNext steps:")
    print("1. Explore the Gradio demos for interactive experimentation")
    print("2. Check the API documentation for integration")
    print("3. Customize configuration for your use case")
    print("4. Deploy to production with Docker")

if __name__ == "__main__":
    main()
