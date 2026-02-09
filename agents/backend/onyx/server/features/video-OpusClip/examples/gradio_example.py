#!/usr/bin/env python3
"""
Gradio Integration Example for Video-OpusClip

This example demonstrates how to use the Gradio interface with different
configurations and showcases various features.
"""

import os
import sys
import asyncio
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

def example_basic_usage():
    """Example of basic Gradio usage."""
    print("🚀 Example 1: Basic Gradio Usage")
    
    try:
        from gradio_integration import launch_gradio
        
        # Launch with basic settings
        launch_gradio(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            debug=False
        )
    except Exception as e:
        print(f"❌ Error: {e}")

def example_simple_interface():
    """Example using the simple interface."""
    print("🎯 Example 2: Simple Interface")
    
    try:
        from gradio_demo import demo
        
        # Launch simple interface
        demo.launch(
            server_name="127.0.0.1",
            server_port=7861,
            share=False,
            debug=True
        )
    except Exception as e:
        print(f"❌ Error: {e}")

def example_custom_configuration():
    """Example with custom configuration."""
    print("⚙️ Example 3: Custom Configuration")
    
    # Set custom environment variables
    os.environ["MAX_WORKERS"] = "4"
    os.environ["BATCH_SIZE"] = "8"
    os.environ["USE_GPU"] = "false"  # Force CPU mode
    os.environ["ENABLE_CACHING"] = "true"
    os.environ["LOG_LEVEL"] = "DEBUG"
    
    try:
        from gradio_integration import launch_gradio
        
        # Launch with custom settings
        launch_gradio(
            server_name="0.0.0.0",
            server_port=7862,
            share=True,  # Create public link
            debug=True
        )
    except Exception as e:
        print(f"❌ Error: {e}")

def example_programmatic_interface():
    """Example of programmatic interface usage."""
    print("🔧 Example 4: Programmatic Interface")
    
    try:
        from gradio_integration import GradioInterface
        
        # Create interface instance
        interface = GradioInterface()
        
        # Access components
        print(f"✅ Interface created successfully")
        print(f"📊 Performance monitor: {interface.performance_monitor}")
        print(f"💾 Cache: {interface.cache}")
        print(f"🎥 Video processor: {interface.video_processor}")
        
        # Get current metrics
        metrics = interface.performance_monitor.get_metrics()
        print(f"📈 Current metrics: {metrics}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def example_batch_processing():
    """Example of batch processing with Gradio."""
    print("📦 Example 5: Batch Processing")
    
    try:
        from gradio_integration import GradioInterface
        import numpy as np
        
        # Create interface
        interface = GradioInterface()
        
        # Simulate batch processing
        videos = [
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8),
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8),
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        ]
        
        results = []
        for i, video in enumerate(videos):
            print(f"Processing video {i+1}/{len(videos)}")
            
            # Process video
            result = interface._process_video(
                video_input=video,
                url_input="",
                target_duration=30.0,
                quality_preset="fast",
                enable_audio=False,
                enable_subtitles=False
            )
            
            results.append(result)
        
        print(f"✅ Batch processing completed: {len(results)} videos")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def example_viral_analysis():
    """Example of viral analysis."""
    print("📈 Example 6: Viral Analysis")
    
    try:
        from gradio_integration import GradioInterface
        
        # Create interface
        interface = GradioInterface()
        
        # Test content
        test_content = "A funny cat video with amazing tricks and cute moments"
        
        # Analyze viral potential
        viral_score, engagement, recommendations, metrics = interface._analyze_viral_potential(
            content=test_content,
            content_type="video",
            platform="tiktok"
        )
        
        print(f"🎯 Viral Score: {viral_score:.2f}")
        print(f"📊 Predicted Engagement: {engagement:.2f}")
        print(f"💡 Recommendations: {recommendations}")
        print(f"📈 Metrics: {metrics}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def example_performance_monitoring():
    """Example of performance monitoring."""
    print("⚡ Example 7: Performance Monitoring")
    
    try:
        from gradio_integration import GradioInterface
        import time
        
        # Create interface
        interface = GradioInterface()
        
        # Monitor performance over time
        for i in range(5):
            metrics = interface._get_performance_metrics()
            
            print(f"📊 Metrics at {i+1}/5:")
            print(f"   CPU Usage: {metrics[0].get('cpu_usage', 0):.1f}%")
            print(f"   Memory Usage: {metrics[0].get('memory_usage', 0):.1f}%")
            print(f"   GPU Usage: {metrics[0].get('gpu_usage', 0):.1f}%")
            
            time.sleep(2)
        
        print("✅ Performance monitoring completed")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def example_configuration_management():
    """Example of configuration management."""
    print("⚙️ Example 8: Configuration Management")
    
    try:
        from gradio_integration import GradioInterface
        
        # Create interface
        interface = GradioInterface()
        
        # Save current settings
        interface._save_settings(
            max_workers=8,
            batch_size=16,
            enable_gpu=True,
            enable_caching=True,
            log_level="INFO",
            cache_size=2000,
            timeout=60.0
        )
        
        print("💾 Settings saved successfully")
        
        # Reset to defaults
        defaults = interface._reset_settings()
        print(f"🔄 Reset to defaults: {defaults}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Run all examples."""
    print("🎬 Video-OpusClip Gradio Integration Examples")
    print("=" * 50)
    
    examples = [
        ("Basic Usage", example_basic_usage),
        ("Simple Interface", example_simple_interface),
        ("Custom Configuration", example_custom_configuration),
        ("Programmatic Interface", example_programmatic_interface),
        ("Batch Processing", example_batch_processing),
        ("Viral Analysis", example_viral_analysis),
        ("Performance Monitoring", example_performance_monitoring),
        ("Configuration Management", example_configuration_management),
    ]
    
    for name, example_func in examples:
        print(f"\n{name}:")
        print("-" * 30)
        
        try:
            example_func()
        except KeyboardInterrupt:
            print("⏹️ Example interrupted by user")
            break
        except Exception as e:
            print(f"❌ Example failed: {e}")
        
        print()

if __name__ == "__main__":
    # Check if running in interactive mode
    if len(sys.argv) > 1:
        # Run specific example
        example_name = sys.argv[1].lower()
        
        if example_name == "basic":
            example_basic_usage()
        elif example_name == "simple":
            example_simple_interface()
        elif example_name == "custom":
            example_custom_configuration()
        elif example_name == "programmatic":
            example_programmatic_interface()
        elif example_name == "batch":
            example_batch_processing()
        elif example_name == "viral":
            example_viral_analysis()
        elif example_name == "performance":
            example_performance_monitoring()
        elif example_name == "config":
            example_configuration_management()
        else:
            print(f"❌ Unknown example: {example_name}")
            print("Available examples: basic, simple, custom, programmatic, batch, viral, performance, config")
    else:
        # Run all examples
        main() 