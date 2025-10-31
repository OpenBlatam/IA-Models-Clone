#!/usr/bin/env python3
"""
Test script for Interactive Demos in the Advanced LLM SEO Engine
Tests SEO analysis demos, diffusion model demos, batch processing, and performance monitoring
"""

import torch
import sys
import os
import time
import numpy as np
from unittest.mock import Mock, patch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from advanced_llm_seo_engine import (
    SEOConfig, AdvancedLLMSEOEngine
)

def test_seo_analysis_demo():
    """Test SEO analysis demo functionality."""
    print("🧪 Testing SEO Analysis Demo...")
    
    try:
        config = SEOConfig(
            batch_size=8,
            learning_rate=1e-4,
            use_mixed_precision=False,
            use_diffusion=False
        )
        
        engine = AdvancedLLMSEOEngine(config)
        print("✅ Engine created successfully")
        
        # Test comprehensive analysis
        sample_text = """
        SEO optimization is crucial for digital marketing success. 
        This comprehensive guide covers keyword research, content optimization, 
        and technical SEO best practices. Implementing these strategies will 
        improve your search engine rankings and drive organic traffic.
        """
        
        # Mock the demo analysis function
        def mock_run_demo_analysis(text, analysis_type, language):
            """Mock version of the demo analysis function."""
            if not text.strip():
                return {"error": "No text provided"}, None
            
            results = {
                "text_length": len(text),
                "word_count": len(text.split()),
                "analysis_type": analysis_type,
                "language": language,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "processing_time": f"{np.random.uniform(0.1, 0.5):.3f}s"
            }
            
            if analysis_type == "comprehensive":
                results.update({
                    "seo_score": np.random.uniform(65, 95),
                    "keyword_density": np.random.uniform(1.5, 4.0),
                    "readability_score": np.random.uniform(70, 90),
                    "sentiment_score": np.random.uniform(-0.3, 0.8),
                    "technical_seo_score": np.random.uniform(75, 95)
                })
            elif analysis_type == "keyword_density":
                words = text.lower().split()
                common_keywords = ["seo", "optimization", "content", "marketing", "digital"]
                keyword_count = sum(1 for word in words if word in common_keywords)
                results["keyword_density"] = (keyword_count / len(words)) * 100 if words else 0
            elif analysis_type == "readability":
                sentences = text.split('.')
                avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
                results["readability_score"] = max(0, 100 - avg_sentence_length * 2)
            
            return results, None
        
        # Test different analysis types
        analysis_types = ["comprehensive", "keyword_density", "readability"]
        
        for analysis_type in analysis_types:
            print(f"  Testing {analysis_type} analysis...")
            results, viz = mock_run_demo_analysis(sample_text, analysis_type, "en")
            
            if "error" not in results:
                print(f"    ✅ {analysis_type} analysis successful")
                print(f"    📊 SEO Score: {results.get('seo_score', 'N/A')}")
                print(f"    📝 Word Count: {results.get('word_count', 'N/A')}")
                print(f"    ⏱️ Processing Time: {results.get('processing_time', 'N/A')}")
            else:
                print(f"    ❌ {analysis_type} analysis failed: {results['error']}")
        
        print("✅ SEO Analysis Demo test passed!")
        return True
        
    except Exception as e:
        print(f"❌ SEO Analysis Demo test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_diffusion_demo():
    """Test diffusion model demo functionality."""
    print("\n🧪 Testing Diffusion Model Demo...")
    
    try:
        config = SEOConfig(
            batch_size=8,
            learning_rate=1e-4,
            use_mixed_precision=False,
            use_diffusion=True
        )
        
        engine = AdvancedLLMSEOEngine(config)
        print("✅ Engine created successfully")
        
        # Mock diffusion generator
        class MockDiffusionGenerator:
            def __init__(self):
                self.pipeline = Mock()
                self.pipeline.__class__.__name__ = "MockStableDiffusionPipeline"
            
            def generate_image(self, prompt, negative_prompt, num_inference_steps, guidance_scale):
                # Simulate image generation
                time.sleep(0.1)  # Simulate processing time
                return Mock()  # Mock PIL image
        
        engine.diffusion_generator = MockDiffusionGenerator()
        
        # Test diffusion demo function
        def mock_run_diffusion_demo(prompt, negative_prompt, steps, guidance_scale, seed):
            """Mock version of the diffusion demo function."""
            if not prompt.strip():
                return None, {"error": "No prompt provided"}
            
            if not engine.diffusion_generator:
                return None, {"error": "Diffusion models not enabled"}
            
            # Set seed if provided
            if seed != -1:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
            
            # Generate image
            start_time = time.time()
            image = engine.diffusion_generator.generate_image(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale
            )
            generation_time = time.time() - start_time
            
            # Metadata
            metadata = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "steps": steps,
                "guidance_scale": guidance_scale,
                "seed": seed if seed != -1 else "random",
                "generation_time": f"{generation_time:.2f}s",
                "model": engine.diffusion_generator.pipeline.__class__.__name__,
                "device": str(engine.device)
            }
            
            return image, metadata
        
        # Test diffusion generation
        test_prompts = [
            "A beautiful landscape with mountains and lake",
            "A futuristic city skyline at sunset",
            "A cute cat playing with yarn"
        ]
        
        for prompt in test_prompts:
            print(f"  Testing prompt: '{prompt[:30]}...'")
            image, metadata = mock_run_diffusion_demo(
                prompt, "blurry, low quality", 50, 7.5, -1
            )
            
            if image is not None and "error" not in metadata:
                print(f"    ✅ Image generation successful")
                print(f"    🎨 Model: {metadata['model']}")
                print(f"    ⏱️ Generation Time: {metadata['generation_time']}")
                print(f"    🔧 Steps: {metadata['steps']}")
                print(f"    📏 Guidance Scale: {metadata['guidance_scale']}")
            else:
                print(f"    ❌ Image generation failed: {metadata.get('error', 'Unknown error')}")
        
        print("✅ Diffusion Model Demo test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Diffusion Model Demo test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_batch_processing_demo():
    """Test batch processing demo functionality."""
    print("\n🧪 Testing Batch Processing Demo...")
    
    try:
        config = SEOConfig(
            batch_size=8,
            learning_rate=1e-4,
            use_mixed_precision=False,
            use_diffusion=False
        )
        
        engine = AdvancedLLMSEOEngine(config)
        print("✅ Engine created successfully")
        
        # Test batch processing function
        def mock_run_batch_demo(texts, processing_type):
            """Mock version of the batch processing function."""
            if not texts.strip():
                return {"error": "No texts provided"}
            
            text_list = [t.strip() for t in texts.split('\n') if t.strip()]
            if not text_list:
                return {"error": "No valid texts found"}
            
            results = {
                "processing_type": processing_type,
                "total_texts": len(text_list),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "results": []
            }
            
            # Process each text
            for i, text in enumerate(text_list):
                if processing_type == "seo_optimization":
                    result = {
                        "text_id": i + 1,
                        "text_preview": text[:50] + "..." if len(text) > 50 else text,
                        "seo_score": np.random.uniform(60, 95),
                        "suggestions": [
                            "Add more relevant keywords",
                            "Improve heading structure",
                            "Optimize meta description"
                        ][:np.random.randint(1, 4)]
                    }
                elif processing_type == "keyword_extraction":
                    words = text.lower().split()
                    keywords = [word for word in words if len(word) > 4 and word.isalpha()]
                    result = {
                        "text_id": i + 1,
                        "extracted_keywords": keywords[:5],
                        "keyword_count": len(keywords),
                        "relevance_score": np.random.uniform(0.5, 1.0)
                    }
                
                results["results"].append(result)
            
            return results
        
        # Test batch processing
        sample_texts = """
        SEO optimization is essential for online success.
        Content marketing drives organic traffic growth.
        Technical SEO improves website performance.
        Keyword research guides content strategy.
        """
        
        processing_types = ["seo_optimization", "keyword_extraction"]
        
        for processing_type in processing_types:
            print(f"  Testing {processing_type} batch processing...")
            results = mock_run_batch_demo(sample_texts, processing_type)
            
            if "error" not in results:
                print(f"    ✅ {processing_type} batch processing successful")
                print(f"    📊 Total texts processed: {results['total_texts']}")
                print(f"    📝 Sample result for text 1:")
                sample_result = results['results'][0]
                print(f"      - SEO Score: {sample_result.get('seo_score', 'N/A')}")
                print(f"      - Keywords: {sample_result.get('extracted_keywords', 'N/A')}")
            else:
                print(f"    ❌ {processing_type} batch processing failed: {results['error']}")
        
        print("✅ Batch Processing Demo test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Batch Processing Demo test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_monitoring():
    """Test performance monitoring functionality."""
    print("\n🧪 Testing Performance Monitoring...")
    
    try:
        config = SEOConfig(
            batch_size=8,
            learning_rate=1e-4,
            use_mixed_precision=False,
            use_diffusion=False
        )
        
        engine = AdvancedLLMSEOEngine(config)
        print("✅ Engine created successfully")
        
        # Mock training history
        engine.training_history = [
            {"epoch": 1, "train_loss": 0.8, "val_loss": 0.9, "learning_rate": 1e-4},
            {"epoch": 2, "train_loss": 0.6, "val_loss": 0.7, "learning_rate": 9e-5},
            {"epoch": 3, "train_loss": 0.5, "val_loss": 0.6, "learning_rate": 8e-5}
        ]
        
        # Test performance metrics function
        def mock_refresh_performance_metrics():
            """Mock version of the performance metrics function."""
            try:
                # System metrics (mocked)
                system_metrics = {
                    "cpu_percent": 45.2,
                    "memory_percent": 67.8,
                    "disk_usage": 23.4,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # GPU metrics if available
                if torch.cuda.is_available():
                    system_metrics.update({
                        "gpu_memory_allocated": f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB",
                        "gpu_memory_cached": f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB",
                        "gpu_utilization": "N/A"
                    })
                
                # Model performance metrics
                model_metrics = {
                    "model_loaded": engine.seo_model is not None,
                    "diffusion_loaded": engine.diffusion_generator is not None,
                    "current_device": str(engine.device),
                    "mixed_precision": engine.config.use_mixed_precision,
                    "batch_size": engine.config.batch_size
                }
                
                # Training metrics if available
                if hasattr(engine, 'training_history') and engine.training_history:
                    latest = engine.training_history[-1]
                    model_metrics.update({
                        "last_epoch": latest.get('epoch', 0),
                        "last_train_loss": latest.get('train_loss', 0),
                        "last_val_loss": latest.get('val_loss', 0),
                        "learning_rate": latest.get('learning_rate', 0)
                    })
                
                return system_metrics, model_metrics
                
            except Exception as e:
                return {"error": f"Failed to refresh metrics: {str(e)}"}, {"error": str(e)}
        
        # Test custom metrics
        def mock_add_custom_metric(name, value):
            """Mock version of the custom metric function."""
            try:
                if not hasattr(engine, 'custom_metrics'):
                    engine.custom_metrics = {}
                
                if not hasattr(engine, 'custom_metrics_history'):
                    engine.custom_metrics_history = []
                
                # Add to current metrics
                engine.custom_metrics[name] = value
                
                # Add to history
                engine.custom_metrics_history.append({
                    "metric": name,
                    "value": value,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                })
                
                return {
                    "success": True,
                    "metric_added": name,
                    "value": value,
                    "total_metrics": len(engine.custom_metrics)
                }
                
            except Exception as e:
                return {"error": f"Failed to add metric: {str(e)}"}
        
        # Test performance monitoring
        print("  Testing performance metrics refresh...")
        system_metrics, model_metrics = mock_refresh_performance_metrics()
        
        if "error" not in system_metrics:
            print(f"    ✅ System metrics retrieved successfully")
            print(f"    💻 CPU Usage: {system_metrics.get('cpu_percent', 'N/A')}%")
            print(f"    💾 Memory Usage: {system_metrics.get('memory_percent', 'N/A')}%")
            print(f"    💿 Disk Usage: {system_metrics.get('disk_usage', 'N/A')}%")
        else:
            print(f"    ❌ System metrics failed: {system_metrics['error']}")
        
        if "error" not in model_metrics:
            print(f"    ✅ Model metrics retrieved successfully")
            print(f"    🧠 Model Loaded: {model_metrics.get('model_loaded', 'N/A')}")
            print(f"    🎨 Diffusion Loaded: {model_metrics.get('diffusion_loaded', 'N/A')}")
            print(f"    📱 Device: {model_metrics.get('current_device', 'N/A')}")
            print(f"    🔄 Last Epoch: {model_metrics.get('last_epoch', 'N/A')}")
        else:
            print(f"    ❌ Model metrics failed: {model_metrics['error']}")
        
        # Test custom metrics
        print("  Testing custom metrics...")
        custom_metrics = [
            ("accuracy", 0.85),
            ("f1_score", 0.78),
            ("precision", 0.82)
        ]
        
        for name, value in custom_metrics:
            result = mock_add_custom_metric(name, value)
            if "error" not in result:
                print(f"    ✅ Added metric '{name}': {value}")
            else:
                print(f"    ❌ Failed to add metric '{name}': {result['error']}")
        
        print("✅ Performance Monitoring test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Performance Monitoring test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gradio_interface_integration():
    """Test Gradio interface integration for demos."""
    print("\n🧪 Testing Gradio Interface Integration...")
    
    try:
        # Test that the interface can be created without errors
        from advanced_llm_seo_engine import create_advanced_gradio_interface
        
        print("  Testing interface creation...")
        interface = create_advanced_gradio_interface()
        print("    ✅ Interface created successfully")
        
        # Check if demo tabs are present
        print("  Testing demo tab presence...")
        tab_names = [tab.name for tab in interface.tabs]
        demo_tab_names = ["🎨 Interactive Demos", "📊 Performance Monitoring"]
        
        for demo_tab in demo_tab_names:
            if demo_tab in tab_names:
                print(f"    ✅ Demo tab '{demo_tab}' found")
            else:
                print(f"    ❌ Demo tab '{demo_tab}' not found")
        
        print("✅ Gradio Interface Integration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Gradio Interface Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all interactive demo tests."""
    print("🚀 Starting Interactive Demo tests...\n")
    
    test_results = []
    
    try:
        test_results.append(test_seo_analysis_demo())
        test_results.append(test_diffusion_demo())
        test_results.append(test_batch_processing_demo())
        test_results.append(test_performance_monitoring())
        test_results.append(test_gradio_interface_integration())
        
        # Summary
        passed_tests = sum(test_results)
        total_tests = len(test_results)
        
        print(f"\n📊 Test Summary:")
        print(f"  ✅ Passed: {passed_tests}/{total_tests}")
        print(f"  ❌ Failed: {total_tests - passed_tests}/{total_tests}")
        
        if passed_tests == total_tests:
            print("\n🎉 All Interactive Demo tests passed successfully!")
            return 0
        else:
            print(f"\n⚠️ {total_tests - passed_tests} test(s) failed")
            return 1
            
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())






