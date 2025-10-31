#!/usr/bin/env python3
"""
Quick Start Transformers for Video-OpusClip

This script demonstrates how to quickly get started with Transformers
in the Video-OpusClip system for text generation, analysis, and processing.
"""

import torch
import time
import logging
from typing import List, Dict, Optional
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def quick_start_basic_transformers():
    """Basic Transformers setup and usage."""
    
    print("🚀 Quick Start: Basic Transformers")
    print("=" * 50)
    
    try:
        # Import Transformers
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        
        print("✅ Transformers imported successfully")
        
        # Test basic model loading
        print("📥 Loading GPT-2 model...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        
        # Add padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("✅ Model loaded successfully")
        
        # Test basic text generation
        prompt = "Generate a viral caption for a funny cat video:"
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=50,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"📝 Generated text: {generated_text}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Transformers not installed: {e}")
        print("Install with: pip install transformers")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def quick_start_pipelines():
    """Quick start with Transformers pipelines."""
    
    print("\n🚀 Quick Start: Transformers Pipelines")
    print("=" * 50)
    
    try:
        from transformers import pipeline
        
        # Text generation pipeline
        print("📝 Testing text generation pipeline...")
        generator = pipeline("text-generation", model="gpt2")
        
        result = generator(
            "Create a viral TikTok caption:",
            max_length=30,
            num_return_sequences=1
        )
        print(f"Generated: {result[0]['generated_text']}")
        
        # Sentiment analysis pipeline
        print("\n😊 Testing sentiment analysis pipeline...")
        classifier = pipeline("sentiment-analysis")
        
        sentiments = classifier([
            "This video is absolutely amazing!",
            "I don't like this content.",
            "It's okay, nothing special."
        ])
        
        for text, sentiment in zip([
            "This video is absolutely amazing!",
            "I don't like this content.",
            "It's okay, nothing special."
        ], sentiments):
            print(f"'{text}' -> {sentiment['label']} ({sentiment['score']:.3f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        return False

def quick_start_video_captions():
    """Quick start for video caption generation."""
    
    print("\n🚀 Quick Start: Video Caption Generation")
    print("=" * 50)
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Load model
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Video descriptions
        video_descriptions = [
            "A cat playing with a laser pointer",
            "A dog learning to skateboard",
            "A bird singing beautifully",
            "A fish doing tricks"
        ]
        
        print("🎬 Generating captions for videos...")
        
        for description in video_descriptions:
            prompt = f"Generate a viral caption for: {description}"
            inputs = tokenizer(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=40,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            caption = tokenizer.decode(outputs[0], skip_special_tokens=True)
            caption = caption.replace(prompt, "").strip()
            
            print(f"📹 {description}")
            print(f"   Caption: {caption}")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Caption generation error: {e}")
        return False

def quick_start_optimization():
    """Quick start with optimization techniques."""
    
    print("\n🚀 Quick Start: Transformers Optimization")
    print("=" * 50)
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Load model with optimization
        print("📥 Loading optimized model...")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained(
            "gpt2",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Enable optimizations
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        
        if hasattr(model, 'enable_attention_slicing'):
            model.enable_attention_slicing()
        
        print("✅ Model optimized successfully")
        
        # Test performance
        print("⚡ Testing performance...")
        start_time = time.time()
        
        prompt = "Generate a caption:"
        inputs = tokenizer(prompt, return_tensors="pt")
        
        if device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=30,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generation_time = time.time() - start_time
        print(f"⏱️  Generation time: {generation_time:.3f} seconds")
        
        return True
        
    except Exception as e:
        print(f"❌ Optimization error: {e}")
        return False

def quick_start_integration():
    """Quick start with Video-OpusClip integration."""
    
    print("\n🚀 Quick Start: Video-OpusClip Integration")
    print("=" * 50)
    
    try:
        # Import Video-OpusClip components
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Simulate Video-OpusClip integration
        class VideoOpusClipTransformers:
            def __init__(self):
                self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
                self.model = AutoModelForCausalLM.from_pretrained("gpt2")
                
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            
            def generate_viral_caption(self, video_description: str, platform: str = "tiktok"):
                """Generate viral caption for video."""
                
                prompts = {
                    "tiktok": f"Create a viral TikTok caption for: {video_description}",
                    "youtube": f"Write engaging YouTube title for: {video_description}",
                    "instagram": f"Generate Instagram caption for: {video_description}"
                }
                
                prompt = prompts.get(platform, f"Generate caption for: {video_description}")
                
                inputs = self.tokenizer(prompt, return_tensors="pt")
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=50,
                        temperature=0.8,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                caption = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                return caption.replace(prompt, "").strip()
        
        # Create instance
        video_opus = VideoOpusClipTransformers()
        
        # Test integration
        test_videos = [
            ("Cat playing with laser", "tiktok"),
            ("Dog learning tricks", "youtube"),
            ("Bird singing", "instagram")
        ]
        
        print("🎬 Testing Video-OpusClip integration...")
        
        for description, platform in test_videos:
            caption = video_opus.generate_viral_caption(description, platform)
            print(f"📹 {description} ({platform})")
            print(f"   Caption: {caption}")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Integration error: {e}")
        return False

def main():
    """Main function to run all quick start examples."""
    
    print("🎯 Transformers Quick Start for Video-OpusClip")
    print("=" * 60)
    print()
    
    # Check PyTorch
    print(f"🔥 PyTorch version: {torch.__version__}")
    print(f"🔥 CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Run all examples
    results = []
    
    results.append(("Basic Transformers", quick_start_basic_transformers()))
    results.append(("Pipelines", quick_start_pipelines()))
    results.append(("Video Captions", quick_start_video_captions()))
    results.append(("Optimization", quick_start_optimization()))
    results.append(("Integration", quick_start_integration()))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Quick Start Results Summary")
    print("=" * 60)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Transformers features working correctly!")
        print("\nNext steps:")
        print("1. Read TRANSFORMERS_GUIDE.md for detailed usage")
        print("2. Integrate with your Video-OpusClip pipeline")
        print("3. Experiment with different models and parameters")
    else:
        print("⚠️  Some features need attention. Check the error messages above.")
        print("\nTroubleshooting:")
        print("1. Install missing dependencies: pip install transformers")
        print("2. Check GPU drivers and CUDA installation")
        print("3. Review TRANSFORMERS_GUIDE.md for troubleshooting")

if __name__ == "__main__":
    main() 