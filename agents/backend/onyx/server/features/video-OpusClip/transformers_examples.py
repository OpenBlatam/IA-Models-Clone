#!/usr/bin/env python3
"""
Transformers Examples for Video-OpusClip

Comprehensive examples demonstrating Transformers usage in the Video-OpusClip system
for text generation, analysis, optimization, and integration.
"""

import torch
import time
import logging
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import json
from dataclasses import dataclass
from functools import lru_cache

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# EXAMPLE 1: BASIC TEXT GENERATION
# =============================================================================

def example_basic_text_generation():
    """Example of basic text generation with Transformers."""
    
    print("🔤 Example 1: Basic Text Generation")
    print("=" * 50)
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Load model
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Generate text
        prompts = [
            "The best way to create viral content is",
            "When making a video, remember to",
            "A successful social media post should"
        ]
        
        for prompt in prompts:
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
            print(f"📝 Prompt: {prompt}")
            print(f"   Generated: {generated_text}")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

# =============================================================================
# EXAMPLE 2: VIDEO CAPTION GENERATION
# =============================================================================

@dataclass
class VideoCaptionRequest:
    """Request for video caption generation."""
    description: str
    platform: str
    style: str
    max_length: int = 50
    temperature: float = 0.8

class VideoCaptionGenerator:
    """Advanced video caption generator using Transformers."""
    
    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the transformer model."""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate_caption(self, request: VideoCaptionRequest) -> str:
        """Generate caption based on request."""
        
        # Create platform-specific prompts
        platform_prompts = {
            "tiktok": f"Create a viral TikTok caption for: {request.description}",
            "youtube": f"Write an engaging YouTube title for: {request.description}",
            "instagram": f"Generate an Instagram caption for: {request.description}",
            "twitter": f"Create a Twitter post about: {request.description}"
        }
        
        prompt = platform_prompts.get(request.platform, f"Generate caption for: {request.description}")
        
        # Add style modifier
        style_modifiers = {
            "funny": " Make it funny and entertaining.",
            "dramatic": " Make it dramatic and attention-grabbing.",
            "informative": " Make it informative and educational.",
            "emotional": " Make it emotional and touching."
        }
        
        prompt += style_modifiers.get(request.style, "")
        
        # Generate caption
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=request.max_length,
                temperature=request.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        caption = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return caption.replace(prompt, "").strip()

def example_video_caption_generation():
    """Example of video caption generation."""
    
    print("🎬 Example 2: Video Caption Generation")
    print("=" * 50)
    
    try:
        generator = VideoCaptionGenerator()
        
        # Test cases
        test_cases = [
            VideoCaptionRequest(
                description="A cat playing with a laser pointer",
                platform="tiktok",
                style="funny"
            ),
            VideoCaptionRequest(
                description="A dog learning to skateboard",
                platform="youtube",
                style="informative"
            ),
            VideoCaptionRequest(
                description="A bird singing beautifully",
                platform="instagram",
                style="emotional"
            )
        ]
        
        for i, request in enumerate(test_cases, 1):
            caption = generator.generate_caption(request)
            print(f"📹 Test {i}: {request.description}")
            print(f"   Platform: {request.platform}")
            print(f"   Style: {request.style}")
            print(f"   Caption: {caption}")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

# =============================================================================
# EXAMPLE 3: SENTIMENT ANALYSIS
# =============================================================================

def example_sentiment_analysis():
    """Example of sentiment analysis for video content."""
    
    print("😊 Example 3: Sentiment Analysis")
    print("=" * 50)
    
    try:
        from transformers import pipeline
        
        # Load sentiment analysis pipeline
        classifier = pipeline("sentiment-analysis")
        
        # Test video descriptions
        video_descriptions = [
            "This video is absolutely amazing! I can't stop watching! 🔥",
            "I don't like this content at all. It's boring.",
            "It's okay, nothing special but not bad either.",
            "This is the best thing I've ever seen! Pure genius!",
            "Terrible video, waste of time."
        ]
        
        print("📊 Analyzing video descriptions...")
        
        for description in video_descriptions:
            result = classifier(description)
            sentiment = result[0]
            
            print(f"📝 '{description}'")
            print(f"   Sentiment: {sentiment['label']}")
            print(f"   Confidence: {sentiment['score']:.3f}")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

# =============================================================================
# EXAMPLE 4: TEXT SUMMARIZATION
# =============================================================================

def example_text_summarization():
    """Example of text summarization for video descriptions."""
    
    print("📝 Example 4: Text Summarization")
    print("=" * 50)
    
    try:
        from transformers import pipeline
        
        # Load summarization pipeline
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        
        # Long video descriptions
        long_descriptions = [
            """
            This incredible video showcases a talented cat who has mastered the art of 
            playing with a laser pointer. The feline demonstrates amazing agility and 
            coordination as it chases the elusive red dot around the room. The cat's 
            determination and focus are truly remarkable, making this a must-watch 
            video for all cat lovers. The way the cat pounces and leaps with such 
            precision is absolutely fascinating and entertaining.
            """,
            """
            In this educational video, we explore the fascinating world of dog training 
            and how dogs can learn complex tricks like skateboarding. The video follows 
            a patient trainer working with a young dog, showing the step-by-step process 
            of teaching such an impressive skill. Viewers will learn about positive 
            reinforcement techniques, the importance of patience in training, and how 
            dogs can develop amazing abilities through proper guidance and encouragement.
            """
        ]
        
        print("📋 Summarizing long video descriptions...")
        
        for i, description in enumerate(long_descriptions, 1):
            print(f"📹 Video {i} Description:")
            print(f"   Original: {description.strip()}")
            
            summary = summarizer(description, max_length=100, min_length=30)
            print(f"   Summary: {summary[0]['summary_text']}")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

# =============================================================================
# EXAMPLE 5: MULTI-LANGUAGE SUPPORT
# =============================================================================

def example_multilingual_support():
    """Example of multi-language support for global audiences."""
    
    print("🌍 Example 5: Multi-Language Support")
    print("=" * 50)
    
    try:
        from transformers import pipeline
        
        # English caption
        english_caption = "This cat is absolutely hilarious! Can't stop laughing! 😂"
        
        # Translation pipelines
        translators = {
            "Spanish": pipeline("translation", model="Helsinki-NLP/opus-mt-en-es"),
            "French": pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr"),
            "German": pipeline("translation", model="Helsinki-NLP/opus-mt-en-de")
        }
        
        print(f"🇺🇸 English: {english_caption}")
        print()
        
        for language, translator in translators.items():
            try:
                translation = translator(english_caption)
                translated_text = translation[0]['translation_text']
                print(f"🌍 {language}: {translated_text}")
            except Exception as e:
                print(f"🌍 {language}: Translation failed - {e}")
        
        print()
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

# =============================================================================
# EXAMPLE 6: OPTIMIZATION TECHNIQUES
# =============================================================================

class OptimizedTransformersManager:
    """Optimized Transformers manager with caching and performance features."""
    
    def __init__(self, cache_dir: str = "./model_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.models = {}
        self.tokenizers = {}
    
    @lru_cache(maxsize=100)
    def get_cached_model(self, model_name: str):
        """Get cached model or load new one."""
        if model_name not in self.models:
            from transformers import AutoModelForCausalLM
            self.models[model_name] = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=str(self.cache_dir),
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True
            )
        return self.models[model_name]
    
    @lru_cache(maxsize=100)
    def get_cached_tokenizer(self, model_name: str):
        """Get cached tokenizer or load new one."""
        if model_name not in self.tokenizers:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=str(self.cache_dir)
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            self.tokenizers[model_name] = tokenizer
        return self.tokenizers[model_name]
    
    def batch_generate(
        self,
        model_name: str,
        prompts: List[str],
        max_length: int = 50,
        batch_size: int = 4
    ) -> List[str]:
        """Generate text in batches for efficiency."""
        
        model = self.get_cached_model(model_name)
        tokenizer = self.get_cached_tokenizer(model_name)
        
        all_generated = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            
            # Tokenize batch
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            # Move to device
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode batch
            batch_generated = []
            for output in outputs:
                text = tokenizer.decode(output, skip_special_tokens=True)
                batch_generated.append(text)
            
            all_generated.extend(batch_generated)
        
        return all_generated

def example_optimization_techniques():
    """Example of optimization techniques."""
    
    print("⚡ Example 6: Optimization Techniques")
    print("=" * 50)
    
    try:
        manager = OptimizedTransformersManager()
        
        # Test prompts
        prompts = [
            "Generate viral caption for cat video:",
            "Create engaging title for dog video:",
            "Write Instagram caption for bird video:",
            "Make TikTok caption for fish video:"
        ]
        
        print("🚀 Testing optimized batch generation...")
        start_time = time.time()
        
        results = manager.batch_generate("gpt2", prompts, max_length=30)
        
        generation_time = time.time() - start_time
        
        print(f"⏱️  Batch generation time: {generation_time:.3f} seconds")
        print(f"📊 Average time per prompt: {generation_time/len(prompts):.3f} seconds")
        print()
        
        for i, (prompt, result) in enumerate(zip(prompts, results), 1):
            print(f"📝 {i}. {prompt}")
            print(f"   Result: {result}")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

# =============================================================================
# EXAMPLE 7: INTEGRATION WITH VIDEO-OPUSCLIP
# =============================================================================

class VideoOpusClipTransformersIntegration:
    """Integration class for Transformers in Video-OpusClip system."""
    
    def __init__(self):
        self.caption_generator = VideoCaptionGenerator()
        self.sentiment_analyzer = None
        self.summarizer = None
        self.setup_analyzers()
    
    def setup_analyzers(self):
        """Setup analysis pipelines."""
        try:
            from transformers import pipeline
            self.sentiment_analyzer = pipeline("sentiment-analysis")
            self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        except Exception as e:
            logger.warning(f"Could not setup analyzers: {e}")
    
    def analyze_video_content(self, video_description: str) -> Dict[str, Any]:
        """Comprehensive video content analysis."""
        
        analysis = {
            "description": video_description,
            "caption_suggestions": [],
            "sentiment": None,
            "summary": None,
            "recommendations": []
        }
        
        # Generate caption suggestions
        platforms = ["tiktok", "youtube", "instagram"]
        styles = ["funny", "dramatic", "informative"]
        
        for platform in platforms:
            for style in styles:
                request = VideoCaptionRequest(
                    description=video_description,
                    platform=platform,
                    style=style,
                    max_length=40
                )
                caption = self.caption_generator.generate_caption(request)
                analysis["caption_suggestions"].append({
                    "platform": platform,
                    "style": style,
                    "caption": caption
                })
        
        # Analyze sentiment
        if self.sentiment_analyzer:
            try:
                sentiment_result = self.sentiment_analyzer(video_description)
                analysis["sentiment"] = sentiment_result[0]
            except Exception as e:
                logger.warning(f"Sentiment analysis failed: {e}")
        
        # Generate summary
        if self.summarizer and len(video_description) > 100:
            try:
                summary_result = self.summarizer(video_description, max_length=80, min_length=20)
                analysis["summary"] = summary_result[0]['summary_text']
            except Exception as e:
                logger.warning(f"Summarization failed: {e}")
        
        # Generate recommendations
        analysis["recommendations"] = self.generate_recommendations(analysis)
        
        return analysis
    
    def generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate content recommendations based on analysis."""
        
        recommendations = []
        
        # Sentiment-based recommendations
        if analysis["sentiment"]:
            sentiment = analysis["sentiment"]["label"]
            if sentiment == "POSITIVE":
                recommendations.append("Content has positive sentiment - great for engagement!")
                recommendations.append("Consider using upbeat music and bright visuals")
            elif sentiment == "NEGATIVE":
                recommendations.append("Content has negative sentiment - consider adding humor")
                recommendations.append("Try to balance with positive elements")
        
        # Platform-specific recommendations
        platform_captions = {}
        for suggestion in analysis["caption_suggestions"]:
            platform = suggestion["platform"]
            if platform not in platform_captions:
                platform_captions[platform] = []
            platform_captions[platform].append(suggestion)
        
        for platform, captions in platform_captions.items():
            if len(captions) > 0:
                recommendations.append(f"Best {platform} caption: {captions[0]['caption']}")
        
        return recommendations

def example_video_opusclip_integration():
    """Example of Transformers integration with Video-OpusClip."""
    
    print("🎯 Example 7: Video-OpusClip Integration")
    print("=" * 50)
    
    try:
        integration = VideoOpusClipTransformersIntegration()
        
        # Test video descriptions
        test_videos = [
            "This incredible video shows a cat who has learned to play the piano! The feline sits at the keyboard and actually plays a recognizable tune. It's absolutely amazing to watch such intelligence and talent in action. The cat's concentration and musical ability are truly remarkable.",
            "A hilarious compilation of dogs failing at simple tasks. From dogs getting stuck in boxes to falling off furniture, this video will have you laughing out loud. The expressions on their faces are priceless and the situations they get themselves into are just too funny.",
            "A beautiful and emotional video of a bird singing to its mate. The melody is hauntingly beautiful and the bond between the birds is touching. This is a rare glimpse into the romantic side of nature that will warm your heart."
        ]
        
        for i, description in enumerate(test_videos, 1):
            print(f"📹 Video {i} Analysis:")
            print(f"   Description: {description[:100]}...")
            print()
            
            analysis = integration.analyze_video_content(description)
            
            # Display results
            if analysis["sentiment"]:
                sentiment = analysis["sentiment"]
                print(f"   😊 Sentiment: {sentiment['label']} ({sentiment['score']:.3f})")
            
            if analysis["summary"]:
                print(f"   📋 Summary: {analysis['summary']}")
            
            print(f"   💡 Recommendations:")
            for rec in analysis["recommendations"][:3]:  # Show first 3
                print(f"      • {rec}")
            
            print(f"   📝 Caption Suggestions:")
            for suggestion in analysis["caption_suggestions"][:2]:  # Show first 2
                print(f"      • {suggestion['platform'].title()} ({suggestion['style']}): {suggestion['caption']}")
            
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Run all Transformers examples."""
    
    print("🎯 Transformers Examples for Video-OpusClip")
    print("=" * 60)
    print()
    
    # Check PyTorch
    print(f"🔥 PyTorch version: {torch.__version__}")
    print(f"🔥 CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Run all examples
    examples = [
        ("Basic Text Generation", example_basic_text_generation),
        ("Video Caption Generation", example_video_caption_generation),
        ("Sentiment Analysis", example_sentiment_analysis),
        ("Text Summarization", example_text_summarization),
        ("Multi-Language Support", example_multilingual_support),
        ("Optimization Techniques", example_optimization_techniques),
        ("Video-OpusClip Integration", example_video_opusclip_integration)
    ]
    
    results = []
    
    for name, example_func in examples:
        print(f"🚀 Running: {name}")
        print("-" * 40)
        
        try:
            success = example_func()
            results.append((name, success))
        except Exception as e:
            print(f"❌ Example failed: {e}")
            results.append((name, False))
        
        print()
    
    # Summary
    print("=" * 60)
    print("📊 Examples Summary")
    print("=" * 60)
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {name}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"\n🎯 Overall: {passed}/{total} examples completed successfully")
    
    if passed == total:
        print("🎉 All Transformers examples working perfectly!")
        print("\nNext steps:")
        print("1. Integrate these examples into your Video-OpusClip pipeline")
        print("2. Customize the models and parameters for your needs")
        print("3. Add more advanced features like fine-tuning")
    else:
        print("⚠️  Some examples need attention. Check the error messages above.")

if __name__ == "__main__":
    main() 