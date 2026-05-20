from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import time
import json
from typing import List, Dict, Any
from transformers import pipeline, set_seed
import torch
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Instagram Captions AI v8.0 - Simple Working Demo

A simplified demonstration of real AI caption generation using transformers.
This version is optimized to work with the installed dependencies.
"""


# Core dependencies

# Set seed for reproducible results
set_seed(42)

print("🧠 Instagram Captions AI v8.0 - Simple Demo")
print("="*60)
print("🔄 Initializing AI models...")

# Initialize the text generation pipeline
try:
    # Use a lightweight model that works well on CPU
    generator = pipeline(
        'text-generation',
        model='distilgpt2',
        tokenizer='distilgpt2',
        device=-1,  # CPU
        max_length=150,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        top_k=50,
        pad_token_id=50256
    )
    print("✅ DistilGPT-2 model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

print("="*60)

class SimpleAICaptions:
    """Simple AI caption generator using real transformers."""
    
    def __init__(self) -> Any:
        self.generator = generator
        self.styles = {
            "casual": "Write a casual, friendly Instagram caption about",
            "professional": "Create a professional Instagram caption about", 
            "playful": "Write a fun, playful Instagram caption about",
            "inspirational": "Write an inspiring Instagram caption about",
            "educational": "Create an educational Instagram caption about",
            "promotional": "Write a compelling promotional Instagram caption about"
        }
        
        self.hashtag_sets = {
            "general": ["#instagood", "#photooftheday", "#love", "#beautiful", "#happy"],
            "lifestyle": ["#lifestyle", "#daily", "#vibes", "#mood", "#inspiration"],
            "business": ["#business", "#success", "#professional", "#growth", "#goals"],
            "creative": ["#creative", "#art", "#design", "#aesthetic", "#unique"]
        }
    
    def generate_caption(self, content: str, style: str = "casual") -> Dict[str, Any]:
        """Generate AI caption using real transformers."""
        
        start_time = time.time()
        
        # Create prompt
        prompt = f"{self.styles.get(style, self.styles['casual'])} {content}:"
        
        try:
            # Generate with transformer
            result = self.generator(
                prompt,
                max_length=len(prompt.split()) + 50,
                num_return_sequences=1,
                truncation=True,
                clean_up_tokenization_spaces=True
            )
            
            # Extract and clean caption
            generated_text = result[0]['generated_text']
            caption = generated_text.replace(prompt, "").strip()
            
            # Clean up caption
            if not caption:
                caption = f"Sharing this amazing {content} ✨"
            
            # Add emoji if missing
            if not any(ord(char) > 127 for char in caption):
                caption += " ✨"
            
            # Generate hashtags
            hashtags = self._generate_hashtags(content, style)
            
            # Calculate quality score (simplified)
            quality_score = self._calculate_quality(caption, content)
            
            processing_time = time.time() - start_time
            
            return {
                "caption": caption,
                "hashtags": hashtags,
                "quality_score": quality_score,
                "style": style,
                "processing_time": processing_time,
                "model": "distilgpt2"
            }
            
        except Exception as e:
            print(f"❌ Generation error: {e}")
            return {
                "caption": f"Beautiful {content} ✨ #inspiration",
                "hashtags": ["#beautiful", "#inspiration", "#moment"],
                "quality_score": 75.0,
                "style": style,
                "processing_time": time.time() - start_time,
                "model": "fallback"
            }
    
    def _generate_hashtags(self, content: str, style: str) -> List[str]:
        """Generate relevant hashtags."""
        
        hashtags = []
        
        # Add style-specific hashtags
        if style in ["professional", "business"]:
            hashtags.extend(self.hashtag_sets["business"][:3])
        elif style == "creative":
            hashtags.extend(self.hashtag_sets["creative"][:3])
        else:
            hashtags.extend(self.hashtag_sets["lifestyle"][:3])
        
        # Add general engagement hashtags
        hashtags.extend(self.hashtag_sets["general"][:5])
        
        # Add content-specific hashtags
        content_words = content.lower().split()
        for word in content_words:
            if len(word) > 4:
                hashtags.append(f"#{word}")
                if len(hashtags) >= 15:
                    break
        
        return hashtags[:15]
    
    def _calculate_quality(self, caption: str, content: str) -> float:
        """Calculate caption quality score."""
        
        score = 70.0
        
        # Length check
        if 30 <= len(caption) <= 200:
            score += 10
        
        # Emoji check
        if any(ord(char) > 127 for char in caption):
            score += 5
        
        # Relevance check (simple keyword matching)
        content_words = set(content.lower().split())
        caption_words = set(caption.lower().split())
        overlap = len(content_words.intersection(caption_words))
        score += min(overlap * 3, 15)
        
        return min(score, 100.0)


def interactive_demo():
    """Run interactive demo."""
    
    ai_generator = SimpleAICaptions()
    
    print("\n🚀 AI Caption Generator Ready!")
    print("📝 Enter your content description (or 'quit' to exit)")
    print("🎨 Available styles: casual, professional, playful, inspirational, educational, promotional")
    print("="*60)
    
    while True:
        print("\n" + "─"*40)
        
        # Get content input
        content = input("📷 Content description: ").strip()
        if content.lower() in ['quit', 'exit', 'q']:
            print("👋 Thanks for using AI Captions v8.0!")
            break
        
        if not content:
            print("⚠️ Please provide a content description")
            continue
        
        # Get style input
        style = input("🎨 Style (default: casual): ").strip().lower()
        if not style:
            style = "casual"
        
        if style not in ai_generator.styles:
            print(f"⚠️ Unknown style '{style}', using 'casual'")
            style = "casual"
        
        print(f"\n🧠 Generating AI caption with style '{style}'...")
        
        # Generate caption
        result = ai_generator.generate_caption(content, style)
        
        # Display results
        print("\n" + "="*60)
        print("🎯 AI GENERATION RESULTS")
        print("="*60)
        print(f"✨ Caption: {result['caption']}")
        print(f"\n🏷️ Hashtags: {' '.join(result['hashtags'])}")
        print(f"\n📊 Quality Score: {result['quality_score']:.1f}/100")
        print(f"⚡ Processing Time: {result['processing_time']:.3f}s")
        print(f"🤖 Model: {result['model']}")
        print("="*60)


def batch_demo():
    """Run batch demonstration."""
    
    ai_generator = SimpleAICaptions()
    
    test_cases = [
        ("Beautiful sunset at the beach", "inspirational"),
        ("Professional headshot photo", "professional"),
        ("Cute puppy playing in park", "playful"),
        ("Homemade pasta dinner", "casual"),
        ("Team meeting success", "professional"),
        ("Art gallery opening", "creative")
    ]
    
    print("\n🚀 Batch AI Caption Generation Demo")
    print("="*60)
    
    total_time = 0
    total_quality = 0
    
    for i, (content, style) in enumerate(test_cases, 1):
        print(f"\n📝 Test {i}/6: {content} ({style})")
        print("─"*40)
        
        result = ai_generator.generate_caption(content, style)
        
        print(f"Caption: {result['caption']}")
        print(f"Quality: {result['quality_score']:.1f}/100")
        print(f"Time: {result['processing_time']:.3f}s")
        
        total_time += result['processing_time']
        total_quality += result['quality_score']
    
    print("\n" + "="*60)
    print("📊 BATCH RESULTS SUMMARY")
    print("="*60)
    print(f"Total Tests: {len(test_cases)}")
    print(f"Average Quality: {total_quality/len(test_cases):.1f}/100")
    print(f"Average Time: {total_time/len(test_cases):.3f}s")
    print(f"Total Time: {total_time:.3f}s")
    print(f"Throughput: {len(test_cases)/total_time:.1f} captions/second")
    print("="*60)


def main():
    """Main demo function."""
    
    print("\n🧠 Instagram Captions AI v8.0 - Demo Options")
    print("="*60)
    print("1. Interactive Demo - Generate captions interactively")
    print("2. Batch Demo - See 6 example generations")
    print("3. Quick Test - Single test generation")
    print("="*60)
    
    choice = input("Choose demo type (1-3, default: 1): ").strip()
    
    if choice == "2":
        batch_demo()
    elif choice == "3":
        # Quick test
        ai_generator = SimpleAICaptions()
        print("\n🧠 Quick Test: Generating caption for 'Beautiful sunset at beach'...")
        result = ai_generator.generate_caption("Beautiful sunset at beach", "inspirational")
        
        print("\n🎯 RESULT:")
        print(f"Caption: {result['caption']}")
        print(f"Hashtags: {' '.join(result['hashtags'][:5])}")
        print(f"Quality: {result['quality_score']:.1f}/100")
        print(f"Time: {result['processing_time']:.3f}s")
    else:
        interactive_demo()


match __name__:
    case "__main__":
    main() 