from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import time
import json
import logging
from typing import Dict, Any, List, Tuple, Optional
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import sys
import os
from transformers_integration import (
from typing import Any, List, Dict, Optional
"""
🤗 Transformers Demo - Facebook Posts Processing
==============================================

This demo showcases the integration with Hugging Face Transformers library
for pre-trained models and tokenizers in Facebook Posts processing.

Features Demonstrated:
- Pre-trained model loading and fine-tuning
- Advanced tokenization strategies
- Text classification and generation
- Sentiment analysis and content moderation
- Multi-language support
- Model evaluation and inference
"""


# Import our transformers integration
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    TransformersConfig,
    FacebookPostsTokenizer,
    FacebookPostsClassifier,
    FacebookPostsGenerator,
    FacebookPostsSentimentAnalyzer,
    FacebookPostsContentModerator,
    TransformersPipeline
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for GPU availability
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")


def generate_sample_posts(num_samples: int = 100) -> List[str]:
    """Generate sample Facebook posts for demonstration."""
    sample_posts = [
        "Amazing product launch today! 🚀 #innovation #tech",
        "Great meeting with the team today. Collaboration is key! 👥",
        "Customer feedback has been incredible. Thank you all! 🙏",
        "New features coming soon. Stay tuned! 🔥",
        "Working on exciting new projects. Can't wait to share! 💡",
        "Team building event was a huge success! 🎉",
        "Product update: Improved performance by 50%! 📈",
        "Customer satisfaction at all-time high! 🏆",
        "Innovation never stops. Always pushing boundaries! 🌟",
        "Thank you to our amazing community! You rock! 🤘",
        "New partnership announcement coming soon! 🤝",
        "Behind the scenes: Our development process 🛠️",
        "User experience improvements live now! ✨",
        "Data shows incredible growth this quarter! 📊",
        "Team collaboration leads to amazing results! 🎯",
        "Customer support team doing fantastic work! 👏",
        "Product roadmap update: Exciting features ahead! 🗺️",
        "Community feedback drives our decisions! 💬",
        "Innovation in action: Real-time analytics! 📱",
        "Success metrics exceeded expectations! 🎊",
        "Breaking news: Major milestone achieved! 🎯",
        "Excited to announce our latest breakthrough! 🚀",
        "Customer success stories inspire us daily! 💪",
        "Technology is transforming our industry! ⚡",
        "Collaboration across teams delivers results! 🤝",
        "Innovation culture drives our success! 💡",
        "Data-driven decisions lead to growth! 📊",
        "User-centric design wins every time! 🎨",
        "Agile methodology accelerates development! ⚡",
        "Continuous improvement is our mantra! 🔄",
        "Customer-first approach never fails! ❤️"
    ]
    
    # Generate variations
    posts = []
    for _ in range(num_samples):
        base_post = random.choice(sample_posts)
        
        # Add random variations
        variations = [
            f"Just {base_post.lower()}",
            f"Update: {base_post}",
            f"Breaking news: {base_post}",
            f"Excited to announce: {base_post}",
            f"Proud to share: {base_post}",
            base_post,
            f"🚀 {base_post}",
            f"💡 {base_post}",
            f"🎉 {base_post}",
            f"🔥 {base_post}",
            f"✨ {base_post}",
            f"📈 {base_post}",
            f"🎯 {base_post}",
            f"💪 {base_post}",
            f"⚡ {base_post}"
        ]
        
        posts.append(random.choice(variations))
    
    return posts


async def demo_tokenizer():
    """Demonstrate advanced tokenization capabilities."""
    logger.info("🔤 Tokenizer Demo")
    logger.info("=" * 50)
    
    # Create configuration
    config = TransformersConfig(
        model_name="bert-base-uncased",
        max_length=128
    )
    
    # Create tokenizer
    tokenizer = FacebookPostsTokenizer(config)
    
    # Sample texts with Facebook-specific content
    sample_texts = [
        "Amazing product launch today! 🚀 #innovation #tech",
        "Great meeting with @john_doe today. Collaboration is key! 👥",
        "Check out our new features at https://example.com",
        "Customer feedback has been incredible. Thank you all! 🙏",
        "Working on exciting new projects. Can't wait to share! 💡 #startup #tech"
    ]
    
    logger.info("Tokenization Examples:")
    for text in sample_texts:
        # Encode text
        encoding = tokenizer.encode_text(text)
        
        # Get token information
        input_ids = encoding['input_ids'][0]
        attention_mask = encoding['attention_mask'][0]
        
        # Decode back to text
        decoded_text = tokenizer.decode_tokens(input_ids.tolist())
        
        logger.info(f"Original: {text}")
        logger.info(f"Processed: {decoded_text}")
        logger.info(f"Tokens: {len(input_ids)}")
        logger.info(f"Special tokens: {sum(1 for id in input_ids if id in tokenizer.tokenizer.all_special_ids)}")
        logger.info("-" * 40)
    
    logger.info(f"Vocabulary size: {tokenizer.get_vocab_size():,}")
    logger.info("")


async def demo_classification():
    """Demonstrate text classification capabilities."""
    logger.info("🏷️ Classification Demo")
    logger.info("=" * 50)
    
    # Create configuration
    config = TransformersConfig(
        model_name="bert-base-uncased",
        model_type="classification",
        num_labels=3,
        label_names=["negative", "neutral", "positive"],
        max_length=256,
        batch_size=8
    )
    
    # Create classifier
    classifier = FacebookPostsClassifier(config)
    
    # Sample posts for classification
    sample_posts = [
        "Amazing product launch today! 🚀 #innovation #tech",
        "Great meeting with the team today. Collaboration is key! 👥",
        "Customer feedback has been incredible. Thank you all! 🙏",
        "Product update: Improved performance by 50%! 📈",
        "Innovation never stops. Always pushing boundaries! 🌟",
        "Thank you to our amazing community! You rock! 🤘",
        "New partnership announcement coming soon! 🤝",
        "Data shows incredible growth this quarter! 📊",
        "Team collaboration leads to amazing results! 🎯",
        "Customer support team doing fantastic work! 👏"
    ]
    
    # Perform classification
    logger.info("Classification Results:")
    results = classifier.predict(sample_posts)
    
    for result in results:
        logger.info(f"Text: {result['text'][:50]}...")
        logger.info(f"Predicted: {result['label_name']} (confidence: {result['confidence']:.3f})")
        logger.info(f"Probabilities: {dict(zip(config.label_names, result['probabilities']))}")
        logger.info("-" * 40)
    
    logger.info("")


async def demo_text_generation():
    """Demonstrate text generation capabilities."""
    logger.info("📝 Text Generation Demo")
    logger.info("=" * 50)
    
    # Create configuration
    config = TransformersConfig(
        model_name="gpt2",
        model_type="generation",
        max_length=128,
        temperature=0.8,
        max_new_tokens=50
    )
    
    # Create generator
    generator = FacebookPostsGenerator(config)
    
    # Sample prompts
    prompts = [
        "Amazing product launch today!",
        "Great meeting with the team",
        "Customer feedback has been",
        "New features coming soon",
        "Working on exciting new projects"
    ]
    
    logger.info("Text Generation Examples:")
    for prompt in prompts:
        try:
            generated_text = generator.generate_text(prompt)
            logger.info(f"Prompt: {prompt}")
            logger.info(f"Generated: {generated_text}")
            logger.info("-" * 40)
        except Exception as e:
            logger.info(f"Generation failed for '{prompt}': {e}")
    
    logger.info("")


async def demo_sentiment_analysis():
    """Demonstrate sentiment analysis capabilities."""
    logger.info("😊 Sentiment Analysis Demo")
    logger.info("=" * 50)
    
    # Create sentiment analyzer
    sentiment_analyzer = FacebookPostsSentimentAnalyzer()
    
    # Sample posts for sentiment analysis
    sample_posts = [
        "Amazing product launch today! 🚀 #innovation #tech",
        "Great meeting with the team today. Collaboration is key! 👥",
        "Customer feedback has been incredible. Thank you all! 🙏",
        "Product update: Improved performance by 50%! 📈",
        "Innovation never stops. Always pushing boundaries! 🌟",
        "Thank you to our amazing community! You rock! 🤘",
        "New partnership announcement coming soon! 🤝",
        "Data shows incredible growth this quarter! 📊",
        "Team collaboration leads to amazing results! 🎯",
        "Customer support team doing fantastic work! 👏"
    ]
    
    # Perform sentiment analysis
    logger.info("Sentiment Analysis Results:")
    results = sentiment_analyzer.analyze_sentiment(sample_posts)
    
    for result in results:
        logger.info(f"Text: {result['text'][:50]}...")
        logger.info(f"Sentiment: {result['sentiment']} (confidence: {result['confidence']:.3f})")
        logger.info(f"Scores - Positive: {result['positive_score']:.3f}, "
                   f"Negative: {result['negative_score']:.3f}, "
                   f"Neutral: {result['neutral_score']:.3f}")
        logger.info("-" * 40)
    
    logger.info("")


async def demo_content_moderation():
    """Demonstrate content moderation capabilities."""
    logger.info("🛡️ Content Moderation Demo")
    logger.info("=" * 50)
    
    # Create content moderator
    content_moderator = FacebookPostsContentModerator()
    
    # Sample posts for content moderation
    sample_posts = [
        "Amazing product launch today! 🚀 #innovation #tech",
        "Great meeting with the team today. Collaboration is key! 👥",
        "Customer feedback has been incredible. Thank you all! 🙏",
        "Product update: Improved performance by 50%! 📈",
        "Innovation never stops. Always pushing boundaries! 🌟",
        "Thank you to our amazing community! You rock! 🤘",
        "New partnership announcement coming soon! 🤝",
        "Data shows incredible growth this quarter! 📊",
        "Team collaboration leads to amazing results! 🎯",
        "Customer support team doing fantastic work! 👏"
    ]
    
    # Perform content moderation
    logger.info("Content Moderation Results:")
    results = content_moderator.moderate_content(sample_posts)
    
    for result in results:
        logger.info(f"Text: {result['text'][:50]}...")
        logger.info(f"Moderation: {result['moderation_label']} (confidence: {result['confidence']:.3f})")
        logger.info(f"Flagged: {'Yes' if result['is_flagged'] else 'No'}")
        logger.info("-" * 40)
    
    logger.info("")


async def demo_comprehensive_pipeline():
    """Demonstrate comprehensive pipeline processing."""
    logger.info("🔄 Comprehensive Pipeline Demo")
    logger.info("=" * 50)
    
    # Create configuration
    config = TransformersConfig(
        model_name="bert-base-uncased",
        model_type="classification",
        num_labels=3,
        label_names=["negative", "neutral", "positive"],
        max_length=256,
        batch_size=8
    )
    
    # Create pipeline
    pipeline = TransformersPipeline(config)
    
    # Setup all models
    classifier = pipeline.setup_classifier()
    generator = pipeline.setup_generator()
    sentiment_analyzer = pipeline.setup_sentiment_analyzer()
    content_moderator = pipeline.setup_content_moderator()
    
    # Sample posts
    sample_posts = [
        "Amazing product launch today! 🚀 #innovation #tech",
        "Great meeting with the team today. Collaboration is key! 👥",
        "Customer feedback has been incredible. Thank you all! 🙏"
    ]
    
    # Process posts with all models
    logger.info("Processing posts with comprehensive pipeline...")
    start_time = time.time()
    
    results = pipeline.process_posts(sample_posts)
    
    processing_time = time.time() - start_time
    
    logger.info(f"Processing completed in {processing_time:.2f} seconds")
    logger.info(f"Processed {len(sample_posts)} posts")
    
    # Display results
    logger.info("\nComprehensive Results:")
    for i, post in enumerate(sample_posts):
        logger.info(f"\nPost {i+1}: {post}")
        
        # Classification results
        if results['classifications']:
            classification = results['classifications'][i]
            logger.info(f"  Classification: {classification['label_name']} "
                       f"(confidence: {classification['confidence']:.3f})")
        
        # Sentiment results
        if results['sentiments']:
            sentiment = results['sentiments'][i]
            logger.info(f"  Sentiment: {sentiment['sentiment']} "
                       f"(confidence: {sentiment['confidence']:.3f})")
        
        # Moderation results
        if results['moderations']:
            moderation = results['moderations'][i]
            logger.info(f"  Moderation: {moderation['moderation_label']} "
                       f"(flagged: {'Yes' if moderation['is_flagged'] else 'No'})")
        
        # Generated responses
        if results['generated_responses']:
            response = results['generated_responses'][i]
            logger.info(f"  Generated Response: {response[:100]}...")
    
    logger.info("")


async def demo_model_comparison():
    """Compare different pre-trained models."""
    logger.info("📊 Model Comparison Demo")
    logger.info("=" * 50)
    
    # Different model configurations
    model_configs = [
        ("BERT Base", "bert-base-uncased"),
        ("DistilBERT", "distilbert-base-uncased"),
        ("RoBERTa Base", "roberta-base"),
        ("ALBERT Base", "albert-base-v2")
    ]
    
    # Sample posts
    sample_posts = [
        "Amazing product launch today! 🚀 #innovation #tech",
        "Great meeting with the team today. Collaboration is key! 👥",
        "Customer feedback has been incredible. Thank you all! 🙏"
    ]
    
    results = {}
    
    for model_name, model_path in model_configs:
        logger.info(f"Testing {model_name}...")
        
        try:
            # Create configuration
            config = TransformersConfig(
                model_name=model_path,
                model_type="classification",
                num_labels=3,
                label_names=["negative", "neutral", "positive"],
                max_length=256
            )
            
            # Create classifier
            classifier = FacebookPostsClassifier(config)
            
            # Test inference speed
            start_time = time.time()
            predictions = classifier.predict(sample_posts)
            inference_time = time.time() - start_time
            
            # Count parameters
            total_params = sum(p.numel() for p in classifier.model.parameters())
            
            results[model_name] = {
                'parameters': total_params,
                'inference_time': inference_time,
                'predictions': predictions
            }
            
            logger.info(f"  Parameters: {total_params:,}")
            logger.info(f"  Inference time: {inference_time:.3f}s")
            
        except Exception as e:
            logger.info(f"  Error: {e}")
            results[model_name] = {'error': str(e)}
    
    # Display comparison
    logger.info("\nModel Comparison Results:")
    logger.info("-" * 60)
    logger.info(f"{'Model':15} | {'Parameters':>12} | {'Inference Time':>15}")
    logger.info("-" * 60)
    
    for model_name, result in results.items():
        if 'error' not in result:
            logger.info(f"{model_name:15} | {result['parameters']:12,} | {result['inference_time']:15.3f}s")
        else:
            logger.info(f"{model_name:15} | {'Error':>12} | {'N/A':>15}")
    
    logger.info("")


async def demo_fine_tuning():
    """Demonstrate model fine-tuning capabilities."""
    logger.info("🎯 Fine-tuning Demo")
    logger.info("=" * 50)
    
    # Generate sample data for fine-tuning
    posts = generate_sample_posts(50)
    
    # Create synthetic labels (0: negative, 1: neutral, 2: positive)
    labels = []
    for post in posts:
        # Simple heuristic for demo
        if any(word in post.lower() for word in ['amazing', 'incredible', 'fantastic', 'great', 'success']):
            labels.append(2)  # positive
        elif any(word in post.lower() for word in ['problem', 'issue', 'failed', 'disappointed']):
            labels.append(0)  # negative
        else:
            labels.append(1)  # neutral
    
    # Split data
    train_size = int(0.8 * len(posts))
    val_size = len(posts) - train_size
    
    train_posts, val_posts = posts[:train_size], posts[train_size:]
    train_labels, val_labels = labels[:train_size], labels[train_size:]
    
    logger.info(f"Training data: {len(train_posts)} posts")
    logger.info(f"Validation data: {len(val_posts)} posts")
    logger.info(f"Label distribution: {np.bincount(labels)}")
    
    # Create configuration for fine-tuning
    config = TransformersConfig(
        model_name="distilbert-base-uncased",  # Smaller model for faster training
        model_type="classification",
        num_labels=3,
        label_names=["negative", "neutral", "positive"],
        max_length=256,
        batch_size=4,
        num_epochs=1,  # Short training for demo
        learning_rate=2e-5
    )
    
    # Create classifier
    classifier = FacebookPostsClassifier(config)
    
    logger.info("Starting fine-tuning...")
    start_time = time.time()
    
    try:
        # Fine-tune model
        classifier.train(train_posts, train_labels, val_posts, val_labels)
        
        training_time = time.time() - start_time
        logger.info(f"Fine-tuning completed in {training_time:.2f} seconds")
        
        # Test fine-tuned model
        test_posts = [
            "Amazing product launch today! 🚀",
            "Great meeting with the team! 👥",
            "Customer feedback has been incredible! 🙏"
        ]
        
        logger.info("\nTesting fine-tuned model:")
        predictions = classifier.predict(test_posts)
        
        for prediction in predictions:
            logger.info(f"Text: {prediction['text']}")
            logger.info(f"Predicted: {prediction['label_name']} "
                       f"(confidence: {prediction['confidence']:.3f})")
        
    except Exception as e:
        logger.info(f"Fine-tuning failed: {e}")
    
    logger.info("")


async def run_transformers_demo():
    """Run the complete Transformers demonstration."""
    logger.info("🤗 TRANSFORMERS DEMO - Facebook Posts Processing")
    logger.info("=" * 60)
    logger.info("Demonstrating Hugging Face Transformers integration for")
    logger.info("pre-trained models and tokenizers in Facebook Posts processing")
    logger.info("=" * 60)
    
    # Run all demos
    await demo_tokenizer()
    await demo_classification()
    await demo_text_generation()
    await demo_sentiment_analysis()
    await demo_content_moderation()
    await demo_comprehensive_pipeline()
    await demo_model_comparison()
    await demo_fine_tuning()
    
    logger.info("🎉 Transformers Demo Completed Successfully!")
    logger.info("All features demonstrated:")
    logger.info("✅ Advanced tokenization with Facebook-specific preprocessing")
    logger.info("✅ Text classification with pre-trained models")
    logger.info("✅ Text generation with temperature sampling")
    logger.info("✅ Sentiment analysis with specialized models")
    logger.info("✅ Content moderation and safety checks")
    logger.info("✅ Comprehensive pipeline processing")
    logger.info("✅ Model comparison and benchmarking")
    logger.info("✅ Fine-tuning capabilities")


async def quick_transformers_demo():
    """Quick demonstration of key Transformers features."""
    logger.info("⚡ QUICK TRANSFORMERS DEMO")
    logger.info("=" * 40)
    
    # Quick sentiment analysis
    sentiment_analyzer = FacebookPostsSentimentAnalyzer()
    
    sample_texts = [
        "Amazing product launch today! 🚀",
        "Great meeting with the team! 👥",
        "Customer feedback has been incredible! 🙏"
    ]
    
    results = sentiment_analyzer.analyze_sentiment(sample_texts)
    
    logger.info("Quick Sentiment Analysis:")
    for result in results:
        logger.info(f"Text: {result['text']}")
        logger.info(f"Sentiment: {result['sentiment']} (confidence: {result['confidence']:.3f})")
    
    logger.info("\n✅ Quick demo completed!")


if __name__ == "__main__":
    # Run the complete demo
    asyncio.run(run_transformers_demo())
    
    # Uncomment for quick demo
    # asyncio.run(quick_transformers_demo()) 