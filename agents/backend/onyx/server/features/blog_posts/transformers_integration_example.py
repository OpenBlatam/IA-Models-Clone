from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import torch
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
    from transformers_integration_system import (
from typing import Any, List, Dict, Optional
import asyncio
"""
🚀 Transformers Integration Example
==================================

Comprehensive example demonstrating the transformers integration system
with various use cases including training, text generation, and classification.
"""


# Import the transformers integration system
try:
        AdvancedTransformersTrainer, TransformersConfig, TransformersPipeline,
        create_transformers_config, get_available_models, validate_transformers_inputs,
        initialize_transformers_system
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: Transformers integration system not available")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demonstrate_basic_training():
    """Demonstrate basic model training with transformers."""
    print("\n" + "="*60)
    print("🏋️ BASIC MODEL TRAINING DEMONSTRATION")
    print("="*60)
    
    if not TRANSFORMERS_AVAILABLE:
        print("❌ Transformers system not available")
        return
    
    try:
        # Create configuration for causal language modeling
        config = TransformersConfig(
            model_name="microsoft/DialoGPT-medium",
            model_type="causal",
            task="text_generation",
            num_epochs=2,  # Reduced for demo
            batch_size=2,   # Reduced for demo
            learning_rate=5e-5,
            use_peft=True,
            lora_r=8,       # Reduced for demo
            lora_alpha=16   # Reduced for demo
        )
        
        # Sample training data
        train_texts = [
            "Hello, how are you today?",
            "What's the weather like?",
            "Tell me a joke",
            "How do I make coffee?",
            "What's your favorite color?",
            "Can you help me with programming?",
            "What's the capital of France?",
            "How do I learn machine learning?"
        ]
        
        validation_texts = [
            "What time is it?",
            "How do I cook pasta?",
            "What's the meaning of life?"
        ]
        
        print(f"📊 Training Configuration:")
        print(f"   Model: {config.model_name}")
        print(f"   Type: {config.model_type}")
        print(f"   Epochs: {config.num_epochs}")
        print(f"   Batch Size: {config.batch_size}")
        print(f"   Learning Rate: {config.learning_rate}")
        print(f"   PEFT: {config.use_peft}")
        
        # Initialize trainer
        trainer = AdvancedTransformersTrainer(config)
        
        # Train the model
        print(f"\n🚀 Starting training with {len(train_texts)} training samples...")
        result = trainer.train(train_texts, val_texts=validation_texts)
        
        print(f"\n📈 Training Results:")
        print(json.dumps(result, indent=2))
        
        if result.get("success", False):
            print("✅ Training completed successfully!")
            return True
        else:
            print("❌ Training failed")
            return False
            
    except Exception as e:
        print(f"❌ Training demonstration failed: {e}")
        return False


def demonstrate_text_generation():
    """Demonstrate text generation with trained model."""
    print("\n" + "="*60)
    print("✨ TEXT GENERATION DEMONSTRATION")
    print("="*60)
    
    if not TRANSFORMERS_AVAILABLE:
        print("❌ Transformers system not available")
        return
    
    try:
        # Check if model exists
        model_path = "./transformers_final_model"
        if not Path(model_path).exists():
            print("❌ No trained model found. Please run training first.")
            return False
        
        # Create pipeline
        config = TransformersConfig()
        pipeline = TransformersPipeline(model_path, config)
        
        # Test prompts
        test_prompts = [
            "Hello, I want to",
            "The weather today is",
            "My favorite hobby is",
            "I'm learning to",
            "The best way to"
        ]
        
        print(f"📝 Generating text for {len(test_prompts)} prompts...")
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n{i}. Prompt: '{prompt}'")
            
            try:
                generated_text = pipeline.generate(
                    prompt,
                    max_new_tokens=50,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )
                
                print(f"   Generated: '{generated_text}'")
                
            except Exception as e:
                print(f"   ❌ Generation failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Text generation demonstration failed: {e}")
        return False


def demonstrate_batch_generation():
    """Demonstrate batch text generation."""
    print("\n" + "="*60)
    print("📦 BATCH TEXT GENERATION DEMONSTRATION")
    print("="*60)
    
    if not TRANSFORMERS_AVAILABLE:
        print("❌ Transformers system not available")
        return
    
    try:
        # Check if model exists
        model_path = "./transformers_final_model"
        if not Path(model_path).exists():
            print("❌ No trained model found. Please run training first.")
            return False
        
        # Create pipeline
        config = TransformersConfig()
        pipeline = TransformersPipeline(model_path, config)
        
        # Batch prompts
        batch_prompts = [
            "The future of AI is",
            "I love programming because",
            "Machine learning helps us",
            "The best programming language is",
            "Artificial intelligence will"
        ]
        
        print(f"📦 Generating text for {len(batch_prompts)} prompts in batch...")
        
        # Generate all texts
        generated_texts = pipeline.batch_generate(
            batch_prompts,
            max_new_tokens=30,
            temperature=0.8
        )
        
        # Display results
        for i, (prompt, generated) in enumerate(zip(batch_prompts, generated_texts), 1):
            print(f"\n{i}. Prompt: '{prompt}'")
            print(f"   Generated: '{generated}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Batch generation demonstration failed: {e}")
        return False


def demonstrate_text_classification():
    """Demonstrate text classification."""
    print("\n" + "="*60)
    print("🏷️ TEXT CLASSIFICATION DEMONSTRATION")
    print("="*60)
    
    if not TRANSFORMERS_AVAILABLE:
        print("❌ Transformers system not available")
        return
    
    try:
        # For classification, we need a different model
        # Let's use a pre-trained BERT model for sentiment analysis
        config = TransformersConfig(
            model_name="microsoft/DistilBERT-base-uncased-finetuned-sst-2-english",
            model_type="sequence_classification",
            task="classification"
        )
        
        trainer = AdvancedTransformersTrainer(config)
        
        # Sample texts for classification
        texts_to_classify = [
            "I love this movie, it's amazing!",
            "This product is terrible and doesn't work.",
            "The weather is nice today.",
            "I'm feeling neutral about this.",
            "This is the best thing ever!",
            "I hate this so much.",
            "It's okay, nothing special.",
            "Absolutely fantastic experience!"
        ]
        
        print(f"🏷️ Classifying {len(texts_to_classify)} texts...")
        
        # Classify texts
        classifications = trainer.predict(texts_to_classify)
        
        # Display results
        for i, (text, result) in enumerate(zip(texts_to_classify, classifications), 1):
            print(f"\n{i}. Text: '{text}'")
            if "error" in result:
                print(f"   ❌ Classification failed: {result['error']}")
            else:
                predicted_class = result.get("predicted_class", "unknown")
                confidence = result.get("confidence", 0.0)
                print(f"   Class: {predicted_class} (confidence: {confidence:.3f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Text classification demonstration failed: {e}")
        return False


def demonstrate_available_models():
    """Demonstrate getting available models."""
    print("\n" + "="*60)
    print("📋 AVAILABLE MODELS DEMONSTRATION")
    print("="*60)
    
    if not TRANSFORMERS_AVAILABLE:
        print("❌ Transformers system not available")
        return
    
    try:
        # Get available models
        models = get_available_models()
        
        print("📋 Available Model Categories:")
        for category, model_list in models.items():
            print(f"\n{category.upper()}:")
            for model in model_list[:5]:  # Show first 5 models
                print(f"   • {model}")
            if len(model_list) > 5:
                print(f"   ... and {len(model_list) - 5} more")
        
        return True
        
    except Exception as e:
        print(f"❌ Available models demonstration failed: {e}")
        return False


def demonstrate_system_status():
    """Demonstrate system status checking."""
    print("\n" + "="*60)
    print("📊 SYSTEM STATUS DEMONSTRATION")
    print("="*60)
    
    if not TRANSFORMERS_AVAILABLE:
        print("❌ Transformers system not available")
        return
    
    try:
        # Initialize system
        print("🚀 Initializing transformers system...")
        success = initialize_transformers_system()
        
        if success:
            print("✅ System initialized successfully")
        else:
            print("❌ System initialization failed")
        
        # Check model existence
        model_path = "./transformers_final_model"
        model_exists = Path(model_path).exists()
        
        print(f"\n📊 System Status:")
        print(f"   Transformers Available: {TRANSFORMERS_AVAILABLE}")
        print(f"   System Initialized: {success}")
        print(f"   Trained Model Exists: {model_exists}")
        print(f"   Model Path: {model_path}")
        
        if torch.cuda.is_available():
            print(f"   CUDA Available: Yes")
            print(f"   GPU Device: {torch.cuda.get_device_name()}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        else:
            print(f"   CUDA Available: No")
        
        return True
        
    except Exception as e:
        print(f"❌ System status demonstration failed: {e}")
        return False


def demonstrate_input_validation():
    """Demonstrate input validation."""
    print("\n" + "="*60)
    print("✅ INPUT VALIDATION DEMONSTRATION")
    print("="*60)
    
    if not TRANSFORMERS_AVAILABLE:
        print("❌ Transformers system not available")
        return
    
    try:
        # Test cases
        test_cases = [
            ("Hello world", "microsoft/DialoGPT-medium", 512),
            ("", "microsoft/DialoGPT-medium", 512),  # Empty text
            ("A" * 1000, "microsoft/DialoGPT-medium", 512),  # Very long text
            ("Hello world", "", 512),  # Empty model name
            ("Hello world", "microsoft/DialoGPT-medium", 100)  # Short max length
        ]
        
        print("✅ Testing input validation...")
        
        for i, (text, model_name, max_length) in enumerate(test_cases, 1):
            is_valid, error_msg = validate_transformers_inputs(text, model_name, max_length)
            
            print(f"\n{i}. Text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            print(f"   Model: '{model_name}'")
            print(f"   Max Length: {max_length}")
            print(f"   Valid: {'✅ Yes' if is_valid else '❌ No'}")
            if not is_valid:
                print(f"   Error: {error_msg}")
        
        return True
        
    except Exception as e:
        print(f"❌ Input validation demonstration failed: {e}")
        return False


def demonstrate_configuration_creation():
    """Demonstrate configuration creation."""
    print("\n" + "="*60)
    print("⚙️ CONFIGURATION CREATION DEMONSTRATION")
    print("="*60)
    
    if not TRANSFORMERS_AVAILABLE:
        print("❌ Transformers system not available")
        return
    
    try:
        # Create different configurations
        configs = [
            ("Causal LM", "microsoft/DialoGPT-medium", "causal", "text_generation"),
            ("Sequence Classification", "bert-base-uncased", "sequence_classification", "classification"),
            ("Token Classification", "bert-base-uncased", "token_classification", "token_classification"),
            ("Question Answering", "bert-base-uncased", "question_answering", "question_answering")
        ]
        
        print("⚙️ Creating different configurations...")
        
        for name, model_name, model_type, task in configs:
            config = create_transformers_config(
                model_name=model_name,
                model_type=model_type,
                task=task,
                num_epochs=3,
                batch_size=4,
                learning_rate=5e-5,
                use_peft=True
            )
            
            print(f"\n{name}:")
            print(f"   Model: {config.model_name}")
            print(f"   Type: {config.model_type}")
            print(f"   Task: {config.task}")
            print(f"   Epochs: {config.num_epochs}")
            print(f"   Batch Size: {config.batch_size}")
            print(f"   Learning Rate: {config.learning_rate}")
            print(f"   PEFT: {config.use_peft}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration creation demonstration failed: {e}")
        return False


def main():
    """Run all demonstrations."""
    print("🚀 TRANSFORMERS INTEGRATION SYSTEM DEMONSTRATION")
    print("="*80)
    
    if not TRANSFORMERS_AVAILABLE:
        print("❌ Transformers integration system not available!")
        print("Please install the required dependencies:")
        print("pip install transformers torch peft")
        return
    
    # Run demonstrations
    demonstrations = [
        ("System Status", demonstrate_system_status),
        ("Available Models", demonstrate_available_models),
        ("Configuration Creation", demonstrate_configuration_creation),
        ("Input Validation", demonstrate_input_validation),
        ("Basic Training", demonstrate_basic_training),
        ("Text Generation", demonstrate_text_generation),
        ("Batch Generation", demonstrate_batch_generation),
        ("Text Classification", demonstrate_text_classification)
    ]
    
    results = {}
    
    for name, demo_func in demonstrations:
        try:
            print(f"\n{'='*20} {name} {'='*20}")
            result = demo_func()
            results[name] = result
        except Exception as e:
            print(f"❌ {name} failed with exception: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "="*80)
    print("📊 DEMONSTRATION SUMMARY")
    print("="*80)
    
    successful = sum(results.values())
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{name}: {status}")
    
    print(f"\nOverall: {successful}/{total} demonstrations passed")
    
    if successful == total:
        print("🎉 All demonstrations completed successfully!")
    else:
        print("⚠️ Some demonstrations failed. Check the output above for details.")


match __name__:
    case "__main__":
    main() 