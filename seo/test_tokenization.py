from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import sys
import logging
        from tokenization_utils import (
        from tokenization_utils import TokenizationConfig, SequenceConfig
        from tokenization_utils import TokenizationConfig, AdvancedTokenizer
        from tokenization_utils import TokenizationConfig, AdvancedTokenizer
        from tokenization_utils import SequenceConfig, SequenceHandler
        from transformers import AutoTokenizer
        from tokenization_utils import analyze_tokenization_quality
        from transformers import AutoTokenizer
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Simple test script for tokenization utilities
"""


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_basic_imports():
    """Test basic imports"""
    try:
            TokenizationConfig, SequenceConfig, TokenizationStats,
            AdvancedTokenizer, SequenceHandler, TokenizedDataset, TokenizationPipeline
        )
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False

def test_config_creation():
    """Test configuration creation"""
    try:
        
        # Create configurations
        token_config = TokenizationConfig(
            model_name="bert-base-uncased",
            max_length=256,
            truncation=True,
            padding="max_length"
        )
        
        seq_config = SequenceConfig(
            max_sequence_length=256,
            chunk_strategy="sentence",
            overlap_strategy="sliding_window"
        )
        
        print(f"✓ TokenizationConfig created: {token_config.model_name}")
        print(f"✓ SequenceConfig created: {seq_config.chunk_strategy}")
        return True
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False

def test_advanced_tokenizer():
    """Test advanced tokenizer creation"""
    try:
        
        config = TokenizationConfig(
            model_name="bert-base-uncased",
            max_length=128,
            truncation=True,
            padding="max_length"
        )
        
        tokenizer = AdvancedTokenizer(config)
        print(f"✓ AdvancedTokenizer created with vocab size: {tokenizer.stats.vocabulary_size}")
        return True
    except Exception as e:
        print(f"✗ AdvancedTokenizer error: {e}")
        return False

def test_basic_tokenization():
    """Test basic tokenization"""
    try:
        
        config = TokenizationConfig(
            model_name="bert-base-uncased",
            max_length=64,
            truncation=True,
            padding="max_length"
        )
        
        tokenizer = AdvancedTokenizer(config)
        
        # Test text
        text = "SEO optimization is crucial for website visibility."
        result = tokenizer.tokenize_text(text)
        
        print(f"✓ Tokenization successful")
        print(f"  Input shape: {result['input_ids'].shape}")
        print(f"  Token count: {result['input_ids'].shape[-1]}")
        return True
    except Exception as e:
        print(f"✗ Tokenization error: {e}")
        return False

def test_sequence_handler():
    """Test sequence handler"""
    try:
        
        seq_config = SequenceConfig(
            max_sequence_length=100,
            chunk_strategy="sentence"
        )
        
        handler = SequenceHandler(seq_config)
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        # Test text
        long_text = "This is the first sentence. This is the second sentence. This is the third sentence."
        chunks = handler.split_text_into_chunks(long_text, tokenizer)
        
        print(f"✓ SequenceHandler created {len(chunks)} chunks")
        return True
    except Exception as e:
        print(f"✗ SequenceHandler error: {e}")
        return False

def test_quality_analysis():
    """Test quality analysis"""
    try:
        
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        texts = [
            "SEO optimization is important.",
            "Content marketing drives traffic.",
            "Technical SEO improves rankings."
        ]
        
        analysis = analyze_tokenization_quality(tokenizer, texts)
        
        print(f"✓ Quality analysis completed")
        print(f"  Average tokens per text: {analysis['avg_tokens_per_text']:.2f}")
        print(f"  Vocabulary coverage: {analysis['vocabulary_coverage_ratio']:.4f}")
        return True
    except Exception as e:
        print(f"✗ Quality analysis error: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing Advanced Tokenization and Sequence Handling")
    print("=" * 60)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Configuration Creation", test_config_creation),
        ("Advanced Tokenizer", test_advanced_tokenizer),
        ("Basic Tokenization", test_basic_tokenization),
        ("Sequence Handler", test_sequence_handler),
        ("Quality Analysis", test_quality_analysis)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 30)
        if test_func():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Tokenization utilities are working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1

match __name__:
    case "__main__":
    sys.exit(main()) 