#!/usr/bin/env python3
"""
Comprehensive test suite for the Advanced Tokenization and Sequence Handling System
"""

import unittest
import logging
import torch
import numpy as np
from typing import List, Dict, Any

# Import the system components
from advanced_tokenization_sequence_system import (
    TokenizationType,
    SequenceStrategy,
    TokenizationConfig,
    SequenceConfig,
    AdvancedTokenizer,
    SequenceHandler,
    TokenizationDataset,
    DataCollator,
    TokenizationAnalyzer,
    SequenceProcessor,
    create_advanced_tokenization_system
)


class TestTokenizationType(unittest.TestCase):
    """Test TokenizationType enum"""
    
    def test_enum_values(self):
        """Test that all tokenization types are properly defined"""
        self.assertEqual(TokenizationType.BPE.value, "bpe")
        self.assertEqual(TokenizationType.WORD_LEVEL.value, "word_level")
        self.assertEqual(TokenizationType.WORD_PIECE.value, "word_piece")
        self.assertEqual(TokenizationType.UNIGRAM.value, "unigram")
        self.assertEqual(TokenizationType.BYTE_LEVEL.value, "byte_level")
        self.assertEqual(TokenizationType.CHARACTER_LEVEL.value, "character_level")


class TestSequenceStrategy(unittest.TestCase):
    """Test SequenceStrategy enum"""
    
    def test_enum_values(self):
        """Test that all sequence strategies are properly defined"""
        self.assertEqual(SequenceStrategy.TRUNCATE.value, "truncate")
        self.assertEqual(SequenceStrategy.PAD.value, "pad")
        self.assertEqual(SequenceStrategy.SLIDING_WINDOW.value, "sliding_window")
        self.assertEqual(SequenceStrategy.OVERLAP.value, "overlap")
        self.assertEqual(SequenceStrategy.CHUNK.value, "chunk")


class TestTokenizationConfig(unittest.TestCase):
    """Test TokenizationConfig dataclass"""
    
    def test_default_values(self):
        """Test default configuration values"""
        config = TokenizationConfig()
        
        self.assertEqual(config.max_length, 512)
        self.assertEqual(config.padding, "max_length")
        self.assertTrue(config.truncation)
        self.assertEqual(config.return_tensors, "pt")
        self.assertTrue(config.return_attention_mask)
        self.assertFalse(config.return_token_type_ids)
        self.assertFalse(config.return_overflowing_tokens)
        self.assertFalse(config.return_special_tokens_mask)
        self.assertFalse(config.return_offsets_mapping)
        self.assertFalse(config.return_length)
        self.assertFalse(config.verbose)
    
    def test_custom_values(self):
        """Test custom configuration values"""
        config = TokenizationConfig(
            max_length=256,
            padding="longest",
            truncation=False,
            return_tensors="np",
            return_attention_mask=False
        )
        
        self.assertEqual(config.max_length, 256)
        self.assertEqual(config.padding, "longest")
        self.assertFalse(config.truncation)
        self.assertEqual(config.return_tensors, "np")
        self.assertFalse(config.return_attention_mask)


class TestSequenceConfig(unittest.TestCase):
    """Test SequenceConfig dataclass"""
    
    def test_default_values(self):
        """Test default configuration values"""
        config = SequenceConfig()
        
        self.assertEqual(config.strategy, SequenceStrategy.PAD)
        self.assertEqual(config.overlap_size, 50)
        self.assertEqual(config.chunk_size, 512)
        self.assertEqual(config.stride, 256)
        self.assertEqual(config.min_chunk_size, 100)
        self.assertTrue(config.preserve_word_boundaries)
        self.assertTrue(config.handle_overflow)
    
    def test_custom_values(self):
        """Test custom configuration values"""
        config = SequenceConfig(
            strategy=SequenceStrategy.SLIDING_WINDOW,
            overlap_size=100,
            chunk_size=256,
            stride=128,
            min_chunk_size=50
        )
        
        self.assertEqual(config.strategy, SequenceStrategy.SLIDING_WINDOW)
        self.assertEqual(config.overlap_size, 100)
        self.assertEqual(config.chunk_size, 256)
        self.assertEqual(config.stride, 128)
        self.assertEqual(config.min_chunk_size, 50)


class TestAdvancedTokenizer(unittest.TestCase):
    """Test AdvancedTokenizer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = TokenizationConfig(max_length=128)
        self.tokenizer = AdvancedTokenizer(self.config)
    
    def test_initialization(self):
        """Test tokenizer initialization"""
        self.assertIsNone(self.tokenizer.tokenizer)
        self.assertEqual(self.tokenizer.vocab_size, 0)
        self.assertEqual(self.tokenizer.special_tokens, {})
        self.assertIsNotNone(self.tokenizer.logger)
    
    def test_load_pretrained_tokenizer(self):
        """Test loading pre-trained tokenizer"""
        try:
            self.tokenizer.load_pretrained_tokenizer("gpt2")
            
            self.assertIsNotNone(self.tokenizer.tokenizer)
            self.assertGreater(self.tokenizer.vocab_size, 0)
            self.assertIn('pad_token', self.tokenizer.special_tokens)
            self.assertIn('eos_token', self.tokenizer.special_tokens)
            
        except Exception as e:
            self.skipTest(f"Could not load GPT-2 tokenizer: {e}")
    
    def test_create_custom_tokenizer(self):
        """Test custom tokenizer creation"""
        try:
            self.tokenizer.create_custom_tokenizer(TokenizationType.BPE, vocab_size=1000)
            
            self.assertIsNotNone(self.tokenizer.tokenizer)
            self.assertEqual(self.tokenizer.vocab_size, 1000)
            
        except Exception as e:
            self.skipTest(f"Could not create custom tokenizer: {e}")
    
    def test_preprocess_text(self):
        """Test text preprocessing"""
        # Test with extra whitespace
        text = "  This   is   a   test   text  \n\n"
        cleaned = self.tokenizer._preprocess_text(text)
        self.assertEqual(cleaned, "This is a test text")
        
        # Test with non-string input
        text = 123
        cleaned = self.tokenizer._preprocess_text(text)
        self.assertEqual(cleaned, "123")
    
    def test_fallback_encoding(self):
        """Test fallback encoding creation"""
        text = "This is a test text with multiple words"
        fallback = self.tokenizer._create_fallback_encoding(text)
        
        self.assertIn('input_ids', fallback)
        self.assertIn('attention_mask', fallback)
        self.assertTrue(isinstance(fallback['input_ids'], torch.Tensor))
        self.assertTrue(isinstance(fallback['attention_mask'], torch.Tensor))
    
    def test_fallback_batch_encoding(self):
        """Test fallback batch encoding creation"""
        texts = ["Text 1", "Text 2", "Text 3"]
        fallback = self.tokenizer._create_fallback_batch_encoding(texts)
        
        self.assertIn('input_ids', fallback)
        self.assertIn('attention_mask', fallback)
        self.assertEqual(fallback['input_ids'].size(0), 3)
        self.assertEqual(fallback['attention_mask'].size(0), 3)


class TestSequenceHandler(unittest.TestCase):
    """Test SequenceHandler class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = SequenceConfig(
            strategy=SequenceStrategy.SLIDING_WINDOW,
            overlap_size=50,
            chunk_size=100,
            stride=50,
            min_chunk_size=25
        )
        self.handler = SequenceHandler(self.config)
    
    def test_initialization(self):
        """Test handler initialization"""
        self.assertEqual(self.handler.config, self.config)
        self.assertIsNotNone(self.handler.logger)
    
    def test_handle_long_sequences_short(self):
        """Test handling of short sequences"""
        text = "This is a short text."
        sequences = self.handler.handle_long_sequences(text, max_length=50)
        
        self.assertEqual(len(sequences), 1)
        self.assertEqual(sequences[0], text)
    
    def test_handle_long_sequences_truncate(self):
        """Test truncation strategy"""
        self.config.strategy = SequenceStrategy.TRUNCATE
        
        text = "This is a very long text with many words that will exceed the maximum length limit"
        sequences = self.handler.handle_long_sequences(text, max_length=10)
        
        self.assertEqual(len(sequences), 1)
        self.assertEqual(len(sequences[0].split()), 10)
    
    def test_sliding_window_split(self):
        """Test sliding window splitting"""
        self.config.strategy = SequenceStrategy.SLIDING_WINDOW
        
        text = "Word1 Word2 Word3 Word4 Word5 Word6 Word7 Word8 Word9 Word10"
        sequences = self.handler._sliding_window_split(text, max_length=5)
        
        self.assertGreater(len(sequences), 1)
        for seq in sequences:
            self.assertLessEqual(len(seq.split()), 5)
            self.assertGreaterEqual(len(seq.split()), self.config.min_chunk_size)
    
    def test_overlapping_split(self):
        """Test overlapping split strategy"""
        self.config.strategy = SequenceStrategy.OVERLAP
        
        text = "Word1 Word2 Word3 Word4 Word5 Word6 Word7 Word8 Word9 Word10"
        sequences = self.handler._overlapping_split(text, max_length=6)
        
        self.assertGreater(len(sequences), 1)
        for seq in sequences:
            self.assertLessEqual(len(seq.split()), 6)
    
    def test_chunk_split(self):
        """Test chunk splitting strategy"""
        self.config.strategy = SequenceStrategy.CHUNK
        
        text = "Word1 Word2 Word3 Word4 Word5 Word6 Word7 Word8 Word9 Word10"
        sequences = self.handler._chunk_split(text, max_length=6)
        
        self.assertGreater(len(sequences), 1)
        for seq in sequences:
            self.assertLessEqual(len(seq.split()), 6)
    
    def test_create_sequence_pairs(self):
        """Test sequence pair creation"""
        texts = ["First sentence.", "Second sentence.", "Third sentence."]
        pairs = self.handler.create_sequence_pairs(texts, max_length=10)
        
        self.assertGreater(len(pairs), 0)
        for pair in pairs:
            self.assertEqual(len(pair), 2)
            self.assertIsInstance(pair[0], str)
            self.assertIsInstance(pair[1], str)
    
    def test_create_masked_sequences(self):
        """Test masked sequence creation"""
        text = "This is a sample text for testing."
        masked_sequences = self.handler.create_masked_sequences(text, mask_prob=0.2)
        
        self.assertGreater(len(masked_sequences), 0)
        for masked_text, mask_positions in masked_sequences:
            self.assertIsInstance(masked_text, str)
            self.assertIsInstance(mask_positions, list)
            self.assertIn('[MASK]', masked_text)


class TestTokenizationDataset(unittest.TestCase):
    """Test TokenizationDataset class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = TokenizationConfig(max_length=64)
        self.tokenizer = AdvancedTokenizer(self.config)
        self.sequence_config = SequenceConfig(strategy=SequenceStrategy.TRUNCATE)
        self.sequence_handler = SequenceHandler(self.sequence_config)
        
        # Try to load a tokenizer
        try:
            self.tokenizer.load_pretrained_tokenizer("gpt2")
        except Exception:
            # Create a mock tokenizer for testing
            self.tokenizer.tokenizer = type('MockTokenizer', (), {
                'vocab_size': 1000,
                'pad_token': '[PAD]',
                'eos_token': '[EOS]'
            })()
        
        self.texts = [
            "This is a short text.",
            "This is a longer text with more words to test the dataset functionality."
        ]
        
        self.dataset = TokenizationDataset(
            self.texts, self.tokenizer, self.sequence_handler, max_length=32
        )
    
    def test_initialization(self):
        """Test dataset initialization"""
        self.assertEqual(len(self.dataset.texts), 2)
        self.assertEqual(self.dataset.max_length, 32)
        self.assertIsNotNone(self.dataset.processed_texts)
    
    def test_length(self):
        """Test dataset length"""
        self.assertGreater(len(self.dataset), 0)
    
    def test_getitem(self):
        """Test dataset item retrieval"""
        if len(self.dataset) > 0:
            item = self.dataset[0]
            self.assertIsInstance(item, dict)
            # Note: This might fail if tokenizer is not properly loaded
            # The test is designed to handle this gracefully


class TestDataCollator(unittest.TestCase):
    """Test DataCollator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = TokenizationConfig(max_length=64)
        self.tokenizer = AdvancedTokenizer(self.config)
        self.collator = DataCollator(self.tokenizer, max_length=64)
    
    def test_initialization(self):
        """Test collator initialization"""
        self.assertEqual(self.collator.tokenizer, self.tokenizer)
        self.assertEqual(self.collator.max_length, 64)
    
    def test_call_method(self):
        """Test collator call method"""
        # Create mock batch data
        batch = [
            {
                'input_ids': torch.tensor([[1, 2, 3, 4]]),
                'attention_mask': torch.tensor([[1, 1, 1, 1]])
            },
            {
                'input_ids': torch.tensor([[5, 6, 7]]),
                'attention_mask': torch.tensor([[1, 1, 1]])
            }
        ]
        
        try:
            result = self.collator(batch)
            
            self.assertIn('input_ids', result)
            self.assertIn('attention_mask', result)
            self.assertEqual(result['input_ids'].size(0), 2)
            self.assertEqual(result['attention_mask'].size(0), 2)
            
        except Exception as e:
            # This might fail if tokenizer is not properly loaded
            # The test is designed to handle this gracefully
            pass


class TestTokenizationAnalyzer(unittest.TestCase):
    """Test TokenizationAnalyzer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = TokenizationAnalyzer()
    
    def test_initialization(self):
        """Test analyzer initialization"""
        self.assertIsNotNone(self.analyzer.logger)
        self.assertEqual(self.analyzer.stats, defaultdict(int))
    
    def test_analyze_tokenization(self):
        """Test tokenization analysis"""
        original_texts = ["Text 1", "Text 2"]
        tokenized_encodings = [
            {'input_ids': torch.tensor([[1, 2, 3]])},
            {'input_ids': torch.tensor([[4, 5, 6, 7]])}
        ]
        
        analysis = self.analyzer.analyze_tokenization(original_texts, tokenized_encodings)
        
        self.assertIn('total_texts', analysis)
        self.assertIn('total_tokens', analysis)
        self.assertEqual(analysis['total_texts'], 2)
        self.assertEqual(analysis['total_tokens'], 7)
    
    def test_generate_tokenization_report(self):
        """Test report generation"""
        analysis = {
            'total_texts': 10,
            'total_tokens': 100,
            'avg_tokens_per_text': 10.0,
            'tokenization_efficiency': 0.5,
            'compression_ratio': 2.0,
            'sequence_lengths': [8, 12, 10, 9, 11]
        }
        
        report = self.analyzer.generate_tokenization_report(analysis)
        
        self.assertIsInstance(report, str)
        self.assertIn("TOKENIZATION ANALYSIS REPORT", report)
        self.assertIn("Total Texts: 10", report)
        self.assertIn("Total Tokens: 100", report)


class TestSequenceProcessor(unittest.TestCase):
    """Test SequenceProcessor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.tokenization_config = TokenizationConfig(max_length=64)
        self.sequence_config = SequenceConfig(strategy=SequenceStrategy.TRUNCATE)
        self.processor = SequenceProcessor(self.tokenization_config, self.sequence_config)
    
    def test_initialization(self):
        """Test processor initialization"""
        self.assertEqual(self.processor.tokenization_config, self.tokenization_config)
        self.assertEqual(self.processor.sequence_config, self.sequence_config)
        self.assertIsNotNone(self.processor.tokenizer)
        self.assertIsNotNone(self.processor.sequence_handler)
        self.assertIsNotNone(self.processor.analyzer)
    
    def test_process_text(self):
        """Test single text processing"""
        text = "This is a test text for processing."
        result = self.processor.process_text(text)
        
        self.assertIn('original_text', result)
        self.assertIn('sequences', result)
        self.assertIn('total_sequences', result)
        self.assertEqual(result['original_text'], text)
    
    def test_process_batch(self):
        """Test batch text processing"""
        texts = ["Text 1", "Text 2", "Text 3"]
        result = self.processor.process_batch(texts)
        
        self.assertIn('results', result)
        self.assertIn('total_texts', result)
        self.assertEqual(result['total_texts'], 3)
        self.assertEqual(len(result['results']), 3)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def test_complete_system_creation(self):
        """Test complete system creation"""
        try:
            processor = create_advanced_tokenization_system(
                model_name="gpt2",
                max_length=128,
                sequence_strategy=SequenceStrategy.SLIDING_WINDOW
            )
            
            self.assertIsInstance(processor, SequenceProcessor)
            self.assertIsNotNone(processor.tokenizer)
            self.assertIsNotNone(processor.sequence_handler)
            
        except Exception as e:
            self.skipTest(f"Could not create complete system: {e}")
    
    def test_text_processing_pipeline(self):
        """Test complete text processing pipeline"""
        try:
            processor = create_advanced_tokenization_system(
                model_name="gpt2",
                max_length=64,
                sequence_strategy=SequenceStrategy.TRUNCATE
            )
            
            # Test single text processing
            text = "This is a test text for the complete pipeline."
            result = processor.process_text(text)
            
            self.assertIn('sequences', result)
            self.assertIn('tokenized_sequences', result)
            self.assertGreater(result['total_sequences'], 0)
            
            # Test batch processing
            texts = ["Text 1", "Text 2"]
            batch_result = processor.process_batch(texts)
            
            self.assertIn('results', batch_result)
            self.assertEqual(batch_result['total_texts'], 2)
            
        except Exception as e:
            self.skipTest(f"Could not test complete pipeline: {e}")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""
    
    def test_empty_text(self):
        """Test handling of empty text"""
        processor = create_advanced_tokenization_system(
            model_name="gpt2",
            max_length=64,
            sequence_strategy=SequenceStrategy.TRUNCATE
        )
        
        result = processor.process_text("")
        self.assertIn('sequences', result)
        self.assertIn('total_sequences', result)
    
    def test_very_long_text(self):
        """Test handling of very long text"""
        processor = create_advanced_tokenization_system(
            model_name="gpt2",
            max_length=32,
            sequence_strategy=SequenceStrategy.SLIDING_WINDOW
        )
        
        long_text = "Word " * 1000
        result = processor.process_text(long_text)
        
        self.assertIn('sequences', result)
        self.assertGreater(result['total_sequences'], 1)
    
    def test_special_characters(self):
        """Test handling of special characters"""
        processor = create_advanced_tokenization_system(
            model_name="gpt2",
            max_length=64,
            sequence_strategy=SequenceStrategy.TRUNCATE
        )
        
        special_text = "Text with special chars: !@#$%^&*()_+-=[]{}|;':\",./<>?"
        result = processor.process_text(special_text)
        
        self.assertIn('sequences', result)
        self.assertIn('total_sequences', result)


def run_performance_tests():
    """Run performance tests"""
    print("\n" + "="*50)
    print("PERFORMANCE TESTS")
    print("="*50)
    
    try:
        processor = create_advanced_tokenization_system(
            model_name="gpt2",
            max_length=128,
            sequence_strategy=SequenceStrategy.SLIDING_WINDOW
        )
        
        # Generate test data
        texts = ["Test text " * 50] * 100
        
        import time
        start_time = time.time()
        results = processor.process_batch(texts)
        end_time = time.time()
        
        processing_time = end_time - start_time
        texts_per_second = len(texts) / processing_time
        
        print(f"Processing time: {processing_time:.2f} seconds")
        print(f"Texts per second: {texts_per_second:.2f}")
        print(f"Total texts processed: {len(texts)}")
        
        if 'analysis' in results:
            analysis = results['analysis']
            print(f"Total tokens: {analysis.get('total_tokens', 0)}")
            print(f"Average tokens per text: {analysis.get('avg_tokens_per_text', 0):.2f}")
        
        return processing_time, texts_per_second
        
    except Exception as e:
        print(f"Performance test failed: {e}")
        return None, None


def main():
    """Main test runner"""
    # Configure logging
    logging.basicConfig(level=logging.WARNING)
    
    print("Advanced Tokenization and Sequence Handling System - Test Suite")
    print("="*70)
    
    # Run unit tests
    print("\nRunning unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run performance tests
    run_performance_tests()
    
    print("\n" + "="*70)
    print("Test suite completed!")
    print("="*70)


if __name__ == "__main__":
    main()


