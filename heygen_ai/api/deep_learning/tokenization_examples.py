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
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import logging
import time
import json
            from .tokenization import TokenizationConfig, WordTokenizer
            from .tokenization import TokenizationConfig, CharacterTokenizer
            from .tokenization import TokenizationConfig, SubwordTokenizer
            from .tokenization import TokenizationConfig, WordTokenizer
            import tempfile
            import os
            from .tokenization import TokenizationConfig, WordTokenizer
            from .tokenization import SequenceHandler
            from .tokenization import SequenceHandler
            from .tokenization import SequenceHandler
            from .tokenization import SequenceHandler
            from .tokenization import SequenceHandler
            from .tokenization import (
            from .tokenization import (
            from .tokenization import TokenizationConfig, WordTokenizer, CharacterTokenizer, SubwordTokenizer
            import psutil
            import gc
            from .tokenization import TokenizationConfig, WordTokenizer, CharacterTokenizer, SubwordTokenizer
from typing import Any, List, Dict, Optional
import asyncio
"""
Tokenization and Sequence Handling Examples for HeyGen AI.

Comprehensive examples demonstrating usage of tokenization and sequence handling
following PEP 8 style guidelines.
"""


logger = logging.getLogger(__name__)


class TokenizationExamples:
    """Examples of tokenization usage."""

    @staticmethod
    def word_tokenizer_example():
        """Word tokenizer example."""
        
        try:
            
            # Create configuration
            config = TokenizationConfig(
                vocab_size=10000,
                min_frequency=2,
                max_sequence_length=512,
                lowercase=True,
                remove_punctuation=False,
                normalize_whitespace=True
            )
            
            # Create word tokenizer
            tokenizer = WordTokenizer(config)
            
            # Sample texts
            texts = [
                "Hello world, this is a test sentence.",
                "Machine learning is amazing and powerful.",
                "Natural language processing with deep learning.",
                "Transformers have revolutionized NLP tasks."
            ]
            
            # Build vocabulary
            tokenizer.build_vocab(texts)
            
            # Test tokenization
            test_text = "Hello world, this is a test sentence."
            tokens = tokenizer.tokenize(test_text)
            token_ids = tokenizer.encode(test_text)
            decoded_text = tokenizer.decode(token_ids)
            
            logger.info(f"Original text: {test_text}")
            logger.info(f"Tokens: {tokens}")
            logger.info(f"Token IDs: {token_ids}")
            logger.info(f"Decoded text: {decoded_text}")
            logger.info(f"Vocabulary size: {len(tokenizer.vocab)}")
            
            return tokenizer, tokens, token_ids, decoded_text
            
        except ImportError as e:
            logger.error(f"Tokenization module not available: {e}")
            return None, None, None, None

    @staticmethod
    def character_tokenizer_example():
        """Character tokenizer example."""
        
        try:
            
            # Create configuration
            config = TokenizationConfig(
                vocab_size=1000,
                min_frequency=1,
                max_sequence_length=512,
                lowercase=True,
                remove_punctuation=False,
                normalize_whitespace=True
            )
            
            # Create character tokenizer
            tokenizer = CharacterTokenizer(config)
            
            # Sample texts
            texts = [
                "Hello world!",
                "Machine learning.",
                "Deep learning models.",
                "Natural language processing."
            ]
            
            # Build vocabulary
            tokenizer.build_vocab(texts)
            
            # Test tokenization
            test_text = "Hello world!"
            tokens = tokenizer.tokenize(test_text)
            token_ids = tokenizer.encode(test_text)
            decoded_text = tokenizer.decode(token_ids)
            
            logger.info(f"Original text: {test_text}")
            logger.info(f"Characters: {tokens}")
            logger.info(f"Character IDs: {token_ids}")
            logger.info(f"Decoded text: {decoded_text}")
            logger.info(f"Vocabulary size: {len(tokenizer.vocab)}")
            
            return tokenizer, tokens, token_ids, decoded_text
            
        except ImportError as e:
            logger.error(f"Tokenization module not available: {e}")
            return None, None, None, None

    @staticmethod
    def subword_tokenizer_example():
        """Subword tokenizer example."""
        
        try:
            
            # Create configuration
            config = TokenizationConfig(
                vocab_size=5000,
                min_frequency=2,
                max_sequence_length=512,
                lowercase=True,
                remove_punctuation=False,
                normalize_whitespace=True
            )
            
            # Create subword tokenizer
            tokenizer = SubwordTokenizer(config)
            
            # Sample texts
            texts = [
                "Hello world, this is a test sentence.",
                "Machine learning is amazing and powerful.",
                "Natural language processing with deep learning.",
                "Transformers have revolutionized NLP tasks.",
                "BERT and GPT are popular transformer models.",
                "Attention mechanisms are key to transformer success."
            ]
            
            # Build vocabulary
            tokenizer.build_vocab(texts)
            
            # Test tokenization
            test_text = "Hello world, this is a test sentence."
            tokens = tokenizer.tokenize(test_text)
            token_ids = tokenizer.encode(test_text)
            decoded_text = tokenizer.decode(token_ids)
            
            logger.info(f"Original text: {test_text}")
            logger.info(f"Subword tokens: {tokens}")
            logger.info(f"Token IDs: {token_ids}")
            logger.info(f"Decoded text: {decoded_text}")
            logger.info(f"Vocabulary size: {len(tokenizer.vocab)}")
            logger.info(f"Number of merges: {len(tokenizer.merges)}")
            
            return tokenizer, tokens, token_ids, decoded_text
            
        except ImportError as e:
            logger.error(f"Tokenization module not available: {e}")
            return None, None, None, None

    @staticmethod
    def batch_encoding_example():
        """Batch encoding example."""
        
        try:
            
            # Create tokenizer
            config = TokenizationConfig(
                vocab_size=5000,
                min_frequency=1,
                max_sequence_length=100,
                lowercase=True
            )
            tokenizer = WordTokenizer(config)
            
            # Sample texts
            texts = [
                "Hello world, this is a test.",
                "Machine learning is amazing.",
                "Deep learning models are powerful.",
                "Natural language processing."
            ]
            
            # Build vocabulary
            tokenizer.build_vocab(texts)
            
            # Batch encoding
            encoded_batch = tokenizer.encode_batch(texts, padding=True, truncation=True)
            
            logger.info(f"Input texts: {texts}")
            logger.info(f"Encoded batch shape: {encoded_batch.shape}")
            logger.info(f"Encoded batch:\n{encoded_batch}")
            
            # Decode batch
            decoded_texts = []
            for i in range(encoded_batch.size(0)):
                decoded_text = tokenizer.decode(encoded_batch[i].tolist())
                decoded_texts.append(decoded_text)
            
            logger.info(f"Decoded texts: {decoded_texts}")
            
            return tokenizer, encoded_batch, decoded_texts
            
        except ImportError as e:
            logger.error(f"Tokenization module not available: {e}")
            return None, None, None

    @staticmethod
    def save_load_tokenizer_example():
        """Save and load tokenizer example."""
        
        try:
            
            
            # Create tokenizer
            config = TokenizationConfig(
                vocab_size=1000,
                min_frequency=1,
                max_sequence_length=100,
                lowercase=True
            )
            tokenizer = WordTokenizer(config)
            
            # Sample texts
            texts = [
                "Hello world, this is a test.",
                "Machine learning is amazing.",
                "Deep learning models are powerful."
            ]
            
            # Build vocabulary
            tokenizer.build_vocab(texts)
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
                temp_filepath = tmp_file.name
            
            try:
                # Save tokenizer
                tokenizer.save(temp_filepath)
                logger.info(f"Tokenizer saved to {temp_filepath}")
                
                # Create new tokenizer and load
                new_tokenizer = WordTokenizer(config)
                new_tokenizer.load(temp_filepath)
                logger.info(f"Tokenizer loaded from {temp_filepath}")
                
                # Test that they work the same
                test_text = "Hello world, this is a test."
                original_ids = tokenizer.encode(test_text)
                loaded_ids = new_tokenizer.encode(test_text)
                
                logger.info(f"Original tokenizer IDs: {original_ids}")
                logger.info(f"Loaded tokenizer IDs: {loaded_ids}")
                logger.info(f"IDs match: {original_ids == loaded_ids}")
                
                return tokenizer, new_tokenizer, original_ids, loaded_ids
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_filepath):
                    os.unlink(temp_filepath)
            
        except ImportError as e:
            logger.error(f"Tokenization module not available: {e}")
            return None, None, None, None


class SequenceHandlerExamples:
    """Examples of sequence handling usage."""

    @staticmethod
    def padding_example():
        """Sequence padding example."""
        
        try:
            
            # Create sequence handler
            handler = SequenceHandler(max_sequence_length=10)
            
            # Sample sequences
            sequences = [
                [1, 2, 3],
                [1, 2, 3, 4, 5],
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # Will be truncated
            ]
            
            # Pad sequences
            padded_sequences = handler.pad_sequences(
                sequences,
                padding="post",
                truncation="post",
                value=0
            )
            
            logger.info(f"Original sequences: {sequences}")
            logger.info(f"Padded sequences shape: {padded_sequences.shape}")
            logger.info(f"Padded sequences:\n{padded_sequences}")
            
            return handler, sequences, padded_sequences
            
        except ImportError as e:
            logger.error(f"Tokenization module not available: {e}")
            return None, None, None

    @staticmethod
    def attention_mask_example():
        """Attention mask example."""
        
        try:
            
            # Create sequence handler
            handler = SequenceHandler(max_sequence_length=10)
            
            # Sample sequences
            sequences = [
                [1, 2, 3, 0, 0],
                [1, 2, 3, 4, 5],
                [1, 2, 0, 0, 0]
            ]
            
            # Convert to tensor
            sequences_tensor = torch.tensor(sequences)
            
            # Create attention mask
            attention_mask = handler.create_attention_mask(sequences_tensor, padding_value=0)
            
            logger.info(f"Sequences:\n{sequences_tensor}")
            logger.info(f"Attention mask:\n{attention_mask}")
            
            return handler, sequences_tensor, attention_mask
            
        except ImportError as e:
            logger.error(f"Tokenization module not available: {e}")
            return None, None, None

    @staticmethod
    def causal_mask_example():
        """Causal mask example."""
        
        try:
            
            # Create sequence handler
            handler = SequenceHandler(max_sequence_length=10)
            
            # Create causal mask
            sequence_length = 5
            causal_mask = handler.create_causal_mask(sequence_length)
            
            logger.info(f"Causal mask for sequence length {sequence_length}:")
            logger.info(f"Mask shape: {causal_mask.shape}")
            logger.info(f"Mask:\n{causal_mask}")
            
            return handler, causal_mask
            
        except ImportError as e:
            logger.error(f"Tokenization module not available: {e}")
            return None, None

    @staticmethod
    def sliding_window_mask_example():
        """Sliding window mask example."""
        
        try:
            
            # Create sequence handler
            handler = SequenceHandler(max_sequence_length=10)
            
            # Create sliding window mask
            sequence_length = 8
            window_size = 3
            stride = 1
            sliding_mask = handler.create_sliding_window_mask(
                sequence_length,
                window_size,
                stride
            )
            
            logger.info(f"Sliding window mask:")
            logger.info(f"Sequence length: {sequence_length}")
            logger.info(f"Window size: {window_size}")
            logger.info(f"Stride: {stride}")
            logger.info(f"Mask shape: {sliding_mask.shape}")
            logger.info(f"Mask:\n{sliding_mask}")
            
            return handler, sliding_mask
            
        except ImportError as e:
            logger.error(f"Tokenization module not available: {e}")
            return None, None

    @staticmethod
    def sequence_splitting_example():
        """Sequence splitting example."""
        
        try:
            
            # Create sequence handler
            handler = SequenceHandler(max_sequence_length=10)
            
            # Sample sequence
            sequence = torch.randn(2, 15, 64)  # (batch_size, seq_len, hidden_dim)
            
            # Split into chunks
            chunk_size = 5
            overlap = 1
            chunks = handler.split_sequences(sequence, chunk_size, overlap)
            
            logger.info(f"Original sequence shape: {sequence.shape}")
            logger.info(f"Chunk size: {chunk_size}")
            logger.info(f"Overlap: {overlap}")
            logger.info(f"Number of chunks: {len(chunks)}")
            
            for i, chunk in enumerate(chunks):
                logger.info(f"Chunk {i} shape: {chunk.shape}")
            
            # Merge chunks
            merged_sequence = handler.merge_sequences(chunks, overlap, strategy="mean")
            
            logger.info(f"Merged sequence shape: {merged_sequence.shape}")
            
            return handler, sequence, chunks, merged_sequence
            
        except ImportError as e:
            logger.error(f"Tokenization module not available: {e}")
            return None, None, None, None


class DatasetExamples:
    """Examples of dataset usage."""

    @staticmethod
    def text_dataset_example():
        """Text dataset example."""
        
        try:
                TokenizationConfig, WordTokenizer, create_text_dataset
            )
            
            # Create tokenizer
            config = TokenizationConfig(
                vocab_size=1000,
                min_frequency=1,
                max_sequence_length=20,
                lowercase=True
            )
            tokenizer = WordTokenizer(config)
            
            # Sample texts
            texts = [
                "Hello world, this is a test sentence.",
                "Machine learning is amazing and powerful.",
                "Deep learning models are very effective.",
                "Natural language processing is fascinating.",
                "Transformers have changed the field of NLP."
            ]
            
            # Build vocabulary
            tokenizer.build_vocab(texts)
            
            # Create dataset
            dataset = create_text_dataset(
                texts=texts,
                tokenizer=tokenizer,
                max_sequence_length=20,
                add_special_tokens=True
            )
            
            logger.info(f"Dataset size: {len(dataset)}")
            logger.info(f"Sample sequence: {dataset[0]}")
            logger.info(f"Sample sequence shape: {dataset[0].shape}")
            
            # Get batch
            batch = dataset.get_batch([0, 1, 2])
            logger.info(f"Batch shape: {batch.shape}")
            logger.info(f"Batch:\n{batch}")
            
            return dataset, tokenizer, batch
            
        except ImportError as e:
            logger.error(f"Tokenization module not available: {e}")
            return None, None, None

    @staticmethod
    def dataloader_example():
        """Data loader example."""
        
        try:
                TokenizationConfig, WordTokenizer, create_text_dataset, create_text_dataloader
            )
            
            # Create tokenizer
            config = TokenizationConfig(
                vocab_size=1000,
                min_frequency=1,
                max_sequence_length=20,
                lowercase=True
            )
            tokenizer = WordTokenizer(config)
            
            # Sample texts
            texts = [
                "Hello world, this is a test sentence.",
                "Machine learning is amazing and powerful.",
                "Deep learning models are very effective.",
                "Natural language processing is fascinating.",
                "Transformers have changed the field of NLP.",
                "BERT and GPT are popular models.",
                "Attention mechanisms are important.",
                "Neural networks learn patterns."
            ]
            
            # Build vocabulary
            tokenizer.build_vocab(texts)
            
            # Create dataset
            dataset = create_text_dataset(
                texts=texts,
                tokenizer=tokenizer,
                max_sequence_length=20,
                add_special_tokens=True
            )
            
            # Create data loader
            dataloader = create_text_dataloader(
                dataset=dataset,
                batch_size=3,
                shuffle=True,
                drop_last=False
            )
            
            logger.info(f"Dataset size: {len(dataset)}")
            logger.info(f"Number of batches: {len(dataloader)}")
            
            # Iterate through batches
            for i, batch in enumerate(dataloader):
                logger.info(f"Batch {i} shape: {batch.shape}")
                logger.info(f"Batch {i}:\n{batch}")
                if i >= 2:  # Show first 3 batches
                    break
            
            return dataset, dataloader
            
        except ImportError as e:
            logger.error(f"Tokenization module not available: {e}")
            return None, None


class PerformanceExamples:
    """Examples of tokenization performance analysis."""

    @staticmethod
    def tokenization_speed_comparison():
        """Compare tokenization speed of different methods."""
        
        try:
            
            # Sample texts
            texts = [
                "Hello world, this is a test sentence for tokenization speed comparison.",
                "Machine learning is amazing and powerful for various applications.",
                "Deep learning models are very effective in solving complex problems.",
                "Natural language processing is fascinating and has many applications.",
                "Transformers have changed the field of NLP significantly."
            ] * 100  # Repeat to have more data
            
            # Test different tokenizers
            tokenizers = {
                "word": WordTokenizer(TokenizationConfig(vocab_size=1000, min_frequency=1)),
                "character": CharacterTokenizer(TokenizationConfig(vocab_size=1000, min_frequency=1)),
                "subword": SubwordTokenizer(TokenizationConfig(vocab_size=1000, min_frequency=1))
            }
            
            results = {}
            
            for name, tokenizer in tokenizers.items():
                logger.info(f"Testing {name} tokenizer...")
                
                # Build vocabulary
                start_time = time.time()
                tokenizer.build_vocab(texts)
                vocab_time = time.time() - start_time
                
                # Test tokenization speed
                start_time = time.time()
                for text in texts:
                    tokens = tokenizer.tokenize(text)
                tokenize_time = time.time() - start_time
                
                # Test encoding speed
                start_time = time.time()
                for text in texts:
                    token_ids = tokenizer.encode(text)
                encode_time = time.time() - start_time
                
                results[name] = {
                    "vocab_time": vocab_time,
                    "tokenize_time": tokenize_time,
                    "encode_time": encode_time,
                    "vocab_size": len(tokenizer.vocab),
                    "avg_tokens_per_text": sum(len(tokenizer.tokenize(text)) for text in texts) / len(texts)
                }
                
                logger.info(f"{name} tokenizer results:")
                logger.info(f"  Vocabulary building time: {vocab_time:.4f}s")
                logger.info(f"  Tokenization time: {tokenize_time:.4f}s")
                logger.info(f"  Encoding time: {encode_time:.4f}s")
                logger.info(f"  Vocabulary size: {len(tokenizer.vocab)}")
                logger.info(f"  Average tokens per text: {results[name]['avg_tokens_per_text']:.2f}")
            
            return results
            
        except ImportError as e:
            logger.error(f"Tokenization module not available: {e}")
            return None

    @staticmethod
    def memory_usage_analysis():
        """Analyze memory usage of different tokenization methods."""
        
        try:
            
            
            # Sample texts
            texts = [
                "Hello world, this is a test sentence for memory analysis.",
                "Machine learning is amazing and powerful for various applications.",
                "Deep learning models are very effective in solving complex problems.",
                "Natural language processing is fascinating and has many applications.",
                "Transformers have changed the field of NLP significantly."
            ] * 50  # Repeat to have more data
            
            # Test different tokenizers
            tokenizers = {
                "word": WordTokenizer(TokenizationConfig(vocab_size=1000, min_frequency=1)),
                "character": CharacterTokenizer(TokenizationConfig(vocab_size=1000, min_frequency=1)),
                "subword": SubwordTokenizer(TokenizationConfig(vocab_size=1000, min_frequency=1))
            }
            
            results = {}
            
            for name, tokenizer in tokenizers.items():
                logger.info(f"Testing {name} tokenizer memory usage...")
                
                # Get initial memory
                process = psutil.Process()
                initial_memory = process.memory_info().rss / 1024 / 1024  # MB
                
                # Build vocabulary
                tokenizer.build_vocab(texts)
                
                # Get memory after vocabulary building
                vocab_memory = process.memory_info().rss / 1024 / 1024  # MB
                
                # Tokenize all texts
                all_tokens = []
                for text in texts:
                    tokens = tokenizer.tokenize(text)
                    all_tokens.append(tokens)
                
                # Get memory after tokenization
                tokenize_memory = process.memory_info().rss / 1024 / 1024  # MB
                
                # Encode all texts
                all_encodings = []
                for text in texts:
                    token_ids = tokenizer.encode(text)
                    all_encodings.append(token_ids)
                
                # Get memory after encoding
                encode_memory = process.memory_info().rss / 1024 / 1024  # MB
                
                results[name] = {
                    "initial_memory_mb": initial_memory,
                    "vocab_memory_mb": vocab_memory,
                    "tokenize_memory_mb": tokenize_memory,
                    "encode_memory_mb": encode_memory,
                    "vocab_memory_increase": vocab_memory - initial_memory,
                    "tokenize_memory_increase": tokenize_memory - vocab_memory,
                    "encode_memory_increase": encode_memory - tokenize_memory,
                    "total_memory_increase": encode_memory - initial_memory
                }
                
                logger.info(f"{name} tokenizer memory results:")
                logger.info(f"  Initial memory: {initial_memory:.2f} MB")
                logger.info(f"  After vocabulary: {vocab_memory:.2f} MB (+{results[name]['vocab_memory_increase']:.2f} MB)")
                logger.info(f"  After tokenization: {tokenize_memory:.2f} MB (+{results[name]['tokenize_memory_increase']:.2f} MB)")
                logger.info(f"  After encoding: {encode_memory:.2f} MB (+{results[name]['encode_memory_increase']:.2f} MB)")
                logger.info(f"  Total increase: {results[name]['total_memory_increase']:.2f} MB")
                
                # Clean up
                del all_tokens, all_encodings
                gc.collect()
            
            return results
            
        except ImportError as e:
            logger.error(f"Required modules not available: {e}")
            return None


def run_tokenization_examples():
    """Run all tokenization examples."""
    
    logger.info("Running Tokenization and Sequence Handling Examples")
    logger.info("=" * 60)
    
    # Tokenization examples
    logger.info("\n1. Word Tokenizer Example:")
    word_tokenizer, word_tokens, word_ids, word_decoded = TokenizationExamples.word_tokenizer_example()
    
    logger.info("\n2. Character Tokenizer Example:")
    char_tokenizer, char_tokens, char_ids, char_decoded = TokenizationExamples.character_tokenizer_example()
    
    logger.info("\n3. Subword Tokenizer Example:")
    subword_tokenizer, subword_tokens, subword_ids, subword_decoded = TokenizationExamples.subword_tokenizer_example()
    
    logger.info("\n4. Batch Encoding Example:")
    batch_tokenizer, batch_encoded, batch_decoded = TokenizationExamples.batch_encoding_example()
    
    logger.info("\n5. Save/Load Tokenizer Example:")
    save_tokenizer, load_tokenizer, save_ids, load_ids = TokenizationExamples.save_load_tokenizer_example()
    
    # Sequence handler examples
    logger.info("\n6. Padding Example:")
    padding_handler, padding_sequences, padding_result = SequenceHandlerExamples.padding_example()
    
    logger.info("\n7. Attention Mask Example:")
    mask_handler, mask_sequences, mask_result = SequenceHandlerExamples.attention_mask_example()
    
    logger.info("\n8. Causal Mask Example:")
    causal_handler, causal_mask = SequenceHandlerExamples.causal_mask_example()
    
    logger.info("\n9. Sliding Window Mask Example:")
    sliding_handler, sliding_mask = SequenceHandlerExamples.sliding_window_mask_example()
    
    logger.info("\n10. Sequence Splitting Example:")
    split_handler, split_sequence, split_chunks, split_merged = SequenceHandlerExamples.sequence_splitting_example()
    
    # Dataset examples
    logger.info("\n11. Text Dataset Example:")
    text_dataset, dataset_tokenizer, dataset_batch = DatasetExamples.text_dataset_example()
    
    logger.info("\n12. Data Loader Example:")
    dataloader_dataset, dataloader = DatasetExamples.dataloader_example()
    
    # Performance examples
    logger.info("\n13. Tokenization Speed Comparison:")
    speed_results = PerformanceExamples.tokenization_speed_comparison()
    
    logger.info("\n14. Memory Usage Analysis:")
    memory_results = PerformanceExamples.memory_usage_analysis()
    
    logger.info("\nAll tokenization examples completed successfully!")
    
    return {
        "tokenizers": {
            "word_tokenizer": word_tokenizer,
            "char_tokenizer": char_tokenizer,
            "subword_tokenizer": subword_tokenizer,
            "batch_tokenizer": batch_tokenizer,
            "save_tokenizer": save_tokenizer,
            "load_tokenizer": load_tokenizer
        },
        "handlers": {
            "padding_handler": padding_handler,
            "mask_handler": mask_handler,
            "causal_handler": causal_handler,
            "sliding_handler": sliding_handler,
            "split_handler": split_handler
        },
        "datasets": {
            "text_dataset": text_dataset,
            "dataloader_dataset": dataloader_dataset,
            "dataset_tokenizer": dataset_tokenizer
        },
        "results": {
            "word_tokens": word_tokens,
            "word_ids": word_ids,
            "char_tokens": char_tokens,
            "char_ids": char_ids,
            "subword_tokens": subword_tokens,
            "subword_ids": subword_ids,
            "batch_encoded": batch_encoded,
            "padding_result": padding_result,
            "mask_result": mask_result,
            "causal_mask": causal_mask,
            "sliding_mask": sliding_mask,
            "split_chunks": split_chunks,
            "split_merged": split_merged,
            "dataset_batch": dataset_batch,
            "speed_results": speed_results,
            "memory_results": memory_results
        }
    }


if __name__ == "__main__":
    # Run examples
    examples = run_tokenization_examples()
    logger.info("Tokenization and Sequence Handling Examples completed!") 