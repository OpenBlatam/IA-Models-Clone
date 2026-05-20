"""
Comprehensive Demonstration of Advanced Tokenization and Sequence Handling System.
"""

import torch
import numpy as np
from advanced_tokenization_system import (
    TokenizationConfig, SequenceConfig, AdvancedTextProcessor,
    AdvancedTokenizer, SequenceProcessor, TextPreprocessor,
    DataCollatorFactory
)
from transformers import AutoTokenizer
import time
import json


class TokenizationDemo:
    """Comprehensive demonstration of the tokenization system."""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Using device: {self.device}")
        
        # Sample texts for demonstration
        self.sample_texts = [
            "The quick brown fox jumps over the lazy dog. This is a sample text for testing tokenization.",
            "Artificial intelligence is transforming the world. Machine learning models are becoming more sophisticated.",
            "Natural language processing enables computers to understand human language. It's a fascinating field.",
            "Deep learning models require large amounts of data and computational resources for training.",
            "Transformers have revolutionized natural language processing with their attention mechanisms."
        ]
        
        # Long text for sequence handling
        self.long_text = """
        Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence 
        concerned with the interactions between computers and human language, in particular how to program computers 
        to process and analyze large amounts of natural language data. Challenges in natural language processing 
        frequently involve speech recognition, natural language understanding, and natural language generation.
        
        The field has seen significant advances in recent years, particularly with the introduction of transformer 
        architectures and large language models. These models have demonstrated remarkable capabilities in various 
        NLP tasks including text generation, translation, summarization, and question answering.
        
        However, training these models requires substantial computational resources and large datasets. The models 
        can have billions of parameters and require specialized hardware for efficient training and inference.
        
        Despite these challenges, NLP continues to evolve rapidly, with new architectures and training methods 
        being developed regularly. The field holds great promise for applications in healthcare, education, 
        customer service, and many other domains.
        """
    
    def demo_basic_tokenization(self):
        """Demonstrate basic tokenization capabilities."""
        print("\n" + "="*60)
        print("🔤 DEMO: Basic Tokenization Capabilities")
        print("="*60)
        
        # Configuration for GPT-2
        config = TokenizationConfig(
            model_name="gpt2",
            max_length=256,
            padding="longest",
            return_attention_mask=True,
            return_token_type_ids=True,
            verbose=True
        )
        
        # Create tokenizer
        tokenizer = AdvancedTokenizer(config)
        
        # Tokenize single text
        print(f"\n📝 Tokenizing single text:")
        text = self.sample_texts[0]
        result = tokenizer.tokenize_with_metadata(text)
        
        print(f"   Original text: {text[:100]}...")
        print(f"   Text length: {len(text)} characters")
        print(f"   Token count: {len(result['input_ids'][0])} tokens")
        print(f"   Vocabulary size: {result['metadata']['vocabulary_size']:,}")
        print(f"   Vocabulary coverage: {result['metadata']['vocabulary_coverage']:.2%}")
        
        # Show special tokens
        print(f"\n🏷️  Special tokens:")
        for token_name, token_value in result['metadata']['special_tokens'].items():
            print(f"   {token_name}: {token_value}")
        
        # Decode tokens back to text
        decoded = tokenizer.decode_tokens(result['input_ids'][0])
        print(f"\n🔄 Decoded text: {decoded[:100]}...")
        
        return tokenizer
    
    def demo_sequence_processing(self):
        """Demonstrate advanced sequence processing."""
        print("\n" + "="*60)
        print("🔀 DEMO: Advanced Sequence Processing")
        print("="*60)
        
        # Configuration
        config = SequenceConfig(
            max_sequence_length=512,
            target_sequence_length=256,
            padding_strategy="longest",
            truncation_strategy="longest_first",
            handle_long_sequences=True,
            sliding_window=True,
            window_size=256,
            window_stride=128
        )
        
        # Create processor
        processor = SequenceProcessor(config)
        
        # Create sample sequences of different lengths
        sequences = [
            torch.randint(0, 1000, (100,)),   # Short sequence
            torch.randint(0, 1000, (300,)),   # Medium sequence
            torch.randint(0, 1000, (600,)),   # Long sequence
            torch.randint(0, 1000, (50,))     # Very short sequence
        ]
        
        print(f"\n📏 Sequence lengths before processing:")
        for i, seq in enumerate(sequences):
            print(f"   Sequence {i+1}: {len(seq)} tokens")
        
        # Process sequences
        processed = processor.process_sequences(sequences)
        print(f"\n✅ Processed sequences shape: {processed.shape}")
        
        # Handle long sequences with sliding window
        print(f"\n🪟 Sliding window for long sequences:")
        long_sequence = torch.randint(0, 1000, (800,))
        windows = processor.create_sliding_windows(long_sequence, window_size=256, stride=128)
        print(f"   Original length: {len(long_sequence)} tokens")
        print(f"   Number of windows: {len(windows)}")
        print(f"   Window shape: {windows[0].shape}")
        
        # Create attention masks
        attention_masks = processor.create_attention_masks(processed)
        print(f"\n🎭 Attention masks shape: {attention_masks.shape}")
        print(f"   Non-zero elements: {attention_masks.sum().item()}")
        
        return processor
    
    def demo_text_preprocessing(self):
        """Demonstrate text preprocessing capabilities."""
        print("\n" + "="*60)
        print("🧹 DEMO: Advanced Text Preprocessing")
        print("="*60)
        
        # Create preprocessor
        preprocessor = TextPreprocessor()
        
        # Sample text with various elements
        sample_text = """
        Check out this amazing website: https://example.com! 
        Contact us at info@example.com or call (555) 123-4567.
        This text has some WEIRD formatting and    extra   spaces.
        It also contains some accented characters: café, naïve, résumé.
        """
        
        print(f"\n📝 Original text:")
        print(f"   {sample_text}")
        
        # Clean text
        cleaned = preprocessor.clean_text(
            sample_text,
            remove_urls=True,
            remove_emails=True,
            remove_phone_numbers=True,
            normalize_whitespace=True
        )
        print(f"\n🧹 Cleaned text:")
        print(f"   {cleaned}")
        
        # Normalize text
        normalized = preprocessor.normalize_text(
            cleaned,
            lowercase=True,
            remove_accents=True,
            normalize_unicode=True
        )
        print(f"\n📏 Normalized text:")
        print(f"   {normalized}")
        
        # Split into chunks
        print(f"\n✂️  Text chunking:")
        chunks = preprocessor.split_into_chunks(
            self.long_text,
            chunk_size=200,
            overlap=50,
            split_by="sentences"
        )
        print(f"   Number of chunks: {len(chunks)}")
        for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
            print(f"   Chunk {i+1}: {len(chunk)} characters")
            print(f"      {chunk[:100]}...")
        
        # Text augmentation
        print(f"\n🎲 Text augmentation:")
        augmented = preprocessor.create_text_augmentations(
            "This is a sample text for augmentation.",
            methods=["random_mask", "random_insert", "random_swap"]
        )
        for i, aug_text in enumerate(augmented):
            print(f"   Augmentation {i+1}: {aug_text}")
        
        return preprocessor
    
    def demo_batch_processing(self):
        """Demonstrate batch processing capabilities."""
        print("\n" + "="*60)
        print("📦 DEMO: Batch Processing & Dataset Creation")
        print("="*60)
        
        # Configuration
        tokenization_config = TokenizationConfig(
            model_name="gpt2",
            max_length=256,
            padding="longest",
            return_attention_mask=True,
            return_token_type_ids=True
        )
        
        sequence_config = SequenceConfig(
            max_sequence_length=512,
            target_sequence_length=256,
            padding_strategy="longest",
            truncation_strategy="longest_first",
            handle_long_sequences=True
        )
        
        # Create processor
        processor = AdvancedTextProcessor(tokenization_config, sequence_config)
        
        # Process batch
        print(f"\n📝 Processing batch of {len(self.sample_texts)} texts:")
        batch_results = processor.process_batch(self.sample_texts, batch_size=2)
        print(f"   Processed {len(batch_results)} texts")
        
        # Show individual results
        for i, result in enumerate(batch_results[:2]):  # Show first 2
            print(f"\n   Text {i+1}:")
            print(f"      Original: {result['preprocessing']['original_text'][:80]}...")
            print(f"      Cleaned: {result['preprocessing']['cleaned_text'][:80]}...")
            print(f"      Tokens: {len(result['input_ids'][0])} tokens")
        
        # Create dataset-ready batch
        print(f"\n🎯 Creating dataset-ready batch:")
        dataset_batch = processor.create_dataset_ready_batch(self.sample_texts)
        
        print(f"   Input IDs shape: {dataset_batch['input_ids'].shape}")
        print(f"   Attention mask shape: {dataset_batch['attention_mask'].shape}")
        if 'token_type_ids' in dataset_batch:
            print(f"   Token type IDs shape: {dataset_batch['token_type_ids'].shape}")
        
        # Show batch statistics
        print(f"\n📊 Batch statistics:")
        print(f"   Batch size: {dataset_batch['input_ids'].shape[0]}")
        print(f"   Sequence length: {dataset_batch['input_ids'].shape[1]}")
        print(f"   Total tokens: {dataset_batch['input_ids'].numel():,}")
        
        return processor, dataset_batch
    
    def demo_data_collators(self):
        """Demonstrate different data collators."""
        print("\n" + "="*60)
        print("🔧 DEMO: Data Collators for Different Tasks")
        print("="*60)
        
        # Load a tokenizer for demonstration
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Create different collators
        collators = {
            "language_modeling": DataCollatorFactory.create_collator(
                "language_modeling", tokenizer, mlm=False
            ),
            "sequence_to_sequence": DataCollatorFactory.create_collator(
                "sequence_to_sequence", tokenizer
            ),
            "token_classification": DataCollatorFactory.create_collator(
                "token_classification", tokenizer
            ),
            "sequence_classification": DataCollatorFactory.create_collator(
                "sequence_classification", tokenizer
            )
        }
        
        print(f"\n🔧 Available data collators:")
        for task_type, collator in collators.items():
            print(f"   {task_type}: {type(collator).__name__}")
        
        # Demonstrate language modeling collator
        print(f"\n📚 Language Modeling Collator:")
        lm_collator = collators["language_modeling"]
        
        # Create sample batch
        sample_batch = [
            {"input_ids": [1, 2, 3, 4, 5]},
            {"input_ids": [1, 2, 3]},
            {"input_ids": [1, 2, 3, 4, 5, 6, 7]}
        ]
        
        collated = lm_collator(sample_batch)
        print(f"   Input shape: {collated['input_ids'].shape}")
        print(f"   Labels shape: {collated['labels'].shape}")
        print(f"   Attention mask shape: {collated['attention_mask'].shape}")
        
        return collators
    
    def demo_performance_benchmarking(self):
        """Demonstrate performance benchmarking."""
        print("\n" + "="*60)
        print("⚡ DEMO: Performance Benchmarking")
        print("="*60)
        
        # Configuration
        tokenization_config = TokenizationConfig(
            model_name="gpt2",
            max_length=512,
            padding="longest",
            verbose=False  # Disable logging for benchmarking
        )
        
        sequence_config = SequenceConfig(
            max_sequence_length=1024,
            target_sequence_length=512,
            handle_long_sequences=True
        )
        
        # Create processor
        processor = AdvancedTextProcessor(tokenization_config, sequence_config)
        
        # Generate test data
        test_texts = [
            "This is a test text for benchmarking the tokenization system. " * 50,
            "Another test text with different content and structure. " * 40,
            "Third test text for comprehensive performance evaluation. " * 45
        ]
        
        print(f"\n📊 Performance benchmarks:")
        
        # Benchmark tokenization
        start_time = time.time()
        for _ in range(10):
            _ = processor.process_batch(test_texts)
        tokenization_time = (time.time() - start_time) / 10
        
        print(f"   Average tokenization time: {tokenization_time:.4f}s")
        print(f"   Texts per second: {len(test_texts) / tokenization_time:.1f}")
        
        # Benchmark sequence processing
        start_time = time.time()
        for _ in range(100):
            _ = processor.create_dataset_ready_batch(test_texts)
        sequence_time = (time.time() - start_time) / 100
        
        print(f"   Average sequence processing time: {sequence_time:.4f}s")
        print(f"   Batches per second: {1 / sequence_time:.1f}")
        
        # Memory usage estimation
        dataset_batch = processor.create_dataset_ready_batch(test_texts)
        total_memory = sum(tensor.numel() * tensor.element_size() for tensor in dataset_batch.values())
        
        print(f"\n💾 Memory usage:")
        print(f"   Total memory: {total_memory / 1024:.2f} KB")
        print(f"   Memory per text: {total_memory / len(test_texts) / 1024:.2f} KB")
        
        return processor
    
    def demo_statistics_and_analysis(self):
        """Demonstrate comprehensive statistics and analysis."""
        print("\n" + "="*60)
        print("📈 DEMO: Statistics & Analysis")
        print("="*60)
        
        # Configuration
        tokenization_config = TokenizationConfig(
            model_name="gpt2",
            max_length=512,
            verbose=False
        )
        
        sequence_config = SequenceConfig(
            max_sequence_length=1024,
            target_sequence_length=512
        )
        
        # Create processor
        processor = AdvancedTextProcessor(tokenization_config, sequence_config)
        
        # Get comprehensive statistics
        print(f"\n📊 Processing statistics for {len(self.sample_texts)} texts:")
        stats = processor.get_processing_statistics(self.sample_texts)
        
        print(f"   Total texts: {stats['total_texts']}")
        print(f"   Total characters: {stats['total_characters']:,}")
        print(f"   Total words: {stats['total_words']:,}")
        print(f"   Average text length: {stats['average_text_length']:.1f} characters")
        print(f"   Text length std: {stats['text_length_std']:.1f}")
        
        if 'aggregate_tokens' in stats:
            print(f"\n🔤 Tokenization statistics:")
            print(f"   Total tokens: {stats['aggregate_tokens']['total_tokens']:,}")
            print(f"   Unique tokens: {stats['aggregate_tokens']['unique_tokens']:,}")
            print(f"   Average tokens per text: {stats['aggregate_tokens']['average_tokens_per_text']:.1f}")
            print(f"   Vocabulary coverage: {stats['aggregate_tokens']['vocabulary_coverage']:.2%}")
        
        # Individual text statistics
        print(f"\n📝 Individual text analysis:")
        for i, text_stats in enumerate(stats['tokenization_stats'][:3]):  # Show first 3
            print(f"   Text {i+1}:")
            print(f"      Tokens: {text_stats['total_tokens']}")
            print(f"      Unique: {text_stats['unique_tokens']}")
            print(f"      Coverage: {text_stats['vocabulary_coverage']:.2%}")
            print(f"      Most common: {text_stats['most_common_tokens'][0] if text_stats['most_common_tokens'] else 'N/A'}")
        
        return stats
    
    def run_all_demos(self):
        """Run all demonstration functions."""
        print("🎯 Advanced Tokenization & Sequence Handling System Demo")
        print("=" * 80)
        
        try:
            # Demo 1: Basic tokenization
            tokenizer = self.demo_basic_tokenization()
            
            # Demo 2: Sequence processing
            sequence_processor = self.demo_sequence_processing()
            
            # Demo 3: Text preprocessing
            text_preprocessor = self.demo_text_preprocessing()
            
            # Demo 4: Batch processing
            processor, dataset_batch = self.demo_batch_processing()
            
            # Demo 5: Data collators
            collators = self.demo_data_collators()
            
            # Demo 6: Performance benchmarking
            benchmark_processor = self.demo_performance_benchmarking()
            
            # Demo 7: Statistics and analysis
            stats = self.demo_statistics_and_analysis()
            
            print("\n" + "="*80)
            print("🎉 All demos completed successfully!")
            print("="*80)
            
            # Summary
            print(f"\n📋 System Summary:")
            print(f"   ✅ Advanced tokenization with multiple model support")
            print(f"   ✅ Intelligent sequence processing and handling")
            print(f"   ✅ Comprehensive text preprocessing")
            print(f"   ✅ Efficient batch processing")
            print(f"   ✅ Multiple data collators for different tasks")
            print(f"   ✅ Performance optimization and benchmarking")
            print(f"   ✅ Detailed statistics and analysis")
            
        except Exception as e:
            print(f"❌ Demo error: {e}")
            raise


def main():
    """Main function to run the demonstration."""
    demo = TokenizationDemo()
    demo.run_all_demos()


if __name__ == "__main__":
    main()






