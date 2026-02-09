import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast,
    BatchEncoding, PreTrainedModel
)
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import BPE, WordLevel, WordPiece, Unigram
from tokenizers.trainers import BpeTrainer, WordLevelTrainer, WordPieceTrainer, UnigramTrainer
from tokenizers.pre_tokenizers import Whitespace, ByteLevel, BertPreTokenizer
from tokenizers.processors import TemplateProcessing
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
import logging
from dataclasses import dataclass
from enum import Enum
import json
import os
from collections import defaultdict
import re
from functools import lru_cache


class TokenizationType(Enum):
    """Enum for different tokenization strategies"""
    BPE = "bpe"
    WORD_LEVEL = "word_level"
    WORD_PIECE = "word_piece"
    UNIGRAM = "unigram"
    BYTE_LEVEL = "byte_level"
    CHARACTER_LEVEL = "character_level"


class SequenceStrategy(Enum):
    """Enum for different sequence handling strategies"""
    TRUNCATE = "truncate"
    PAD = "pad"
    SLIDING_WINDOW = "sliding_window"
    OVERLAP = "overlap"
    CHUNK = "chunk"


@dataclass
class TokenizationConfig:
    """Configuration for tokenization settings"""
    max_length: int = 512
    padding: str = "max_length"  # max_length, longest, do_not_pad
    truncation: bool = True
    return_tensors: str = "pt"  # pt, tf, np
    return_attention_mask: bool = True
    return_token_type_ids: bool = False
    return_overflowing_tokens: bool = False
    return_special_tokens_mask: bool = False
    return_offsets_mapping: bool = False
    return_length: bool = False
    verbose: bool = False


@dataclass
class SequenceConfig:
    """Configuration for sequence handling"""
    strategy: SequenceStrategy = SequenceStrategy.PAD
    overlap_size: int = 50
    chunk_size: int = 512
    stride: int = 256
    min_chunk_size: int = 100
    preserve_word_boundaries: bool = True
    handle_overflow: bool = True


class AdvancedTokenizer:
    """Advanced tokenizer with multiple tokenization strategies"""
    
    def __init__(self, config: TokenizationConfig):
        self.config = config
        self.tokenizer = None
        self.vocab_size = 0
        self.special_tokens = {}
        self.logger = logging.getLogger(__name__)
        
    def load_pretrained_tokenizer(self, model_name: str) -> None:
        """Load a pre-trained tokenizer from HuggingFace"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.vocab_size = self.tokenizer.vocab_size
            self.special_tokens = {
                'pad_token': self.tokenizer.pad_token,
                'eos_token': self.tokenizer.eos_token,
                'bos_token': self.tokenizer.bos_token,
                'unk_token': self.tokenizer.unk_token,
                'sep_token': self.tokenizer.sep_token,
                'cls_token': self.tokenizer.cls_token,
                'mask_token': self.tokenizer.mask_token
            }
            
            # Ensure pad token is set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.logger.info(f"Loaded tokenizer: {model_name} with vocab size: {self.vocab_size}")
            
        except Exception as e:
            self.logger.error(f"Error loading tokenizer: {e}")
            raise
    
    def create_custom_tokenizer(self, tokenization_type: TokenizationType, 
                               vocab_size: int = 30000) -> None:
        """Create a custom tokenizer from scratch"""
        try:
            if tokenization_type == TokenizationType.BPE:
                tokenizer = HFTokenizer(BPE())
                trainer = BpeTrainer(
                    vocab_size=vocab_size,
                    special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
                )
            elif tokenization_type == TokenizationType.WORD_LEVEL:
                tokenizer = HFTokenizer(WordLevel())
                trainer = WordLevelTrainer(
                    vocab_size=vocab_size,
                    special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
                )
            elif tokenization_type == TokenizationType.WORD_PIECE:
                tokenizer = HFTokenizer(WordPiece())
                trainer = WordPieceTrainer(
                    vocab_size=vocab_size,
                    special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
                )
            elif tokenization_type == TokenizationType.UNIGRAM:
                tokenizer = HFTokenizer(Unigram())
                trainer = UnigramTrainer(
                    vocab_size=vocab_size,
                    special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
                )
            else:
                raise ValueError(f"Unsupported tokenization type: {tokenization_type}")
            
            # Set pre-tokenizer
            if tokenization_type == TokenizationType.BYTE_LEVEL:
                tokenizer.pre_tokenizer = ByteLevel()
            else:
                tokenizer.pre_tokenizer = Whitespace()
            
            # Set post-processor
            tokenizer.post_processor = TemplateProcessing(
                single="[BOS] $A [EOS]",
                pair="[BOS] $A [SEP] $B:1 [EOS]:1",
                special_tokens=[
                    ("[BOS]", tokenizer.token_to_id("[BOS]")),
                    ("[EOS]", tokenizer.token_to_id("[EOS]")),
                    ("[SEP]", tokenizer.token_to_id("[SEP]"))
                ]
            )
            
            self.tokenizer = tokenizer
            self.vocab_size = vocab_size
            self.logger.info(f"Created custom {tokenization_type.value} tokenizer")
            
        except Exception as e:
            self.logger.error(f"Error creating custom tokenizer: {e}")
            raise
    
    def tokenize_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize a single text with proper error handling"""
        try:
            if self.tokenizer is None:
                raise ValueError("Tokenizer not initialized")
            
            # Clean and preprocess text
            cleaned_text = self._preprocess_text(text)
            
            # Tokenize with configuration
            encoding = self.tokenizer(
                cleaned_text,
                max_length=self.config.max_length,
                padding=self.config.padding,
                truncation=self.config.truncation,
                return_tensors=self.config.return_tensors,
                return_attention_mask=self.config.return_attention_mask,
                return_token_type_ids=self.config.return_token_type_ids,
                return_overflowing_tokens=self.config.return_overflowing_tokens,
                return_special_tokens_mask=self.config.return_special_tokens_mask,
                return_offsets_mapping=self.config.return_offsets_mapping,
                return_length=self.config.return_length,
                verbose=self.config.verbose
            )
            
            return encoding
            
        except Exception as e:
            self.logger.error(f"Error tokenizing text: {e}")
            # Return fallback encoding
            return self._create_fallback_encoding(text)
    
    def batch_tokenize(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize multiple texts in batch"""
        try:
            if self.tokenizer is None:
                raise ValueError("Tokenizer not initialized")
            
            # Clean and preprocess all texts
            cleaned_texts = [self._preprocess_text(text) for text in texts]
            
            # Batch tokenize
            encoding = self.tokenizer(
                cleaned_texts,
                max_length=self.config.max_length,
                padding=self.config.padding,
                truncation=self.config.truncation,
                return_tensors=self.config.return_tensors,
                return_attention_mask=self.config.return_attention_mask,
                return_token_type_ids=self.config.return_token_type_ids,
                return_overflowing_tokens=self.config.return_overflowing_tokens,
                return_special_tokens_mask=self.config.return_special_tokens_mask,
                return_offsets_mapping=self.config.return_offsets_mapping,
                return_length=self.config.return_length,
                verbose=self.config.verbose
            )
            
            return encoding
            
        except Exception as e:
            self.logger.error(f"Error batch tokenizing: {e}")
            # Return fallback batch encoding
            return self._create_fallback_batch_encoding(texts)
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text before tokenization"""
        if not isinstance(text, str):
            text = str(text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Basic text cleaning
        text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        
        return text
    
    def _create_fallback_encoding(self, text: str) -> Dict[str, torch.Tensor]:
        """Create fallback encoding when tokenization fails"""
        fallback_tokens = [0] * min(len(text.split()), self.config.max_length)
        fallback_attention = [1] * len(fallback_tokens)
        
        return {
            'input_ids': torch.tensor([fallback_tokens]),
            'attention_mask': torch.tensor([fallback_attention])
        }
    
    def _create_fallback_batch_encoding(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Create fallback batch encoding when tokenization fails"""
        fallback_input_ids = []
        fallback_attention_masks = []
        
        for text in texts:
            tokens = [0] * min(len(text.split()), self.config.max_length)
            attention = [1] * len(tokens)
            fallback_input_ids.append(tokens)
            fallback_attention_masks.append(attention)
        
        return {
            'input_ids': torch.tensor(fallback_input_ids),
            'attention_mask': torch.tensor(fallback_attention_masks)
        }
    
    def decode_tokens(self, token_ids: Union[List[int], torch.Tensor], 
                     skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text"""
        try:
            if isinstance(token_ids, torch.Tensor):
                token_ids = token_ids.tolist()
            
            if self.tokenizer is None:
                raise ValueError("Tokenizer not initialized")
            
            decoded_text = self.tokenizer.decode(
                token_ids, 
                skip_special_tokens=skip_special_tokens
            )
            
            return decoded_text
            
        except Exception as e:
            self.logger.error(f"Error decoding tokens: {e}")
            return "[DECODING_ERROR]"
    
    def get_vocab(self) -> Dict[str, int]:
        """Get the vocabulary mapping"""
        if self.tokenizer is None:
            return {}
        
        try:
            if hasattr(self.tokenizer, 'get_vocab'):
                return self.tokenizer.get_vocab()
            elif hasattr(self.tokenizer, 'vocab'):
                return self.tokenizer.vocab
            else:
                return {}
        except Exception as e:
            self.logger.error(f"Error getting vocab: {e}")
            return {}


class SequenceHandler:
    """Advanced sequence handling for text data"""
    
    def __init__(self, config: SequenceConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def handle_long_sequences(self, text: str, max_length: int) -> List[str]:
        """Handle sequences longer than max_length using specified strategy"""
        try:
            if len(text.split()) <= max_length:
                return [text]
            
            if self.config.strategy == SequenceStrategy.TRUNCATE:
                return self._truncate_sequence(text, max_length)
            elif self.config.strategy == SequenceStrategy.SLIDING_WINDOW:
                return self._sliding_window_split(text, max_length)
            elif self.config.strategy == SequenceStrategy.OVERLAP:
                return self._overlapping_split(text, max_length)
            elif self.config.strategy == SequenceStrategy.CHUNK:
                return self._chunk_split(text, max_length)
            else:
                return self._truncate_sequence(text, max_length)
                
        except Exception as e:
            self.logger.error(f"Error handling long sequence: {e}")
            return [text[:max_length * 4]]  # Rough character-based fallback
    
    def _truncate_sequence(self, text: str, max_length: int) -> List[str]:
        """Truncate sequence to max_length"""
        words = text.split()
        if len(words) <= max_length:
            return [text]
        
        truncated_words = words[:max_length]
        return [' '.join(truncated_words)]
    
    def _sliding_window_split(self, text: str, max_length: int) -> List[str]:
        """Split text using sliding window approach"""
        words = text.split()
        if len(words) <= max_length:
            return [text]
        
        sequences = []
        stride = self.config.stride
        
        for i in range(0, len(words) - max_length + 1, stride):
            sequence = words[i:i + max_length]
            if len(sequence) >= self.config.min_chunk_size:
                sequences.append(' '.join(sequence))
        
        # Handle remaining words
        if len(words) % stride != 0:
            remaining = words[-(max_length - stride):]
            if len(remaining) >= self.config.min_chunk_size:
                sequences.append(' '.join(remaining))
        
        return sequences if sequences else [text[:max_length * 4]]
    
    def _overlapping_split(self, text: str, max_length: int) -> List[str]:
        """Split text with overlapping sequences"""
        words = text.split()
        if len(words) <= max_length:
            return [text]
        
        sequences = []
        overlap = self.config.overlap_size
        
        for i in range(0, len(words), max_length - overlap):
            sequence = words[i:i + max_length]
            if len(sequence) >= self.config.min_chunk_size:
                sequences.append(' '.join(sequence))
        
        return sequences if sequences else [text[:max_length * 4]]
    
    def _chunk_split(self, text: str, max_length: int) -> List[str]:
        """Split text into chunks preserving word boundaries"""
        words = text.split()
        if len(words) <= max_length:
            return [text]
        
        sequences = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # +1 for space
            
            if current_length + word_length > max_length and current_chunk:
                sequences.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length
        
        if current_chunk:
            sequences.append(' '.join(current_chunk))
        
        return sequences if sequences else [text[:max_length * 4]]
    
    def create_sequence_pairs(self, texts: List[str], max_length: int) -> List[Tuple[str, str]]:
        """Create sequence pairs for tasks like next sentence prediction"""
        try:
            pairs = []
            
            for i in range(len(texts) - 1):
                text_a = texts[i]
                text_b = texts[i + 1]
                
                # Handle long sequences
                text_a_chunks = self.handle_long_sequences(text_a, max_length // 2)
                text_b_chunks = self.handle_long_sequences(text_b, max_length // 2)
                
                for chunk_a in text_a_chunks:
                    for chunk_b in text_b_chunks:
                        pairs.append((chunk_a, chunk_b))
            
            return pairs
            
        except Exception as e:
            self.logger.error(f"Error creating sequence pairs: {e}")
            return []
    
    def create_masked_sequences(self, text: str, mask_prob: float = 0.15) -> List[Tuple[str, List[int]]]:
        """Create masked sequences for masked language modeling"""
        try:
            words = text.split()
            if len(words) <= 1:
                return [(text, [])]
            
            masked_sequences = []
            num_masks = max(1, int(len(words) * mask_prob))
            
            # Create multiple masked versions
            for _ in range(min(3, num_masks)):
                masked_words = words.copy()
                mask_positions = []
                
                # Randomly select positions to mask
                positions = np.random.choice(len(words), num_masks, replace=False)
                
                for pos in positions:
                    if pos < len(masked_words):
                        masked_words[pos] = '[MASK]'
                        mask_positions.append(pos)
                
                masked_sequences.append((' '.join(masked_words), mask_positions))
            
            return masked_sequences
            
        except Exception as e:
            self.logger.error(f"Error creating masked sequences: {e}")
            return [(text, [])]


class TokenizationDataset(Dataset):
    """Dataset for tokenized text data"""
    
    def __init__(self, texts: List[str], tokenizer: AdvancedTokenizer, 
                 sequence_handler: SequenceHandler, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.sequence_handler = sequence_handler
        self.max_length = max_length
        self.processed_texts = self._process_texts()
    
    def _process_texts(self) -> List[Dict[str, Any]]:
        """Process all texts and handle long sequences"""
        processed = []
        
        for text in self.texts:
            # Handle long sequences
            sequences = self.sequence_handler.handle_long_sequences(text, self.max_length)
            
            for seq in sequences:
                # Tokenize sequence
                encoding = self.tokenizer.tokenize_text(seq)
                
                processed.append({
                    'text': seq,
                    'encoding': encoding,
                    'length': len(seq.split())
                })
        
        return processed
    
    def __len__(self) -> int:
        return len(self.processed_texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.processed_texts[idx]
        encoding = item['encoding']
        
        # Ensure all tensors have the same shape
        batch_size = 1
        for key, value in encoding.items():
            if isinstance(value, torch.Tensor):
                if value.dim() == 1:
                    encoding[key] = value.unsqueeze(0)
                elif value.dim() == 2 and value.size(0) == 1:
                    pass  # Already correct shape
                else:
                    encoding[key] = value[:batch_size]
        
        return encoding


class DataCollator:
    """Custom data collator for batching tokenized data"""
    
    def __init__(self, tokenizer: AdvancedTokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate a batch of tokenized data"""
        try:
            # Find the maximum length in the batch
            max_len = max(
                item['input_ids'].size(-1) if 'input_ids' in item else 0
                for item in batch
            )
            
            # Limit to max_length
            max_len = min(max_len, self.max_length)
            
            # Initialize batch tensors
            batch_size = len(batch)
            input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
            attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
            
            # Fill batch tensors
            for i, item in enumerate(batch):
                if 'input_ids' in item and 'attention_mask' in item:
                    seq_len = min(item['input_ids'].size(-1), max_len)
                    input_ids[i, :seq_len] = item['input_ids'][0, :seq_len]
                    attention_mask[i, :seq_len] = item['attention_mask'][0, :seq_len]
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
            
        except Exception as e:
            logging.error(f"Error in data collator: {e}")
            # Return fallback batch
            return self._create_fallback_batch(batch)
    
    def _create_fallback_batch(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Create fallback batch when collation fails"""
        batch_size = len(batch)
        fallback_input_ids = torch.zeros(batch_size, self.max_length, dtype=torch.long)
        fallback_attention_mask = torch.ones(batch_size, self.max_length, dtype=torch.long)
        
        return {
            'input_ids': fallback_input_ids,
            'attention_mask': fallback_attention_mask
        }


class TokenizationAnalyzer:
    """Analyzer for tokenization quality and statistics"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.stats = defaultdict(int)
    
    def analyze_tokenization(self, original_texts: List[str], 
                           tokenized_encodings: List[Dict[str, torch.Tensor]]) -> Dict[str, Any]:
        """Analyze tokenization quality and statistics"""
        try:
            analysis = {
                'total_texts': len(original_texts),
                'total_tokens': 0,
                'avg_tokens_per_text': 0,
                'vocabulary_usage': defaultdict(int),
                'sequence_lengths': [],
                'special_token_usage': defaultdict(int),
                'tokenization_efficiency': 0,
                'compression_ratio': 0
            }
            
            total_characters = 0
            
            for i, (text, encoding) in enumerate(zip(original_texts, tokenized_encodings)):
                if 'input_ids' in encoding:
                    tokens = encoding['input_ids'].flatten().tolist()
                    analysis['total_tokens'] += len(tokens)
                    analysis['sequence_lengths'].append(len(tokens))
                    
                    # Count vocabulary usage
                    for token_id in tokens:
                        analysis['vocabulary_usage'][token_id] += 1
                    
                    # Count special tokens
                    if hasattr(self.tokenizer, 'special_tokens_map'):
                        special_tokens = self.tokenizer.special_tokens_map.values()
                        for token_id in tokens:
                            if token_id in special_tokens:
                                analysis['special_token_usage'][token_id] += 1
                
                total_characters += len(text)
            
            # Calculate averages and ratios
            if analysis['total_texts'] > 0:
                analysis['avg_tokens_per_text'] = analysis['total_tokens'] / analysis['total_texts']
                analysis['tokenization_efficiency'] = analysis['total_tokens'] / total_characters
                analysis['compression_ratio'] = total_characters / analysis['total_tokens']
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing tokenization: {e}")
            return {}
    
    def generate_tokenization_report(self, analysis: Dict[str, Any]) -> str:
        """Generate a human-readable tokenization report"""
        try:
            report = []
            report.append("=" * 50)
            report.append("TOKENIZATION ANALYSIS REPORT")
            report.append("=" * 50)
            
            report.append(f"Total Texts: {analysis.get('total_texts', 0)}")
            report.append(f"Total Tokens: {analysis.get('total_tokens', 0)}")
            report.append(f"Average Tokens per Text: {analysis.get('avg_tokens_per_text', 0):.2f}")
            report.append(f"Tokenization Efficiency: {analysis.get('tokenization_efficiency', 0):.4f}")
            report.append(f"Compression Ratio: {analysis.get('compression_ratio', 0):.2f}")
            
            if 'sequence_lengths' in analysis:
                lengths = analysis['sequence_lengths']
                if lengths:
                    report.append(f"Min Sequence Length: {min(lengths)}")
                    report.append(f"Max Sequence Length: {max(lengths)}")
                    report.append(f"Median Sequence Length: {np.median(lengths):.2f}")
            
            report.append("=" * 50)
            return "\n".join(report)
            
        except Exception as e:
            self.logger.error(f"Error generating report: {e}")
            return "Error generating tokenization report"


class SequenceProcessor:
    """Main processor for tokenization and sequence handling"""
    
    def __init__(self, tokenization_config: TokenizationConfig, 
                 sequence_config: SequenceConfig):
        self.tokenization_config = tokenization_config
        self.sequence_config = sequence_config
        self.tokenizer = AdvancedTokenizer(tokenization_config)
        self.sequence_handler = SequenceHandler(sequence_config)
        self.analyzer = TokenizationAnalyzer()
        self.logger = logging.getLogger(__name__)
    
    def process_text(self, text: str) -> Dict[str, Any]:
        """Process a single text through the complete pipeline"""
        try:
            # Handle long sequences
            sequences = self.sequence_handler.handle_long_sequences(
                text, self.tokenization_config.max_length
            )
            
            # Tokenize each sequence
            tokenized_sequences = []
            for seq in sequences:
                encoding = self.tokenizer.tokenize_text(seq)
                tokenized_sequences.append({
                    'original_text': seq,
                    'encoding': encoding,
                    'decoded_text': self.tokenizer.decode_tokens(
                        encoding['input_ids'].flatten()
                    )
                })
            
            return {
                'original_text': text,
                'sequences': sequences,
                'tokenized_sequences': tokenized_sequences,
                'total_sequences': len(sequences)
            }
            
        except Exception as e:
            self.logger.error(f"Error processing text: {e}")
            return {'error': str(e), 'original_text': text}
    
    def process_batch(self, texts: List[str]) -> Dict[str, Any]:
        """Process multiple texts through the complete pipeline"""
        try:
            results = []
            all_encodings = []
            
            for text in texts:
                result = self.process_text(text)
                results.append(result)
                
                # Collect encodings for analysis
                if 'tokenized_sequences' in result:
                    for seq_result in result['tokenized_sequences']:
                        all_encodings.append(seq_result['encoding'])
            
            # Analyze tokenization
            analysis = self.analyzer.analyze_tokenization(texts, all_encodings)
            
            return {
                'results': results,
                'analysis': analysis,
                'total_texts': len(texts)
            }
            
        except Exception as e:
            self.logger.error(f"Error processing batch: {e}")
            return {'error': str(e), 'total_texts': len(texts)}
    
    def create_dataset(self, texts: List[str], max_length: int = None) -> TokenizationDataset:
        """Create a dataset from texts"""
        if max_length is None:
            max_length = self.tokenization_config.max_length
        
        return TokenizationDataset(
            texts, self.tokenizer, self.sequence_handler, max_length
        )
    
    def create_data_loader(self, texts: List[str], batch_size: int = 32, 
                          max_length: int = None, shuffle: bool = True) -> DataLoader:
        """Create a DataLoader from texts"""
        dataset = self.create_dataset(texts, max_length)
        collator = DataCollator(self.tokenizer, max_length or self.tokenization_config.max_length)
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collator,
            num_workers=4,
            pin_memory=True
        )


def create_advanced_tokenization_system(
    model_name: str = "gpt2",
    max_length: int = 512,
    tokenization_strategy: TokenizationType = TokenizationType.BPE,
    sequence_strategy: SequenceStrategy = SequenceStrategy.PAD
) -> SequenceProcessor:
    """Factory function to create a complete tokenization system"""
    
    # Tokenization configuration
    tokenization_config = TokenizationConfig(
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        return_attention_mask=True
    )
    
    # Sequence configuration
    sequence_config = SequenceConfig(
        strategy=sequence_strategy,
        overlap_size=50,
        chunk_size=max_length,
        stride=max_length // 2,
        min_chunk_size=100
    )
    
    # Create processor
    processor = SequenceProcessor(tokenization_config, sequence_config)
    
    # Load or create tokenizer
    if model_name != "custom":
        processor.tokenizer.load_pretrained_tokenizer(model_name)
    else:
        processor.tokenizer.create_custom_tokenizer(tokenization_strategy)
    
    return processor


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create tokenization system
    processor = create_advanced_tokenization_system(
        model_name="gpt2",
        max_length=256,
        sequence_strategy=SequenceStrategy.SLIDING_WINDOW
    )
    
    # Sample texts
    sample_texts = [
        "This is a short text for testing.",
        "This is a much longer text that will need to be processed using the advanced sequence handling capabilities of our tokenization system. It contains multiple sentences and will demonstrate how the system handles texts that exceed the maximum length limit.",
        "Another example text with different content and structure.",
        "A very long text that will definitely exceed the maximum token limit and require sophisticated sequence handling strategies including sliding windows, overlapping chunks, and proper word boundary preservation to ensure high-quality tokenization results."
    ]
    
    # Process texts
    results = processor.process_batch(sample_texts)
    
    # Print results
    print("Tokenization Results:")
    for i, result in enumerate(results['results']):
        print(f"\nText {i+1}:")
        print(f"Original: {result['original_text'][:100]}...")
        print(f"Sequences: {result['total_sequences']}")
        print(f"Tokenized sequences: {len(result['tokenized_sequences'])}")
    
    # Print analysis
    if 'analysis' in results:
        report = processor.analyzer.generate_tokenization_report(results['analysis'])
        print(f"\n{report}")
    
    # Create dataset and dataloader
    dataset = processor.create_dataset(sample_texts)
    dataloader = processor.create_data_loader(sample_texts, batch_size=2)
    
    print(f"\nDataset size: {len(dataset)}")
    print(f"Dataloader batches: {len(dataloader)}")
    
    # Test batch processing
    for batch_idx, batch in enumerate(dataloader):
        print(f"\nBatch {batch_idx + 1}:")
        print(f"Input IDs shape: {batch['input_ids'].shape}")
        print(f"Attention mask shape: {batch['attention_mask'].shape}")
        if batch_idx >= 1:  # Show only first 2 batches
            break


