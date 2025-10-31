from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any, Callable, Iterator
import logging
import re
import json
import pickle
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
import numpy as np
from typing import Any, List, Dict, Optional
import asyncio
"""
Tokenization and Sequence Handling for HeyGen AI.

Implementation of comprehensive tokenization and sequence handling for text data
following PEP 8 style guidelines and best practices.
"""


logger = logging.getLogger(__name__)


@dataclass
class TokenizationConfig:
    """Configuration for tokenization."""
    vocab_size: int = 50000
    min_frequency: int = 2
    max_sequence_length: int = 512
    padding_token: str = "<PAD>"
    unknown_token: str = "<UNK>"
    start_token: str = "<START>"
    end_token: str = "<END>"
    mask_token: str = "<MASK>"
    special_tokens: Optional[List[str]] = None
    lowercase: bool = True
    remove_punctuation: bool = False
    normalize_whitespace: bool = True


class BaseTokenizer:
    """Base tokenizer class."""

    def __init__(self, config: TokenizationConfig):
        """Initialize base tokenizer.

        Args:
            config: Tokenization configuration.
        """
        self.config = config
        self.vocab = {}
        self.reverse_vocab = {}
        self.special_tokens = {
            "pad": config.padding_token,
            "unk": config.unknown_token,
            "start": config.start_token,
            "end": config.end_token,
            "mask": config.mask_token
        }
        
        if config.special_tokens:
            self.special_tokens.update({token: token for token in config.special_tokens})

    def preprocess_text(self, text: str) -> str:
        """Preprocess text before tokenization.

        Args:
            text: Input text.

        Returns:
            str: Preprocessed text.
        """
        if self.config.lowercase:
            text = text.lower()
        
        if self.config.normalize_whitespace:
            text = re.sub(r'\s+', ' ', text).strip()
        
        if self.config.remove_punctuation:
            text = re.sub(r'[^\w\s]', '', text)
        
        return text

    def build_vocab(self, texts: List[str]) -> None:
        """Build vocabulary from texts.

        Args:
            texts: List of input texts.
        """
        raise NotImplementedError("Subclasses must implement build_vocab")

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into tokens.

        Args:
            text: Input text.

        Returns:
            List[str]: List of tokens.
        """
        raise NotImplementedError("Subclasses must implement tokenize")

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs.

        Args:
            text: Input text.

        Returns:
            List[int]: List of token IDs.
        """
        tokens = self.tokenize(text)
        return [self.vocab.get(token, self.vocab[self.special_tokens["unk"]]) for token in tokens]

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text.

        Args:
            token_ids: List of token IDs.

        Returns:
            str: Decoded text.
        """
        tokens = [self.reverse_vocab.get(token_id, self.special_tokens["unk"]) for token_id in token_ids]
        return " ".join(tokens)

    def encode_batch(self, texts: List[str], padding: bool = True, truncation: bool = True) -> torch.Tensor:
        """Encode a batch of texts.

        Args:
            texts: List of input texts.
            padding: Whether to pad sequences.
            truncation: Whether to truncate sequences.

        Returns:
            torch.Tensor: Batch of encoded sequences.
        """
        encoded_sequences = []
        for text in texts:
            token_ids = self.encode(text)
            if truncation and len(token_ids) > self.config.max_sequence_length:
                token_ids = token_ids[:self.config.max_sequence_length]
            encoded_sequences.append(token_ids)
        
        if padding:
            max_length = max(len(seq) for seq in encoded_sequences)
            padded_sequences = []
            for seq in encoded_sequences:
                padded_seq = seq + [self.vocab[self.special_tokens["pad"]]] * (max_length - len(seq))
                padded_sequences.append(padded_seq)
            return torch.tensor(padded_sequences)
        
        return encoded_sequences

    def save(self, filepath: str) -> None:
        """Save tokenizer to file.

        Args:
            filepath: Path to save tokenizer.
        """
        tokenizer_data = {
            "config": self.config,
            "vocab": self.vocab,
            "reverse_vocab": self.reverse_vocab,
            "special_tokens": self.special_tokens
        }
        
        with open(filepath, 'wb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            pickle.dump(tokenizer_data, f)
        
        logger.info(f"Tokenizer saved to {filepath}")

    def load(self, filepath: str) -> None:
        """Load tokenizer from file.

        Args:
            filepath: Path to load tokenizer from.
        """
        with open(filepath, 'rb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            tokenizer_data = pickle.load(f)
        
        self.config = tokenizer_data["config"]
        self.vocab = tokenizer_data["vocab"]
        self.reverse_vocab = tokenizer_data["reverse_vocab"]
        self.special_tokens = tokenizer_data["special_tokens"]
        
        logger.info(f"Tokenizer loaded from {filepath}")


class WordTokenizer(BaseTokenizer):
    """Word-based tokenizer."""

    def __init__(self, config: TokenizationConfig):
        """Initialize word tokenizer.

        Args:
            config: Tokenization configuration.
        """
        super().__init__(config)

    def build_vocab(self, texts: List[str]) -> None:
        """Build vocabulary from texts.

        Args:
            texts: List of input texts.
        """
        # Count word frequencies
        word_counts = Counter()
        for text in texts:
            preprocessed_text = self.preprocess_text(text)
            words = preprocessed_text.split()
            word_counts.update(words)
        
        # Add special tokens
        vocab = {token: idx for idx, token in enumerate(self.special_tokens.values())}
        
        # Add words that meet minimum frequency
        for word, count in word_counts.most_common():
            if count >= self.config.min_frequency and len(vocab) < self.config.vocab_size:
                vocab[word] = len(vocab)
        
        self.vocab = vocab
        self.reverse_vocab = {idx: word for word, idx in vocab.items()}
        
        logger.info(f"Built vocabulary with {len(self.vocab)} tokens")

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words.

        Args:
            text: Input text.

        Returns:
            List[str]: List of word tokens.
        """
        preprocessed_text = self.preprocess_text(text)
        return preprocessed_text.split()


class CharacterTokenizer(BaseTokenizer):
    """Character-based tokenizer."""

    def __init__(self, config: TokenizationConfig):
        """Initialize character tokenizer.

        Args:
            config: Tokenization configuration.
        """
        super().__init__(config)

    def build_vocab(self, texts: List[str]) -> None:
        """Build vocabulary from texts.

        Args:
            texts: List of input texts.
        """
        # Count character frequencies
        char_counts = Counter()
        for text in texts:
            preprocessed_text = self.preprocess_text(text)
            char_counts.update(preprocessed_text)
        
        # Add special tokens
        vocab = {token: idx for idx, token in enumerate(self.special_tokens.values())}
        
        # Add characters
        for char, count in char_counts.most_common():
            if count >= self.config.min_frequency and len(vocab) < self.config.vocab_size:
                vocab[char] = len(vocab)
        
        self.vocab = vocab
        self.reverse_vocab = {idx: char for char, idx in vocab.items()}
        
        logger.info(f"Built vocabulary with {len(self.vocab)} characters")

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into characters.

        Args:
            text: Input text.

        Returns:
            List[str]: List of character tokens.
        """
        preprocessed_text = self.preprocess_text(text)
        return list(preprocessed_text)


class SubwordTokenizer(BaseTokenizer):
    """Subword tokenizer using Byte Pair Encoding (BPE)."""

    def __init__(self, config: TokenizationConfig):
        """Initialize subword tokenizer.

        Args:
            config: Tokenization configuration.
        """
        super().__init__(config)
        self.merges = {}
        self.pattern = re.compile(r'\S+|\n')

    def build_vocab(self, texts: List[str]) -> None:
        """Build vocabulary using BPE algorithm.

        Args:
            texts: List of input texts.
        """
        # Initialize vocabulary with characters
        vocab = {token: idx for idx, token in enumerate(self.special_tokens.values())}
        
        # Count character frequencies
        char_counts = Counter()
        for text in texts:
            preprocessed_text = self.preprocess_text(text)
            char_counts.update(preprocessed_text)
        
        # Add characters to vocabulary
        for char, count in char_counts.most_common():
            if count >= self.config.min_frequency and len(vocab) < self.config.vocab_size:
                vocab[char] = len(vocab)
        
        # Build BPE merges
        self.merges = self._build_bpe_merges(texts, vocab)
        
        # Update vocabulary with merged tokens
        for merge in self.merges:
            if len(vocab) < self.config.vocab_size:
                vocab[merge] = len(vocab)
        
        self.vocab = vocab
        self.reverse_vocab = {idx: token for token, idx in vocab.items()}
        
        logger.info(f"Built BPE vocabulary with {len(self.vocab)} tokens and {len(self.merges)} merges")

    def _build_bpe_merges(self, texts: List[str], vocab: Dict[str, int]) -> Dict[str, str]:
        """Build BPE merges.

        Args:
            texts: List of input texts.
            vocab: Initial vocabulary.

        Returns:
            Dict[str, str]: BPE merges.
        """
        # Tokenize texts into words
        words = []
        for text in texts:
            preprocessed_text = self.preprocess_text(text)
            words.extend(self.pattern.findall(preprocessed_text))
        
        # Initialize word vocabularies
        word_vocabs = {}
        for word in words:
            word_vocab = list(word)
            word_vocabs[word] = word_vocab
        
        # Build merges
        merges = {}
        num_merges = self.config.vocab_size - len(vocab)
        
        for i in range(num_merges):
            # Count bigram frequencies
            bigram_counts = Counter()
            for word_vocab in word_vocabs.values():
                for j in range(len(word_vocab) - 1):
                    bigram = (word_vocab[j], word_vocab[j + 1])
                    bigram_counts[bigram] += 1
            
            if not bigram_counts:
                break
            
            # Find most frequent bigram
            most_frequent_bigram = bigram_counts.most_common(1)[0][0]
            merged_token = ''.join(most_frequent_bigram)
            
            # Add merge
            merges[merged_token] = most_frequent_bigram
            
            # Update word vocabularies
            for word, word_vocab in word_vocabs.items():
                new_word_vocab = []
                j = 0
                while j < len(word_vocab):
                    if j < len(word_vocab) - 1 and (word_vocab[j], word_vocab[j + 1]) == most_frequent_bigram:
                        new_word_vocab.append(merged_token)
                        j += 2
                    else:
                        new_word_vocab.append(word_vocab[j])
                        j += 1
                word_vocabs[word] = new_word_vocab
        
        return merges

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using BPE.

        Args:
            text: Input text.

        Returns:
            List[str]: List of subword tokens.
        """
        preprocessed_text = self.preprocess_text(text)
        words = self.pattern.findall(preprocessed_text)
        
        tokens = []
        for word in words:
            word_tokens = self._tokenize_word(word)
            tokens.extend(word_tokens)
        
        return tokens

    def _tokenize_word(self, word: str) -> List[str]:
        """Tokenize a single word using BPE.

        Args:
            word: Input word.

        Returns:
            List[str]: List of subword tokens.
        """
        word_vocab = list(word)
        
        # Apply merges
        for merged_token, (token1, token2) in self.merges.items():
            new_word_vocab = []
            i = 0
            while i < len(word_vocab):
                if i < len(word_vocab) - 1 and word_vocab[i] == token1 and word_vocab[i + 1] == token2:
                    new_word_vocab.append(merged_token)
                    i += 2
                else:
                    new_word_vocab.append(word_vocab[i])
                    i += 1
            word_vocab = new_word_vocab
        
        return word_vocab


class SequenceHandler:
    """Handler for sequence operations."""

    def __init__(self, max_sequence_length: int = 512):
        """Initialize sequence handler.

        Args:
            max_sequence_length: Maximum sequence length.
        """
        self.max_sequence_length = max_sequence_length

    def pad_sequences(
        self,
        sequences: List[List[int]],
        padding: str = "post",
        truncation: str = "post",
        value: int = 0
    ) -> torch.Tensor:
        """Pad sequences to the same length.

        Args:
            sequences: List of sequences.
            padding: Padding strategy ("pre" or "post").
            truncation: Truncation strategy ("pre" or "post").
            value: Padding value.

        Returns:
            torch.Tensor: Padded sequences.
        """
        # Truncate sequences if necessary
        if truncation == "post":
            sequences = [seq[:self.max_sequence_length] for seq in sequences]
        elif truncation == "pre":
            sequences = [seq[-self.max_sequence_length:] for seq in sequences]
        
        # Find maximum length
        max_length = max(len(seq) for seq in sequences)
        
        # Pad sequences
        padded_sequences = []
        for seq in sequences:
            if padding == "post":
                padded_seq = seq + [value] * (max_length - len(seq))
            else:  # pre
                padded_seq = [value] * (max_length - len(seq)) + seq
            padded_sequences.append(padded_seq)
        
        return torch.tensor(padded_sequences)

    def create_attention_mask(self, sequences: torch.Tensor, padding_value: int = 0) -> torch.Tensor:
        """Create attention mask for padded sequences.

        Args:
            sequences: Input sequences.
            padding_value: Padding value.

        Returns:
            torch.Tensor: Attention mask.
        """
        return (sequences != padding_value).long()

    def create_causal_mask(self, sequence_length: int) -> torch.Tensor:
        """Create causal mask for autoregressive models.

        Args:
            sequence_length: Length of the sequence.

        Returns:
            torch.Tensor: Causal mask.
        """
        mask = torch.triu(torch.ones(sequence_length, sequence_length), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))

    def create_sliding_window_mask(
        self,
        sequence_length: int,
        window_size: int,
        stride: int = 1
    ) -> torch.Tensor:
        """Create sliding window mask.

        Args:
            sequence_length: Length of the sequence.
            window_size: Size of the sliding window.
            stride: Stride of the sliding window.

        Returns:
            torch.Tensor: Sliding window mask.
        """
        mask = torch.zeros(sequence_length, sequence_length)
        
        for i in range(0, sequence_length, stride):
            start = max(0, i - window_size // 2)
            end = min(sequence_length, i + window_size // 2 + 1)
            mask[i, start:end] = 1
        
        return mask

    def split_sequences(
        self,
        sequences: torch.Tensor,
        chunk_size: int,
        overlap: int = 0
    ) -> List[torch.Tensor]:
        """Split sequences into chunks.

        Args:
            sequences: Input sequences.
            chunk_size: Size of each chunk.
            overlap: Overlap between chunks.

        Returns:
            List[torch.Tensor]: List of sequence chunks.
        """
        chunks = []
        for i in range(0, sequences.size(1), chunk_size - overlap):
            chunk = sequences[:, i:i + chunk_size]
            if chunk.size(1) == chunk_size:
                chunks.append(chunk)
        
        return chunks

    def merge_sequences(
        self,
        chunks: List[torch.Tensor],
        overlap: int = 0,
        strategy: str = "mean"
    ) -> torch.Tensor:
        """Merge sequence chunks.

        Args:
            chunks: List of sequence chunks.
            overlap: Overlap between chunks.
            strategy: Merging strategy ("mean", "max", "last").

        Returns:
            torch.Tensor: Merged sequence.
        """
        if not chunks:
            return torch.tensor([])
        
        # Calculate total length
        chunk_size = chunks[0].size(1)
        total_length = chunk_size + (len(chunks) - 1) * (chunk_size - overlap)
        
        # Initialize output tensor
        merged = torch.zeros(chunks[0].size(0), total_length, chunks[0].size(2))
        
        # Merge chunks
        for i, chunk in enumerate(chunks):
            start_idx = i * (chunk_size - overlap)
            end_idx = start_idx + chunk_size
            
            if i == 0:
                merged[:, start_idx:end_idx] = chunk
            else:
                if strategy == "mean":
                    merged[:, start_idx:start_idx + overlap] = (
                        merged[:, start_idx:start_idx + overlap] + chunk[:, :overlap]
                    ) / 2
                elif strategy == "max":
                    merged[:, start_idx:start_idx + overlap] = torch.max(
                        merged[:, start_idx:start_idx + overlap], chunk[:, :overlap]
                    )
                elif strategy == "last":
                    merged[:, start_idx:start_idx + overlap] = chunk[:, :overlap]
                
                merged[:, start_idx + overlap:end_idx] = chunk[:, overlap:]
        
        return merged


class TextDataset:
    """Dataset for text data with tokenization."""

    def __init__(
        self,
        texts: List[str],
        tokenizer: BaseTokenizer,
        max_sequence_length: int = 512,
        add_special_tokens: bool = True
    ):
        """Initialize text dataset.

        Args:
            texts: List of input texts.
            tokenizer: Tokenizer instance.
            max_sequence_length: Maximum sequence length.
            add_special_tokens: Whether to add special tokens.
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.add_special_tokens = add_special_tokens
        
        # Tokenize all texts
        self.tokenized_texts = []
        for text in texts:
            token_ids = self.tokenizer.encode(text)
            if add_special_tokens:
                token_ids = (
                    [self.tokenizer.vocab[self.tokenizer.special_tokens["start"]]] +
                    token_ids +
                    [self.tokenizer.vocab[self.tokenizer.special_tokens["end"]]]
                )
            
            if len(token_ids) > max_sequence_length:
                token_ids = token_ids[:max_sequence_length]
            
            self.tokenized_texts.append(token_ids)

    def __len__(self) -> int:
        """Get dataset length.

        Returns:
            int: Dataset length.
        """
        return len(self.tokenized_texts)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get item at index.

        Args:
            idx: Index.

        Returns:
            torch.Tensor: Tokenized sequence.
        """
        return torch.tensor(self.tokenized_texts[idx])

    def get_batch(self, indices: List[int]) -> torch.Tensor:
        """Get batch of sequences.

        Args:
            indices: List of indices.

        Returns:
            torch.Tensor: Batch of sequences.
        """
        sequences = [self.tokenized_texts[i] for i in indices]
        return self.tokenizer.encode_batch(sequences, padding=True, truncation=True)


class TextDataLoader:
    """Data loader for text data."""

    def __init__(
        self,
        dataset: TextDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        drop_last: bool = False
    ):
        """Initialize text data loader.

        Args:
            dataset: Text dataset.
            batch_size: Batch size.
            shuffle: Whether to shuffle data.
            drop_last: Whether to drop last incomplete batch.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[torch.Tensor]:
        """Get iterator.

        Returns:
            Iterator[torch.Tensor]: Data iterator.
        """
        indices = list(range(len(self.dataset)))
        
        if self.shuffle:
            np.random.shuffle(indices)
        
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            
            if self.drop_last and len(batch_indices) < self.batch_size:
                continue
            
            yield self.dataset.get_batch(batch_indices)

    def __len__(self) -> int:
        """Get number of batches.

        Returns:
            int: Number of batches.
        """
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def create_tokenizer(
    tokenizer_type: str,
    config: TokenizationConfig
) -> BaseTokenizer:
    """Create tokenizer based on type.

    Args:
        tokenizer_type: Type of tokenizer ("word", "character", "subword").
        config: Tokenization configuration.

    Returns:
        BaseTokenizer: Created tokenizer.
    """
    if tokenizer_type == "word":
        return WordTokenizer(config)
    elif tokenizer_type == "character":
        return CharacterTokenizer(config)
    elif tokenizer_type == "subword":
        return SubwordTokenizer(config)
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")


def create_sequence_handler(max_sequence_length: int = 512) -> SequenceHandler:
    """Create sequence handler.

    Args:
        max_sequence_length: Maximum sequence length.

    Returns:
        SequenceHandler: Created sequence handler.
    """
    return SequenceHandler(max_sequence_length)


def create_text_dataset(
    texts: List[str],
    tokenizer: BaseTokenizer,
    max_sequence_length: int = 512,
    add_special_tokens: bool = True
) -> TextDataset:
    """Create text dataset.

    Args:
        texts: List of input texts.
        tokenizer: Tokenizer instance.
        max_sequence_length: Maximum sequence length.
        add_special_tokens: Whether to add special tokens.

    Returns:
        TextDataset: Created text dataset.
    """
    return TextDataset(texts, tokenizer, max_sequence_length, add_special_tokens)


def create_text_dataloader(
    dataset: TextDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    drop_last: bool = False
) -> TextDataLoader:
    """Create text data loader.

    Args:
        dataset: Text dataset.
        batch_size: Batch size.
        shuffle: Whether to shuffle data.
        drop_last: Whether to drop last incomplete batch.

    Returns:
        TextDataLoader: Created text data loader.
    """
    return TextDataLoader(dataset, batch_size, shuffle, drop_last) 