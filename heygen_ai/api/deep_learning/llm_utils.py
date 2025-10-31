from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import math
import logging
import numpy as np
import json
import re
from typing import Any, List, Dict, Optional
import asyncio
"""
LLM Utilities for HeyGen AI.

Advanced utilities for Large Language Models (LLMs) including tokenization,
generation, and inference following PEP 8 style guidelines.
"""


logger = logging.getLogger(__name__)


class Tokenizer:
    """Simple tokenizer for LLM models."""

    def __init__(
        self,
        vocabulary: Dict[str, int],
        special_tokens: Optional[Dict[str, str]] = None,
        pad_token: str = "<PAD>",
        unk_token: str = "<UNK>",
        bos_token: str = "<BOS>",
        eos_token: str = "<EOS>",
        mask_token: str = "<MASK>"
    ):
        """Initialize tokenizer.

        Args:
            vocabulary: Vocabulary mapping tokens to IDs.
            special_tokens: Special tokens mapping.
            pad_token: Padding token.
            unk_token: Unknown token.
            bos_token: Beginning of sequence token.
            eos_token: End of sequence token.
            mask_token: Mask token.
        """
        self.vocabulary = vocabulary
        self.reverse_vocabulary = {v: k for k, v in vocabulary.items()}
        
        # Special tokens
        self.special_tokens = special_tokens or {}
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.mask_token = mask_token
        
        # Token IDs
        self.pad_token_id = vocabulary.get(pad_token, 0)
        self.unk_token_id = vocabulary.get(unk_token, 1)
        self.bos_token_id = vocabulary.get(bos_token, 2)
        self.eos_token_id = vocabulary.get(eos_token, 3)
        self.mask_token_id = vocabulary.get(mask_token, 4)

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = True,
        padding: bool = False
    ) -> List[int]:
        """Encode text to token IDs.

        Args:
            text: Input text.
            add_special_tokens: Whether to add special tokens.
            max_length: Maximum sequence length.
            truncation: Whether to truncate sequences.
            padding: Whether to pad sequences.

        Returns:
            List[int]: Token IDs.
        """
        # Simple word-based tokenization
        tokens = text.split()
        
        # Convert tokens to IDs
        token_ids = []
        for token in tokens:
            token_id = self.vocabulary.get(token, self.unk_token_id)
            token_ids.append(token_id)
        
        # Add special tokens
        if add_special_tokens:
            token_ids = [self.bos_token_id] + token_ids + [self.eos_token_id]
        
        # Truncation
        if max_length is not None and truncation and len(token_ids) > max_length:
            if add_special_tokens:
                token_ids = token_ids[:max_length - 1] + [self.eos_token_id]
            else:
                token_ids = token_ids[:max_length]
        
        # Padding
        if padding and max_length is not None and len(token_ids) < max_length:
            token_ids = token_ids + [self.pad_token_id] * (max_length - len(token_ids))
        
        return token_ids

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True
    ) -> str:
        """Decode token IDs to text.

        Args:
            token_ids: Token IDs.
            skip_special_tokens: Whether to skip special tokens.

        Returns:
            str: Decoded text.
        """
        tokens = []
        for token_id in token_ids:
            token = self.reverse_vocabulary.get(token_id, self.unk_token)
            if skip_special_tokens and token in [self.pad_token, self.unk_token, self.bos_token, self.eos_token, self.mask_token]:
                continue
            tokens.append(token)
        
        return " ".join(tokens)

    def batch_encode(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = True,
        padding: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Encode batch of texts.

        Args:
            texts: List of input texts.
            add_special_tokens: Whether to add special tokens.
            max_length: Maximum sequence length.
            truncation: Whether to truncate sequences.
            padding: Whether to pad sequences.

        Returns:
            Dict[str, torch.Tensor]: Encoded batch.
        """
        # Encode all texts
        encoded_texts = [
            self.encode(
                text=text,
                add_special_tokens=add_special_tokens,
                max_length=max_length,
                truncation=truncation,
                padding=False
            )
            for text in texts
        ]
        
        # Find maximum length
        if max_length is None:
            max_length = max(len(encoded) for encoded in encoded_texts)
        
        # Pad sequences
        if padding:
            for i in range(len(encoded_texts)):
                if len(encoded_texts[i]) < max_length:
                    encoded_texts[i] = encoded_texts[i] + [self.pad_token_id] * (max_length - len(encoded_texts[i]))
        
        # Convert to tensors
        input_ids = torch.tensor(encoded_texts, dtype=torch.long)
        attention_mask = (input_ids != self.pad_token_id).long()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }


class GenerationConfig:
    """Configuration for text generation."""

    def __init__(
        self,
        max_length: int = 100,
        min_length: int = 1,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        length_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        do_sample: bool = True,
        num_beams: int = 1,
        early_stopping: bool = False,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None
    ):
        """Initialize generation config.

        Args:
            max_length: Maximum generation length.
            min_length: Minimum generation length.
            temperature: Sampling temperature.
            top_k: Top-k sampling parameter.
            top_p: Top-p sampling parameter.
            repetition_penalty: Repetition penalty.
            length_penalty: Length penalty.
            no_repeat_ngram_size: N-gram repetition prevention.
            do_sample: Whether to use sampling.
            num_beams: Number of beams for beam search.
            early_stopping: Whether to use early stopping.
            pad_token_id: Padding token ID.
            eos_token_id: End of sequence token ID.
        """
        self.max_length = max_length
        self.min_length = min_length
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.length_penalty = length_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.do_sample = do_sample
        self.num_beams = num_beams
        self.early_stopping = early_stopping
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id


class TextGenerator:
    """Text generation utilities for LLMs."""

    def __init__(self, model: nn.Module, tokenizer: Tokenizer):
        """Initialize text generator.

        Args:
            model: Language model.
            tokenizer: Tokenizer.
        """
        self.model = model
        self.tokenizer = tokenizer

    def generate(
        self,
        input_text: str,
        generation_config: GenerationConfig,
        **kwargs
    ) -> str:
        """Generate text.

        Args:
            input_text: Input text.
            generation_config: Generation configuration.
            **kwargs: Additional arguments.

        Returns:
            str: Generated text.
        """
        # Encode input
        input_ids = self.tokenizer.encode(input_text, add_special_tokens=True)
        input_ids = torch.tensor([input_ids], dtype=torch.long)
        
        # Generate
        if generation_config.num_beams > 1:
            output_ids = self._beam_search(input_ids, generation_config)
        else:
            output_ids = self._greedy_search(input_ids, generation_config)
        
        # Decode output
        generated_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        
        return generated_text

    def _greedy_search(
        self,
        input_ids: torch.Tensor,
        generation_config: GenerationConfig
    ) -> List[int]:
        """Greedy search generation.

        Args:
            input_ids: Input token IDs.
            generation_config: Generation configuration.

        Returns:
            List[int]: Generated token IDs.
        """
        self.model.eval()
        
        with torch.no_grad():
            current_ids = input_ids.clone()
            
            for _ in range(generation_config.max_length - input_ids.shape[1]):
                # Get logits
                outputs = self.model(current_ids)
                next_token_logits = outputs[:, -1, :]
                
                # Apply repetition penalty
                if generation_config.repetition_penalty != 1.0:
                    for i in range(current_ids.shape[0]):
                        for previous_token in set(current_ids[i].tolist()):
                            if previous_token in self.tokenizer.vocabulary.values():
                                next_token_logits[i, previous_token] /= generation_config.repetition_penalty
                
                # Apply temperature
                next_token_logits = next_token_logits / generation_config.temperature
                
                # Apply top-k filtering
                if generation_config.top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, generation_config.top_k, dim=-1)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                
                # Apply top-p filtering
                if generation_config.top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > generation_config.top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample or greedy decode
                if generation_config.do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append next token
                current_ids = torch.cat([current_ids, next_token], dim=-1)
                
                # Check for EOS
                if generation_config.eos_token_id is not None and (next_token == generation_config.eos_token_id).any():
                    break
        
        return current_ids[0].tolist()

    def _beam_search(
        self,
        input_ids: torch.Tensor,
        generation_config: GenerationConfig
    ) -> List[int]:
        """Beam search generation.

        Args:
            input_ids: Input token IDs.
            generation_config: Generation configuration.

        Returns:
            List[int]: Generated token IDs.
        """
        self.model.eval()
        
        with torch.no_grad():
            batch_size = input_ids.shape[0]
            num_beams = generation_config.num_beams
            
            # Expand input for beam search
            input_ids = input_ids.repeat(num_beams, 1)
            
            # Initialize beam states
            beam_scores = torch.zeros(batch_size * num_beams, device=input_ids.device)
            beam_scores[1::num_beams] = -1e9  # Only first beam is active initially
            
            current_ids = input_ids.clone()
            
            for step in range(generation_config.max_length - input_ids.shape[1]):
                # Get logits
                outputs = self.model(current_ids)
                next_token_logits = outputs[:, -1, :]
                
                # Apply repetition penalty
                if generation_config.repetition_penalty != 1.0:
                    for i in range(current_ids.shape[0]):
                        for previous_token in set(current_ids[i].tolist()):
                            if previous_token in self.tokenizer.vocabulary.values():
                                next_token_logits[i, previous_token] /= generation_config.repetition_penalty
                
                # Apply temperature
                next_token_logits = next_token_logits / generation_config.temperature
                
                # Get top-k candidates
                vocab_size = next_token_logits.shape[-1]
                next_token_logits = next_token_logits.view(batch_size, num_beams, vocab_size)
                next_token_scores = F.log_softmax(next_token_logits, dim=-1)
                
                # Add beam scores
                next_token_scores = next_token_scores + beam_scores.unsqueeze(-1)
                
                # Reshape for beam selection
                next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)
                
                # Select top beams
                next_token_scores, next_token_indices = torch.topk(
                    next_token_scores, num_beams, dim=-1
                )
                
                # Update beam states
                beam_indices = next_token_indices // vocab_size
                token_indices = next_token_indices % vocab_size
                
                # Update current IDs and scores
                new_current_ids = []
                new_beam_scores = []
                
                for batch_idx in range(batch_size):
                    for beam_idx in range(num_beams):
                        beam_id = beam_indices[batch_idx, beam_idx]
                        token_id = token_indices[batch_idx, beam_idx]
                        
                        # Get current sequence
                        current_sequence = current_ids[batch_idx * num_beams + beam_id]
                        new_sequence = torch.cat([current_sequence, token_id.unsqueeze(0)])
                        new_current_ids.append(new_sequence)
                        
                        # Update score
                        new_beam_scores.append(next_token_scores[batch_idx, beam_idx])
                
                current_ids = torch.stack(new_current_ids)
                beam_scores = torch.stack(new_beam_scores)
                
                # Check for EOS
                if generation_config.eos_token_id is not None:
                    eos_mask = (current_ids == generation_config.eos_token_id).any(dim=-1)
                    if eos_mask.any() and generation_config.early_stopping:
                        break
        
        # Return best sequence
        best_beam_idx = torch.argmax(beam_scores.view(batch_size, num_beams), dim=-1)
        best_sequences = []
        
        for batch_idx in range(batch_size):
            best_sequence = current_ids[batch_idx * num_beams + best_beam_idx[batch_idx]]
            best_sequences.append(best_sequence.tolist())
        
        return best_sequences[0]


class LLMInference:
    """LLM inference utilities."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Tokenizer,
        device: Optional[torch.device] = None
    ):
        """Initialize LLM inference.

        Args:
            model: Language model.
            tokenizer: Tokenizer.
            device: Device to run inference on.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()

    def generate_text(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        num_beams: int = 1,
        do_sample: bool = True
    ) -> str:
        """Generate text from prompt.

        Args:
            prompt: Input prompt.
            max_length: Maximum generation length.
            temperature: Sampling temperature.
            top_k: Top-k sampling parameter.
            top_p: Top-p sampling parameter.
            num_beams: Number of beams for beam search.
            do_sample: Whether to use sampling.

        Returns:
            str: Generated text.
        """
        # Create generation config
        generation_config = GenerationConfig(
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_beams=num_beams,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        # Create text generator
        generator = TextGenerator(self.model, self.tokenizer)
        
        # Generate text
        generated_text = generator.generate(prompt, generation_config)
        
        return generated_text

    def batch_generate(
        self,
        prompts: List[str],
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        num_beams: int = 1,
        do_sample: bool = True
    ) -> List[str]:
        """Generate text for multiple prompts.

        Args:
            prompts: List of input prompts.
            max_length: Maximum generation length.
            temperature: Sampling temperature.
            top_k: Top-k sampling parameter.
            top_p: Top-p sampling parameter.
            num_beams: Number of beams for beam search.
            do_sample: Whether to use sampling.

        Returns:
            List[str]: Generated texts.
        """
        generated_texts = []
        
        for prompt in prompts:
            generated_text = self.generate_text(
                prompt=prompt,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                num_beams=num_beams,
                do_sample=do_sample
            )
            generated_texts.append(generated_text)
        
        return generated_texts

    def get_embeddings(
        self,
        texts: List[str],
        max_length: Optional[int] = None
    ) -> torch.Tensor:
        """Get embeddings for texts.

        Args:
            texts: List of input texts.
            max_length: Maximum sequence length.

        Returns:
            torch.Tensor: Text embeddings.
        """
        # Encode texts
        encoded = self.tokenizer.batch_encode(
            texts=texts,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
            padding=True
        )
        
        # Move to device
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            
            # Use last hidden state as embeddings
            if isinstance(outputs, dict):
                embeddings = outputs["sequence_output"]
            else:
                embeddings = outputs
            
            # Average pooling over sequence length
            embeddings = (embeddings * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        
        return embeddings

    def compute_perplexity(
        self,
        texts: List[str],
        max_length: Optional[int] = None
    ) -> float:
        """Compute perplexity for texts.

        Args:
            texts: List of input texts.
            max_length: Maximum sequence length.

        Returns:
            float: Average perplexity.
        """
        # Encode texts
        encoded = self.tokenizer.batch_encode(
            texts=texts,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
            padding=True
        )
        
        # Move to device
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        
        # Compute perplexity
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for i in range(input_ids.shape[0]):
                # Get sequence
                sequence = input_ids[i:i+1]
                mask = attention_mask[i:i+1]
                
                # Get logits
                outputs = self.model(sequence, attention_mask=mask)
                
                if isinstance(outputs, dict):
                    logits = outputs["mlm_logits"]
                else:
                    logits = outputs
                
                # Shift for language modeling
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = sequence[..., 1:].contiguous()
                shift_mask = mask[..., 1:].contiguous()
                
                # Compute loss
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=self.tokenizer.pad_token_id
                )
                
                # Count tokens
                num_tokens = shift_mask.sum().item()
                
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
        
        # Compute average perplexity
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        
        return perplexity


class LLMTraining:
    """LLM training utilities."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Tokenizer,
        device: Optional[torch.device] = None
    ):
        """Initialize LLM training.

        Args:
            model: Language model.
            tokenizer: Tokenizer.
            device: Device to run training on.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move model to device
        self.model.to(self.device)

    def prepare_training_data(
        self,
        texts: List[str],
        max_length: int = 512,
        stride: int = 128
    ) -> torch.utils.data.Dataset:
        """Prepare training data.

        Args:
            texts: List of training texts.
            max_length: Maximum sequence length.
            stride: Stride for sliding window.

        Returns:
            torch.utils.data.Dataset: Training dataset.
        """
        # Tokenize all texts
        all_tokens = []
        for text in texts:
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            all_tokens.extend(tokens)
        
        # Create sliding windows
        training_data = []
        
        for i in range(0, len(all_tokens) - max_length + 1, stride):
            window = all_tokens[i:i + max_length]
            
            # Create input and target
            input_ids = window[:-1]
            target_ids = window[1:]
            
            training_data.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "target_ids": torch.tensor(target_ids, dtype=torch.long)
            })
        
        return training_data

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        loss_function: Optional[Callable] = None
    ) -> float:
        """Perform training step.

        Args:
            batch: Training batch.
            optimizer: Optimizer.
            loss_function: Loss function.

        Returns:
            float: Training loss.
        """
        self.model.train()
        
        # Move batch to device
        input_ids = batch["input_ids"].to(self.device)
        target_ids = batch["target_ids"].to(self.device)
        
        # Forward pass
        outputs = self.model(input_ids)
        
        # Compute loss
        if loss_function is None:
            loss = F.cross_entropy(
                outputs.view(-1, outputs.size(-1)),
                target_ids.view(-1),
                ignore_index=self.tokenizer.pad_token_id
            )
        else:
            loss = loss_function(outputs, target_ids)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()

    def evaluate(
        self,
        texts: List[str],
        max_length: int = 512
    ) -> Dict[str, float]:
        """Evaluate model.

        Args:
            texts: Evaluation texts.
            max_length: Maximum sequence length.

        Returns:
            Dict[str, float]: Evaluation metrics.
        """
        self.model.eval()
        
        # Compute perplexity
        perplexity = LLMInference(self.model, self.tokenizer, self.device).compute_perplexity(
            texts, max_length
        )
        
        return {
            "perplexity": perplexity
        }


def create_simple_vocabulary(texts: List[str], min_frequency: int = 1) -> Dict[str, int]:
    """Create simple vocabulary from texts.

    Args:
        texts: List of texts.
        min_frequency: Minimum token frequency.

    Returns:
        Dict[str, int]: Vocabulary.
    """
    # Count token frequencies
    token_counts = {}
    
    for text in texts:
        tokens = text.split()
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1
    
    # Filter by minimum frequency
    filtered_tokens = [
        token for token, count in token_counts.items()
        if count >= min_frequency
    ]
    
    # Create vocabulary
    vocabulary = {
        "<PAD>": 0,
        "<UNK>": 1,
        "<BOS>": 2,
        "<EOS>": 3,
        "<MASK>": 4
    }
    
    for i, token in enumerate(sorted(filtered_tokens)):
        vocabulary[token] = i + 5
    
    return vocabulary


def save_vocabulary(vocabulary: Dict[str, int], filepath: str):
    """Save vocabulary to file.

    Args:
        vocabulary: Vocabulary to save.
        filepath: File path.
    """
    with open(filepath, 'w', encoding='utf-8') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        json.dump(vocabulary, f, ensure_ascii=False, indent=2)


def load_vocabulary(filepath: str) -> Dict[str, int]:
    """Load vocabulary from file.

    Args:
        filepath: File path.

    Returns:
        Dict[str, int]: Loaded vocabulary.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        vocabulary = json.load(f)
    
    # Convert string keys back to integers
    return {k: int(v) if isinstance(v, str) and v.isdigit() else v for k, v in vocabulary.items()} 