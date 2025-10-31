from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from collections import defaultdict
import re
import json
import torch
from transformers import (
from tokenizers import Tokenizer as HFTokenizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
from ..models.sequence import EmailSequence, SequenceStep
from ..models.subscriber import Subscriber
from ..models.template import EmailTemplate
            import subprocess
from typing import Any, List, Dict, Optional
"""
Tokenization Engine for Email Sequence System

Advanced tokenization and sequence handling for text data with support for
multiple tokenizers, sequence optimization, and efficient text processing.
"""


    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    BatchEncoding
)


logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


@dataclass
class TokenizationConfig:
    """Configuration for tokenization engine"""
    model_name: str = "microsoft/DialoGPT-medium"
    max_length: int = 512
    truncation: bool = True
    padding: str = "max_length"
    return_tensors: str = "pt"
    return_attention_mask: bool = True
    return_token_type_ids: bool = False
    add_special_tokens: bool = True
    use_fast_tokenizer: bool = True
    cache_dir: Optional[str] = None


@dataclass
class SequenceInfo:
    """Information about tokenized sequence"""
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: Optional[torch.Tensor] = None
    length: int = 0
    token_count: int = 0
    special_tokens: Dict[str, int] = None
    vocabulary_coverage: float = 0.0


class AdvancedTokenizer:
    """Advanced tokenizer with multiple tokenization strategies"""
    
    def __init__(self, config: TokenizationConfig):
        
    """__init__ function."""
self.config = config
        self.tokenizer = self._load_tokenizer()
        self.spacy_nlp = self._load_spacy()
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),
            stop_words='english'
        )
        
        # Tokenization statistics
        self.tokenization_stats = defaultdict(int)
        self.vocabulary_stats = defaultdict(int)
        
        logger.info(f"Advanced Tokenizer initialized with {config.model_name}")
    
    def _load_tokenizer(self) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
        """Load the appropriate tokenizer"""
        try:
            if self.config.use_fast_tokenizer:
                tokenizer = AutoTokenizer.from_pretrained(
                    self.config.model_name,
                    use_fast=True,
                    cache_dir=self.config.cache_dir
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(
                    self.config.model_name,
                    use_fast=False,
                    cache_dir=self.config.cache_dir
                )
            
            # Add padding token if not present
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            return tokenizer
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
    
    def _load_spacy(self) -> Any:
        """Load spaCy model for advanced NLP"""
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Installing...")
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            return spacy.load("en_core_web_sm")
    
    async def tokenize_text(
        self, 
        text: str, 
        strategy: str = "standard"
    ) -> SequenceInfo:
        """Tokenize text using specified strategy"""
        
        if strategy == "standard":
            return await self._standard_tokenization(text)
        elif strategy == "advanced":
            return await self._advanced_tokenization(text)
        elif strategy == "semantic":
            return await self._semantic_tokenization(text)
        elif strategy == "email_specific":
            return await self._email_specific_tokenization(text)
        else:
            raise ValueError(f"Unknown tokenization strategy: {strategy}")
    
    async def _standard_tokenization(self, text: str) -> SequenceInfo:
        """Standard tokenization using transformers tokenizer"""
        
        # Clean and preprocess text
        cleaned_text = self._preprocess_text(text)
        
        # Tokenize
        encoding = self.tokenizer(
            cleaned_text,
            max_length=self.config.max_length,
            truncation=self.config.truncation,
            padding=self.config.padding,
            return_tensors=self.config.return_tensors,
            return_attention_mask=self.config.return_attention_mask,
            return_token_type_ids=self.config.return_token_type_ids,
            add_special_tokens=self.config.add_special_tokens
        )
        
        # Update statistics
        self._update_tokenization_stats(encoding, "standard")
        
        return SequenceInfo(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
            token_type_ids=encoding.get("token_type_ids"),
            length=len(encoding["input_ids"][0]),
            token_count=encoding["attention_mask"][0].sum().item(),
            special_tokens=self._count_special_tokens(encoding["input_ids"][0]),
            vocabulary_coverage=self._calculate_vocabulary_coverage(encoding["input_ids"][0])
        )
    
    async def _advanced_tokenization(self, text: str) -> SequenceInfo:
        """Advanced tokenization with NLP preprocessing"""
        
        # Use spaCy for advanced preprocessing
        doc = self.spacy_nlp(text)
        
        # Extract entities, POS tags, and dependencies
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        pos_tags = [(token.text, token.pos_) for token in doc]
        
        # Create enhanced text with entity markers
        enhanced_text = self._enhance_text_with_entities(text, entities)
        
        # Tokenize enhanced text
        encoding = self.tokenizer(
            enhanced_text,
            max_length=self.config.max_length,
            truncation=self.config.truncation,
            padding=self.config.padding,
            return_tensors=self.config.return_tensors,
            return_attention_mask=self.config.return_attention_mask,
            return_token_type_ids=self.config.return_token_type_ids,
            add_special_tokens=self.config.add_special_tokens
        )
        
        # Update statistics
        self._update_tokenization_stats(encoding, "advanced")
        
        return SequenceInfo(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
            token_type_ids=encoding.get("token_type_ids"),
            length=len(encoding["input_ids"][0]),
            token_count=encoding["attention_mask"][0].sum().item(),
            special_tokens=self._count_special_tokens(encoding["input_ids"][0]),
            vocabulary_coverage=self._calculate_vocabulary_coverage(encoding["input_ids"][0])
        )
    
    async def _semantic_tokenization(self, text: str) -> SequenceInfo:
        """Semantic tokenization with meaning preservation"""
        
        # Extract semantic features
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        
        # Create semantic representation
        semantic_text = self._create_semantic_representation(sentences, words)
        
        # Tokenize semantic text
        encoding = self.tokenizer(
            semantic_text,
            max_length=self.config.max_length,
            truncation=self.config.truncation,
            padding=self.config.padding,
            return_tensors=self.config.return_tensors,
            return_attention_mask=self.config.return_attention_mask,
            return_token_type_ids=self.config.return_token_type_ids,
            add_special_tokens=self.config.add_special_tokens
        )
        
        # Update statistics
        self._update_tokenization_stats(encoding, "semantic")
        
        return SequenceInfo(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
            token_type_ids=encoding.get("token_type_ids"),
            length=len(encoding["input_ids"][0]),
            token_count=encoding["attention_mask"][0].sum().item(),
            special_tokens=self._count_special_tokens(encoding["input_ids"][0]),
            vocabulary_coverage=self._calculate_vocabulary_coverage(encoding["input_ids"][0])
        )
    
    async def _email_specific_tokenization(self, text: str) -> SequenceInfo:
        """Email-specific tokenization with domain knowledge"""
        
        # Extract email-specific features
        email_features = self._extract_email_features(text)
        
        # Create email-specific representation
        email_text = self._create_email_representation(text, email_features)
        
        # Tokenize email text
        encoding = self.tokenizer(
            email_text,
            max_length=self.config.max_length,
            truncation=self.config.truncation,
            padding=self.config.padding,
            return_tensors=self.config.return_tensors,
            return_attention_mask=self.config.return_attention_mask,
            return_token_type_ids=self.config.return_token_type_ids,
            add_special_tokens=self.config.add_special_tokens
        )
        
        # Update statistics
        self._update_tokenization_stats(encoding, "email_specific")
        
        return SequenceInfo(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
            token_type_ids=encoding.get("token_type_ids"),
            length=len(encoding["input_ids"][0]),
            token_count=encoding["attention_mask"][0].sum().item(),
            special_tokens=self._count_special_tokens(encoding["input_ids"][0]),
            vocabulary_coverage=self._calculate_vocabulary_coverage(encoding["input_ids"][0])
        )
    
    async def batch_tokenize(
        self, 
        texts: List[str], 
        strategy: str = "standard"
    ) -> List[SequenceInfo]:
        """Tokenize multiple texts in batch"""
        
        tasks = [self.tokenize_text(text, strategy) for text in texts]
        results = await asyncio.gather(*tasks)
        
        return results
    
    async def tokenize_email_sequence(
        self, 
        sequence: EmailSequence, 
        subscriber: Subscriber,
        template: EmailTemplate
    ) -> Dict[str, SequenceInfo]:
        """Tokenize complete email sequence with context"""
        
        sequence_tokens = {}
        
        for step in sequence.steps:
            # Create context-aware text
            context_text = self._create_context_text(step, subscriber, template)
            
            # Tokenize with email-specific strategy
            tokens = await self.tokenize_text(context_text, "email_specific")
            
            sequence_tokens[f"step_{step.order}"] = tokens
        
        return sequence_tokens
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for tokenization"""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Normalize quotes and dashes
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r'[''']', "'", text)
        text = re.sub(r'–|—', '-', text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Normalize URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '[URL]', text)
        
        return text
    
    def _enhance_text_with_entities(self, text: str, entities: List[Tuple[str, str]]) -> str:
        """Enhance text with entity markers"""
        
        enhanced_text = text
        
        for entity_text, entity_type in entities:
            if entity_type in ['PERSON', 'ORG', 'GPE', 'PRODUCT']:
                marker = f"[{entity_type}:{entity_text}]"
                enhanced_text = enhanced_text.replace(entity_text, marker)
        
        return enhanced_text
    
    def _create_semantic_representation(self, sentences: List[str], words: List[str]) -> str:
        """Create semantic representation of text"""
        
        # Extract key concepts
        key_concepts = self._extract_key_concepts(sentences)
        
        # Create semantic structure
        semantic_parts = []
        
        # Add main content
        semantic_parts.append(" ".join(sentences))
        
        # Add key concepts
        if key_concepts:
            semantic_parts.append(f"Key concepts: {', '.join(key_concepts)}")
        
        return " | ".join(semantic_parts)
    
    def _extract_key_concepts(self, sentences: List[str]) -> List[str]:
        """Extract key concepts from sentences"""
        
        # Simple keyword extraction (in production, use more sophisticated methods)
        all_words = []
        for sentence in sentences:
            words = word_tokenize(sentence.lower())
            all_words.extend([w for w in words if len(w) > 3])
        
        # Count word frequencies
        word_freq = defaultdict(int)
        for word in all_words:
            word_freq[word] += 1
        
        # Return top concepts
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:10]]
    
    def _extract_email_features(self, text: str) -> Dict[str, Any]:
        """Extract email-specific features"""
        
        features = {
            "has_greeting": bool(re.search(r'\b(hi|hello|dear|good morning|good afternoon)\b', text.lower())),
            "has_call_to_action": bool(re.search(r'\b(click|download|sign up|register|subscribe|buy|order)\b', text.lower())),
            "has_urgency": bool(re.search(r'\b(limited time|offer ends|act now|hurry|urgent)\b', text.lower())),
            "has_personalization": bool(re.search(r'\b(your|you|personal|customized)\b', text.lower())),
            "has_social_proof": bool(re.search(r'\b(customers|users|people|testimonials|reviews)\b', text.lower())),
            "has_benefits": bool(re.search(r'\b(benefit|advantage|feature|improve|enhance)\b', text.lower())),
            "sentence_count": len(sent_tokenize(text)),
            "word_count": len(word_tokenize(text)),
            "avg_sentence_length": len(word_tokenize(text)) / max(len(sent_tokenize(text)), 1)
        }
        
        return features
    
    def _create_email_representation(self, text: str, features: Dict[str, Any]) -> str:
        """Create email-specific text representation"""
        
        # Add feature markers
        markers = []
        
        if features["has_greeting"]:
            markers.append("[GREETING]")
        if features["has_call_to_action"]:
            markers.append("[CTA]")
        if features["has_urgency"]:
            markers.append("[URGENCY]")
        if features["has_personalization"]:
            markers.append("[PERSONALIZATION]")
        if features["has_social_proof"]:
            markers.append("[SOCIAL_PROOF]")
        if features["has_benefits"]:
            markers.append("[BENEFITS]")
        
        # Create enhanced representation
        enhanced_text = f"{' '.join(markers)} {text}"
        
        return enhanced_text
    
    def _create_context_text(
        self, 
        step: SequenceStep, 
        subscriber: Subscriber, 
        template: EmailTemplate
    ) -> str:
        """Create context-aware text for email step"""
        
        context_parts = []
        
        # Add subscriber context
        context_parts.append(f"Subscriber: {subscriber.first_name} {subscriber.last_name}")
        context_parts.append(f"Company: {subscriber.company}")
        context_parts.append(f"Interests: {', '.join(subscriber.interests)}")
        
        # Add template context
        context_parts.append(f"Template: {template.name}")
        context_parts.append(f"Category: {template.category}")
        
        # Add step context
        context_parts.append(f"Step: {step.order}")
        context_parts.append(f"Delay: {step.delay_hours} hours")
        
        # Add content
        if step.content:
            context_parts.append(f"Content: {step.content}")
        
        return " | ".join(context_parts)
    
    def _update_tokenization_stats(self, encoding: BatchEncoding, strategy: str):
        """Update tokenization statistics"""
        
        self.tokenization_stats[f"{strategy}_total"] += 1
        self.tokenization_stats[f"{strategy}_tokens"] += encoding["attention_mask"][0].sum().item()
        
        # Update vocabulary statistics
        for token_id in encoding["input_ids"][0]:
            self.vocabulary_stats[token_id.item()] += 1
    
    def _count_special_tokens(self, input_ids: torch.Tensor) -> Dict[str, int]:
        """Count special tokens in sequence"""
        
        special_tokens = {
            "pad": 0,
            "unk": 0,
            "cls": 0,
            "sep": 0,
            "mask": 0
        }
        
        for token_id in input_ids:
            token_id = token_id.item()
            if token_id == self.tokenizer.pad_token_id:
                special_tokens["pad"] += 1
            elif token_id == self.tokenizer.unk_token_id:
                special_tokens["unk"] += 1
            elif token_id == self.tokenizer.cls_token_id:
                special_tokens["cls"] += 1
            elif token_id == self.tokenizer.sep_token_id:
                special_tokens["sep"] += 1
            elif token_id == self.tokenizer.mask_token_id:
                special_tokens["mask"] += 1
        
        return special_tokens
    
    def _calculate_vocabulary_coverage(self, input_ids: torch.Tensor) -> float:
        """Calculate vocabulary coverage"""
        
        unique_tokens = set(input_ids.tolist())
        total_vocab_size = self.tokenizer.vocab_size
        
        return len(unique_tokens) / total_vocab_size if total_vocab_size > 0 else 0.0
    
    async def get_tokenization_report(self) -> Dict[str, Any]:
        """Generate comprehensive tokenization report"""
        
        return {
            "tokenization_stats": dict(self.tokenization_stats),
            "vocabulary_stats": {
                "unique_tokens": len(self.vocabulary_stats),
                "most_common_tokens": dict(sorted(
                    self.vocabulary_stats.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:20])
            },
            "tokenizer_info": {
                "model_name": self.config.model_name,
                "vocab_size": self.tokenizer.vocab_size,
                "max_length": self.config.max_length,
                "special_tokens": {
                    "pad_token": self.tokenizer.pad_token,
                    "unk_token": self.tokenizer.unk_token,
                    "cls_token": self.tokenizer.cls_token,
                    "sep_token": self.tokenizer.sep_token,
                    "mask_token": self.tokenizer.mask_token
                }
            },
            "performance_metrics": {
                "total_tokenizations": sum(self.tokenization_stats.values()),
                "average_tokens_per_text": sum(
                    self.tokenization_stats[f"{s}_tokens"] 
                    for s in ["standard", "advanced", "semantic", "email_specific"]
                ) / max(sum(
                    self.tokenization_stats[f"{s}_total"] 
                    for s in ["standard", "advanced", "semantic", "email_specific"]
                ), 1)
            }
        }
    
    def decode_tokens(self, input_ids: torch.Tensor) -> str:
        """Decode tokens back to text"""
        return self.tokenizer.decode(input_ids, skip_special_tokens=True)
    
    def get_token_info(self, token_id: int) -> Dict[str, Any]:
        """Get information about a specific token"""
        
        token_text = self.tokenizer.decode([token_id])
        
        return {
            "token_id": token_id,
            "token_text": token_text,
            "frequency": self.vocabulary_stats.get(token_id, 0),
            "is_special": token_id in [
                self.tokenizer.pad_token_id,
                self.tokenizer.unk_token_id,
                self.tokenizer.cls_token_id,
                self.tokenizer.sep_token_id,
                self.tokenizer.mask_token_id
            ]
        } 