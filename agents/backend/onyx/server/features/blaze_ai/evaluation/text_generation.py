"""
Text generation evaluation metrics.

This module provides comprehensive evaluation metrics for text generation models
including perplexity, BLEU, ROUGE, BERTScore, and custom content quality metrics.
"""

from __future__ import annotations

import re
import math
from typing import Dict, List, Optional, Tuple, Union
from collections import Counter

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.rouge_score import rouge_scorer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from bert_score import score as bert_score
    BERT_SCORE_AVAILABLE = True
except ImportError:
    BERT_SCORE_AVAILABLE = False


def calculate_perplexity(model, data_loader: DataLoader, device: str) -> float:
    """Calculate perplexity for a language model."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in data_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            
            # Forward pass
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            
            # Count non-padding tokens
            num_tokens = (labels != -100).sum().item()
            
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    if total_tokens == 0:
        return float('inf')
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return perplexity


def calculate_bleu_score(references: List[str], candidates: List[str], 
                        smoothing: bool = True) -> Dict[str, float]:
    """Calculate BLEU scores for text generation evaluation."""
    if not NLTK_AVAILABLE:
        return {"bleu": 0.0, "bleu_1": 0.0, "bleu_2": 0.0, "bleu_3": 0.0, "bleu_4": 0.0}
    
    smoothing_function = SmoothingFunction().method1 if smoothing else None
    
    # Tokenize references and candidates
    ref_tokens = [ref.split() for ref in references]
    cand_tokens = [cand.split() for cand in candidates]
    
    # Calculate BLEU scores for different n-gram orders
    bleu_scores = {}
    for n in range(1, 5):
        try:
            score = sentence_bleu(ref_tokens, cand_tokens, 
                                smoothing_function=smoothing_function,
                                weights=tuple([1.0/n] * n))
            bleu_scores[f"bleu_{n}"] = score
        except:
            bleu_scores[f"bleu_{n}"] = 0.0
    
    # Calculate overall BLEU-4
    try:
        overall_bleu = sentence_bleu(ref_tokens, cand_tokens, 
                                   smoothing_function=smoothing_function)
        bleu_scores["bleu"] = overall_bleu
    except:
        bleu_scores["bleu"] = 0.0
    
    return bleu_scores


def calculate_rouge_scores(references: List[str], candidates: List[str]) -> Dict[str, float]:
    """Calculate ROUGE scores for text generation evaluation."""
    if not NLTK_AVAILABLE:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge_scores = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    
    for ref, cand in zip(references, candidates):
        scores = scorer.score(ref, cand)
        for metric in rouge_scores:
            rouge_scores[metric] += scores[metric].fmeasure
    
    # Average scores
    num_samples = len(references)
    if num_samples > 0:
        for metric in rouge_scores:
            rouge_scores[metric] /= num_samples
    
    return rouge_scores


def calculate_bert_score(references: List[str], candidates: List[str], 
                        model_type: str = "bert-base-uncased") -> Dict[str, float]:
    """Calculate BERTScore for semantic similarity."""
    if not BERT_SCORE_AVAILABLE:
        return {"bert_score": 0.0}
    
    try:
        P, R, F1 = bert_score(candidates, references, model_type=model_type, verbose=False)
        
        return {
            "bert_score_precision": P.mean().item(),
            "bert_score_recall": R.mean().item(), 
            "bert_score_f1": F1.mean().item(),
            "bert_score": F1.mean().item()  # Main metric
        }
    except:
        return {"bert_score": 0.0}


def calculate_content_quality_metrics(generated_texts: List[str], 
                                    reference_texts: List[str]) -> Dict[str, float]:
    """Calculate custom content quality metrics."""
    metrics = {}
    
    # Text length metrics
    gen_lengths = [len(text.split()) for text in generated_texts]
    ref_lengths = [len(text.split()) for text in reference_texts]
    
    metrics["avg_generated_length"] = np.mean(gen_lengths)
    metrics["avg_reference_length"] = np.mean(ref_lengths)
    metrics["length_ratio"] = np.mean(gen_lengths) / max(np.mean(ref_lengths), 1)
    
    # Vocabulary diversity
    all_gen_words = " ".join(generated_texts).lower().split()
    all_ref_words = " ".join(reference_texts).lower().split()
    
    gen_vocab_size = len(set(all_gen_words))
    ref_vocab_size = len(set(all_ref_words))
    
    metrics["generated_vocab_size"] = gen_vocab_size
    metrics["reference_vocab_size"] = ref_vocab_size
    metrics["vocab_diversity_ratio"] = gen_vocab_size / max(ref_vocab_size, 1)
    
    # Repetition metrics
    repetition_scores = []
    for text in generated_texts:
        words = text.lower().split()
        if len(words) < 2:
            repetition_scores.append(0.0)
            continue
        
        # Calculate n-gram repetition
        bigrams = list(zip(words[:-1], words[1:]))
        unique_bigrams = set(bigrams)
        repetition_score = 1.0 - (len(unique_bigrams) / max(len(bigrams), 1))
        repetition_scores.append(repetition_score)
    
    metrics["avg_repetition_score"] = np.mean(repetition_scores)
    
    # Readability metrics (Flesch Reading Ease approximation)
    readability_scores = []
    for text in generated_texts:
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            readability_scores.append(0.0)
            continue
        
        words = text.split()
        syllables = sum(len(re.findall(r'[aeiouy]+', word.lower())) for word in words)
        
        if len(words) == 0 or len(sentences) == 0:
            readability_scores.append(0.0)
        else:
            # Simplified Flesch Reading Ease
            flesch_score = 206.835 - (1.015 * (len(words) / len(sentences))) - (84.6 * (syllables / len(words)))
            readability_scores.append(max(0, min(100, flesch_score)))
    
    metrics["avg_readability_score"] = np.mean(readability_scores)
    
    return metrics


def calculate_semantic_similarity(generated_texts: List[str], 
                                 reference_texts: List[str]) -> Dict[str, float]:
    """Calculate semantic similarity metrics using simple heuristics."""
    metrics = {}
    
    # Word overlap metrics
    overlap_scores = []
    for gen_text, ref_text in zip(generated_texts, reference_texts):
        gen_words = set(gen_text.lower().split())
        ref_words = set(ref_text.lower().split())
        
        if len(ref_words) == 0:
            overlap_scores.append(0.0)
            continue
        
        # Jaccard similarity
        intersection = len(gen_words.intersection(ref_words))
        union = len(gen_words.union(ref_words))
        jaccard = intersection / union if union > 0 else 0.0
        
        # Precision and recall
        precision = intersection / len(gen_words) if len(gen_words) > 0 else 0.0
        recall = intersection / len(ref_words) if len(ref_words) > 0 else 0.0
        
        # F1 score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        overlap_scores.append({
            "jaccard": jaccard,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })
    
    # Average scores
    if overlap_scores:
        metrics["avg_jaccard_similarity"] = np.mean([s["jaccard"] for s in overlap_scores])
        metrics["avg_precision"] = np.mean([s["precision"] for s in overlap_scores])
        metrics["avg_recall"] = np.mean([s["recall"] for s in overlap_scores])
        metrics["avg_f1_similarity"] = np.mean([s["f1"] for s in overlap_scores])
    else:
        metrics.update({
            "avg_jaccard_similarity": 0.0,
            "avg_precision": 0.0,
            "avg_recall": 0.0,
            "avg_f1_similarity": 0.0
        })
    
    return metrics


@torch.inference_mode()
def evaluate_text_generation(model, data_loader: DataLoader, device: str,
                           references: Optional[List[str]] = None,
                           candidates: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Comprehensive evaluation of text generation models.
    
    Args:
        model: The text generation model to evaluate
        data_loader: DataLoader containing evaluation data
        device: Device to run evaluation on
        references: Optional list of reference texts
        candidates: Optional list of generated candidate texts
    
    Returns:
        Dictionary containing all evaluation metrics
    """
    model.eval()
    
    # Calculate perplexity if we have a data loader
    perplexity = float('inf')
    if data_loader is not None:
        perplexity = calculate_perplexity(model, data_loader, device)
    
    # Initialize metrics
    metrics = {"perplexity": perplexity}
    
    # Calculate text-based metrics if references and candidates are provided
    if references and candidates:
        # BLEU scores
        bleu_scores = calculate_bleu_score(references, candidates)
        metrics.update(bleu_scores)
        
        # ROUGE scores
        rouge_scores = calculate_rouge_scores(references, candidates)
        metrics.update(rouge_scores)
        
        # BERTScore
        bert_scores = calculate_bert_score(references, candidates)
        metrics.update(bert_scores)
        
        # Content quality metrics
        quality_metrics = calculate_content_quality_metrics(candidates, references)
        metrics.update(quality_metrics)
        
        # Semantic similarity
        similarity_metrics = calculate_semantic_similarity(candidates, references)
        metrics.update(similarity_metrics)
    
    return metrics


def evaluate_text_generation_batch(generated_texts: List[str], 
                                  reference_texts: List[str],
                                  model_type: str = "bert-base-uncased") -> Dict[str, float]:
    """
    Evaluate a batch of generated texts against references.
    
    Args:
        generated_texts: List of generated text samples
        reference_texts: List of reference text samples
        model_type: BERT model type for BERTScore
    
    Returns:
        Dictionary containing all evaluation metrics
    """
    if len(generated_texts) != len(reference_texts):
        raise ValueError("Number of generated texts must match number of reference texts")
    
    metrics = {}
    
    # BLEU scores
    bleu_scores = calculate_bleu_score(reference_texts, generated_texts)
    metrics.update(bleu_scores)
    
    # ROUGE scores
    rouge_scores = calculate_rouge_scores(reference_texts, generated_texts)
    metrics.update(rouge_scores)
    
    # BERTScore
    bert_scores = calculate_bert_score(reference_texts, generated_texts, model_type)
    metrics.update(bert_scores)
    
    # Content quality metrics
    quality_metrics = calculate_content_quality_metrics(generated_texts, reference_texts)
    metrics.update(quality_metrics)
    
    # Semantic similarity
    similarity_metrics = calculate_semantic_similarity(generated_texts, reference_texts)
    metrics.update(similarity_metrics)
    
    return metrics
