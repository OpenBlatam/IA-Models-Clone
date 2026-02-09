from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from dataclasses import dataclass
import json
import re
from collections import Counter
import math

            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            from rouge_score import rouge_scorer
            from nltk.translate.meteor_score import meteor_score
            from nltk.tokenize import word_tokenize
            from pycocoevalcap.cider.cider import Cider
            from bert_score import score
            from bleurt import score
        from sklearn.metrics import (
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
            from scipy.stats import pearsonr
from typing import Any, List, Dict, Optional
import asyncio
logger = logging.getLogger(__name__)


@dataclass
class MetricsConfig:
  Configuration for task-specific metrics.    task_type: str = "caption_generation"  # caption_generation, text_classification, sentiment_analysis
    use_bleu: bool = True
    use_rouge: bool = True
    use_meteor: bool = True
    use_cider: bool = true
    use_bertscore: bool = True
    use_bleurt: bool = False
    max_length: int =512
    language: str =en"


class CaptionGenerationMetrics:
  prehensive metrics for caption generation tasks."   
    def __init__(self, config: MetricsConfig):
        
    """__init__ function."""
self.config = config
        self.metrics = {}
    
    def compute_all_metrics(self, generated_captions: List[str], reference_captions: Liststr]) -> Dict[str, float]:
    ompute all caption generation metrics."
        metrics = {}
        
        # Basic text metrics
        metrics.update(self._compute_basic_metrics(generated_captions, reference_captions))
        
        # BLEU Score
        if self.config.use_bleu:
            metrics['bleu] = self._compute_bleu_score(generated_captions, reference_captions)
        
        # ROUGE Score
        if self.config.use_rouge:
            metrics.update(self._compute_rouge_scores(generated_captions, reference_captions))
        
        # METEOR Score
        if self.config.use_meteor:
            metrics[meteor] = self._compute_meteor_score(generated_captions, reference_captions)
        
        # CIDEr Score
        if self.config.use_cider:
            metrics['cider] = self._compute_cider_score(generated_captions, reference_captions)
        
        # BERTScore
        if self.config.use_bertscore:
            metrics['bertscore] = self._compute_bertscore(generated_captions, reference_captions)
        
        # BLEURT Score
        if self.config.use_bleurt:
            metrics[bleurt] = self._compute_bleurt_score(generated_captions, reference_captions)
        
        self.metrics = metrics
        return metrics
    
    def _compute_basic_metrics(self, generated: List[str], references: Liststr]) -> Dict[str, float]:
      Compute basic text metrics."
        metrics = {}
        
        # Length statistics
        gen_lengths = [len(caption.split()) for caption in generated]
        ref_lengths = [len(ref.split()) for ref in references]
        
        metrics['avg_generated_length'] = np.mean(gen_lengths)
        metrics['std_generated_length'] = np.std(gen_lengths)
        metrics['avg_reference_length'] = np.mean(ref_lengths)
        metrics['length_ratio'] = np.mean(gen_lengths) / np.mean(ref_lengths)
        
        # Vocabulary statistics
        gen_vocab = set()
        ref_vocab = set()
        
        for caption in generated:
            gen_vocab.update(caption.lower().split())
        for ref in references:
            ref_vocab.update(ref.lower().split())
        
        metrics['generated_vocab_size] = len(gen_vocab)
        metrics['reference_vocab_size] = len(ref_vocab)
        metrics[vocab_overlap] = len(gen_vocab.intersection(ref_vocab)) / len(gen_vocab.union(ref_vocab))
        
        # Repetition metrics
        metrics['repetition_ratio]= self._compute_repetition_ratio(generated)
        
        return metrics
    
    def _compute_bleu_score(self, generated: List[str], references: List[str]) -> float:
     Compute BLEU score."""
        try:
            smoothie = SmoothingFunction().method1
            
            scores =            for gen, ref in zip(generated, references):
                gen_tokens = gen.split()
                ref_tokens = ref.split()
                score = sentence_bleu([ref_tokens], gen_tokens, smoothing_function=smoothie)
                scores.append(score)
            
            return np.mean(scores)
        except ImportError:
            logger.warning("NLTK not available for BLEU calculation")
            return 0 
    def _compute_rouge_scores(self, generated: List[str], references: Liststr]) -> Dict[str, float]:
      mpute ROUGE scores."""
        try:
            scorer = rouge_scorer.RougeScorer(['rouge1,rouge2rougeL'], use_stemmer=True)
            
            rouge1_scores = []
            rouge2_scores = []
            rougeL_scores = []
            
            for gen, ref in zip(generated, references):
                score = scorer.score(ref, gen)
                rouge1_scores.append(score[rouge1'].fmeasure)
                rouge2_scores.append(score[rouge2'].fmeasure)
                rougeL_scores.append(score[rougeL'].fmeasure)
            
            return[object Object]
                rouge1: np.mean(rouge1_scores),
                rouge2: np.mean(rouge2_scores),
                rougeL': np.mean(rougeL_scores)
            }
        except ImportError:
            logger.warning("rouge-score not available for ROUGE calculation")
            return {'rouge1: 00, 'rouge2: 00ougeL':00 
    def _compute_meteor_score(self, generated: List[str], references: List[str]) -> float:
       mpute METEOR score."""
        try:
            
            scores =            for gen, ref in zip(generated, references):
                gen_tokens = word_tokenize(gen.lower())
                ref_tokens = word_tokenize(ref.lower())
                score = meteor_score([ref_tokens], gen_tokens)
                scores.append(score)
            
            return np.mean(scores)
        except ImportError:
            logger.warning("NLTK not available for METEOR calculation")
            return 0 
    def _compute_cider_score(self, generated: List[str], references: List[str]) -> float:
      ompute CIDEr score."""
        try:
            
            cider_scorer = Cider()
            scores = cider_scorer.compute_score(references, generated)
            return scores0     except ImportError:
            logger.warning(pycocoevalcap not available for CIDEr calculation")
            return 0 
    def _compute_bertscore(self, generated: List[str], references: List[str]) -> float:
     Compute BERTScore."""
        try:
            
            P, R, F1 = score(generated, references, lang=self.config.language, verbose=True)
            return F1.mean().item()
        except ImportError:
            logger.warning("bert-score not available for BERTScore calculation")
            return 0 
    def _compute_bleurt_score(self, generated: List[str], references: List[str]) -> float:
       mpute BLEURT score."""
        try:
            
            checkpoint = bleurt-base-128"
            scorer = score.BleurtScorer(checkpoint)
            scores = scorer.score(references=references, candidates=generated)
            return np.mean(scores)
        except ImportError:
            logger.warning("bleurt not available for BLEURT calculation")
            return 0 
    def _compute_repetition_ratio(self, captions: List[str]) -> float:
Compute repetition ratio in generated captions."""
        repetition_ratios = []
        
        for caption in captions:
            words = caption.split()
            if len(words) < 2        repetition_ratios.append(0.0          continue
            
            word_counts = Counter(words)
            repeated_words = sum(count - 1 for count in word_counts.values())
            repetition_ratios.append(repeated_words / len(words))
        
        return np.mean(repetition_ratios)


class TextClassificationMetrics:
ext classification tasks."   
    def __init__(self, config: MetricsConfig):
        
    """__init__ function."""
self.config = config
        self.metrics = {}
    
    def compute_all_metrics(self, predictions: List[int], targets: List[int], 
                           class_names: OptionalList[str]] = None) -> Dict[str, float]:
    ext classification metrics."""
            accuracy_score, precision_recall_fscore_support,
            confusion_matrix, classification_report, roc_auc_score
        )
        
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(targets, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(targets, predictions, average='weighted)
        metrics['precision'] = precision
        metrics[recall'] = recall
        metrics['f1_score'] = f1        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            targets, predictions, average=None
        )
        
        if class_names:
            for i, class_name in enumerate(class_names):
                metrics[f'precision_{class_name}] = precision_per_class[i]
                metrics[f'recall_{class_name}'] = recall_per_class[i]
                metrics[ff1[object Object]class_name}'] = f1_per_class[i]
        
        # Confusion matrix
        cm = confusion_matrix(targets, predictions)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Classification report
        if class_names:
            report = classification_report(targets, predictions, target_names=class_names, output_dict=True)
            metrics['classification_report'] = report
        
        self.metrics = metrics
        return metrics


class SentimentAnalysisMetrics:
rics for sentiment analysis tasks."   
    def __init__(self, config: MetricsConfig):
        
    """__init__ function."""
self.config = config
        self.metrics = {}
    
    def compute_all_metrics(self, predictions: List[int], targets: List[int],
                           sentiment_scores: Optional[List[float]] = None) -> Dict[str, float]:
Computesentiment analysis specific metrics."""
        
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(targets, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(targets, predictions, average='weighted)
        metrics['precision'] = precision
        metrics[recall'] = recall
        metrics['f1_score'] = f1        
        # Sentiment-specific metrics
        if sentiment_scores:
            metrics['sentiment_correlation] = self._compute_sentiment_correlation(predictions, sentiment_scores)
            metrics['sentiment_consistency] = self._compute_sentiment_consistency(predictions, targets)
        
        # Polarity analysis
        metrics.update(self._compute_polarity_metrics(predictions, targets))
        
        self.metrics = metrics
        return metrics
    
    def _compute_sentiment_correlation(self, predictions: List[int], sentiment_scores: List[float]) -> float:
ompute correlation between predictions and sentiment scores."""
        try:
            correlation, _ = pearsonr(predictions, sentiment_scores)
            return correlation
        except ImportError:
            logger.warning("scipy not available for correlation calculation")
            return 0 
    def _compute_sentiment_consistency(self, predictions: List[int], targets: List[int]) -> float:
Compute sentiment consistency."
        correct_predictions = sum(1 for p, t in zip(predictions, targets) if p == t)
        return correct_predictions / len(predictions)
    
    def _compute_polarity_metrics(self, predictions: List[int], targets: Listint]) -> Dict[str, float]:
 polarity-specific metrics."
        metrics = {}
        
        # Positive sentiment metrics
        positive_pred = sum(1 for p in predictions if p > 0)
        positive_true = sum(1or t in targets if t > 0)
        
        metrics['positive_ratio_predicted] = positive_pred / len(predictions)
        metrics['positive_ratio_actual] = positive_true / len(targets)
        
        # Negative sentiment metrics
        negative_pred = sum(1 for p in predictions if p < 0)
        negative_true = sum(1or t in targets if t < 0)
        
        metrics['negative_ratio_predicted] = negative_pred / len(predictions)
        metrics['negative_ratio_actual] = negative_true / len(targets)
        
        return metrics


class InstagramCaptionMetrics:
pecialized metrics for Instagram caption generation."   
    def __init__(self, config: MetricsConfig):
        
    """__init__ function."""
self.config = config
        self.metrics = {}
    
    def compute_instagram_metrics(self, generated_captions: List[str], reference_captions: Liststr]) -> Dict[str, float]:
ComputeInstagram-specific caption metrics."
        metrics = {}
        
        # Basic caption metrics
        caption_metrics = CaptionGenerationMetrics(config)
        metrics.update(caption_metrics.compute_all_metrics(generated_captions, reference_captions))
        
        # Instagram-specific metrics
        metrics.update(self._compute_engagement_metrics(generated_captions))
        metrics.update(self._compute_hashtag_metrics(generated_captions))
        metrics.update(self._compute_emoji_metrics(generated_captions))
        metrics.update(self._compute_call_to_action_metrics(generated_captions))
        
        self.metrics = metrics
        return metrics
    
    def _compute_engagement_metrics(self, captions: Liststr]) -> Dict[str, float]:
Compute engagement-related metrics."
        metrics = {}
        
        # Question marks (encourages comments)
        question_counts =caption.count('?') for caption in captions]
        metrics[avg_questions_per_caption'] = np.mean(question_counts)
        metrics['captions_with_questions'] = sum(1 for count in question_counts if count >0) / len(captions)
        
        # Exclamation marks (shows enthusiasm)
        exclamation_counts =caption.count('!') for caption in captions]
        metrics['avg_exclamations_per_caption'] = np.mean(exclamation_counts)
        metrics['captions_with_exclamations'] = sum(1 for count in exclamation_counts if count >0) / len(captions)
        
        # Mentions (@username)
        mention_counts = len(re.findall(r'@\w+', caption)) for caption in captions]
        metrics['avg_mentions_per_caption'] = np.mean(mention_counts)
        metrics['captions_with_mentions'] = sum(1 for count in mention_counts if count >0) / len(captions)
        
        return metrics
    
    def _compute_hashtag_metrics(self, captions: Liststr]) -> Dict[str, float]:
        te hashtag-related metrics."
        metrics = {}
        
        # Hashtag counts
        hashtag_counts = len(re.findall(r'#\w+', caption)) for caption in captions]
        metrics['avg_hashtags_per_caption] = np.mean(hashtag_counts)
        metrics[captions_with_hashtags'] = sum(1 for count in hashtag_counts if count >0) / len(captions)
        
        # Hashtag density
        total_words = sum(len(caption.split()) for caption in captions)
        total_hashtags = sum(hashtag_counts)
        metrics[hashtag_density'] = total_hashtags / total_words if total_words > 00        
        # Hashtag variety
        all_hashtags = []
        for caption in captions:
            all_hashtags.extend(re.findall(r#w+ caption))
        metrics['unique_hashtag_ratio'] = len(set(all_hashtags)) / len(all_hashtags) if all_hashtags else 0   
        return metrics
    
    def _compute_emoji_metrics(self, captions: Liststr]) -> Dict[str, float]:
      pute emoji-related metrics."
        metrics = {}
        
        # Emoji counts (basic emoji detection)
        emoji_pattern = re.compile(r'[^\w\s,.-]')
        emoji_counts = [len(emoji_pattern.findall(caption)) for caption in captions]
        metrics['avg_emojis_per_caption] = np.mean(emoji_counts)
        metrics['captions_with_emojis'] = sum(1 for count in emoji_counts if count >0) / len(captions)
        
        return metrics
    
    def _compute_call_to_action_metrics(self, captions: Liststr]) -> Dict[str, float]:
        ute call-to-action metrics."
        metrics = {}
        
        # Common CTA phrases
        cta_phrases = [
            r'like\s+if', r'comment\s+if', r'share\s+if', r'follow\s+if,          r'double\s+tap, rsave\s+this, rtag\s+someone', r'dm\s+me',
            rlink\s+in\s+bio', r'swipe\s+left, r'swipe\s+right'
        ]
        
        cta_counts = []
        for caption in captions:
            caption_lower = caption.lower()
            cta_count = sum(1 for phrase in cta_phrases if re.search(phrase, caption_lower))
            cta_counts.append(cta_count)
        
        metrics['avg_cta_per_caption'] = np.mean(cta_counts)
        metricscaptions_with_cta'] = sum(1ount in cta_counts if count >0) / len(captions)
        
        return metrics


class MetricsAggregator:
  regate and compare multiple metric sets."   
    def __init__(self) -> Any:
        self.metric_sets = {}
    
    def add_metric_set(self, name: str, metrics: Dict[str, float]):
     
    """add_metric_set function."""
d a set of metrics.       self.metric_sets[name] = metrics
    
    def compare_models(self) -> Dict[str, Any]:
Compare different models based on their metrics.       if len(self.metric_sets) < 2:
            return {"error: "Need at least 2 metric sets to compare"}
        
        comparison = {}
        
        # Get all unique metric names
        all_metrics = set()
        for metrics in self.metric_sets.values():
            all_metrics.update(metrics.keys())
        
        # Compare each metric
        for metric in all_metrics:
            values = [object Object]          for name, metrics in self.metric_sets.items():
                values[name] = metrics.get(metric, 0      
            comparison[metric] =[object Object]
                valuess,
               best_model': max(values.items(), key=lambda x: x[1])[0],
                worst_model': min(values.items(), key=lambda x: x[1])[0],
          range': max(values.values()) - min(values.values()),
             mean': np.mean(list(values.values())),
                stdnp.std(list(values.values()))
            }
        
        return comparison
    
    def generate_report(self, output_path: str =./metrics_comparison.json"):
   
    """generate_report function."""
te a comprehensive metrics report.       report = {
        metric_sets': self.metric_sets,
            comparison': self.compare_models(),
           summary': self._generate_summary()
        }
        
        with open(output_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(report, f, indent=2)
        
        return report
    
    def _generate_summary(self) -> Dict[str, Any]:
   Generate a summary of all metrics."
        summary = {}
        
        for metric_set_name, metrics in self.metric_sets.items():
            summary[metric_set_name] =[object Object]
                num_metrics': len(metrics),
             avg_score': np.mean(list(metrics.values())),
                best_metric': max(metrics.items(), key=lambda x: x[1]),
             worst_metric': min(metrics.items(), key=lambda x: x[1       }
        
        return summary


# Example usage functions
def evaluate_caption_generation(
    generated_captions: List[str],
    reference_captions: List[str],
    task_type: str = "instagram"
) -> Dict[str, float]:
    "caption generation with task-specific metrics."
    
    config = MetricsConfig(task_type=task_type)
    
    if task_type == "instagram":
        evaluator = InstagramCaptionMetrics(config)
        return evaluator.compute_instagram_metrics(generated_captions, reference_captions)
    else:
        evaluator = CaptionGenerationMetrics(config)
        return evaluator.compute_all_metrics(generated_captions, reference_captions)


def evaluate_text_classification(
    predictions: Listint],
    targets: List[int],
    class_names: OptionalList[str]] = None
) -> Dict[str, float]:
  ext classification with appropriate metrics."
    
    config = MetricsConfig(task_type="text_classification)  evaluator = TextClassificationMetrics(config)
    return evaluator.compute_all_metrics(predictions, targets, class_names)


def evaluate_sentiment_analysis(
    predictions: Listint],
    targets: List[int],
    sentiment_scores: Optional[List[float]] = None
) -> Dict[str, float]:
    """Evaluate sentiment analysis with specialized metrics."
    
    config = MetricsConfig(task_type="sentiment_analysis)  evaluator = SentimentAnalysisMetrics(config)
    return evaluator.compute_all_metrics(predictions, targets, sentiment_scores)


def create_comprehensive_evaluation_report(
    model_results: Dict[str, Dict[str, Any]],
    output_path: str = ./comprehensive_evaluation.json"
) -> Dict[str, Any]:
    """Create a comprehensive evaluation report for multiple models."   
    aggregator = MetricsAggregator()
    
    for model_name, results in model_results.items():
        if 'metrics' in results:
            aggregator.add_metric_set(model_name, results['metrics'])
    
    report = aggregator.generate_report(output_path)
    return report 