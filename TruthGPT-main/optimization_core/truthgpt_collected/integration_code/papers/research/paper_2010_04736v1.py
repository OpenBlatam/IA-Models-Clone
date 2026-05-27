import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Any, Callable
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np

class Paper_2010_04736v1Config:
    """
    Configuration for the rationale evaluation module.
    
    Attributes:
        model_name_or_path: HuggingFace model identifier for the black-box classifier.
        device: 'cuda' or 'cpu'.
        pad_token_id: Token ID used for padding (auto-deduced from tokenizer if None).
        mask_token_id: Token ID to use when masking input (usually [MASK]).
        max_length: Maximum sequence length for tokenization.
        batch_size: Batch size for inference.
        use_human_gold: Whether to include human rationale overlap metrics.
        metrics: List of metrics to compute.
                 Supported: 'comprehensiveness', 'sufficiency', 'human_overlap_f1', 'human_overlap_auc'.
        overlap_threshold: IoU threshold to consider a token as part of a rationale span.
    """
    def __init__(self,
                 model_name_or_path: str = "bert-base-uncased",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 pad_token_id: Optional[int] = None,
                 mask_token_id: Optional[int] = None,
                 max_length: int = 512,
                 batch_size: int = 16,
                 use_human_gold: bool = True,
                 metrics: Optional[List[str]] = None,
                 overlap_threshold: float = 0.5):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id if mask_token_id is not None else 103  # BERT [MASK]
        self.max_length = max_length
        self.batch_size = batch_size
        self.use_human_gold = use_human_gold
        self.metrics = metrics if metrics is not None else ['comprehensiveness', 'sufficiency', 'human_overlap_f1']
        self.overlap_threshold = overlap_threshold


class Paper_2010_04736v1Module(nn.Module):
    """
    PyTorch module implementing rationale evaluation metrics as described in
    "Evaluating and Characterizing Human Rationales" (cs.CL).

    The module supports two families of metrics:
    1. **Automated behavior-based metrics**: comprehensiveness (how much rationales affect prediction)
       and sufficiency (how much rationales alone preserve prediction).
    2. **Human-gold comparison metrics**: token-level overlap (F1, AUC) between generated rationales
       and human annotations.

    The module is designed as a standalone evaluation wrapper that takes a black-box classifier
    (any HuggingFace model) and computes the requested metrics on a dataset.

    Outputs a dictionary of scalar scores for each metric.
    """
    def __init__(self, config: Paper_2010_04736v1Config):
        super().__init__()
        self.config = config

        # Load the black-box model and tokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained(config.model_name_or_path).to(config.device)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
        self.model.eval()

        # Auto-detect pad token id if not provided
        if config.pad_token_id is None:
            config.pad_token_id = self.tokenizer.pad_token_id
        self.pad_token_id = config.pad_token_id

        # Ensure pad token is set
        if self.pad_token_id is None:
            self.pad_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0

        self.mask_token_id = config.mask_token_id if config.mask_token_id is not None else self.tokenizer.mask_token_id
        self.max_length = config.max_length
        self.device = config.device
        self.batch_size = config.batch_size

    @torch.no_grad()
    def forward(self,
                texts: List[str],
                predicted_masks: List[torch.Tensor],
                human_masks: Optional[List[torch.Tensor]] = None) -> Dict[str, float]:
        """
        Compute rationale evaluation metrics for a batch of inputs.

        Args:
            texts: List of input texts.
            predicted_masks: List of binary masks (0/1) of length equal to the number of tokens after tokenization.
                             Each mask indicates which tokens are selected as the rationale.
            human_masks: Optional list of ground-truth human rationale masks (same shape as predicted_masks).

        Returns:
            Dictionary mapping metric names to scalar values (averaged over inputs).
        """
        # Tokenize all texts in one go (list of dicts)
        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_token_type_ids=False,  # not needed for BERT-based models
            return_attention_mask=True
        )
        input_ids = tokenized["input_ids"].to(self.device)          # (B, L)
        attention_mask = tokenized["attention_mask"].to(self.device) # (B, L)
        batch_size, seq_len = input_ids.shape

        # Sanity: predicted_masks must be list of tensors
        if len(predicted_masks) != batch_size:
            raise ValueError(f"Number of predicted_masks ({len(predicted_masks)}) must equal batch size ({batch_size})")
        for i, m in enumerate(predicted_masks):
            if m.numel() != seq_len:
                raise ValueError(f"predicted_masks[{i}] has length {m.numel()} but sequence length is {seq_len}")

        # Stack masks and move to device
        pred_masks = torch.stack(predicted_masks).to(self.device)   # (B, L)

        # Compute original model predictions (logits, probabilities)
        # Using softmax to get probabilities for the predicted label
        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits                                   # (B, num_classes)
        probs = F.softmax(logits, dim=-1)                         # (B, num_classes)
        pred_labels = logits.argmax(dim=-1)                       # (B,)

        # Gather probability of the predicted label for each sample
        original_confidence = probs.gather(1, pred_labels.unsqueeze(1)).squeeze(1)  # (B,)

        # Prepare results dict
        results = {}
        if 'comprehensiveness' in self.config.metrics:
            # Comprehensiveness: drop (zero out) the rationales and measure confidence drop
            # We replace rationale tokens with [PAD] (non-informative)
            comp_mask = (1 - pred_masks).bool()  # mask out rationales: keep non-rationale
            # New input: keep non-rationale tokens, replace rationales with pad
            comp_input_ids = input_ids.clone()
            # For positions where pred_masks==1 (rationale), set to pad_token_id
            comp_input_ids[pred_masks.bool()] = self.pad_token_id
            # Must also adjust attention_mask: set attention to zero for rationales (since they are now pad)
            comp_attention_mask = attention_mask.clone()
            comp_attention_mask[pred_masks.bool()] = 0
            # Forward
            comp_outputs = self.model(comp_input_ids, attention_mask=comp_attention_mask)
            comp_logits = comp_outputs.logits
            comp_probs = F.softmax(comp_logits, dim=-1)
            comp_confidence = comp_probs.gather(1, pred_labels.unsqueeze(1)).squeeze(1)
            # Comprehensiveness = original_confidence - comp_confidence (positive = rationales are important)
            comprehensiveness = (original_confidence - comp_confidence).mean().item()
            results['comprehensiveness'] = comprehensiveness

        if 'sufficiency' in self.config.metrics:
            # Sufficiency: keep only rationale tokens, rest become [PAD] (or [MASK])
            suff_mask = pred_masks.bool()
            # New input: keep rationale tokens, replace non-rationale with [PAD] (mask token is also fine)
            suff_input_ids = input_ids.clone()
            suff_input_ids[~suff_mask] = self.pad_token_id
            # Attention mask: only attend to rationale tokens
            suff_attention_mask = attention_mask.clone()
            suff_attention_mask[~suff_mask] = 0
            # Forward
            suff_outputs = self.model(suff_input_ids, attention_mask=suff_attention_mask)
            suff_logits = suff_outputs.logits
            suff_probs = F.softmax(suff_logits, dim=-1)
            suff_confidence = suff_probs.gather(1, pred_labels.unsqueeze(1)).squeeze(1)
            # Sufficiency = suff_confidence (higher is better: rationale alone is enough)
            sufficiency = suff_confidence.mean().item()
            results['sufficiency'] = sufficiency

        # Human overlap metrics
        if human_masks is not None and self.config.use_human_gold:
            if len(human_masks) != batch_size:
                raise ValueError(f"Number of human_masks ({len(human_masks)}) must equal batch size ({batch_size})")
            human = torch.stack(human_masks).to(self.device)        # (B, L)
            # Overlap metrics: F1, AUC (if we treat masks as binary decisions)
            if 'human_overlap_f1' in self.config.metrics:
                # Compute token-level precision, recall, F1
                pred = pred_masks.bool()
                true = human.bool()
                # Intersection, union, etc.
                intersection = (pred & true).sum(dim=1).float()  # (B,)
                pred_sum = pred.sum(dim=1).float()
                true_sum = true.sum(dim=1).float()
                # Avoid division by zero
                precision = torch.where(pred_sum > 0, intersection / pred_sum, torch.zeros_like(intersection))
                recall = torch.where(true_sum > 0, intersection / true_sum, torch.zeros_like(intersection))
                f1 = torch.where((precision + recall) > 0,
                                 2 * precision * recall / (precision + recall),
                                 torch.zeros_like(precision))
                mean_f1 = f1.mean().item()
                results['human_overlap_f1'] = mean_f1

            if 'human_overlap_auc' in self.config.metrics:
                # For AUC we need continuous scores. We assume the predicted mask can be a confidence score.
                # If not continuous, we threshold.
                if predicted_masks[0].dtype == torch.float:
                    # Use predicted scores directly
                    pred_scores = pred_masks  # (B, L)
                else:
                    # Convert binary mask to scores: can treat as 0/1 probabilities
                    pred_scores = pred_masks.float()
                # Compute AUC using ROC curve (simplified: average precision across thresholds)
                # We'll compute average precision (area under precision-recall curve) as proxy
                # Flatten over batch
                all_pred = pred_scores.flatten()
                all_true = human.bool().flatten()
                # Sort by descending prediction score
                sorted_indices = torch.argsort(all_pred, descending=True)
                sorted_true = all_true[sorted_indices]
                # Compute cumulative precision and recall
                cum_true = torch.cumsum(sorted_true.float(), dim=0)
                total_pos = cum_true[-1].item() if cum_true.numel() > 0 else 1.0
                if total_pos == 0:
                    auc = 0.0
                else:
                    # Precision at each rank
                    precisions = cum_true / torch.arange(1, cum_true.shape[0]+1, dtype=torch.float, device=cum_true.device)
                    # Average precision (AP)
                    ap = (precisions * sorted_true.float()).sum() / total_pos
                    auc = ap.item()
                results['human_overlap_auc'] = auc

        # If no metrics computed, return empty dict
        return results

    @torch.no_grad()
    def evaluate(self, dataset: List[Dict]) -> Dict[str, float]:
        """
        Convenience method to evaluate over full dataset. 
        Each entry is dict with 'text', 'predicted_mask', optionally 'human_mask'.
        Returns aggregated metrics.
        """
        all_metrics = []
        for i in range(0, len(dataset), self.batch_size):
            batch = dataset[i:i+self.batch_size]
            texts = [b['text'] for b in batch]
            pred_masks = [torch.tensor(b['predicted_mask'], dtype=torch.long) for b in batch]
            human_masks = None
            if self.config.use_human_gold and 'human_mask' in batch[0]:
                human_masks = [torch.tensor(b['human_mask'], dtype=torch.long) for b in batch]
            metrics = self.forward(texts, pred_masks, human_masks)
            all_metrics.append(metrics)

        # Aggregate (average) across batches
        aggregated = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            aggregated[key] = np.mean(values)
        return aggregated


if __name__ == "__main__":
    # Quick demonstration of the module
    print("Paper_2010_04736v1Module: Evaluating and Characterizing Human Rationales")
    print("=" * 70)

    # Configuration
    config = Paper_2010_04736v1Config(
        model_name_or_path="bert-base-uncased",
        device="cpu",  # change to cuda if available
        metrics=['comprehensiveness', 'sufficiency', 'human_overlap_f1', 'human_overlap_auc'],
        use_human_gold=True
    )

    # Initialize module (this downloads BERT model)
    module = Paper_2010_04736v1Module(config)

    # Example data: two sentences
    texts = [
        "The movie was not good, but the acting was superb.",
        "I loved the special effects and the story was engaging."
    ]

    # Tokenize to get length
    tokenized = module.tokenizer(texts, padding=True, truncation=True, max_length=128, return_length=True)
    lengths = tokenized['length']  # actual token lengths (including [CLS] and [SEP])

    # Create dummy predicted masks: keep only the first two tokens after [CLS] as rationales
    predicted_masks = []
    for length in lengths:
        mask = torch.zeros(length, dtype=torch.long)
        # mark tokens 1..2 (excluding [CLS]) as rationales
        if length > 2:
            mask[1:3] = 1
        predicted_masks.append(mask)

    # Create dummy human masks (same pattern for demonstration)
    human_masks = []
    for length in lengths:
        mask = torch.zeros(length, dtype=torch.long)
        if length > 2:
            mask[1:4] = 1   # human highlights first three tokens
        human_masks.append(mask)

    # Compute evaluation metrics
    metrics = module.forward(texts, predicted_masks, human_masks)
    print("Results (single batch):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Full dataset evaluation (simulate)
    dataset = [
        {"text": t, "predicted_mask": pm.tolist(), "human_mask": hm.tolist()}
        for t, pm, hm in zip(texts, predicted_masks, human_masks)
    ]
    aggregated = module.evaluate(dataset)
    print("\nAggregated over dataset:")
    for k, v in aggregated.items():
        print(f"  {k}: {v:.4f}")