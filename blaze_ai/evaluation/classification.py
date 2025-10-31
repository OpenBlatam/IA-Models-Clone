from __future__ import annotations

from typing import Dict

import torch
from torch.utils.data import DataLoader

from ..utils.metrics import classification_metrics


@torch.inference_mode()
def evaluate_classification(model, data_loader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    for batch in data_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["input_ids"], batch.get("attention_mask"))
        loss = criterion(logits, batch["labels"])
        total_loss += float(loss.item()) * batch["labels"].size(0)
        all_logits.append(logits)
        all_labels.append(batch["labels"])
    logits_cat = torch.cat(all_logits, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    avg_loss = total_loss / max(1, labels_cat.numel())
    cls_metrics = classification_metrics(logits_cat, labels_cat, average="macro")
    return {"val_loss": avg_loss, **cls_metrics}


