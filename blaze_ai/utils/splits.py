from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import random


def split_train_val_test(
    labels: Sequence[int],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    stratify: bool = True,
) -> Tuple[List[int], List[int], List[int]]:
    n = len(labels)
    assert 0.0 < train_ratio < 1.0 and 0.0 <= val_ratio < 1.0
    assert train_ratio + val_ratio < 1.0
    indices = list(range(n))
    rng = random.Random(seed)

    if stratify:
        try:
            from sklearn.model_selection import train_test_split  # type: ignore
            train_idx, tmp_idx = train_test_split(
                indices, test_size=(1.0 - train_ratio), random_state=seed, stratify=labels
            )
            # Compute relative val ratio among remaining set
            remaining_labels = [labels[i] for i in tmp_idx]
            val_size = val_ratio / (1.0 - train_ratio)
            val_idx, test_idx = train_test_split(
                tmp_idx, test_size=(1.0 - val_size), random_state=seed, stratify=remaining_labels
            )
            return list(train_idx), list(val_idx), list(test_idx)
        except Exception:
            pass

    # fallback: shuffle and split without stratification
    rng.shuffle(indices)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]
    return train_idx, val_idx, test_idx


def stratified_kfold_indices(
    labels: Sequence[int],
    num_folds: int = 5,
    seed: int = 42,
) -> List[Tuple[List[int], List[int]]]:
    try:
        from sklearn.model_selection import StratifiedKFold  # type: ignore
        import numpy as np  # type: ignore

        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        X = np.arange(len(labels))
        y = np.array(labels)
        folds = []
        for train_idx, val_idx in skf.split(X, y):
            folds.append((train_idx.tolist(), val_idx.tolist()))
        return folds
    except Exception:
        # simple round-robin fallback (not strictly stratified)
        label_to_indices: dict[int, List[int]] = {}
        for i, y in enumerate(labels):
            label_to_indices.setdefault(int(y), []).append(i)
        rng = random.Random(seed)
        for lst in label_to_indices.values():
            rng.shuffle(lst)
        buckets: List[List[int]] = [[] for _ in range(num_folds)]
        for cls_indices in label_to_indices.values():
            for i, idx in enumerate(cls_indices):
                buckets[i % num_folds].append(idx)
        all_indices = set(range(len(labels)))
        folds: List[Tuple[List[int], List[int]]] = []
        for k in range(num_folds):
            val_idx = buckets[k]
            train_idx = list(all_indices.difference(val_idx))
            folds.append((train_idx, val_idx))
        return folds


