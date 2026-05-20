from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch


def pad_length_to_multiple(length: int, multiple: int) -> int:
    if multiple <= 1:
        return length
    return int(math.ceil(length / multiple) * multiple)


def pad_to_multiple_2d(x: torch.Tensor, multiple: int, pad_value: int = 0) -> torch.Tensor:
    """Pad 2D tensor [batch, seq] on seq dim to nearest multiple."""
    if multiple <= 1:
        return x
    bsz, seqlen = x.shape
    target = pad_length_to_multiple(int(seqlen), int(multiple))
    if target == seqlen:
        return x
    pad_amt = target - seqlen
    pad_tensor = x.new_full((bsz, pad_amt), int(pad_value))
    return torch.cat([x, pad_tensor], dim=1)


def truncate_and_add_eos(ids: torch.Tensor, eos_id: int, max_length: Optional[int]) -> torch.Tensor:
    """Ensure EOS at end and truncate to max_length if set.

    ids: [seq]
    """
    seq = ids
    if max_length is not None and max_length > 0:
        seq = seq[: max_length - 1]
    if seq.numel() == 0 or int(seq[-1].item()) != int(eos_id):
        seq = torch.cat([seq, torch.tensor([int(eos_id)], dtype=seq.dtype, device=seq.device)])
    return seq


def create_attention_mask(input_ids: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
    """Create attention mask from input_ids [batch, seq]: 1 for tokens, 0 for pad."""
    return (input_ids != int(pad_token_id)).to(input_ids.dtype)


