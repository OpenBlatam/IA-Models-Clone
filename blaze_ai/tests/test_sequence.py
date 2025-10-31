from __future__ import annotations

import torch

from blaze_ai.utils.sequence import pad_length_to_multiple, pad_to_multiple_2d, truncate_and_add_eos, create_attention_mask


def test_pad_length_to_multiple() -> None:
    assert pad_length_to_multiple(10, 8) == 16
    assert pad_length_to_multiple(16, 8) == 16


def test_pad_to_multiple_2d() -> None:
    x = torch.ones(2, 10, dtype=torch.long)
    y = pad_to_multiple_2d(x, 8, pad_value=0)
    assert y.shape[1] == 16
    assert int(y[:, 10:].sum().item()) == 0


def test_truncate_and_add_eos_and_mask() -> None:
    ids = torch.tensor([1, 2, 3], dtype=torch.long)
    seq = truncate_and_add_eos(ids, eos_id=9, max_length=4)
    assert list(seq.tolist()) == [1, 2, 9]
    batch = torch.tensor([[1, 2, 0], [5, 0, 0]], dtype=torch.long)
    mask = create_attention_mask(batch, pad_token_id=0)
    assert mask.tolist() == [[1, 1, 0], [1, 0, 0]]


