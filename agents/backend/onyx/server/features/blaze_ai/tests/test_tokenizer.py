from __future__ import annotations

class DummyTok:
    pad_token_id = None
    eos_token_id = 2
    padding_side = "right"


def test_prepare_tokenizer_sets_pad_and_side() -> None:
    from blaze_ai.utils.tokenizer import prepare_tokenizer

    t = DummyTok()
    t2 = prepare_tokenizer(t, padding_side="left")
    assert t2.pad_token_id == 2
    assert t2.padding_side == "left"


