from __future__ import annotations

import torch

from chronorisk.reckoning.timepe import TimeAwarePositional


def test_elapsed_time_changes_encoding() -> None:
    torch.manual_seed(0)
    module = TimeAwarePositional(ehr_features=6, dim=12)
    ehr = torch.zeros(1, 4, 6)
    short = torch.tensor([[0.0, 0.1, 0.1, 0.1]])
    long = torch.tensor([[0.0, 2.0, 2.0, 2.0]])
    encoded_short = module(ehr, short)
    encoded_long = module(ehr, long)
    difference = (encoded_short - encoded_long).abs().mean().item()
    assert difference > 1e-3


def test_zero_delta_is_constant_position() -> None:
    torch.manual_seed(1)
    module = TimeAwarePositional(ehr_features=6, dim=12)
    ehr = torch.zeros(1, 3, 6)
    delta = torch.zeros(1, 3)
    encoded = module(ehr, delta)
    first = encoded[0, 0]
    last = encoded[0, -1]
    assert torch.allclose(first, last, atol=1e-5)
