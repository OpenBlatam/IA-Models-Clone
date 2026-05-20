from __future__ import annotations

import torch

from blaze_ai.models.ddpm import DDPM, DDPMSchedule, DDPMScheduleConfig


def test_forward_diffusion_increases_noise() -> None:
    torch.manual_seed(0)
    schedule = DDPMSchedule(DDPMScheduleConfig(timesteps=100))
    model = DDPM(schedule=schedule)
    x0 = torch.randn(2, 3, 16, 16).clamp(-1, 1)
    t = torch.full((2,), 80, dtype=torch.long)
    xt = model.q_sample(x0, t)
    diff = (xt - x0).abs().mean().item()
    assert diff > 0.05


def test_reverse_step_shapes_and_finite() -> None:
    torch.manual_seed(0)
    schedule = DDPMSchedule(DDPMScheduleConfig(timesteps=10))
    model = DDPM(schedule=schedule)
    x = torch.randn(2, 3, 8, 8)
    t = torch.full((2,), 5, dtype=torch.long)
    x_prev = model.p_sample(x, t)
    assert x_prev.shape == x.shape
    assert torch.isfinite(x_prev).all()


def test_sampling_range_clamped() -> None:
    torch.manual_seed(0)
    schedule = DDPMSchedule(DDPMScheduleConfig(timesteps=10))
    model = DDPM(schedule=schedule)
    imgs = model.sample((1, 3, 8, 8))
    assert imgs.min().item() >= -1.0 - 1e-6
    assert imgs.max().item() <= 1.0 + 1e-6


