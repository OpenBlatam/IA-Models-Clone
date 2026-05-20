"""
Tests for device utilities.
"""

import pytest
import torch
from audio_separator.utils.device_utils import (
    get_device,
    get_device_info,
    move_to_device
)


def test_get_device():
    """Test getting device."""
    device = get_device("auto")
    assert isinstance(device, torch.device)
    
    device = get_device("cpu")
    assert device.type == "cpu"


def test_get_device_info():
    """Test getting device info."""
    info = get_device_info()
    assert "cpu" in info
    assert "cuda" in info
    assert isinstance(info["cuda"], bool)
    assert isinstance(info["cpu"], bool)


def test_move_to_device():
    """Test moving tensor to device."""
    tensor = torch.tensor([1, 2, 3])
    device = get_device("cpu")
    moved = move_to_device(tensor, device)
    assert moved.device.type == "cpu"

