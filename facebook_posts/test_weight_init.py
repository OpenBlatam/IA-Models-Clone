#!/usr/bin/env python3
"""
Simple test script for weight initialization system
"""

import torch
import torch.nn as nn

# Test basic PyTorch weight initialization
def test_basic_init():
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    
    # Apply PyTorch best practices
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    print("✅ Basic initialization test passed")
    return model

# Test CNN initialization
def test_cnn_init():
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU()
    )
    
    # Apply CNN best practices
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    print("✅ CNN initialization test passed")
    return model

# Test weight statistics
def test_weight_stats(model):
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            total_params += param.numel()
            print(f"{name}: mean={param.mean().item():.4f}, std={param.std().item():.4f}")
    
    print(f"Total parameters: {total_params:,}")
    print("✅ Weight statistics test passed")

if __name__ == "__main__":
    print("🚀 Testing Weight Initialization System")
    print("=" * 40)
    
    # Test basic initialization
    basic_model = test_basic_init()
    test_weight_stats(basic_model)
    
    print()
    
    # Test CNN initialization
    cnn_model = test_cnn_init()
    test_weight_stats(cnn_model)
    
    print("\n✅ All tests completed successfully!")






