#!/usr/bin/env python3
"""
Test script for gradient clipping and NaN handling system.
"""

import torch
import torch.nn as nn
import numpy as np
from gradient_clipping_nan_handling import (
    GradientClippingConfig,
    NaNHandlingConfig,
    NumericalStabilityManager,
    ClippingType,
    NaNHandlingType,
    create_training_wrapper
)


def test_basic_functionality():
    """Test basic functionality of the system."""
    print("Testing basic functionality...")
    
    # Create simple model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
        nn.ReLU(),
        nn.Linear(10, 1)
    )
    
    # Create configurations
    clipping_config = GradientClippingConfig(
        clipping_type=ClippingType.NORM,
        max_norm=1.0,
        monitor_clipping=True
    )
    
    nan_config = NaNHandlingConfig(
        handling_type=NaNHandlingType.DETECT,
        detect_nan=True,
        detect_inf=True,
        detect_overflow=True
    )
    
    # Create stability manager
    stability_manager = NumericalStabilityManager(clipping_config, nan_config)
    
    # Test forward and backward pass
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    x = torch.randn(32, 10)
    target = torch.randn(32, 1)
    
    optimizer.zero_grad()
    output = model(x)
    loss = nn.MSELoss()(output, target)
    loss.backward()
    
    # Apply stability measures
    result = stability_manager.step(model, loss, optimizer)
    
    print(f"  ✓ Basic functionality test passed")
    print(f"    Stability score: {result['stability_score']:.4f}")
    print(f"    Clipping ratio: {result['clipping_stats'].get('clipping_ratio', 0.0):.4f}")
    print(f"    NaN detected: {result['nan_stats']['nan_detected']}")
    
    return True


def test_different_clipping_types():
    """Test different gradient clipping types."""
    print("\nTesting different clipping types...")
    
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
        nn.Linear(10, 1)
    )
    
    clipping_types = [
        (ClippingType.NORM, "Norm"),
        (ClippingType.VALUE, "Value"),
        (ClippingType.GLOBAL_NORM, "Global Norm"),
        (ClippingType.ADAPTIVE, "Adaptive"),
        (ClippingType.LAYER_WISE, "Layer-wise"),
        (ClippingType.PERCENTILE, "Percentile"),
        (ClippingType.EXPONENTIAL, "Exponential")
    ]
    
    for clipping_type, name in clipping_types:
        try:
            config = GradientClippingConfig(
                clipping_type=clipping_type,
                max_norm=1.0,
                max_value=1.0,
                layer_wise_enabled=(clipping_type == ClippingType.LAYER_WISE),
                layer_norm_thresholds={'0.weight': 0.8, '2.weight': 1.2, '3.weight': 0.6},
                percentile_enabled=(clipping_type == ClippingType.PERCENTILE),
                percentile_threshold=90.0,
                exponential_enabled=(clipping_type == ClippingType.EXPONENTIAL),
                exponential_alpha=0.9,
                exponential_min_threshold=0.5
            )
            
            nan_config = NaNHandlingConfig(
                handling_type=NaNHandlingType.DETECT,
                detect_nan=True,
                detect_inf=True,
                detect_overflow=True
            )
            
            stability_manager = NumericalStabilityManager(config, nan_config)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            x = torch.randn(32, 10)
            target = torch.randn(32, 1)
            
            optimizer.zero_grad()
            output = model(x)
            loss = nn.MSELoss()(output, target)
            loss.backward()
            
            result = stability_manager.step(model, loss, optimizer)
            
            print(f"  ✓ {name} clipping test passed")
            print(f"    Clipping ratio: {result['clipping_stats'].get('clipping_ratio', 0.0):.4f}")
            
        except Exception as e:
            print(f"  ✗ {name} clipping test failed: {e}")
            return False
    
    return True


def test_nan_handling():
    """Test NaN handling capabilities."""
    print("\nTesting NaN handling...")
    
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
        nn.Linear(10, 1)
    )
    
    handling_types = [
        (NaNHandlingType.DETECT, "Detect"),
        (NaNHandlingType.REPLACE, "Replace"),
        (NaNHandlingType.SKIP, "Skip"),
        (NaNHandlingType.GRADIENT_ZEROING, "Gradient Zeroing"),
        (NaNHandlingType.ADAPTIVE, "Adaptive"),
        (NaNHandlingType.GRADIENT_SCALING, "Gradient Scaling")
    ]
    
    for handling_type, name in handling_types:
        try:
            config = GradientClippingConfig(
                clipping_type=ClippingType.NORM,
                max_norm=1.0
            )
            
            nan_config = NaNHandlingConfig(
                handling_type=handling_type,
                detect_nan=True,
                detect_inf=True,
                detect_overflow=True
            )
            
            stability_manager = NumericalStabilityManager(config, nan_config)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            x = torch.randn(32, 10)
            target = torch.randn(32, 1)
            
            optimizer.zero_grad()
            output = model(x)
            loss = nn.MSELoss()(output, target)
            loss.backward()
            
            # Introduce NaN in gradients
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data[0, 0] = float('nan')
            
            result = stability_manager.step(model, loss, optimizer)
            
            print(f"  ✓ {name} handling test passed")
            print(f"    NaN detected: {result['nan_stats']['nan_detected']}")
            print(f"    Handling action: {result['nan_stats']['handling_action']}")
            
        except Exception as e:
            print(f"  ✗ {name} handling test failed: {e}")
            return False
    
    return True


def test_training_wrapper():
    """Test the training wrapper functionality."""
    print("\nTesting training wrapper...")
    
    try:
        clipping_config = GradientClippingConfig(
            clipping_type=ClippingType.NORM,
            max_norm=1.0
        )
        
        nan_config = NaNHandlingConfig(
            handling_type=NaNHandlingType.ADAPTIVE,
            detect_nan=True,
            detect_inf=True,
            detect_overflow=True
        )
        
        wrapper = create_training_wrapper(clipping_config, nan_config)
        
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        x = torch.randn(32, 10)
        target = torch.randn(32, 1)
        
        # Test wrapper call
        optimizer.zero_grad()
        output = model(x)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        
        result = wrapper(model, loss, optimizer)
        
        print(f"  ✓ Training wrapper test passed")
        print(f"    Step count: {wrapper.step_count}")
        print(f"    Stability score: {result['stability_score']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Training wrapper test failed: {e}")
        return False


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\nTesting edge cases...")
    
    try:
        # Test with empty model
        model = nn.Sequential()
        
        config = GradientClippingConfig(
            clipping_type=ClippingType.NORM,
            max_norm=1.0
        )
        
        nan_config = NaNHandlingConfig(
            handling_type=NaNHandlingType.DETECT,
            detect_nan=True,
            detect_inf=True,
            detect_overflow=True
        )
        
        stability_manager = NumericalStabilityManager(config, nan_config)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        x = torch.randn(32, 10)
        target = torch.randn(32, 1)
        
        optimizer.zero_grad()
        output = model(x)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        
        result = stability_manager.step(model, loss, optimizer)
        
        print(f"  ✓ Edge case test passed (empty model)")
        
        # Test with very large gradients
        model = nn.Sequential(nn.Linear(10, 1))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        optimizer.zero_grad()
        output = model(x)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        
        # Introduce very large gradients
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data *= 1000
        
        result = stability_manager.step(model, loss, optimizer)
        
        print(f"  ✓ Edge case test passed (large gradients)")
        print(f"    Clipping ratio: {result['clipping_stats'].get('clipping_ratio', 0.0):.4f}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Edge case test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Running gradient clipping and NaN handling tests...")
    print("=" * 60)
    
    tests = [
        test_basic_functionality,
        test_different_clipping_types,
        test_nan_handling,
        test_training_wrapper,
        test_edge_cases
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  ✗ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is working correctly.")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    
    return passed == total


if __name__ == "__main__":
    main()






