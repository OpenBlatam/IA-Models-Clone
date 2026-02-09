from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import asyncio
import aiohttp
import aiofiles
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
            import json
            import csv
            from io import StringIO
            import json
from typing import Any, List, Dict, Optional
import logging
"""
Async/Sync Refactoring Examples
==============================

This file demonstrates how to refactor existing code to follow:
- `def` for CPU-bound operations
- `async def` for I/O-bound operations
- RORO pattern for all interfaces
- Named exports for utilities
"""


# ============================================================================
# NAMED EXPORTS
# ============================================================================

__all__ = [
    # CPU-bound functions
    "calculate_loss_function",
    "normalize_tensor_data",
    "validate_model_configuration",
    "compute_gradient_descent",
    "calculate_metrics",
    
    # I/O-bound functions
    "load_training_data_async",
    "save_model_checkpoint_async",
    "fetch_external_api_data",
    "download_model_weights_async",
    
    # Mixed operations
    "train_model_with_async_data_loading",
    "evaluate_model_with_async_metrics",
]

# ============================================================================
# BEFORE AND AFTER REFACTORING EXAMPLES
# ============================================================================

# ❌ BEFORE: Mixed responsibilities, no RORO pattern
def process_training_data_old(data_path, model, config) -> Any:
    """Old implementation with mixed CPU and I/O operations."""
    # I/O operation (should be async)
    with open(data_path, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        data = json.load(f)
    
    # CPU operations
    processed_data = normalize_data(data)
    predictions = model(processed_data)
    loss = nn.MSELoss()(predictions, data['targets'])
    
    # I/O operation (should be async)
    with open('results.json', 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        json.dump({'loss': loss.item()}, f)
    
    return loss.item()

# ✅ AFTER: Separated responsibilities with RORO pattern
def normalize_tensor_data(params: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize tensor data - CPU-bound pure function."""
    try:
        tensor_data = params["tensor_data"]
        normalization_type = params.get("normalization_type", "standard")
        
        if normalization_type == "standard":
            mean = torch.mean(tensor_data)
            std = torch.std(tensor_data)
            normalized = (tensor_data - mean) / (std + 1e-8)
        elif normalization_type == "minmax":
            min_val = torch.min(tensor_data)
            max_val = torch.max(tensor_data)
            normalized = (tensor_data - min_val) / (max_val - min_val + 1e-8)
        
        return {
            "is_successful": True,
            "result": normalized,
            "error": None
        }
    except Exception as exc:
        return {
            "is_successful": False,
            "result": None,
            "error": str(exc)
        }

def calculate_loss_function(params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate loss function - CPU-bound pure function."""
    try:
        predictions = params["predictions"]
        targets = params["targets"]
        loss_type = params.get("loss_type", "mse")
        
        if loss_type == "mse":
            loss = nn.MSELoss()(predictions, targets)
        elif loss_type == "cross_entropy":
            loss = nn.CrossEntropyLoss()(predictions, targets)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
        
        return {
            "is_successful": True,
            "result": loss.item(),
            "error": None
        }
    except Exception as exc:
        return {
            "is_successful": False,
            "result": None,
            "error": str(exc)
        }

async def load_training_data_async(params: Dict[str, Any]) -> Dict[str, Any]:
    """Load training data from file - I/O-bound async function."""
    try:
        file_path = params["file_path"]
        data_format = params.get("data_format", "json")
        
        # I/O-bound operation
        async with aiofiles.open(file_path, 'r') as file:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            content = await file.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        
        if data_format == "json":
            data = json.loads(content)
        elif data_format == "csv":
            data = list(csv.DictReader(StringIO(content)))
        else:
            raise ValueError(f"Unsupported data format: {data_format}")
        
        return {
            "is_successful": True,
            "result": data,
            "error": None
        }
    except Exception as exc:
        return {
            "is_successful": False,
            "result": None,
            "error": str(exc)
        }

async def save_model_checkpoint_async(params: Dict[str, Any]) -> Dict[str, Any]:
    """Save model checkpoint to disk - I/O-bound async function."""
    try:
        model_state = params["model_state"]
        checkpoint_path = params["checkpoint_path"]
        
        # I/O-bound operation
        async with aiofiles.open(checkpoint_path, 'w') as file:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            await file.write(json.dumps(model_state))
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        
        return {
            "is_successful": True,
            "result": checkpoint_path,
            "error": None
        }
    except Exception as exc:
        return {
            "is_successful": False,
            "result": None,
            "error": str(exc)
        }

async def train_model_with_async_data_loading(params: Dict[str, Any]) -> Dict[str, Any]:
    """Train model with async data loading - mixed CPU and I/O operations."""
    try:
        model = params["model"]
        data_path = params["data_path"]
        training_config = params["training_config"]
        
        # I/O-bound: Load training data
        data_result = await load_training_data_async({"file_path": data_path})
        if not data_result["is_successful"]:
            return data_result
        
        training_data = data_result["result"]
        
        # CPU-bound: Process data
        processed_result = normalize_tensor_data({
            "tensor_data": training_data["features"],
            "normalization_type": "standard"
        })
        if not processed_result["is_successful"]:
            return processed_result
        
        processed_data = processed_result["result"]
        
        # CPU-bound: Training loop
        for epoch in range(training_config.get("epochs", 10)):
            # CPU-intensive training
            predictions = model(processed_data)
            loss_result = calculate_loss_function({
                "predictions": predictions,
                "targets": training_data["targets"],
                "loss_type": "mse"
            })
            
            if not loss_result["is_successful"]:
                return loss_result
        
        # I/O-bound: Save model
        save_result = await save_model_checkpoint_async({
            "model_state": model.state_dict(),
            "checkpoint_path": training_config.get("checkpoint_path", "model.pth")
        })
        
        return save_result
        
    except Exception as exc:
        return {
            "is_successful": False,
            "result": None,
            "error": str(exc)
        }

# ============================================================================
# REFACTORING PATTERNS
# ============================================================================

def demonstrate_refactoring_patterns():
    """Demonstrate common refactoring patterns."""
    
    patterns = {
        "cpu_bound_operations": {
            "description": "Pure functions for CPU-intensive operations",
            "examples": [
                "calculate_loss_function",
                "normalize_tensor_data", 
                "validate_model_configuration",
                "compute_gradient_descent",
                "calculate_metrics"
            ],
            "signature": "def function_name(params: Dict[str, Any]) -> Dict[str, Any]:",
            "return_pattern": {
                "is_successful": True,
                "result": computed_value,
                "error": None
            }
        },
        "io_bound_operations": {
            "description": "Async functions for I/O operations",
            "examples": [
                "load_training_data_async",
                "save_model_checkpoint_async",
                "fetch_external_api_data",
                "download_model_weights_async"
            ],
            "signature": "async def function_name(params: Dict[str, Any]) -> Dict[str, Any]:",
            "return_pattern": {
                "is_successful": True,
                "result": loaded_data,
                "error": None
            }
        },
        "mixed_operations": {
            "description": "Combining CPU and I/O operations",
            "examples": [
                "train_model_with_async_data_loading",
                "evaluate_model_with_async_metrics"
            ],
            "pattern": "I/O → CPU → I/O",
            "benefits": [
                "Clear separation of concerns",
                "Better error handling",
                "Improved performance",
                "Easier testing"
            ]
        }
    }
    
    return patterns

# ============================================================================
# PRACTICAL REFACTORING EXAMPLES
# ============================================================================

# ❌ BEFORE: Function with mixed responsibilities
def evaluate_model_old(model, data_path, metrics_path) -> Any:
    """Old evaluation function with mixed responsibilities."""
    # I/O: Load data
    with open(data_path, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        data = json.load(f)
    
    # CPU: Model inference
    predictions = model(torch.tensor(data['features']))
    
    # CPU: Calculate metrics
    accuracy = (predictions.argmax(dim=1) == torch.tensor(data['targets'])).float().mean()
    
    # I/O: Save results
    with open(metrics_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        json.dump({'accuracy': accuracy.item()}, f)
    
    return accuracy.item()

# ✅ AFTER: Separated responsibilities with RORO pattern
def calculate_metrics(params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate evaluation metrics - CPU-bound pure function."""
    try:
        predictions = params["predictions"]
        targets = params["targets"]
        metrics = params.get("metrics", ["accuracy"])
        
        results = {}
        
        if "accuracy" in metrics:
            correct = (predictions.argmax(dim=1) == targets).sum()
            accuracy = correct / len(targets)
            results["accuracy"] = accuracy.item()
        
        if "precision" in metrics:
            precision = torch.sum((predictions.argmax(dim=1) == targets) & (targets == 1)) / torch.sum(predictions.argmax(dim=1) == 1)
            results["precision"] = precision.item()
        
        if "recall" in metrics:
            recall = torch.sum((predictions.argmax(dim=1) == targets) & (targets == 1)) / torch.sum(targets == 1)
            results["recall"] = recall.item()
        
        return {
            "is_successful": True,
            "result": results,
            "error": None
        }
    except Exception as exc:
        return {
            "is_successful": False,
            "result": None,
            "error": str(exc)
        }

async def evaluate_model_with_async_metrics(params: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate model with async metrics calculation - mixed CPU and I/O operations."""
    try:
        model = params["model"]
        data_path = params["data_path"]
        metrics_path = params["metrics_path"]
        
        # I/O-bound: Load test data
        data_result = await load_training_data_async({"file_path": data_path})
        if not data_result["is_successful"]:
            return data_result
        
        test_data = data_result["result"]
        
        # CPU-bound: Model inference
        predictions = model(torch.tensor(test_data["features"]))
        
        # CPU-bound: Calculate metrics
        metrics_result = calculate_metrics({
            "predictions": predictions,
            "targets": torch.tensor(test_data["targets"]),
            "metrics": ["accuracy", "precision", "recall"]
        })
        
        if not metrics_result["is_successful"]:
            return metrics_result
        
        metrics = metrics_result["result"]
        
        # I/O-bound: Save results
        save_result = await save_model_checkpoint_async({
            "model_state": metrics,
            "checkpoint_path": metrics_path
        })
        
        return save_result
        
    except Exception as exc:
        return {
            "is_successful": False,
            "result": None,
            "error": str(exc)
        }

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

async def demonstrate_refactored_usage():
    """Demonstrate how to use the refactored functions."""
    
    print("🔄 Demonstrating Refactored Async/Sync Patterns")
    print("=" * 50)
    
    # Example 1: CPU-bound operation
    print("\n📊 CPU-Bound Operation:")
    loss_result = calculate_loss_function({
        "predictions": torch.randn(10, 3),
        "targets": torch.randint(0, 3, (10,)),
        "loss_type": "cross_entropy"
    })
    print(f"Loss calculation: {loss_result['is_successful']}")
    if loss_result["is_successful"]:
        print(f"Loss value: {loss_result['result']}")
    
    # Example 2: I/O-bound operation (simulated)
    print("\n🌐 I/O-Bound Operation:")
    # Note: This would require actual files
    print("I/O operations require actual files/data")
    print("Pattern demonstrated in the code above")
    
    # Example 3: Mixed operation
    print("\n🔄 Mixed Operation:")
    print("Combines I/O → CPU → I/O pattern")
    print("See train_model_with_async_data_loading() for example")

def show_refactoring_benefits():
    """Show the benefits of refactoring."""
    
    benefits = {
        "performance": [
            "CPU-bound operations don't block I/O",
            "I/O-bound operations don't block CPU",
            "Better resource utilization",
            "Improved concurrency"
        ],
        "maintainability": [
            "Clear separation of concerns",
            "Easier to test individual functions",
            "Better error handling",
            "More modular code"
        ],
        "scalability": [
            "Async operations handle concurrency",
            "CPU operations can be parallelized",
            "Better resource management",
            "Improved throughput"
        ],
        "reliability": [
            "Consistent error handling",
            "RORO pattern ensures predictable returns",
            "Named exports prevent accidental imports",
            "Clear function responsibilities"
        ]
    }
    
    return benefits

if __name__ == "__main__":
    # Demonstrate refactoring patterns
    patterns = demonstrate_refactoring_patterns()
    benefits = show_refactoring_benefits()
    
    print("✅ Async/Sync refactoring examples created successfully!")
    print(f"📊 CPU-bound patterns: {len(patterns['cpu_bound_operations']['examples'])} examples")
    print(f"🌐 I/O-bound patterns: {len(patterns['io_bound_operations']['examples'])} examples")
    print(f"🔄 Mixed patterns: {len(patterns['mixed_operations']['examples'])} examples")
    
    print("\n🎯 Key Refactoring Benefits:")
    for category, items in benefits.items():
        print(f"\n{category.title()}:")
        for item in items:
            print(f"  • {item}")
    
    # Run async demonstration
    asyncio.run(demonstrate_refactored_usage()) 