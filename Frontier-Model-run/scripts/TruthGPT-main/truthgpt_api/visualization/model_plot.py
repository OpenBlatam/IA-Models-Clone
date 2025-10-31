"""
Model Plotting for TruthGPT API
==============================

TensorFlow-like model plotting utilities.
"""

import torch
import matplotlib.pyplot as plt
from typing import Any, Optional, Dict, List
import numpy as np


def plot_model(model: Any, 
               to_file: Optional[str] = None,
               show_shapes: bool = False,
               show_layer_names: bool = True,
               rankdir: str = 'TB',
               expand_nested: bool = False,
               dpi: int = 96,
               layer_range: Optional[List[int]] = None) -> None:
    """
    Plot model architecture.
    
    Similar to tf.keras.utils.plot_model, this function
    creates a visual representation of the model architecture.
    
    Args:
        model: Model to plot
        to_file: File to save the plot
        show_shapes: Whether to show input/output shapes
        show_layer_names: Whether to show layer names
        rankdir: Direction of the plot ('TB', 'BT', 'LR', 'RL')
        expand_nested: Whether to expand nested models
        dpi: DPI for the plot
        layer_range: Range of layers to show
    """
    print(f"📊 Plotting model architecture...")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get model information
    if hasattr(model, 'layers_list'):
        layers = model.layers_list
    elif hasattr(model, 'layers'):
        layers = model.layers
    else:
        layers = []
    
    # Plot layers
    y_pos = 0
    layer_info = []
    
    for i, layer in enumerate(layers):
        if layer_range and i not in range(layer_range[0], layer_range[1]):
            continue
        
        layer_name = str(layer)
        layer_type = type(layer).__name__
        
        # Create layer box
        box = plt.Rectangle((0, y_pos), 2, 0.8, 
                          facecolor='lightblue', 
                          edgecolor='black',
                          linewidth=1)
        ax.add_patch(box)
        
        # Add layer name
        ax.text(1, y_pos + 0.4, layer_name, 
                ha='center', va='center', 
                fontsize=10, fontweight='bold')
        
        # Add layer type
        ax.text(1, y_pos + 0.2, layer_type, 
                ha='center', va='center', 
                fontsize=8, style='italic')
        
        layer_info.append({
            'name': layer_name,
            'type': layer_type,
            'position': y_pos
        })
        
        y_pos += 1.2
    
    # Set plot properties
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, y_pos + 0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Add title
    ax.set_title(f'Model Architecture: {type(model).__name__}', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Add legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='lightblue', 
                     edgecolor='black', label='Layer')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Save or show plot
    if to_file:
        plt.savefig(to_file, dpi=dpi, bbox_inches='tight')
        print(f"✅ Model plot saved to {to_file}")
    else:
        plt.show()
    
    # Print model summary
    print(f"\n📋 Model Summary:")
    print(f"   Total layers: {len(layers)}")
    print(f"   Model type: {type(model).__name__}")
    
    if show_layer_names:
        print(f"\n🔍 Layer Details:")
        for i, info in enumerate(layer_info):
            print(f"   {i+1}. {info['name']} ({info['type']})")
    
    plt.close()


def plot_model_parameters(model: Any, 
                         to_file: Optional[str] = None,
                         show_details: bool = True) -> None:
    """
    Plot model parameter distribution.
    
    Args:
        model: Model to analyze
        to_file: File to save the plot
        show_details: Whether to show detailed statistics
    """
    print(f"📊 Plotting model parameters...")
    
    # Get model parameters
    parameters = []
    layer_names = []
    
    if hasattr(model, 'named_parameters'):
        for name, param in model.named_parameters():
            if param.requires_grad:
                parameters.append(param.data.flatten().cpu().numpy())
                layer_names.append(name)
    elif hasattr(model, 'parameters'):
        for i, param in enumerate(model.parameters()):
            if param.requires_grad:
                parameters.append(param.data.flatten().cpu().numpy())
                layer_names.append(f'Layer_{i}')
    
    if not parameters:
        print("⚠️ No trainable parameters found")
        return
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Model Parameter Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Parameter distribution
    ax1 = axes[0, 0]
    all_params = np.concatenate(parameters)
    ax1.hist(all_params, bins=50, alpha=0.7, color='blue')
    ax1.set_title('Parameter Distribution')
    ax1.set_xlabel('Parameter Value')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Layer-wise parameter count
    ax2 = axes[0, 1]
    param_counts = [len(p) for p in parameters]
    ax2.bar(range(len(param_counts)), param_counts, color='green', alpha=0.7)
    ax2.set_title('Parameters per Layer')
    ax2.set_xlabel('Layer Index')
    ax2.set_ylabel('Parameter Count')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Layer-wise parameter magnitude
    ax3 = axes[1, 0]
    param_magnitudes = [np.mean(np.abs(p)) for p in parameters]
    ax3.bar(range(len(param_magnitudes)), param_magnitudes, color='red', alpha=0.7)
    ax3.set_title('Average Parameter Magnitude per Layer')
    ax3.set_xlabel('Layer Index')
    ax3.set_ylabel('Average |Parameter|')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Parameter statistics
    ax4 = axes[1, 1]
    stats = {
        'Mean': np.mean(all_params),
        'Std': np.std(all_params),
        'Min': np.min(all_params),
        'Max': np.max(all_params)
    }
    ax4.bar(stats.keys(), stats.values(), color='orange', alpha=0.7)
    ax4.set_title('Parameter Statistics')
    ax4.set_ylabel('Value')
    ax4.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f"Total Parameters: {len(all_params):,}\n"
    stats_text += f"Mean: {stats['Mean']:.4f}\n"
    stats_text += f"Std: {stats['Std']:.4f}\n"
    stats_text += f"Min: {stats['Min']:.4f}\n"
    stats_text += f"Max: {stats['Max']:.4f}"
    
    ax4.text(0.02, 0.98, stats_text, transform=ax4.transAxes, 
             verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save or show plot
    if to_file:
        plt.savefig(to_file, dpi=300, bbox_inches='tight')
        print(f"✅ Parameter plot saved to {to_file}")
    else:
        plt.show()
    
    # Print detailed statistics
    if show_details:
        print(f"\n📋 Parameter Statistics:")
        print(f"   Total parameters: {len(all_params):,}")
        print(f"   Mean: {stats['Mean']:.6f}")
        print(f"   Standard deviation: {stats['Std']:.6f}")
        print(f"   Minimum: {stats['Min']:.6f}")
        print(f"   Maximum: {stats['Max']:.6f}")
        
        print(f"\n🔍 Layer-wise Analysis:")
        for i, (name, count, magnitude) in enumerate(zip(layer_names, param_counts, param_magnitudes)):
            print(f"   {i+1}. {name}: {count:,} parameters, avg magnitude: {magnitude:.6f}")
    
    plt.close()









