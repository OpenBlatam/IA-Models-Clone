#!/usr/bin/env python3
"""
Gradio Integration for Gradient Clipping and NaN/Inf Handling System
Interactive web interface for experimenting with numerical stability configurations.
"""

import gradio as gr
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple
import json
import time
from pathlib import Path

# Import our gradient clipping and NaN handling system
from gradient_clipping_nan_handling import (
    GradientClippingConfig,
    NaNHandlingConfig,
    NumericalStabilityManager,
    ClippingType,
    NaNHandlingType,
    create_training_wrapper
)


class InteractiveTrainingSimulator:
    """Interactive training simulator with real-time visualization."""
    
    def __init__(self):
        self.current_model = None
        self.current_optimizer = None
        self.stability_manager = None
        self.training_history = {
            'steps': [],
            'losses': [],
            'stability_scores': [],
            'clipping_ratios': [],
            'nan_counts': [],
            'inf_counts': [],
            'overflow_counts': []
        }
        self.current_step = 0
        
    def create_model(self, model_type: str, input_size: int, hidden_size: int, output_size: int) -> str:
        """Create a new model based on user specifications."""
        try:
            if model_type == "Sequential":
                self.current_model = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_size // 2, output_size)
                )
            elif model_type == "Deep":
                layers = []
                current_size = input_size
                for i in range(3):
                    layers.extend([
                        nn.Linear(current_size, hidden_size),
                        nn.ReLU(),
                        nn.Dropout(0.2)
                    ])
                    current_size = hidden_size
                layers.append(nn.Linear(current_size, output_size))
                self.current_model = nn.Sequential(*layers)
            elif model_type == "Wide":
                self.current_model = nn.Sequential(
                    nn.Linear(input_size, hidden_size * 2),
                    nn.ReLU(),
                    nn.Linear(hidden_size * 2, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, output_size)
                )
            
            # Initialize optimizer
            self.current_optimizer = torch.optim.Adam(self.current_model.parameters(), lr=0.01)
            
            # Reset training history
            self.reset_training_history()
            
            param_count = sum(p.numel() for p in self.current_model.parameters())
            return f"✅ Model created successfully!\nModel type: {model_type}\nParameters: {param_count:,}\nInput size: {input_size}\nHidden size: {hidden_size}\nOutput size: {output_size}"
            
        except Exception as e:
            return f"❌ Error creating model: {str(e)}"
    
    def reset_training_history(self):
        """Reset training history."""
        self.training_history = {
            'steps': [],
            'losses': [],
            'stability_scores': [],
            'clipping_ratios': [],
            'nan_counts': [],
            'inf_counts': [],
            'overflow_counts': []
        }
        self.current_step = 0
    
    def configure_stability_manager(self, clipping_type: str, max_norm: float, 
                                  nan_handling_type: str, adaptive_threshold: float) -> str:
        """Configure the numerical stability manager."""
        try:
            # Create clipping configuration
            clipping_config = GradientClippingConfig(
                clipping_type=getattr(ClippingType, clipping_type.upper()),
                max_norm=max_norm,
                adaptive_threshold=adaptive_threshold,
                monitor_clipping=True,
                log_clipping_stats=True,
                save_clipping_history=True
            )
            
            # Create NaN handling configuration
            nan_config = NaNHandlingConfig(
                handling_type=getattr(NaNHandlingType, nan_handling_type.upper()),
                detect_nan=True,
                detect_inf=True,
                detect_overflow=True,
                monitor_nan=True,
                log_nan_stats=True,
                save_nan_history=True
            )
            
            # Create stability manager
            self.stability_manager = NumericalStabilityManager(clipping_config, nan_config)
            
            return f"✅ Stability manager configured!\nClipping type: {clipping_type}\nMax norm: {max_norm}\nNaN handling: {nan_handling_type}\nAdaptive threshold: {adaptive_threshold}"
            
        except Exception as e:
            return f"❌ Error configuring stability manager: {str(e)}"
    
    def run_training_step(self, batch_size: int, introduce_nan_prob: float, 
                         introduce_inf_prob: float, introduce_overflow_prob: float) -> Tuple[str, Dict[str, Any]]:
        """Run a single training step with the current configuration."""
        if self.current_model is None or self.stability_manager is None:
            return "❌ Please create a model and configure stability manager first.", {}
        
        try:
            # Generate synthetic data
            x = torch.randn(batch_size, self.current_model[0].in_features)
            target = torch.randn(batch_size, self.current_model[-1].out_features)
            
            # Forward pass
            self.current_optimizer.zero_grad()
            output = self.current_model(x)
            loss = nn.MSELoss()(output, target)
            
            # Backward pass
            loss.backward()
            
            # Introduce numerical issues based on probabilities
            self._introduce_numerical_issues(introduce_nan_prob, introduce_inf_prob, introduce_overflow_prob)
            
            # Apply stability measures
            stability_result = self.stability_manager.step(self.current_model, loss, self.current_optimizer)
            
            # Optimizer step
            self.current_optimizer.step()
            
            # Update training history
            self.current_step += 1
            self.training_history['steps'].append(self.current_step)
            self.training_history['losses'].append(loss.item())
            self.training_history['stability_scores'].append(stability_result['stability_score'])
            self.training_history['clipping_ratios'].append(stability_result['clipping_stats'].get('clipping_ratio', 0.0))
            self.training_history['nan_counts'].append(1 if stability_result['nan_stats']['nan_detected'] else 0)
            self.training_history['inf_counts'].append(1 if stability_result['nan_stats']['inf_detected'] else 0)
            self.training_history['overflow_counts'].append(1 if stability_result['nan_stats']['overflow_detected'] else 0)
            
            # Prepare result message
            result_msg = f"✅ Training step {self.current_step} completed!\n"
            result_msg += f"Loss: {loss.item():.6f}\n"
            result_msg += f"Stability Score: {stability_result['stability_score']:.4f}\n"
            result_msg += f"Clipping Ratio: {stability_result['clipping_stats'].get('clipping_ratio', 0.0):.4f}\n"
            result_msg += f"NaN Detected: {stability_result['nan_stats']['nan_detected']}\n"
            result_msg += f"Inf Detected: {stability_result['nan_stats']['inf_detected']}\n"
            result_msg += f"Overflow Detected: {stability_result['nan_stats']['overflow_detected']}\n"
            result_msg += f"Handling Action: {stability_result['nan_stats']['handling_action']}"
            
            return result_msg, stability_result
            
        except Exception as e:
            return f"❌ Error in training step: {str(e)}", {}
    
    def _introduce_numerical_issues(self, nan_prob: float, inf_prob: float, overflow_prob: float):
        """Introduce numerical issues in gradients based on probabilities."""
        for param in self.current_model.parameters():
            if param.grad is not None:
                # Introduce NaN
                if np.random.random() < nan_prob:
                    param.grad.data[0, 0] = float('nan')
                
                # Introduce Inf
                if np.random.random() < inf_prob:
                    param.grad.data[0, 0] = float('inf')
                
                # Introduce overflow
                if np.random.random() < overflow_prob:
                    param.grad.data *= 1e6
    
    def run_multiple_steps(self, num_steps: int, batch_size: int, 
                          introduce_nan_prob: float, introduce_inf_prob: float, 
                          introduce_overflow_prob: float, progress=gr.Progress()) -> str:
        """Run multiple training steps."""
        if self.current_model is None or self.stability_manager is None:
            return "❌ Please create a model and configure stability manager first."
        
        try:
            results = []
            for step in range(num_steps):
                progress(step / num_steps, desc=f"Training step {step + 1}/{num_steps}")
                
                # Generate synthetic data
                x = torch.randn(batch_size, self.current_model[0].in_features)
                target = torch.randn(batch_size, self.current_model[-1].out_features)
                
                # Forward pass
                self.current_optimizer.zero_grad()
                output = self.current_model(x)
                loss = nn.MSELoss()(output, target)
                
                # Backward pass
                loss.backward()
                
                # Introduce numerical issues
                self._introduce_numerical_issues(introduce_nan_prob, introduce_inf_prob, introduce_overflow_prob)
                
                # Apply stability measures
                stability_result = self.stability_manager.step(self.current_model, loss, self.current_optimizer)
                
                # Optimizer step
                self.current_optimizer.step()
                
                # Update history
                self.current_step += 1
                self.training_history['steps'].append(self.current_step)
                self.training_history['losses'].append(loss.item())
                self.training_history['stability_scores'].append(stability_result['stability_score'])
                self.training_history['clipping_ratios'].append(stability_result['clipping_stats'].get('clipping_ratio', 0.0))
                self.training_history['nan_counts'].append(1 if stability_result['nan_stats']['nan_detected'] else 0)
                self.training_history['inf_counts'].append(1 if stability_result['nan_stats']['inf_detected'] else 0)
                self.training_history['overflow_counts'].append(1 if stability_result['nan_stats']['overflow_detected'] else 0)
                
                results.append(f"Step {step + 1}: Loss={loss.item():.6f}, Stability={stability_result['stability_score']:.4f}")
                
                # Small delay to prevent overwhelming the interface
                time.sleep(0.01)
            
            return f"✅ Completed {num_steps} training steps!\n\n" + "\n".join(results[-10:])  # Show last 10 results
            
        except Exception as e:
            return f"❌ Error in multiple training steps: {str(e)}"
    
    def generate_training_plots(self) -> Tuple[plt.Figure, plt.Figure, plt.Figure]:
        """Generate comprehensive training visualization plots."""
        if not self.training_history['steps']:
            # Return empty plots if no training data
            fig1, ax1 = plt.subplots(figsize=(8, 6))
            ax1.text(0.5, 0.5, 'No training data available', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Training Progress')
            
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            ax2.text(0.5, 0.5, 'No training data available', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Numerical Stability')
            
            fig3, ax3 = plt.subplots(figsize=(8, 6))
            ax3.text(0.5, 0.5, 'No training data available', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Numerical Issues')
            
            return fig1, fig2, fig3
        
        # Plot 1: Training Progress
        fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Loss over time
        ax1.plot(self.training_history['steps'], self.training_history['losses'], 'b-', label='Training Loss')
        ax1.set_title('Training Loss Over Time')
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Stability score over time
        ax2.plot(self.training_history['steps'], self.training_history['stability_scores'], 'g-', label='Stability Score')
        ax2.set_title('Numerical Stability Score Over Time')
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Stability Score')
        ax2.set_ylim(0, 1)
        ax2.legend()
        ax2.grid(True)
        
        fig1.tight_layout()
        
        # Plot 2: Numerical Stability Metrics
        fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Clipping ratio over time
        ax1.plot(self.training_history['steps'], self.training_history['clipping_ratios'], 'r-', label='Clipping Ratio')
        ax1.set_title('Gradient Clipping Ratio Over Time')
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Clipping Ratio')
        ax1.legend()
        ax1.grid(True)
        
        # Stability score distribution
        ax2.hist(self.training_history['stability_scores'], bins=20, alpha=0.7, edgecolor='black', color='green')
        ax2.set_title('Stability Score Distribution')
        ax2.set_xlabel('Stability Score')
        ax2.set_ylabel('Frequency')
        ax2.grid(True)
        
        fig2.tight_layout()
        
        # Plot 3: Numerical Issues
        fig3, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Numerical issues over time
        ax1.plot(self.training_history['steps'], self.training_history['nan_counts'], 'r-', label='NaN', alpha=0.7)
        ax1.plot(self.training_history['steps'], self.training_history['inf_counts'], 'orange', label='Inf', alpha=0.7)
        ax1.plot(self.training_history['steps'], self.training_history['overflow_counts'], 'yellow', label='Overflow', alpha=0.7)
        ax1.set_title('Numerical Issues Over Time')
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Count')
        ax1.legend()
        ax1.grid(True)
        
        # Cumulative numerical issues
        cumulative_nan = np.cumsum(self.training_history['nan_counts'])
        cumulative_inf = np.cumsum(self.training_history['inf_counts'])
        cumulative_overflow = np.cumsum(self.training_history['overflow_counts'])
        
        ax2.plot(self.training_history['steps'], cumulative_nan, 'r-', label='Cumulative NaN', alpha=0.7)
        ax2.plot(self.training_history['steps'], cumulative_inf, 'orange', label='Cumulative Inf', alpha=0.7)
        ax2.plot(self.training_history['steps'], cumulative_overflow, 'yellow', label='Cumulative Overflow', alpha=0.7)
        ax2.set_title('Cumulative Numerical Issues')
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Cumulative Count')
        ax2.legend()
        ax2.grid(True)
        
        fig3.tight_layout()
        
        return fig1, fig2, fig3
    
    def get_training_summary(self) -> str:
        """Get a summary of the training session."""
        if not self.training_history['steps']:
            return "No training data available."
        
        total_steps = len(self.training_history['steps'])
        avg_loss = np.mean(self.training_history['losses'])
        avg_stability = np.mean(self.training_history['stability_scores'])
        avg_clipping = np.mean(self.training_history['clipping_ratios'])
        total_nan = sum(self.training_history['nan_counts'])
        total_inf = sum(self.training_history['inf_counts'])
        total_overflow = sum(self.training_history['overflow_counts'])
        
        summary = f"📊 Training Session Summary\n"
        summary += f"{'='*40}\n"
        summary += f"Total Steps: {total_steps}\n"
        summary += f"Average Loss: {avg_loss:.6f}\n"
        summary += f"Average Stability Score: {avg_stability:.4f}\n"
        summary += f"Average Clipping Ratio: {avg_clipping:.4f}\n"
        summary += f"Total NaN Issues: {total_nan}\n"
        summary += f"Total Inf Issues: {total_inf}\n"
        summary += f"Total Overflow Issues: {total_overflow}\n"
        summary += f"Numerical Issues Rate: {(total_nan + total_inf + total_overflow) / total_steps * 100:.2f}%"
        
        return summary
    
    def save_training_session(self, filename: str) -> str:
        """Save the current training session to a file."""
        try:
            session_data = {
                'training_history': self.training_history,
                'model_info': {
                    'type': str(type(self.current_model)),
                    'parameters': sum(p.numel() for p in self.current_model.parameters()) if self.current_model else 0
                },
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_steps': self.current_step
            }
            
            with open(filename, 'w') as f:
                json.dump(session_data, f, indent=2, default=str)
            
            return f"✅ Training session saved to {filename}"
            
        except Exception as e:
            return f"❌ Error saving training session: {str(e)}"


def create_gradio_interface():
    """Create the Gradio interface."""
    simulator = InteractiveTrainingSimulator()
    
    with gr.Blocks(title="Gradient Clipping & NaN Handling Simulator", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # 🚀 Gradient Clipping & NaN/Inf Handling Simulator
        
        Interactive web interface for experimenting with numerical stability configurations in deep learning training.
        
        **Features:**
        - 🎯 Multiple gradient clipping strategies
        - 🚨 Advanced NaN/Inf detection and handling
        - 📊 Real-time training visualization
        - 🔧 Configurable training parameters
        - 📈 Comprehensive analytics and plots
        """)
        
        with gr.Tabs():
            # Tab 1: Model Setup
            with gr.Tab("🏗️ Model Setup"):
                gr.Markdown("### Configure your neural network model")
                
                with gr.Row():
                    with gr.Column():
                        model_type = gr.Dropdown(
                            choices=["Sequential", "Deep", "Wide"],
                            value="Sequential",
                            label="Model Architecture"
                        )
                        input_size = gr.Slider(minimum=1, maximum=100, value=10, step=1, label="Input Size")
                        hidden_size = gr.Slider(minimum=10, maximum=500, value=50, step=10, label="Hidden Size")
                        output_size = gr.Slider(minimum=1, maximum=50, value=1, step=1, label="Output Size")
                        
                        create_model_btn = gr.Button("🚀 Create Model", variant="primary")
                        model_status = gr.Textbox(label="Model Status", lines=3)
                    
                    with gr.Column():
                        gr.Markdown("### Model Information")
                        model_info = gr.Markdown("No model created yet.")
                
                create_model_btn.click(
                    fn=simulator.create_model,
                    inputs=[model_type, input_size, hidden_size, output_size],
                    outputs=model_status
                )
            
            # Tab 2: Stability Configuration
            with gr.Tab("⚙️ Stability Configuration"):
                gr.Markdown("### Configure numerical stability parameters")
                
                with gr.Row():
                    with gr.Column():
                        clipping_type = gr.Dropdown(
                            choices=["NORM", "VALUE", "GLOBAL_NORM", "ADAPTIVE", "LAYER_WISE", "PERCENTILE", "EXPONENTIAL"],
                            value="NORM",
                            label="Gradient Clipping Type"
                        )
                        max_norm = gr.Slider(minimum=0.1, maximum=10.0, value=1.0, step=0.1, label="Max Norm")
                        adaptive_threshold = gr.Slider(minimum=0.01, maximum=2.0, value=0.1, step=0.01, label="Adaptive Threshold")
                        
                    with gr.Column():
                        nan_handling_type = gr.Dropdown(
                            choices=["DETECT", "REPLACE", "SKIP", "GRADIENT_ZEROING", "ADAPTIVE", "GRADIENT_SCALING"],
                            value="ADAPTIVE",
                            label="NaN/Inf Handling Type"
                        )
                        
                        config_btn = gr.Button("🔧 Configure Stability Manager", variant="primary")
                        config_status = gr.Textbox(label="Configuration Status", lines=3)
                
                config_btn.click(
                    fn=simulator.configure_stability_manager,
                    inputs=[clipping_type, max_norm, nan_handling_type, adaptive_threshold],
                    outputs=config_status
                )
            
            # Tab 3: Training
            with gr.Tab("🏃‍♂️ Training"):
                gr.Markdown("### Run training steps with numerical stability monitoring")
                
                with gr.Row():
                    with gr.Column():
                        batch_size = gr.Slider(minimum=1, maximum=128, value=32, step=1, label="Batch Size")
                        introduce_nan_prob = gr.Slider(minimum=0.0, maximum=1.0, value=0.1, step=0.01, label="NaN Probability")
                        introduce_inf_prob = gr.Slider(minimum=0.0, maximum=1.0, value=0.05, step=0.01, label="Inf Probability")
                        introduce_overflow_prob = gr.Slider(minimum=0.0, maximum=1.0, value=0.1, step=0.01, label="Overflow Probability")
                        
                        with gr.Row():
                            single_step_btn = gr.Button("▶️ Single Step", variant="primary")
                            multiple_steps_btn = gr.Button("⏩ Multiple Steps", variant="primary")
                        
                        num_steps = gr.Slider(minimum=1, maximum=100, value=10, step=1, label="Number of Steps")
                        
                    with gr.Column():
                        training_status = gr.Textbox(label="Training Status", lines=8)
                        training_summary = gr.Textbox(label="Training Summary", lines=8)
                
                single_step_btn.click(
                    fn=simulator.run_training_step,
                    inputs=[batch_size, introduce_nan_prob, introduce_inf_prob, introduce_overflow_prob],
                    outputs=[training_status, training_summary]
                )
                
                multiple_steps_btn.click(
                    fn=simulator.run_multiple_steps,
                    inputs=[num_steps, batch_size, introduce_nan_prob, introduce_inf_prob, introduce_overflow_prob],
                    outputs=training_status
                )
            
            # Tab 4: Visualization
            with gr.Tab("📊 Visualization"):
                gr.Markdown("### Training progress and numerical stability analysis")
                
                with gr.Row():
                    plot_btn = gr.Button("📈 Generate Plots", variant="primary")
                    summary_btn = gr.Button("📋 Training Summary", variant="primary")
                
                with gr.Row():
                    plot1 = gr.Plot(label="Training Progress")
                    plot2 = gr.Plot(label="Numerical Stability")
                
                with gr.Row():
                    plot3 = gr.Plot(label="Numerical Issues")
                    summary_text = gr.Textbox(label="Training Summary", lines=10)
                
                plot_btn.click(
                    fn=simulator.generate_training_plots,
                    outputs=[plot1, plot2, plot3]
                )
                
                summary_btn.click(
                    fn=simulator.get_training_summary,
                    outputs=summary_text
                )
            
            # Tab 5: Export & Save
            with gr.Tab("💾 Export & Save"):
                gr.Markdown("### Save your training session and results")
                
                with gr.Row():
                    filename = gr.Textbox(
                        value="training_session.json",
                        label="Filename",
                        placeholder="Enter filename to save training session"
                    )
                    save_btn = gr.Button("💾 Save Session", variant="primary")
                
                save_status = gr.Textbox(label="Save Status", lines=3)
                
                save_btn.click(
                    fn=simulator.save_training_session,
                    inputs=filename,
                    outputs=save_status
                )
        
        # Footer
        gr.Markdown("""
        ---
        **Gradient Clipping & NaN/Inf Handling Simulator** | Built with Gradio
        
        This interface allows you to experiment with different numerical stability configurations
        and observe their effects on training dynamics in real-time.
        """)
    
    return interface


def main():
    """Main function to launch the Gradio interface."""
    interface = create_gradio_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
        show_tips=True
    )


if __name__ == "__main__":
    main()






