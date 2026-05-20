"""
Advanced Gradio Interface
Enhanced UI with model comparison, attention visualization, and training monitoring
"""

from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    logger.warning("Gradio not available")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class AdvancedGradioUI:
    """
    Advanced Gradio interface with:
    - Model comparison
    - Attention visualization
    - Training monitoring
    - Performance metrics
    """
    
    def __init__(self):
        if not GRADIO_AVAILABLE:
            raise ImportError("Gradio required")
        
        self.setup_services()
    
    def setup_services(self):
        """Setup ML services"""
        try:
            from ..services.ml_service import get_ml_service
            from ..core.deep_models import get_deep_analyzer
            from ..performance.profiler import PerformanceMonitor
            
            self.ml_service = get_ml_service()
            self.deep_analyzer = get_deep_analyzer()
            self.performance_monitor = PerformanceMonitor()
        except Exception as e:
            logger.warning(f"Could not setup services: {str(e)}")
            self.ml_service = None
    
    def compare_models(
        self,
        audio_file,
        models: List[str]
    ) -> Tuple[str, Optional[str]]:
        """Compare multiple models on same input"""
        if audio_file is None:
            return "Please upload an audio file", None
        
        try:
            from ..core.ml_audio_analyzer import AudioFeatureExtractor
            
            extractor = AudioFeatureExtractor()
            features = extractor.extract_features(audio_file.name)
            
            # Create feature vector
            feature_vector = np.concatenate([
                features.mfcc.mean(axis=1),
                features.chroma.mean(axis=1),
                features.spectral_contrast.mean(axis=1),
                features.tonnetz.mean(axis=1),
                [features.tempo]
            ])
            
            results = {}
            for model_name in models:
                if model_name in self.deep_analyzer.models:
                    model = self.deep_analyzer.models[model_name]
                    
                    import torch
                    input_tensor = torch.FloatTensor(feature_vector).unsqueeze(0)
                    
                    with torch.no_grad():
                        if hasattr(model, 'forward'):
                            output = model(input_tensor)
                            if isinstance(output, dict):
                                results[model_name] = output
                            else:
                                results[model_name] = {"output": output.cpu().numpy().tolist()}
            
            # Format results
            output_text = "## Model Comparison Results\n\n"
            for model_name, result in results.items():
                output_text += f"### {model_name}\n"
                output_text += f"```json\n{result}\n```\n\n"
            
            return output_text, None
        
        except Exception as e:
            logger.error(f"Model comparison error: {str(e)}", exc_info=True)
            return f"Error: {str(e)}", None
    
    def visualize_performance_metrics(self) -> Optional[str]:
        """Visualize performance metrics"""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        try:
            # Get performance stats
            gpu_memory = self.performance_monitor.get_gpu_memory()
            system_info = self.performance_monitor.get_system_info()
            
            # Create visualization
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # GPU Memory
            if gpu_memory.get("available"):
                memory_data = [
                    gpu_memory.get("allocated_gb", 0),
                    gpu_memory.get("free_gb", 0)
                ]
                axes[0].bar(["Allocated", "Free"], memory_data, color=["#FF6B6B", "#4ECDC4"])
                axes[0].set_title("GPU Memory Usage")
                axes[0].set_ylabel("GB")
            
            # System Info
            if system_info.get("cuda_available"):
                info_text = f"""
                PyTorch: {system_info.get('pytorch_version', 'N/A')}
                CUDA: {system_info.get('cuda_version', 'N/A')}
                GPUs: {system_info.get('gpu_count', 0)}
                GPU: {system_info.get('gpu_name', 'N/A')}
                """
                axes[1].text(0.1, 0.5, info_text, fontsize=12, verticalalignment='center')
                axes[1].set_title("System Information")
                axes[1].axis('off')
            
            plt.tight_layout()
            
            # Save to temp file
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            plt.savefig(temp_file.name, dpi=150, bbox_inches="tight")
            plt.close()
            
            return temp_file.name
        
        except Exception as e:
            logger.error(f"Performance visualization error: {str(e)}")
            return None
    
    def create_advanced_interface(self) -> gr.Blocks:
        """Create advanced Gradio interface"""
        with gr.Blocks(title="Music Analyzer AI - Advanced", theme=gr.themes.Soft()) as interface:
            gr.Markdown("# 🎵 Music Analyzer AI - Advanced Features")
            
            with gr.Tabs():
                # Tab 1: Model Comparison
                with gr.Tab("Model Comparison"):
                    audio_input = gr.Audio(
                        label="Upload Audio File",
                        type="filepath"
                    )
                    model_selection = gr.CheckboxGroup(
                        choices=["genre_classifier", "mood_detector", "multi_task", "transformer_encoder"],
                        label="Select Models to Compare",
                        value=["genre_classifier", "mood_detector"]
                    )
                    compare_btn = gr.Button("Compare Models", variant="primary")
                    comparison_output = gr.Markdown(label="Comparison Results")
                    
                    compare_btn.click(
                        self.compare_models,
                        inputs=[audio_input, model_selection],
                        outputs=[comparison_output, gr.Image(visible=False)]
                    )
                
                # Tab 2: Performance Monitoring
                with gr.Tab("Performance Monitoring"):
                    refresh_btn = gr.Button("Refresh Metrics", variant="primary")
                    performance_plot = gr.Image(label="Performance Metrics")
                    
                    refresh_btn.click(
                        self.visualize_performance_metrics,
                        inputs=[],
                        outputs=[performance_plot]
                    )
                
                # Tab 3: Model Information
                with gr.Tab("Model Information"):
                    gr.Markdown("## Available Models")
                    gr.Markdown("""
                    ### Deep Genre Classifier
                    - **Architecture**: 6-8 layer neural network with residual connections
                    - **Input**: 169 features
                    - **Output**: 10 genres
                    - **Parameters**: ~500K
                    
                    ### Deep Mood Detector
                    - **Architecture**: CNN + LSTM (multi-modal)
                    - **Input**: 13 MFCC features
                    - **Output**: 6 moods
                    - **Parameters**: ~1M
                    
                    ### Multi-Task Model
                    - **Architecture**: Shared encoder + 5 task heads
                    - **Tasks**: Genre, Mood, Energy, Complexity, Instruments
                    - **Parameters**: ~2M
                    
                    ### Transformer Encoder
                    - **Architecture**: 4-layer transformer with self-attention
                    - **Attention Heads**: 8
                    - **Parameters**: ~3M
                    """)
            
            gr.Markdown("---")
            gr.Markdown("**Music Analyzer AI v2.4.0** - Advanced Deep Learning Features")
        
        return interface
    
    def launch(self, share: bool = False, server_name: str = "0.0.0.0", server_port: int = 7861):
        """Launch advanced Gradio interface"""
        interface = self.create_advanced_interface()
        interface.launch(
            share=share,
            server_name=server_name,
            server_port=server_port
        )


def create_advanced_gradio_app():
    """Create and return advanced Gradio app"""
    ui = AdvancedGradioUI()
    return ui.create_advanced_interface()

