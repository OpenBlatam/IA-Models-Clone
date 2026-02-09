"""
Gradio Interface for Music Analyzer AI
Interactive web interface for music analysis and visualization
"""

from typing import Optional, Tuple
import logging
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    logger.warning("Gradio not available")

try:
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class MusicAnalyzerGradioUI:
    """
    Gradio interface for Music Analyzer AI:
    - Audio upload and analysis
    - Real-time visualization
    - Model inference
    - Results display
    """
    
    def __init__(self):
        if not GRADIO_AVAILABLE:
            raise ImportError("Gradio is required for UI")
        
        self.setup_services()
    
    def setup_services(self):
        """Setup ML services"""
        try:
            from ..services.ml_service import get_ml_service
            from ..services.spotify_service import SpotifyService
            from ..core.ml_audio_analyzer import AudioFeatureExtractor
            
            self.ml_service = get_ml_service()
            self.spotify_service = SpotifyService()
            self.feature_extractor = AudioFeatureExtractor()
        except Exception as e:
            logger.warning(f"Could not setup services: {str(e)}")
            self.ml_service = None
    
    def analyze_audio_file(self, audio_file) -> Tuple[str, Optional[str]]:
        """Analyze uploaded audio file"""
        if audio_file is None:
            return "Please upload an audio file", None
        
        try:
            # Run comprehensive analysis
            result = self.ml_service.analyze_track_comprehensive(
                audio_path=audio_file.name,
                use_pipeline=True
            )
            
            if not result["success"]:
                return f"Error: {result.get('error', 'Unknown error')}", None
            
            # Format results
            analysis = result.get("analysis", {})
            output_text = self._format_analysis(analysis)
            
            # Create visualization
            plot_path = self._create_visualization(analysis)
            
            return output_text, plot_path
        
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}", exc_info=True)
            return f"Error analyzing audio: {str(e)}", None
    
    def analyze_spotify_track(self, track_id: str) -> Tuple[str, Optional[str]]:
        """Analyze track from Spotify"""
        if not track_id:
            return "Please enter a Spotify track ID", None
        
        try:
            # Get Spotify data
            spotify_data = self.spotify_service.get_track_full_analysis(track_id)
            spotify_features = spotify_data.get("audio_features", {})
            
            # Run ML analysis
            result = self.ml_service.analyze_track_comprehensive(
                spotify_features=spotify_features,
                use_pipeline=False
            )
            
            if not result["success"]:
                return f"Error: {result.get('error', 'Unknown error')}", None
            
            # Format results
            analysis = result.get("analysis", {})
            output_text = self._format_analysis(analysis, spotify_data)
            
            # Create visualization
            plot_path = self._create_visualization(analysis)
            
            return output_text, plot_path
        
        except Exception as e:
            logger.error(f"Spotify analysis error: {str(e)}", exc_info=True)
            return f"Error analyzing track: {str(e)}", None
    
    def _format_analysis(self, analysis: dict, spotify_data: Optional[dict] = None) -> str:
        """Format analysis results as text"""
        lines = ["# Music Analysis Results\n"]
        
        # Multi-task results
        if "multi_task" in analysis:
            mt = analysis["multi_task"]
            lines.append("## Multi-Task Predictions\n")
            lines.append(f"**Genre ID**: {mt.get('genre', {}).get('predicted', 'N/A')}\n")
            lines.append(f"**Mood ID**: {mt.get('mood', {}).get('predicted', 'N/A')}\n")
            lines.append(f"**Energy**: {mt.get('energy', 0):.2f}\n")
            lines.append(f"**Complexity**: {mt.get('complexity', 0):.2f}\n")
        
        # ML Analysis
        if "ml_analysis" in analysis:
            ml = analysis["ml_analysis"]
            lines.append("\n## ML Analysis\n")
            lines.append(f"**Genre**: {ml.get('genre', 'N/A')}\n")
            lines.append(f"**Mood**: {ml.get('mood', 'N/A')}\n")
            lines.append(f"**Energy Level**: {ml.get('energy', 0):.2f}\n")
            lines.append(f"**Complexity Score**: {ml.get('complexity', 0):.2f}\n")
            if ml.get('instruments'):
                lines.append(f"**Instruments**: {', '.join(ml['instruments'])}\n")
        
        # Spotify data
        if spotify_data:
            track_info = spotify_data.get("track_info", {})
            lines.append("\n## Track Information\n")
            lines.append(f"**Name**: {track_info.get('name', 'N/A')}\n")
            lines.append(f"**Artists**: {', '.join(track_info.get('artists', []))}\n")
        
        return "\n".join(lines)
    
    def _create_visualization(self, analysis: dict) -> Optional[str]:
        """Create visualization of analysis results"""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle("Music Analysis Visualization", fontsize=16)
            
            # Energy and Complexity
            if "multi_task" in analysis:
                mt = analysis["multi_task"]
                axes[0, 0].bar(
                    ["Energy", "Complexity"],
                    [mt.get("energy", 0), mt.get("complexity", 0)],
                    color=["#FF6B6B", "#4ECDC4"]
                )
                axes[0, 0].set_title("Energy & Complexity")
                axes[0, 0].set_ylim(0, 1)
            
            # Genre probabilities
            if "genre" in analysis:
                genre_probs = analysis["genre"].get("probabilities", [])
                if genre_probs:
                    genres = [f"G{i}" for i in range(len(genre_probs))]
                    axes[0, 1].bar(genres, genre_probs)
                    axes[0, 1].set_title("Genre Probabilities")
                    axes[0, 1].set_ylabel("Probability")
            
            # Mood distribution
            if "multi_task" in analysis:
                mood_logits = analysis["multi_task"].get("mood", {}).get("logits", [])
                if mood_logits:
                    moods = [f"M{i}" for i in range(len(mood_logits))]
                    axes[1, 0].bar(moods, mood_logits)
                    axes[1, 0].set_title("Mood Distribution")
                    axes[1, 0].set_ylabel("Score")
            
            # Instrument detection
            if "multi_task" in analysis:
                instruments = analysis["multi_task"].get("instruments", {})
                if instruments:
                    inst_probs = instruments.get("probabilities", [])
                    if inst_probs:
                        inst_names = [f"I{i}" for i in range(len(inst_probs))]
                        axes[1, 1].barh(inst_names, inst_probs)
                        axes[1, 1].set_title("Instrument Detection")
                        axes[1, 1].set_xlabel("Probability")
            
            plt.tight_layout()
            
            # Save to temp file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            plt.savefig(temp_file.name, dpi=150, bbox_inches="tight")
            plt.close()
            
            return temp_file.name
        
        except Exception as e:
            logger.error(f"Visualization error: {str(e)}")
            return None
    
    def create_interface(self) -> gr.Blocks:
        """Create Gradio interface"""
        with gr.Blocks(title="Music Analyzer AI", theme=gr.themes.Soft()) as interface:
            gr.Markdown("# 🎵 Music Analyzer AI - Advanced ML Analysis")
            
            with gr.Tabs():
                # Tab 1: Audio File Analysis
                with gr.Tab("Audio File Analysis"):
                    audio_input = gr.Audio(
                        label="Upload Audio File",
                        type="filepath"
                    )
                    analyze_btn = gr.Button("Analyze Audio", variant="primary")
                    output_text = gr.Markdown(label="Analysis Results")
                    output_plot = gr.Image(label="Visualization")
                    
                    analyze_btn.click(
                        self.analyze_audio_file,
                        inputs=[audio_input],
                        outputs=[output_text, output_plot]
                    )
                
                # Tab 2: Spotify Track Analysis
                with gr.Tab("Spotify Track Analysis"):
                    track_id_input = gr.Textbox(
                        label="Spotify Track ID",
                        placeholder="Enter track ID (e.g., 4uLU6hMCjMI75M1A2tKUQC)"
                    )
                    spotify_analyze_btn = gr.Button("Analyze Track", variant="primary")
                    spotify_output_text = gr.Markdown(label="Analysis Results")
                    spotify_output_plot = gr.Image(label="Visualization")
                    
                    spotify_analyze_btn.click(
                        self.analyze_spotify_track,
                        inputs=[track_id_input],
                        outputs=[spotify_output_text, spotify_output_plot]
                    )
                
                # Tab 3: Model Information
                with gr.Tab("Model Information"):
                    gr.Markdown("## Available Models")
                    gr.Markdown("""
                    - **Deep Genre Classifier**: 6-8 layer neural network
                    - **Deep Mood Detector**: CNN + LSTM architecture
                    - **Multi-Task Model**: Shared encoder with 5 task heads
                    - **Transformer Encoder**: 4-layer transformer with attention
                    """)
            
            gr.Markdown("---")
            gr.Markdown("**Music Analyzer AI v2.1.0** - Powered by Deep Learning")
        
        return interface
    
    def launch(self, share: bool = False, server_name: str = "0.0.0.0", server_port: int = 7860):
        """Launch Gradio interface"""
        interface = self.create_interface()
        interface.launch(
            share=share,
            server_name=server_name,
            server_port=server_port
        )


def create_gradio_app():
    """Create and return Gradio app"""
    ui = MusicAnalyzerGradioUI()
    return ui.create_interface()

