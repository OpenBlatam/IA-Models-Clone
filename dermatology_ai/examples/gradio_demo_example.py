"""
Gradio Demo Example
Interactive demo for skin analysis
"""

from ml import ViTSkinAnalyzer
from ml.visualization import GradioDemo
from core.skin_analyzer import SkinAnalyzer

def main():
    """Create and launch Gradio demo"""
    
    # 1. Create analyzer
    analyzer = SkinAnalyzer(use_advanced=True)
    
    # 2. Create demo
    demo = GradioDemo(
        analyzer=analyzer,
        title="Dermatology AI - Skin Analysis",
        description="Upload an image to analyze skin quality and get recommendations"
    )
    
    # 3. Launch
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )

if __name__ == "__main__":
    main()













