"""
Example: Gradio Interface for Quality Control AI
"""

from quality_control_ai import (
    QualityInspector,
    CameraConfig,
    DetectionConfig,
    create_gradio_app
)


def main():
    """Launch Gradio interface"""
    print("Initializing Quality Control AI...")
    
    # Create inspector
    camera_config = CameraConfig()
    detection_config = DetectionConfig()
    
    inspector = QualityInspector(camera_config, detection_config)
    
    # Create Gradio app
    app = create_gradio_app(inspector)
    
    # Launch
    print("Launching Gradio interface...")
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )


if __name__ == "__main__":
    main()

