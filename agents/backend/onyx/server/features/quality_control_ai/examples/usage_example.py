"""
Usage Example

Complete example showing how to use the refactored Quality Control AI system.
"""

import numpy as np
from quality_control_ai import (
    ApplicationServiceFactory,
    InspectionRequest,
    create_app,
)

def example_basic_inspection():
    """Basic inspection example."""
    print("=== Basic Inspection Example ===")
    
    # Create factory
    factory = ApplicationServiceFactory()
    
    # Get inspection service
    inspection_service = factory.create_inspection_application_service()
    
    # Create a sample image (in real usage, load from file or camera)
    sample_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Create request
    request = InspectionRequest(
        image_data=sample_image,
        image_format="numpy",
        include_visualization=False,
    )
    
    # Inspect
    try:
        response = inspection_service.inspect_image(request)
        print(f"Inspection ID: {response.inspection_id}")
        print(f"Quality Score: {response.quality_score:.2f}")
        print(f"Quality Status: {response.quality_status}")
        print(f"Defects: {len(response.defects)}")
        print(f"Anomalies: {len(response.anomalies)}")
        print(f"Is Acceptable: {response.is_acceptable}")
        print(f"Recommendation: {response.recommendation}")
    except Exception as e:
        print(f"Error: {e}")


def example_batch_inspection():
    """Batch inspection example."""
    print("\n=== Batch Inspection Example ===")
    
    factory = ApplicationServiceFactory()
    inspection_service = factory.create_inspection_application_service()
    
    # Create multiple sample images
    images = [
        np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        for _ in range(5)
    ]
    
    # Create batch request
    from quality_control_ai import BatchInspectionRequest
    
    batch_request = BatchInspectionRequest(
        images=[
            InspectionRequest(image_data=img, image_format="numpy")
            for img in images
        ],
        parallel=True,
        max_workers=4,
    )
    
    # Inspect batch
    try:
        batch_response = inspection_service.inspect_batch(batch_request)
        print(f"Total Processed: {batch_response.total_processed}")
        print(f"Total Succeeded: {batch_response.total_succeeded}")
        print(f"Total Failed: {batch_response.total_failed}")
        print(f"Average Quality Score: {batch_response.average_quality_score:.2f}")
    except Exception as e:
        print(f"Error: {e}")


def example_api_usage():
    """API usage example."""
    print("\n=== API Usage Example ===")
    
    # Create FastAPI app
    app = create_app()
    
    print("FastAPI app created!")
    print("Start with: uvicorn quality_control_ai.presentation.api:app")
    print("API docs at: http://localhost:8000/docs")


def example_camera_stream():
    """Camera stream example."""
    print("\n=== Camera Stream Example ===")
    
    factory = ApplicationServiceFactory()
    inspection_service = factory.create_inspection_application_service()
    
    try:
        # Start stream
        stream_info = inspection_service.start_inspection_stream(
            camera_index=0,
            resolution=(1920, 1080),
            fps=30,
        )
        print(f"Stream started: {stream_info}")
        
        # Later, stop stream
        # stop_info = inspection_service.stop_inspection_stream(camera_index=0)
        # print(f"Stream stopped: {stop_info}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    example_basic_inspection()
    example_batch_inspection()
    example_api_usage()
    example_camera_stream()



