"""
API Usage Example
=================

Example of using the API endpoints.
"""

import requests
import base64
from pathlib import Path


def example_api_upscale():
    """Example: Upscale via API."""
    url = "http://localhost:8003/api/v1/enhanced/upscale"
    
    # Read image
    with open("input.jpg", "rb") as f:
        files = {"image": ("input.jpg", f, "image/jpeg")}
        data = {
            "scale_factor": 4.0,
            "use_recommendations": "true",
            "validate_quality": "true"
        }
        
        response = requests.post(url, files=files, data=data)
        result = response.json()
        
        if result["success"]:
            # Decode and save image
            img_data = base64.b64decode(result["upscaled_image_base64"])
            with open("output_api.jpg", "wb") as out:
                out.write(img_data)
            
            print(f"✅ Upscaled via API")
            print(f"Quality: {result['quality_score']:.2f}")
            print(f"Time: {result['processing_time']:.2f}s")
        else:
            print(f"❌ Error: {result.get('error')}")


def example_api_recommendations():
    """Example: Get recommendations via API."""
    url = "http://localhost:8003/api/v1/enhanced/recommendations"
    
    with open("input.jpg", "rb") as f:
        files = {"image": ("input.jpg", f, "image/jpeg")}
        data = {
            "target_scale": 4.0,
            "prioritize_speed": "false"
        }
        
        response = requests.post(url, files=files, data=data)
        recommendation = response.json()
        
        print("Recommendation:")
        print(f"  Method: {recommendation['method']}")
        print(f"  Expected Quality: {recommendation['expected_quality']:.2f}")
        print(f"  Expected Time: {recommendation['expected_time']:.2f}s")
        print(f"  Confidence: {recommendation['confidence']:.2f}")


def example_api_status():
    """Example: Get system status via API."""
    url = "http://localhost:8003/api/v1/enhanced/status"
    
    response = requests.get(url)
    status = response.json()
    
    print("System Status:")
    print(f"  Total operations: {status['system_metrics']['total_operations']}")
    print(f"  Success rate: {status['system_metrics']['success_rate']:.2%}")
    print(f"  Throughput: {status['system_metrics']['throughput']:.2f} ops/s")
    print(f"  Cache hit rate: {status['cache']['hit_rate']:.2%}")


if __name__ == "__main__":
    print("API Examples")
    print("="*60)
    
    try:
        example_api_status()
        print()
        example_api_recommendations()
        print()
        example_api_upscale()
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Make sure the server is running:")
        print("   uvicorn image_upscaling_ai.api.upscaling_api:app --host 0.0.0.0 --port 8003")


