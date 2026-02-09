"""Example usage of the Plastic Surgery Visualization AI API."""

import requests
import json
from pathlib import Path


# API base URL
BASE_URL = "http://localhost:8025"


def example_create_visualization_from_url():
    """Example: Create visualization from image URL."""
    url = f"{BASE_URL}/api/v1/visualize"
    
    payload = {
        "surgery_type": "rhinoplasty",
        "intensity": 0.7,
        "image_url": "https://example.com/user-photo.jpg",
        "target_areas": ["nose"],
        "additional_notes": "Make nose slightly smaller and more refined"
    }
    
    response = requests.post(url, json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nVisualization ID: {result['visualization_id']}")
        print(f"Image URL: {result['image_url']}")
        print(f"Processing time: {result.get('processing_time', 'N/A')}s")


def example_upload_image():
    """Example: Upload image and create visualization."""
    url = f"{BASE_URL}/api/v1/visualize/upload"
    
    # Path to your image file
    image_path = Path("path/to/your/image.jpg")
    
    if not image_path.exists():
        print(f"Image file not found: {image_path}")
        return
    
    with open(image_path, "rb") as f:
        files = {"file": (image_path.name, f, "image/jpeg")}
        data = {
            "surgery_type": "facelift",
            "intensity": 0.6
        }
        
        response = requests.post(url, files=files, data=data)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")


def example_get_visualization():
    """Example: Get a previously created visualization."""
    visualization_id = "your-visualization-id-here"
    url = f"{BASE_URL}/api/v1/visualize/{visualization_id}"
    
    response = requests.get(url)
    
    if response.status_code == 200:
        # Save the image
        output_path = Path(f"output_{visualization_id}.png")
        with open(output_path, "wb") as f:
            f.write(response.content)
        print(f"Image saved to: {output_path}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)


def example_get_surgery_types():
    """Example: Get available surgery types."""
    url = f"{BASE_URL}/api/v1/surgery-types"
    
    response = requests.get(url)
    print(f"Status: {response.status_code}")
    print(f"Available surgery types:")
    print(json.dumps(response.json(), indent=2))


def example_health_check():
    """Example: Check service health."""
    url = f"{BASE_URL}/health/"
    
    response = requests.get(url)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


if __name__ == "__main__":
    print("Plastic Surgery Visualization AI - Example Usage\n")
    
    # Check health
    print("1. Health Check:")
    example_health_check()
    print("\n")
    
    # Get surgery types
    print("2. Available Surgery Types:")
    example_get_surgery_types()
    print("\n")
    
    # Create visualization (uncomment to use)
    # print("3. Create Visualization:")
    # example_create_visualization_from_url()
    # print("\n")
    
    # Upload image (uncomment to use)
    # print("4. Upload Image:")
    # example_upload_image()
    # print("\n")

