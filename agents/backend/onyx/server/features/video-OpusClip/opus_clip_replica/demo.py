"""
Opus Clip Replica Demo

Simple demo script showing how to use the Opus Clip replica API.
"""

import asyncio
import requests
import json
from pathlib import Path

# API base URL
API_BASE = "http://localhost:8000"

def test_health():
    """Test API health."""
    try:
        response = requests.get(f"{API_BASE}/health")
        print("Health Check:")
        print(json.dumps(response.json(), indent=2))
        print()
    except Exception as e:
        print(f"Health check failed: {e}")

def test_viral_score():
    """Test viral score calculation."""
    try:
        data = {
            "content": "This is an amazing video that will go viral! Check out this incredible content that everyone needs to see!",
            "platform": "tiktok"
        }
        
        response = requests.post(f"{API_BASE}/viral-score", json=data)
        print("Viral Score Test:")
        print(json.dumps(response.json(), indent=2))
        print()
    except Exception as e:
        print(f"Viral score test failed: {e}")

def test_video_analysis():
    """Test video analysis (requires video file)."""
    try:
        # This would require an actual video file
        data = {
            "video_file": "sample_video.mp4",  # Replace with actual video path
            "max_clips": 5,
            "min_duration": 3.0,
            "max_duration": 30.0
        }
        
        response = requests.post(f"{API_BASE}/analyze", json=data)
        print("Video Analysis Test:")
        print(json.dumps(response.json(), indent=2))
        print()
    except Exception as e:
        print(f"Video analysis test failed: {e}")

def test_export():
    """Test clip export."""
    try:
        data = {
            "clips": [
                {
                    "clip_id": "segment_1",
                    "start_time": 10.0,
                    "end_time": 25.0,
                    "duration": 15.0,
                    "engagement_score": 0.8
                },
                {
                    "clip_id": "segment_2", 
                    "start_time": 45.0,
                    "end_time": 60.0,
                    "duration": 15.0,
                    "engagement_score": 0.7
                }
            ],
            "platform": "youtube",
            "format": "mp4",
            "quality": "high"
        }
        
        response = requests.post(f"{API_BASE}/export", json=data)
        print("Export Test:")
        print(json.dumps(response.json(), indent=2))
        print()
    except Exception as e:
        print(f"Export test failed: {e}")

def main():
    """Run all tests."""
    print("🎬 Opus Clip Replica Demo")
    print("=" * 50)
    print()
    
    # Test health
    test_health()
    
    # Test viral score
    test_viral_score()
    
    # Test video analysis (commented out - requires video file)
    # test_video_analysis()
    
    # Test export
    test_export()
    
    print("✅ Demo completed!")

if __name__ == "__main__":
    main()


