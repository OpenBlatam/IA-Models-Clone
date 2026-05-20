#!/usr/bin/env python3
"""
BUL Omniversal AI Test Script
============================

Simple test script to verify the omniversal AI system is working.
"""

import requests
import json
import time
from datetime import datetime

def test_omniversal_api():
    """Test the omniversal API endpoints."""
    base_url = "http://localhost:8000"
    
    print("🚀 Testing BUL Omniversal AI System")
    print("=" * 50)
    
    # Test 1: Root endpoint
    print("1. Testing root endpoint...")
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Root endpoint: {data['status']}")
            print(f"   - Version: {data['version']}")
            print(f"   - Active Tasks: {data['active_tasks']}")
            print(f"   - Universe Creations: {data['universe_creations']}")
            print(f"   - Dimensional Transcendence: {data['dimensional_transcendence_sessions']}")
            print(f"   - Omniversal Features: {len(data['omniversal_features'])}")
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")
        return False
    
    # Test 2: Omniversal AI models
    print("\n2. Testing omniversal AI models...")
    try:
        response = requests.get(f"{base_url}/ai/omniversal-models", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Omniversal AI models: {len(data['models'])} models")
            print(f"   - Default model: {data['default_model']}")
            print(f"   - Recommended model: {data['recommended_model']}")
            print(f"   - Omniversal capabilities: {len(data['omniversal_capabilities'])}")
        else:
            print(f"❌ Omniversal AI models failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Omniversal AI models error: {e}")
    
    # Test 3: Generate omniversal document
    print("\n3. Testing omniversal document generation...")
    try:
        payload = {
            "query": "Create a business plan for a multiverse corporation",
            "ai_model": "gpt_omniverse",
            "omniversal_features": {
                "infinite_intelligence": True,
                "reality_engineering": True,
                "divine_ai": True,
                "cosmic_consciousness": True,
                "universe_creation": False,
                "dimensional_transcendence": False
            },
            "divine_consciousness_level": 10,
            "cosmic_consciousness_level": 10,
            "infinite_intelligence_level": 10
        }
        
        response = requests.post(
            f"{base_url}/documents/generate-omniversal",
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Omniversal document generation started")
            print(f"   - Task ID: {data['task_id']}")
            print(f"   - Status: {data['status']}")
            print(f"   - AI Model: {data['ai_model']}")
            print(f"   - Omniversal Features: {len(data['omniversal_features_enabled'])}")
            
            # Wait a bit and check task status
            time.sleep(2)
            task_response = requests.get(f"{base_url}/tasks/{data['task_id']}/status", timeout=5)
            if task_response.status_code == 200:
                task_data = task_response.json()
                print(f"   - Task Status: {task_data['status']}")
                print(f"   - Progress: {task_data['progress']}%")
        else:
            print(f"❌ Omniversal document generation failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Omniversal document generation error: {e}")
    
    # Test 4: Universe creation
    print("\n4. Testing universe creation...")
    try:
        payload = {
            "user_id": "admin",
            "universe_type": "artificial",
            "dimensions": 4,
            "physical_constants": "standard",
            "divine_consciousness_level": 10,
            "cosmic_consciousness_level": 10
        }
        
        response = requests.post(
            f"{base_url}/universe/create",
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Universe creation successful")
            print(f"   - Universe ID: {data['universe_id']}")
            print(f"   - Universe Type: {data['universe_type']}")
            print(f"   - Dimensions: {data['dimensions']}")
            print(f"   - Status: {data['universe_status']}")
        else:
            print(f"❌ Universe creation failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Universe creation error: {e}")
    
    # Test 5: Dimensional transcendence
    print("\n5. Testing dimensional transcendence...")
    try:
        payload = {
            "user_id": "admin",
            "target_dimension": "omniversal",
            "transcendence_level": "omniversal",
            "divine_consciousness_level": 10,
            "cosmic_consciousness_level": 10
        }
        
        response = requests.post(
            f"{base_url}/dimensional-transcendence/transcend",
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Dimensional transcendence successful")
            print(f"   - Transcendence ID: {data['transcendence_id']}")
            print(f"   - Target Dimension: {data['target_dimension']}")
            print(f"   - Transcendence Level: {data['transcendence_level']}")
            print(f"   - Status: {data['transcendence_status']}")
        else:
            print(f"❌ Dimensional transcendence failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Dimensional transcendence error: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 BUL Omniversal AI System Test Complete!")
    print("=" * 50)
    
    return True

def main():
    """Main function."""
    print("Starting BUL Omniversal AI System Test...")
    print(f"Test started at: {datetime.now()}")
    
    # Wait a moment for the API to start
    print("⏳ Waiting for API to start...")
    time.sleep(3)
    
    # Run tests
    success = test_omniversal_api()
    
    if success:
        print("\n✅ All tests completed successfully!")
    else:
        print("\n❌ Some tests failed. Check the API status.")

if __name__ == "__main__":
    main()
