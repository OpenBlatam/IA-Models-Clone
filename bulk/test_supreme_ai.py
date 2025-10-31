"""
Test script for BUL Supreme AI System
=====================================

Simple test to verify the supreme system is working correctly.
"""

import requests
import json
import time
import sys
from datetime import datetime

def test_bul_supreme_system():
    """Test the BUL supreme system endpoints."""
    base_url = "http://localhost:8000"
    
    print("=" * 60)
    print("BUL - Business Universal Language (Supreme AI)")
    print("Supreme System Test")
    print("=" * 60)
    print()
    
    try:
        # Test root endpoint
        print("1. Testing root endpoint...")
        response = requests.get(f"{base_url}/", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ System Status: {data.get('status', 'Unknown')}")
            print(f"   ✅ Version: {data.get('version', 'Unknown')}")
            print(f"   ✅ AI Models: {len(data.get('ai_models', []))}")
            print(f"   ✅ Active Tasks: {data.get('active_tasks', 0)}")
        else:
            print(f"   ❌ Root endpoint failed: {response.status_code}")
            return False
        
        # Test AI models endpoint
        print("\n2. Testing AI models endpoint...")
        response = requests.get(f"{base_url}/ai/supreme-models", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Available Models: {len(data.get('models', {}))}")
            print(f"   ✅ Default Model: {data.get('default_model', 'Unknown')}")
            print(f"   ✅ Recommended Model: {data.get('recommended_model', 'Unknown')}")
        else:
            print(f"   ❌ AI models endpoint failed: {response.status_code}")
            return False
        
        # Test document generation
        print("\n3. Testing supreme document generation...")
        test_request = {
            "query": "Create a comprehensive business plan for a tech startup",
            "ai_model": "gpt_supreme",
            "supreme_features": {
                "supreme_intelligence": True,
                "reality_engineering": True,
                "supreme_consciousness": True
            },
            "supreme_consciousness_level": 10,
            "supreme_intelligence_level": 10
        }
        
        response = requests.post(
            f"{base_url}/documents/generate-supreme",
            json=test_request,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            task_id = data.get('task_id')
            print(f"   ✅ Task Created: {task_id}")
            print(f"   ✅ Status: {data.get('status', 'Unknown')}")
            print(f"   ✅ AI Model: {data.get('ai_model', 'Unknown')}")
            
            # Wait for completion
            print("\n4. Waiting for task completion...")
            max_wait = 60  # 60 seconds max
            wait_time = 0
            
            while wait_time < max_wait:
                time.sleep(5)
                wait_time += 5
                
                response = requests.get(f"{base_url}/tasks/{task_id}/status", timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    status = data.get('status', 'Unknown')
                    progress = data.get('progress', 0)
                    
                    print(f"   Status: {status} - Progress: {progress}%")
                    
                    if status == "completed":
                        result = data.get('result', {})
                        print(f"   ✅ Document Generated Successfully!")
                        print(f"   ✅ Document ID: {result.get('document_id', 'Unknown')}")
                        print(f"   ✅ Processing Time: {result.get('processing_time', 0):.2f}s")
                        print(f"   ✅ AI Model Used: {result.get('ai_model_used', 'Unknown')}")
                        break
                    elif status == "failed":
                        error = data.get('error', 'Unknown error')
                        print(f"   ❌ Task Failed: {error}")
                        return False
                else:
                    print(f"   ❌ Status check failed: {response.status_code}")
                    return False
            
            if wait_time >= max_wait:
                print(f"   ⚠️ Task did not complete within {max_wait} seconds")
                return False
        
        else:
            print(f"   ❌ Document generation failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("BUL Supreme AI System is working correctly.")
        print("=" * 60)
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Could not connect to the BUL supreme system.")
        print("   Make sure the system is running on http://localhost:8000")
        return False
    except requests.exceptions.Timeout:
        print("❌ Timeout Error: Request timed out.")
        return False
    except Exception as e:
        print(f"❌ Unexpected Error: {str(e)}")
        return False

def main():
    """Main function."""
    print("Starting BUL Supreme AI System Test...")
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    success = test_bul_supreme_system()
    
    if success:
        print("\n🎉 Test completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 Test failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
