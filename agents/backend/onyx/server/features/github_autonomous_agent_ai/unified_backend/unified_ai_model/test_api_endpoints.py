import requests
import json
import time

BASE_URL = "http://localhost:8050/api/v1"

def test_health():
    print("\nTesting Health Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")

def test_chat():
    print("\nTesting Chat Endpoint...")
    payload = {
        "message": "Hello, are you working?",
        "stream": False
    }
    try:
        response = requests.post(f"{BASE_URL}/chat", json=payload)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {json.dumps(response.json(), indent=2)}")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("=== Verifying Unified AI Model API ===")
    test_health()
    test_chat()
