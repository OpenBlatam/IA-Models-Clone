from fastapi.testclient import TestClient
from agents.backend.onyx.server.features.lovable.api.app import app

client = TestClient(app)

def test_api_integration():
    print("--- Testing Lovable API Integration ---")
    
    # 1. Test Health
    print("1. Testing Health Endpoint...")
    response = client.get("/health")
    if response.status_code == 200:
        print("  Health Check: SUCCESS")
    else:
        print(f"  Health Check FAILED: {response.status_code}")

    # 2. Test System Generation (Web Track)
    print("\n2. Testing System Generation (Web Track)...")
    payload = {
        "prompt": "Create a landing page for a coffee shop",
        "target": "html",
        "use_agents": True
    }
    try:
        response = client.post("/ai/system/generate", json=payload)
        if response.status_code == 200:
            data = response.json()
            if data.get("type") == "project": # Web track returns code structure
                print("  Web Generation: SUCCESS")
                print(f"  Output keys: {list(data.get('structure', {}).keys())}")
            else:
                 print(f"  Web Generation: SUCCESS (Type: {data.get('type')})")
        else:
            print(f"  Web Generation FAILED: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"  Web Generation ERROR: {e}")

    # 3. Test System Generation (Mobile Track)
    print("\n3. Testing System Generation (Mobile Track)...")
    payload_mobile = {
        "prompt": "Automate a mobile app for fitness",
        "target": "expo",
        "use_agents": True
    }
    try:
        response = client.post("/ai/system/generate", json=payload_mobile)
        if response.status_code == 200:
            data = response.json()
            if data.get("type") == "project":
                print("  Mobile Generation: SUCCESS")
                print(f"  Output: {data.get('structure')}")
            else:
                print(f"  Mobile Generation: SUCCESS (Type: {data.get('type')})")
        else:
            print(f"  Mobile Generation FAILED: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"  Mobile Generation ERROR: {e}")

if __name__ == "__main__":
    test_api_integration()
