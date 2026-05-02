#!/usr/bin/env python3
"""
Script to check if the Onyx backend is running and accessible.
"""

import requests
import sys
import os

API_URL = os.getenv("API_SERVER_URL", "http://localhost:3000/api")
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080")

def check_backend():
    """Check if backend is accessible."""
    
    print("Checking Onyx backend status...")
    print(f"Frontend API URL: {API_URL}")
    print(f"Direct Backend URL: {BACKEND_URL}")
    print()
    
    # Try frontend proxy first
    print(f"1. Checking frontend API proxy ({API_URL}/health)...")
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            print("   ✅ Frontend API proxy is accessible")
            return True
        else:
            print(f"   ⚠️  Frontend API proxy responded with status {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("   ❌ Cannot connect to frontend API proxy")
    except requests.exceptions.Timeout:
        print("   ⏱️  Frontend API proxy timed out")
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
    
    # Try direct backend
    print(f"\n2. Checking direct backend ({BACKEND_URL}/health)...")
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if response.status_code == 200:
            print("   ✅ Direct backend is accessible")
            print(f"\n⚠️  Note: Frontend proxy might not be running, but backend is.")
            return True
        else:
            print(f"   ⚠️  Backend responded with status {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("   ❌ Cannot connect to direct backend")
    except requests.exceptions.Timeout:
        print("   ⏱️  Backend timed out")
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
    
    print("\n❌ Backend is not accessible.")
    print("\nTo start the backend:")
    print("  - Docker: docker-compose up")
    print("  - Local: Check backend/README.md for setup instructions")
    print("  - Verify logs in backend/log/api_server_debug.log")
    
    return False

if __name__ == "__main__":
    success = check_backend()
    sys.exit(0 if success else 1)


















