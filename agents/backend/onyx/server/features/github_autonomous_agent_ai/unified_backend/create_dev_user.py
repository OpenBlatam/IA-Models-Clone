#!/usr/bin/env python3
"""
Script to create a development user for Onyx.
This creates a user with email 'a@test.com' and password 'a' for local development.
"""

import requests
import sys
import os

# Add the backend directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Default development user credentials
DEV_EMAIL = "a@test.com"
DEV_PASSWORD = "a"

# Default API URL (frontend proxy)
API_URL = os.getenv("API_SERVER_URL", "http://localhost:3000/api")

def create_dev_user():
    """Create the development user if it doesn't exist."""
    
    print(f"Attempting to create development user: {DEV_EMAIL}")
    print(f"Using API URL: {API_URL}")
    
    register_url = f"{API_URL}/auth/register"
    
    body = {
        "email": DEV_EMAIL,
        "username": DEV_EMAIL,
        "password": DEV_PASSWORD,
    }
    
    try:
        response = requests.post(
            url=register_url,
            json=body,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 200:
            user_data = response.json()
            print(f"✅ Successfully created development user!")
            print(f"   User ID: {user_data.get('id', 'N/A')}")
            print(f"   Email: {user_data.get('email', DEV_EMAIL)}")
            print(f"\nYou can now login with:")
            print(f"   Email: {DEV_EMAIL}")
            print(f"   Password: {DEV_PASSWORD}")
            return True
        elif response.status_code == 400:
            error_detail = response.json().get("detail", {})
            if isinstance(error_detail, dict) and error_detail.get("reason") == "REGISTER_USER_ALREADY_EXISTS":
                print(f"ℹ️  User {DEV_EMAIL} already exists.")
                print(f"\nYou can login with:")
                print(f"   Email: {DEV_EMAIL}")
                print(f"   Password: {DEV_PASSWORD}")
                return True
            else:
                print(f"❌ Failed to create user. Error: {error_detail}")
                return False
        else:
            print(f"❌ Failed to create user. Status: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Error: Could not connect to the API server.")
        print(f"   Make sure the backend is running at {API_URL}")
        print(f"   Try: docker-compose up or check your backend logs")
        return False
    except requests.exceptions.Timeout:
        print(f"❌ Error: Request timed out. The server might be slow to respond.")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    success = create_dev_user()
    sys.exit(0 if success else 1)


















