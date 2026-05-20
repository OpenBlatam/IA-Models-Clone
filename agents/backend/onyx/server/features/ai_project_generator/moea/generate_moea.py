"""Script to generate MOEA project via API"""
import requests
import json
import time
import sys

API_URL = "http://localhost:8020"

def wait_for_server(max_attempts=15, delay=1):
    """Wait for server to be ready"""
    print("Waiting for server to be ready...")
    for i in range(max_attempts):
        try:
            response = requests.get(f"{API_URL}/health", timeout=2)
            if response.status_code == 200:
                print("✅ Server is ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(delay)
        if (i + 1) % 3 == 0:
            print(f"   Attempt {i+1}/{max_attempts}...")
    return False

def generate_moea_project():
    """Generate MOEA project via API"""
    if not wait_for_server():
        print("❌ Server not responding. Make sure it's running.")
        print("   Start server with: python main.py")
        return False
    
    # Generate MOEA project
    print("\n🚀 Generating MOEA project...")
    print("   This may take a few minutes...")
    
    try:
        response = requests.post(
            f"{API_URL}/api/v1/generate",
            json={
                "description": (
                    "A Multi-Objective Evolutionary Algorithm (MOEA) system for solving "
                    "optimization problems with multiple conflicting objectives. The system should "
                    "support various MOEA algorithms like NSGA-II, NSGA-III, MOEA/D, and SPEA2. "
                    "It should include visualization of Pareto fronts, performance metrics "
                    "calculation (hypervolume, IGD, GD), comparison tools, and interactive "
                    "parameter tuning. The system should handle real-time optimization, batch "
                    "processing, and export results in various formats."
                ),
                "project_name": "moea_optimization_system",
                "author": "Blatam Academy",
                "priority": 5,
                "generate_tests": True,
                "include_docker": True,
                "include_docs": True
            },
            timeout=300  # 5 minutes timeout
        )
        
        if response.status_code == 200:
            data = response.json()
            print("\n✅ Project queued successfully!")
            print(f"   Project ID: {data.get('project_id', 'N/A')}")
            print(f"   Status: {data.get('status', 'N/A')}")
            print(f"   Message: {data.get('message', 'N/A')}")
            
            project_id = data.get('project_id')
            if project_id:
                print(f"\n📊 Monitoring project generation...")
                print(f"   Check status: {API_URL}/api/v1/project/{project_id}")
                print(f"   Or visit: {API_URL}/dashboard")
                
                # Wait a bit and check status
                time.sleep(2)
                try:
                    status_response = requests.get(
                        f"{API_URL}/api/v1/project/{project_id}",
                        timeout=10
                    )
                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        print(f"\n   Current status: {status_data.get('status', 'unknown')}")
                except:
                    pass
            
            return True
        else:
            print(f"\n❌ Error: {response.status_code}")
            print(response.text)
            return False
            
    except requests.exceptions.Timeout:
        print("\n❌ Request timeout. The project might still be generating.")
        print("   Check the server logs or dashboard for status.")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = generate_moea_project()
    sys.exit(0 if success else 1)

