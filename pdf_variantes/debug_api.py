#!/usr/bin/env python3
"""
API Debugging Tools
==================
Interactive debugging tools for the API.
"""

import os
import sys
import json
import requests
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


class APIDebugger:
    """Debugging tools for the API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Authorization": "Bearer debug_token_123"
        })
        self.request_history: list = []
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health."""
        try:
            response = self.session.get(f"{self.base_url}/health")
            return {
                "status": response.status_code,
                "data": response.json() if response.status_code == 200 else response.text,
                "headers": dict(response.headers),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "error": str(e),
                "status": "unreachable",
                "timestamp": datetime.now().isoformat()
            }
    
    def test_endpoint(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Test an API endpoint."""
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method.upper() == "GET":
                response = self.session.get(url, params=data)
            elif method.upper() == "POST":
                if files:
                    response = self.session.post(url, files=files)
                else:
                    response = self.session.post(url, json=data)
            elif method.upper() == "PUT":
                response = self.session.put(url, json=data)
            elif method.upper() == "DELETE":
                response = self.session.delete(url)
            else:
                return {"error": f"Unsupported method: {method}"}
            
            result = {
                "method": method.upper(),
                "endpoint": endpoint,
                "status": response.status_code,
                "headers": dict(response.headers),
                "timestamp": datetime.now().isoformat()
            }
            
            # Try to parse JSON
            try:
                result["data"] = response.json()
            except:
                result["data"] = response.text[:500]  # First 500 chars
            
            # Add to history
            self.request_history.append(result)
            
            return result
        except Exception as e:
            return {
                "error": str(e),
                "method": method.upper(),
                "endpoint": endpoint,
                "timestamp": datetime.now().isoformat()
            }
    
    def test_upload(self, file_path: Path) -> Dict[str, Any]:
        """Test file upload."""
        if not file_path.exists():
            return {"error": f"File not found: {file_path}"}
        
        with open(file_path, "rb") as f:
            files = {"file": (file_path.name, f, "application/pdf")}
            return self.test_endpoint("POST", "/pdf/upload", files=files)
    
    def test_variant(self, file_id: str, variant_type: str = "summary") -> Dict[str, Any]:
        """Test variant generation."""
        data = {
            "variant_type": variant_type,
            "options": {}
        }
        return self.test_endpoint("POST", f"/pdf/{file_id}/variants", data=data)
    
    def test_topics(self, file_id: str) -> Dict[str, Any]:
        """Test topic extraction."""
        return self.test_endpoint("GET", f"/pdf/{file_id}/topics")
    
    def test_preview(self, file_id: str, page_number: int = 1) -> Dict[str, Any]:
        """Test preview."""
        return self.test_endpoint("GET", f"/pdf/{file_id}/preview?page_number={page_number}")
    
    def get_request_history(self) -> list:
        """Get request history."""
        return self.request_history
    
    def save_history(self, file_path: Path):
        """Save request history to file."""
        with open(file_path, "w") as f:
            json.dump(self.request_history, f, indent=2)
        print(f"✅ History saved to {file_path}")
    
    def print_summary(self):
        """Print summary of requests."""
        print("\n" + "=" * 60)
        print("📊 Request Summary")
        print("=" * 60)
        
        if not self.request_history:
            print("No requests made yet")
            return
        
        for i, req in enumerate(self.request_history, 1):
            status = req.get("status", "N/A")
            method = req.get("method", "N/A")
            endpoint = req.get("endpoint", "N/A")
            
            status_emoji = "✅" if 200 <= status < 300 else "❌" if status >= 400 else "⚠️"
            
            print(f"{i}. {status_emoji} {method} {endpoint} - Status: {status}")
        
        print("=" * 60)


def interactive_debug():
    """Interactive debugging session with enhanced features."""
    debugger = APIDebugger()
    
    # Colors for better UX
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    def print_header(text):
        print(f"\n{CYAN}{'=' * 70}{RESET}")
        print(f"{BOLD}{CYAN}{text}{RESET}")
        print(f"{CYAN}{'=' * 70}{RESET}")
    
    def print_success(text):
        print(f"{GREEN}✅ {text}{RESET}")
    
    def print_error(text):
        print(f"{RED}❌ {text}{RESET}")
    
    def print_info(text):
        print(f"{BLUE}ℹ️  {text}{RESET}")
    
    print_header("🐛 API Debugging Tool - Enhanced")
    print(f"{BOLD}Commands:{RESET}")
    print("  health          - Check API health")
    print("  upload <file>    - Upload a PDF file")
    print("  variant <id>     - Generate variant for file")
    print("  topics <id>      - Extract topics from file")
    print("  preview <id>     - Get preview of file")
    print("  test <method> <endpoint> [data] - Test custom endpoint")
    print("  history         - Show request history")
    print("  save <file>      - Save history to file")
    print("  summary         - Show summary")
    print("  clear           - Clear history")
    print("  help            - Show this help")
    print("  quit/exit        - Exit")
    print()
    
    # Check health first
    print_info("Checking API health...")
    health = debugger.health_check()
    if health.get("status") == 200:
        data = health.get("data", {})
        print_success(f"API is healthy (Status: {data.get('status', 'ok')})")
        if "version" in data:
            print_info(f"Version: {data.get('version')}")
    else:
        print_error(f"API health check failed: {health.get('error', 'Unknown error')}")
        print_info("Make sure the API is running on http://localhost:8000")
    print()
    
    while True:
        try:
            command = input(f"{CYAN}debug>{RESET} ").strip()
            
            if not command:
                continue
            
            parts = command.split()
            cmd = parts[0].lower()
            
            if cmd in ["quit", "exit", "q"]:
                print_success("Goodbye!")
                break
            
            elif cmd == "help":
                print_header("Help - Available Commands")
                print("See the command list above or type a command to see usage.")
            
            elif cmd == "health":
                print_info("Checking API health...")
                result = debugger.health_check()
                if result.get("status") == 200:
                    print_success("API is healthy")
                    print(json.dumps(result, indent=2))
                else:
                    print_error("API health check failed")
                    print(json.dumps(result, indent=2))
            
            elif cmd == "upload" and len(parts) > 1:
                file_path = Path(parts[1])
                if not file_path.exists():
                    print_error(f"File not found: {file_path}")
                    continue
                print_info(f"Uploading {file_path.name}...")
                result = debugger.test_upload(file_path)
                if result.get("status") == 200:
                    print_success("Upload successful")
                    file_id = result.get("data", {}).get("file_id") or result.get("data", {}).get("id")
                    if file_id:
                        print_info(f"File ID: {file_id}")
                else:
                    print_error(f"Upload failed: {result.get('status')}")
                print(json.dumps(result, indent=2))
            
            elif cmd == "variant" and len(parts) > 1:
                file_id = parts[1]
                variant_type = parts[2] if len(parts) > 2 else "summary"
                print_info(f"Generating {variant_type} variant for {file_id}...")
                result = debugger.test_variant(file_id, variant_type)
                if result.get("status") in [200, 201, 202]:
                    print_success("Variant generation started")
                else:
                    print_error(f"Variant generation failed: {result.get('status')}")
                print(json.dumps(result, indent=2))
            
            elif cmd == "topics" and len(parts) > 1:
                file_id = parts[1]
                print_info(f"Extracting topics from {file_id}...")
                result = debugger.test_topics(file_id)
                if result.get("status") in [200, 201, 202]:
                    print_success("Topic extraction successful")
                    topics = result.get("data", {}).get("topics", [])
                    if topics:
                        print_info(f"Found {len(topics)} topics")
                else:
                    print_error(f"Topic extraction failed: {result.get('status')}")
                print(json.dumps(result, indent=2))
            
            elif cmd == "preview" and len(parts) > 1:
                file_id = parts[1]
                page = int(parts[2]) if len(parts) > 2 else 1
                print_info(f"Getting preview of page {page} for {file_id}...")
                result = debugger.test_preview(file_id, page)
                if result.get("status") in [200, 201, 202]:
                    print_success("Preview retrieved")
                else:
                    print_error(f"Preview failed: {result.get('status')}")
                print(json.dumps(result, indent=2))
            
            elif cmd == "test" and len(parts) > 2:
                method = parts[1]
                endpoint = parts[2]
                data = json.loads(parts[3]) if len(parts) > 3 else None
                print_info(f"Testing {method} {endpoint}...")
                result = debugger.test_endpoint(method, endpoint, data)
                if result.get("status") and 200 <= result.get("status") < 300:
                    print_success(f"Request successful: {result.get('status')}")
                elif result.get("status"):
                    print_error(f"Request failed: {result.get('status')}")
                print(json.dumps(result, indent=2))
            
            elif cmd == "history":
                history = debugger.get_request_history()
                if history:
                    print_info(f"Request history ({len(history)} requests):")
                    print(json.dumps(history, indent=2))
                else:
                    print_info("No requests in history yet")
            
            elif cmd == "save" and len(parts) > 1:
                file_path = Path(parts[1])
                debugger.save_history(file_path)
                print_success(f"History saved to {file_path}")
            
            elif cmd == "summary":
                debugger.print_summary()
            
            elif cmd == "clear":
                debugger.request_history.clear()
                print_success("History cleared")
            
            else:
                print_error(f"Unknown command: {command}")
                print_info("Type 'help' for available commands")
            
            print()
        
        except KeyboardInterrupt:
            print(f"\n{YELLOW}👋 Goodbye!{RESET}")
            break
        except json.JSONDecodeError as e:
            print_error(f"Invalid JSON: {e}")
        except Exception as e:
            print_error(f"Error: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        # Non-interactive mode
        debugger = APIDebugger()
        
        if sys.argv[1] == "health":
            result = debugger.health_check()
            print(json.dumps(result, indent=2))
        
        elif sys.argv[1] == "test" and len(sys.argv) > 3:
            method = sys.argv[2]
            endpoint = sys.argv[3]
            data = json.loads(sys.argv[4]) if len(sys.argv) > 4 else None
            result = debugger.test_endpoint(method, endpoint, data)
            print(json.dumps(result, indent=2))
        
        else:
            print("Usage:")
            print("  python debug_api.py                    - Interactive mode")
            print("  python debug_api.py health              - Check health")
            print("  python debug_api.py test GET /health    - Test endpoint")
    else:
        # Interactive mode
        interactive_debug()


if __name__ == "__main__":
    main()

