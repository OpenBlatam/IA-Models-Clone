#!/bin/bash
# API management script
# Provides REST API interface for deployment operations

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"

CLOUD_DIR="${SCRIPT_DIR}/.."

# Load configuration
load_env_file "${CLOUD_DIR}/.env"

# Default values
readonly API_PORT="${API_PORT:-8080}"
readonly API_HOST="${API_HOST:-0.0.0.0}"

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS] COMMAND

API management for deployment operations.

COMMANDS:
    start               Start API server
    stop                Stop API server
    status              Check API status
    test                Test API endpoints

OPTIONS:
    -p, --port PORT     API port (default: 8080)
    -h, --host HOST     API host (default: 0.0.0.0)
    --help              Show this help message

EXAMPLES:
    $0 start
    $0 start --port 9000
    $0 status
    $0 test

EOF
}

# Parse arguments
parse_args() {
    COMMAND=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -p|--port)
                API_PORT="$2"
                shift 2
                ;;
            -h|--host)
                API_HOST="$2"
                shift 2
                ;;
            --help)
                usage
                exit 0
                ;;
            start|stop|status|test)
                COMMAND="$1"
                shift
                break
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
}

# Start API server
start_api() {
    log_info "Starting API server on ${API_HOST}:${API_PORT}..."
    
    # Create simple Python API server
    cat > /tmp/deployment_api.py << 'PYTHON_API'
#!/usr/bin/env python3
import json
import subprocess
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

class DeploymentAPIHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        query = parse_qs(parsed_path.query)
        
        if path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "healthy"}).encode())
        
        elif path == '/status':
            # Get deployment status
            result = subprocess.run(['./scripts/deployment_status.sh'], 
                                  capture_output=True, text=True, cwd=os.getcwd())
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "ok",
                "output": result.stdout
            }).encode())
        
        elif path == '/metrics':
            # Get metrics
            result = subprocess.run(['./scripts/metrics.sh', '--json'], 
                                  capture_output=True, text=True, cwd=os.getcwd())
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            try:
                metrics = json.loads(result.stdout)
                self.wfile.write(json.dumps(metrics).encode())
            except:
                self.wfile.write(json.dumps({"error": "Could not parse metrics"}).encode())
        
        else:
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Not found"}).encode())
    
    def do_POST(self):
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        if path == '/deploy':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode())
            
            # Trigger deployment
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "deployment_triggered",
                "message": "Deployment initiated"
            }).encode())
        
        else:
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Not found"}).encode())
    
    def log_message(self, format, *args):
        pass  # Suppress default logging

if __name__ == '__main__':
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    host = sys.argv[2] if len(sys.argv) > 2 else '0.0.0.0'
    
    server = HTTPServer((host, port), DeploymentAPIHandler)
    print(f"API server running on http://{host}:{port}")
    server.serve_forever()
PYTHON_API
    
    chmod +x /tmp/deployment_api.py
    
    # Start server in background
    python3 /tmp/deployment_api.py "${API_PORT}" "${API_HOST}" > /tmp/api.log 2>&1 &
    echo $! > /tmp/api.pid
    
    log_info "API server started (PID: $(cat /tmp/api.pid))"
    log_info "API available at http://${API_HOST}:${API_PORT}"
}

# Stop API server
stop_api() {
    if [ -f /tmp/api.pid ]; then
        local pid
        pid=$(cat /tmp/api.pid)
        if kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}"
            rm /tmp/api.pid
            log_info "API server stopped"
        else
            log_warn "API server not running"
            rm /tmp/api.pid
        fi
    else
        log_warn "API server PID file not found"
    fi
}

# Check API status
check_status() {
    if [ -f /tmp/api.pid ]; then
        local pid
        pid=$(cat /tmp/api.pid)
        if kill -0 "${pid}" 2>/dev/null; then
            log_info "API server is running (PID: ${pid})"
            
            # Test health endpoint
            if curl -sf "http://${API_HOST}:${API_PORT}/health" > /dev/null 2>&1; then
                log_info "API health check: ✓ Healthy"
            else
                log_warn "API health check: ✗ Unhealthy"
            fi
        else
            log_warn "API server is not running"
        fi
    else
        log_warn "API server is not running"
    fi
}

# Test API endpoints
test_api() {
    log_info "Testing API endpoints..."
    
    local base_url="http://${API_HOST}:${API_PORT}"
    
    # Test health
    if curl -sf "${base_url}/health" > /dev/null 2>&1; then
        log_info "✓ Health endpoint working"
    else
        log_error "✗ Health endpoint failed"
    fi
    
    # Test status
    if curl -sf "${base_url}/status" > /dev/null 2>&1; then
        log_info "✓ Status endpoint working"
    else
        log_error "✗ Status endpoint failed"
    fi
    
    # Test metrics
    if curl -sf "${base_url}/metrics" > /dev/null 2>&1; then
        log_info "✓ Metrics endpoint working"
    else
        log_error "✗ Metrics endpoint failed"
    fi
}

# Main function
main() {
    parse_args "$@"
    
    if [ -z "${COMMAND}" ]; then
        error_exit 1 "Command is required"
    fi
    
    case "${COMMAND}" in
        start)
            start_api
            ;;
        stop)
            stop_api
            ;;
        status)
            check_status
            ;;
        test)
            test_api
            ;;
        *)
            error_exit 1 "Unknown command: ${COMMAND}"
            ;;
    esac
}

main "$@"


