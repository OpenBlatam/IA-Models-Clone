#!/bin/bash
# Setup Webhook Listener for Automatic Deployment
# This script sets up a webhook listener service on the EC2 instance

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="${PROJECT_NAME:-ai-project-generator}"
SERVICE_NAME="github-webhook-deploy"
WEBHOOK_PORT="${WEBHOOK_PORT:-9000}"
WEBHOOK_SECRET="${WEBHOOK_SECRET:-}"

# Create systemd service
create_service() {
    cat > "/etc/systemd/system/$SERVICE_NAME.service" <<EOF
[Unit]
Description=GitHub Webhook Deployment Service
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=$SCRIPT_DIR
Environment="GIT_REPO_URL=\${GIT_REPO_URL}"
Environment="GIT_BRANCH=\${GIT_BRANCH:-main}"
Environment="PROJECT_NAME=$PROJECT_NAME"
ExecStart=$SCRIPT_DIR/github_webhook_deploy.sh
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl enable "$SERVICE_NAME"
    systemctl start "$SERVICE_NAME"
}

# Create webhook HTTP listener (optional, for direct webhook calls)
create_webhook_listener() {
    cat > "$SCRIPT_DIR/webhook_http_listener.py" <<'PYTHON_EOF'
#!/usr/bin/env python3
"""
Simple HTTP webhook listener for GitHub webhooks
"""
import http.server
import socketserver
import json
import hmac
import hashlib
import subprocess
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

WEBHOOK_SECRET = os.environ.get('WEBHOOK_SECRET', '')
PORT = int(os.environ.get('WEBHOOK_PORT', '9000'))
DEPLOY_SCRIPT = os.path.join(os.path.dirname(__file__), 'github_webhook_deploy.sh')

class WebhookHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != '/webhook':
            self.send_response(404)
            self.end_headers()
            return

        content_length = int(self.headers.get('Content-Length', 0))
        payload = self.rfile.read(content_length)
        
        # Verify webhook signature
        if WEBHOOK_SECRET:
            signature = self.headers.get('X-Hub-Signature-256', '')
            if not self.verify_signature(payload, signature):
                logger.warning("Invalid webhook signature")
                self.send_response(401)
                self.end_headers()
                return

        # Parse payload
        try:
            event = json.loads(payload.decode('utf-8'))
            ref = event.get('ref', '')
            
            # Only deploy on push to main
            if ref == 'refs/heads/main' and event.get('pusher'):
                logger.info("Push to main detected, triggering deployment")
                self.trigger_deployment()
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'deployment_triggered'}).encode())
            else:
                logger.info(f"Ignoring event: {ref}")
                self.send_response(200)
                self.end_headers()
        except Exception as e:
            logger.error(f"Error processing webhook: {e}")
            self.send_response(500)
            self.end_headers()

    def verify_signature(self, payload, signature):
        if not signature:
            return False
        expected_signature = 'sha256=' + hmac.new(
            WEBHOOK_SECRET.encode(),
            payload,
            hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(signature, expected_signature)

    def trigger_deployment(self):
        try:
            subprocess.Popen(
                [DEPLOY_SCRIPT],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            logger.info("Deployment script triggered")
        except Exception as e:
            logger.error(f"Failed to trigger deployment: {e}")

    def log_message(self, format, *args):
        logger.info(f"{self.address_string()} - {format % args}")

if __name__ == '__main__':
    with socketserver.TCPServer(("", PORT), WebhookHandler) as httpd:
        logger.info(f"Webhook listener started on port {PORT}")
        httpd.serve_forever()
PYTHON_EOF

    chmod +x "$SCRIPT_DIR/webhook_http_listener.py"
}

# Setup polling service (alternative to webhooks)
create_polling_service() {
    cat > "/etc/systemd/system/$SERVICE_NAME-polling.service" <<EOF
[Unit]
Description=GitHub Polling Deployment Service
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=$SCRIPT_DIR
Environment="GIT_REPO_URL=\${GIT_REPO_URL}"
Environment="GIT_BRANCH=\${GIT_BRANCH:-main}"
Environment="PROJECT_NAME=$PROJECT_NAME"
ExecStart=/bin/bash -c 'while true; do $SCRIPT_DIR/github_webhook_deploy.sh; sleep 300; done'
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl enable "${SERVICE_NAME}-polling"
}

# Main setup
main() {
    echo "Setting up webhook listener for $PROJECT_NAME..."
    
    # Make scripts executable
    chmod +x "$SCRIPT_DIR/github_webhook_deploy.sh"
    
    # Create service
    create_service
    
    # Create webhook HTTP listener (optional)
    if command -v python3 > /dev/null 2>&1; then
        create_webhook_listener
        echo "Webhook HTTP listener created at $SCRIPT_DIR/webhook_http_listener.py"
    fi
    
    # Create polling service (alternative)
    create_polling_service
    
    echo "Setup complete!"
    echo ""
    echo "To start the service:"
    echo "  sudo systemctl start $SERVICE_NAME"
    echo ""
    echo "To use polling instead:"
    echo "  sudo systemctl start ${SERVICE_NAME}-polling"
    echo ""
    echo "To view logs:"
    echo "  sudo journalctl -u $SERVICE_NAME -f"
}

main "$@"

