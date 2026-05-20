"""
Slack Integration
Send test results to Slack
"""

import json
import requests
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

class SlackIntegration:
    """Integrate with Slack for notifications"""
    
    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url or None
    
    def send_test_summary(self, test_results: Dict, channel: str = None) -> bool:
        """Send test summary to Slack"""
        if not self.webhook_url:
            print("⚠️  Slack webhook URL not configured")
            return False
        
        total = test_results.get('total_tests', 0)
        passed = total - test_results.get('failures', 0) - test_results.get('errors', 0) - test_results.get('skipped', 0)
        failed = test_results.get('failures', 0)
        errors = test_results.get('errors', 0)
        success_rate = test_results.get('success_rate', 0)
        
        # Determine color
        if success_rate >= 95:
            color = "good"  # Green
        elif success_rate >= 85:
            color = "warning"  # Yellow
        else:
            color = "danger"  # Red
        
        payload = {
            "attachments": [{
                "color": color,
                "title": "🧪 Test Run Summary",
                "fields": [
                    {
                        "title": "Total Tests",
                        "value": str(total),
                        "short": True
                    },
                    {
                        "title": "Passed",
                        "value": str(passed),
                        "short": True
                    },
                    {
                        "title": "Failed",
                        "value": str(failed),
                        "short": True
                    },
                    {
                        "title": "Errors",
                        "value": str(errors),
                        "short": True
                    },
                    {
                        "title": "Success Rate",
                        "value": f"{success_rate:.1f}%",
                        "short": True
                    },
                    {
                        "title": "Execution Time",
                        "value": f"{test_results.get('execution_time', 0):.2f}s",
                        "short": True
                    }
                ],
                "footer": "TruthGPT Test System",
                "ts": int(datetime.now().timestamp())
            }]
        }
        
        if channel:
            payload["channel"] = channel
        
        try:
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"⚠️  Failed to send to Slack: {e}")
            return False
    
    def send_alert(self, message: str, severity: str = "warning") -> bool:
        """Send alert to Slack"""
        if not self.webhook_url:
            return False
        
        color_map = {
            "info": "#36a64f",
            "warning": "#ff9900",
            "error": "#ff0000",
            "critical": "#8b0000"
        }
        
        payload = {
            "attachments": [{
                "color": color_map.get(severity, "#ff9900"),
                "title": f"🔔 Test Alert: {severity.upper()}",
                "text": message,
                "footer": "TruthGPT Test System",
                "ts": int(datetime.now().timestamp())
            }]
        }
        
        try:
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"⚠️  Failed to send alert to Slack: {e}")
            return False

def create_slack_config(webhook_url: str, config_file: str = "slack_config.json") -> Path:
    """Create Slack configuration file"""
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    
    config = {
        'webhook_url': webhook_url,
        'enabled': True
    }
    
    config_path = project_root / config_file
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    
    return config_path

def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Slack Integration')
    parser.add_argument('--webhook', type=str, help='Slack webhook URL')
    parser.add_argument('--test', action='store_true', help='Send test message')
    
    args = parser.parse_args()
    
    if args.webhook:
        config_path = create_slack_config(args.webhook)
        print(f"✅ Slack configuration saved to: {config_path}")
    
    if args.test:
        integration = SlackIntegration(args.webhook)
        test_results = {
            'total_tests': 204,
            'passed': 200,
            'failures': 2,
            'errors': 0,
            'skipped': 2,
            'success_rate': 98.0,
            'execution_time': 45.3
        }
        success = integration.send_test_summary(test_results)
        if success:
            print("✅ Test message sent to Slack")

if __name__ == "__main__":
    main()







