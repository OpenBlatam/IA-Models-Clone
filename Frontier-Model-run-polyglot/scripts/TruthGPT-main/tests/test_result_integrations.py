"""
Test Result Integrations
Integrate with external services (Jira, PagerDuty, etc.)
"""

import json
import requests
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import base64


class TestResultIntegrations:
    """Integrate test results with external services"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.config_file = project_root / "config" / "integrations.json"
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        """Load integration configuration"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def _save_config(self):
        """Save integration configuration"""
        self.config_file.parent.mkdir(exist_ok=True)
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2)
    
    def configure_jira(
        self,
        url: str,
        email: str,
        api_token: str,
        project_key: str
    ):
        """Configure Jira integration"""
        if 'jira' not in self.config:
            self.config['jira'] = {}
        
        self.config['jira'].update({
            'url': url,
            'email': email,
            'api_token': api_token,
            'project_key': project_key,
            'enabled': True
        })
        
        self._save_config()
        print("✅ Jira integration configured")
    
    def create_jira_issue(
        self,
        test_name: str,
        error_message: str,
        summary: str = None
    ) -> Optional[str]:
        """Create Jira issue for failed test"""
        if not self.config.get('jira', {}).get('enabled'):
            print("⚠️ Jira integration not configured")
            return None
        
        jira_config = self.config['jira']
        url = f"{jira_config['url']}/rest/api/3/issue"
        
        auth_string = f"{jira_config['email']}:{jira_config['api_token']}"
        auth_header = base64.b64encode(auth_string.encode()).decode()
        
        issue_data = {
            "fields": {
                "project": {"key": jira_config['project_key']},
                "summary": summary or f"Test Failed: {test_name}",
                "description": {
                    "type": "doc",
                    "version": 1,
                    "content": [
                        {
                            "type": "paragraph",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"Test: {test_name}\n\nError: {error_message}"
                                }
                            ]
                        }
                    ]
                },
                "issuetype": {"name": "Bug"}
            }
        }
        
        try:
            response = requests.post(
                url,
                json=issue_data,
                headers={
                    "Authorization": f"Basic {auth_header}",
                    "Content-Type": "application/json"
                }
            )
            
            if response.status_code == 201:
                issue_key = response.json().get('key')
                print(f"✅ Created Jira issue: {issue_key}")
                return issue_key
            else:
                print(f"❌ Failed to create Jira issue: {response.text}")
                return None
        except Exception as e:
            print(f"❌ Error creating Jira issue: {e}")
            return None
    
    def configure_pagerduty(
        self,
        integration_key: str,
        service_id: str = None
    ):
        """Configure PagerDuty integration"""
        if 'pagerduty' not in self.config:
            self.config['pagerduty'] = {}
        
        self.config['pagerduty'].update({
            'integration_key': integration_key,
            'service_id': service_id,
            'enabled': True
        })
        
        self._save_config()
        print("✅ PagerDuty integration configured")
    
    def send_pagerduty_alert(
        self,
        summary: str,
        severity: str = 'error',
        details: Dict = None
    ) -> bool:
        """Send PagerDuty alert"""
        if not self.config.get('pagerduty', {}).get('enabled'):
            print("⚠️ PagerDuty integration not configured")
            return False
        
        pagerduty_config = self.config['pagerduty']
        url = "https://events.pagerduty.com/v2/enqueue"
        
        payload = {
            "routing_key": pagerduty_config['integration_key'],
            "event_action": "trigger",
            "payload": {
                "summary": summary,
                "severity": severity,
                "source": "test-suite",
                "custom_details": details or {}
            }
        }
        
        try:
            response = requests.post(url, json=payload)
            
            if response.status_code == 202:
                print("✅ PagerDuty alert sent")
                return True
            else:
                print(f"❌ Failed to send PagerDuty alert: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Error sending PagerDuty alert: {e}")
            return False
    
    def configure_webhook(
        self,
        url: str,
        headers: Dict = None,
        enabled: bool = True
    ):
        """Configure webhook integration"""
        if 'webhook' not in self.config:
            self.config['webhook'] = {}
        
        self.config['webhook'].update({
            'url': url,
            'headers': headers or {},
            'enabled': enabled
        })
        
        self._save_config()
        print("✅ Webhook integration configured")
    
    def send_webhook(
        self,
        results: Dict,
        event_type: str = 'test_complete'
    ) -> bool:
        """Send results to webhook"""
        if not self.config.get('webhook', {}).get('enabled'):
            print("⚠️ Webhook integration not configured")
            return False
        
        webhook_config = self.config['webhook']
        
        payload = {
            'event_type': event_type,
            'timestamp': datetime.now().isoformat(),
            'results': results
        }
        
        try:
            response = requests.post(
                webhook_config['url'],
                json=payload,
                headers=webhook_config.get('headers', {}),
                timeout=10
            )
            
            if response.status_code in [200, 201, 202]:
                print("✅ Webhook sent successfully")
                return True
            else:
                print(f"❌ Webhook failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Error sending webhook: {e}")
            return False
    
    def process_failed_tests(
        self,
        results: Dict,
        create_jira: bool = False,
        send_pagerduty: bool = False
    ) -> Dict:
        """Process failed tests and send to integrations"""
        test_details = results.get('test_details', {})
        failed_tests = [
            (name, data) for name, data in test_details.items()
            if data.get('status') in ('failed', 'error')
        ]
        
        processed = {
            'total_failed': len(failed_tests),
            'jira_issues': [],
            'pagerduty_alerts': 0
        }
        
        for test_name, test_data in failed_tests:
            error_message = test_data.get('error_message', 'Unknown error')
            
            if create_jira:
                issue_key = self.create_jira_issue(test_name, error_message)
                if issue_key:
                    processed['jira_issues'].append({
                        'test_name': test_name,
                        'issue_key': issue_key
                    })
            
            if send_pagerduty and len(failed_tests) > 0:
                # Send one alert for all failures
                if processed['pagerduty_alerts'] == 0:
                    summary = f"{len(failed_tests)} test(s) failed"
                    if self.send_pagerduty_alert(summary, 'error', {
                        'failed_tests': len(failed_tests),
                        'run_name': results.get('run_name', 'unknown')
                    }):
                        processed['pagerduty_alerts'] = 1
        
        return processed


def main():
    """CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Result Integrations')
    parser.add_argument('--configure-jira', nargs=4, metavar=('URL', 'EMAIL', 'TOKEN', 'PROJECT'),
                       help='Configure Jira integration')
    parser.add_argument('--configure-pagerduty', nargs=1, metavar='KEY',
                       help='Configure PagerDuty integration')
    parser.add_argument('--configure-webhook', nargs=1, metavar='URL',
                       help='Configure webhook integration')
    parser.add_argument('--process-failures', type=str, metavar='RESULTS_FILE',
                       help='Process failed tests from results file')
    parser.add_argument('--project-root', type=str, help='Project root directory')
    
    args = parser.parse_args()
    
    project_root = Path(args.project_root) if args.project_root else Path(__file__).parent
    
    integrations = TestResultIntegrations(project_root)
    
    if args.configure_jira:
        url, email, token, project = args.configure_jira
        integrations.configure_jira(url, email, token, project)
    elif args.configure_pagerduty:
        integrations.configure_pagerduty(args.configure_pagerduty[0])
    elif args.configure_webhook:
        integrations.configure_webhook(args.configure_webhook[0])
    elif args.process_failures:
        with open(args.process_failures, 'r', encoding='utf-8') as f:
            results = json.load(f)
        processed = integrations.process_failed_tests(results, create_jira=True, send_pagerduty=True)
        print(f"\n📊 Processed {processed['total_failed']} failed tests")
        print(f"  Jira issues created: {len(processed['jira_issues'])}")
        print(f"  PagerDuty alerts sent: {processed['pagerduty_alerts']}")
    else:
        print("Use --help to see available options")


if __name__ == '__main__':
    main()

