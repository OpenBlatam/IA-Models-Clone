"""
Test Notifier
Sends notifications when tests fail or regress
"""

import json
import smtplib
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class TestNotifier:
    """Send notifications for test results"""
    
    def __init__(self, project_root: Path, config: Optional[Dict] = None):
        self.project_root = project_root
        self.config = config or {}
        self.notification_history = []
    
    def notify_test_failure(
        self,
        test_name: str,
        error_message: str,
        recipients: Optional[List[str]] = None
    ) -> bool:
        """Notify about test failure"""
        message = f"""
Test Failure Alert

Test: {test_name}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Error: {error_message[:500]}

Please investigate this test failure.
"""
        
        return self._send_notification(
            subject=f"Test Failure: {test_name}",
            message=message,
            recipients=recipients
        )
    
    def notify_regression(
        self,
        regression_type: str,
        details: Dict,
        recipients: Optional[List[str]] = None
    ) -> bool:
        """Notify about test regression"""
        message = f"""
Test Regression Alert

Type: {regression_type}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Details:
{json.dumps(details, indent=2)}

Please review this regression.
"""
        
        return self._send_notification(
            subject=f"Test Regression: {regression_type}",
            message=message,
            recipients=recipients
        )
    
    def notify_summary(
        self,
        test_results: Dict,
        recipients: Optional[List[str]] = None
    ) -> bool:
        """Notify about test run summary"""
        total = test_results.get('total_tests', 0)
        passed = total - test_results.get('failures', 0) - test_results.get('errors', 0)
        failed = test_results.get('failures', 0)
        errors = test_results.get('errors', 0)
        success_rate = test_results.get('success_rate', 0)
        
        status = "✅ PASSED" if failed == 0 and errors == 0 else "❌ FAILED"
        
        message = f"""
Test Run Summary

Status: {status}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Results:
  Total Tests: {total}
  Passed: {passed}
  Failed: {failed}
  Errors: {errors}
  Success Rate: {success_rate:.1f}%

Execution Time: {test_results.get('execution_time', 0):.2f}s
"""
        
        return self._send_notification(
            subject=f"Test Run Summary: {status}",
            message=message,
            recipients=recipients
        )
    
    def _send_notification(
        self,
        subject: str,
        message: str,
        recipients: Optional[List[str]] = None
    ) -> bool:
        """Send notification via configured method"""
        recipients = recipients or self.config.get('recipients', [])
        
        if not recipients:
            # Just log to file
            return self._log_notification(subject, message)
        
        method = self.config.get('method', 'file')
        
        if method == 'email':
            return self._send_email(subject, message, recipients)
        elif method == 'file':
            return self._log_notification(subject, message)
        else:
            return self._log_notification(subject, message)
    
    def _send_email(
        self,
        subject: str,
        message: str,
        recipients: List[str]
    ) -> bool:
        """Send email notification"""
        try:
            smtp_config = self.config.get('smtp', {})
            smtp_server = smtp_config.get('server', 'localhost')
            smtp_port = smtp_config.get('port', 587)
            smtp_user = smtp_config.get('user', '')
            smtp_password = smtp_config.get('password', '')
            from_email = smtp_config.get('from', smtp_user)
            
            msg = MIMEMultipart()
            msg['From'] = from_email
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = subject
            msg.attach(MIMEText(message, 'plain'))
            
            server = smtplib.SMTP(smtp_server, smtp_port)
            if smtp_user and smtp_password:
                server.starttls()
                server.login(smtp_user, smtp_password)
            
            server.send_message(msg)
            server.quit()
            
            return True
        except Exception as e:
            print(f"⚠️  Failed to send email: {e}")
            # Fallback to file logging
            return self._log_notification(subject, message)
    
    def _log_notification(self, subject: str, message: str) -> bool:
        """Log notification to file"""
        try:
            log_file = self.project_root / "test_notifications.log"
            
            notification = {
                'timestamp': datetime.now().isoformat(),
                'subject': subject,
                'message': message
            }
            
            self.notification_history.append(notification)
            
            # Append to log file
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"Time: {notification['timestamp']}\n")
                f.write(f"Subject: {subject}\n")
                f.write(f"{'='*80}\n")
                f.write(f"{message}\n")
            
            return True
        except Exception as e:
            print(f"⚠️  Failed to log notification: {e}")
            return False
    
    def get_notification_history(self, limit: int = 10) -> List[Dict]:
        """Get recent notification history"""
        return self.notification_history[-limit:]

def create_notifier_config(
    method: str = 'file',
    recipients: Optional[List[str]] = None,
    smtp_config: Optional[Dict] = None
) -> Dict:
    """Create notifier configuration"""
    return {
        'method': method,
        'recipients': recipients or [],
        'smtp': smtp_config or {}
    }

def main():
    """Example usage"""
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    
    # Create notifier with file logging
    config = create_notifier_config(method='file')
    notifier = TestNotifier(project_root, config)
    
    # Example: Notify about test failure
    notifier.notify_test_failure(
        test_name="test_inference_basic",
        error_message="AssertionError: Expected output shape (10, 5), got (10, 4)"
    )
    
    # Example: Notify about summary
    test_results = {
        'total_tests': 204,
        'passed': 200,
        'failures': 2,
        'errors': 0,
        'skipped': 2,
        'success_rate': 98.0,
        'execution_time': 45.3
    }
    notifier.notify_summary(test_results)
    
    print("✅ Notifications sent (check test_notifications.log)")

if __name__ == "__main__":
    main()







