#!/usr/bin/env python3
"""
Notificador de Resultados
Envía notificaciones de resultados de tests a múltiples canales
"""

import sys
import json
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


class ResultsNotifier:
    """Notificador de resultados de tests"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config = self._load_config(config_path)
    
    def _load_config(self, config_path: Optional[Path]) -> Dict:
        """Cargar configuración"""
        default_config = {
            'email': {
                'enabled': False,
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'username': '',
                'password': '',
                'from': '',
                'to': []
            },
            'slack': {
                'enabled': False,
                'webhook_url': ''
            },
            'teams': {
                'enabled': False,
                'webhook_url': ''
            }
        }
        
        if config_path and config_path.exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def notify_email(self, results: Dict, subject: str = None) -> bool:
        """Enviar notificación por email"""
        if not self.config['email']['enabled']:
            return False
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config['email']['from']
            msg['To'] = ', '.join(self.config['email']['to'])
            msg['Subject'] = subject or f"Test Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            body = self._format_email_body(results)
            msg.attach(MIMEText(body, 'html'))
            
            server = smtplib.SMTP(self.config['email']['smtp_server'], self.config['email']['smtp_port'])
            server.starttls()
            server.login(self.config['email']['username'], self.config['email']['password'])
            server.send_message(msg)
            server.quit()
            
            print("✅ Notificación por email enviada")
            return True
        except Exception as e:
            print(f"❌ Error enviando email: {e}")
            return False
    
    def notify_slack(self, results: Dict) -> bool:
        """Enviar notificación a Slack"""
        if not self.config['slack']['enabled']:
            return False
        
        try:
            import requests
            
            stats = results.get('stats', {})
            total = stats.get('total_runs', 0)
            successful = stats.get('successful', 0)
            failed = stats.get('failed', 0)
            success_rate = (successful / total * 100) if total > 0 else 0
            
            color = "good" if success_rate >= 90 else "warning" if success_rate >= 70 else "danger"
            
            payload = {
                "attachments": [{
                    "color": color,
                    "title": "Test Results Summary",
                    "fields": [
                        {"title": "Total Runs", "value": str(total), "short": True},
                        {"title": "Successful", "value": str(successful), "short": True},
                        {"title": "Failed", "value": str(failed), "short": True},
                        {"title": "Success Rate", "value": f"{success_rate:.1f}%", "short": True}
                    ],
                    "footer": "TruthGPT Test Suite",
                    "ts": int(datetime.now().timestamp())
                }]
            }
            
            response = requests.post(self.config['slack']['webhook_url'], json=payload)
            response.raise_for_status()
            
            print("✅ Notificación a Slack enviada")
            return True
        except Exception as e:
            print(f"❌ Error enviando a Slack: {e}")
            return False
    
    def notify_teams(self, results: Dict) -> bool:
        """Enviar notificación a Microsoft Teams"""
        if not self.config['teams']['enabled']:
            return False
        
        try:
            import requests
            
            stats = results.get('stats', {})
            total = stats.get('total_runs', 0)
            successful = stats.get('successful', 0)
            failed = stats.get('failed', 0)
            success_rate = (successful / total * 100) if total > 0 else 0
            
            payload = {
                "@type": "MessageCard",
                "@context": "https://schema.org/extensions",
                "summary": "Test Results",
                "themeColor": "0078D4",
                "title": "Test Results Summary",
                "sections": [{
                    "facts": [
                        {"name": "Total Runs", "value": str(total)},
                        {"name": "Successful", "value": str(successful)},
                        {"name": "Failed", "value": str(failed)},
                        {"name": "Success Rate", "value": f"{success_rate:.1f}%"}
                    ]
                }]
            }
            
            response = requests.post(self.config['teams']['webhook_url'], json=payload)
            response.raise_for_status()
            
            print("✅ Notificación a Teams enviada")
            return True
        except Exception as e:
            print(f"❌ Error enviando a Teams: {e}")
            return False
    
    def _format_email_body(self, results: Dict) -> str:
        """Formatear cuerpo del email"""
        stats = results.get('stats', {})
        total = stats.get('total_runs', 0)
        successful = stats.get('successful', 0)
        failed = stats.get('failed', 0)
        success_rate = (successful / total * 100) if total > 0 else 0
        
        return f"""
        <html>
        <body>
            <h2>Test Results Summary</h2>
            <table border="1" cellpadding="10">
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Total Runs</td><td>{total}</td></tr>
                <tr><td>Successful</td><td>{successful}</td></tr>
                <tr><td>Failed</td><td>{failed}</td></tr>
                <tr><td>Success Rate</td><td>{success_rate:.1f}%</td></tr>
            </table>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </body>
        </html>
        """
    
    def notify_all(self, results: Dict):
        """Enviar notificaciones a todos los canales configurados"""
        print("📧 Enviando notificaciones...\n")
        
        self.notify_email(results)
        self.notify_slack(results)
        self.notify_teams(results)
        
        print("\n✅ Notificaciones completadas")


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Notificar resultados de tests')
    parser.add_argument('--input', type=Path, required=True,
                       help='Archivo JSON de resultados')
    parser.add_argument('--config', type=Path,
                       help='Archivo de configuración JSON')
    parser.add_argument('--email', action='store_true',
                       help='Enviar solo email')
    parser.add_argument('--slack', action='store_true',
                       help='Enviar solo a Slack')
    parser.add_argument('--teams', action='store_true',
                       help='Enviar solo a Teams')
    
    args = parser.parse_args()
    
    # Cargar resultados
    with open(args.input, 'r') as f:
        results = json.load(f)
    
    notifier = ResultsNotifier(args.config)
    
    if args.email:
        notifier.notify_email(results)
    elif args.slack:
        notifier.notify_slack(results)
    elif args.teams:
        notifier.notify_teams(results)
    else:
        notifier.notify_all(results)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

