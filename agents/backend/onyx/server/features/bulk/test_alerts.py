"""
Sistema de Alertas
Envía alertas cuando se detectan problemas
"""

import requests
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, List, Optional
from datetime import datetime

BASE_URL = "http://localhost:8000"

class AlertSystem:
    """Sistema de alertas."""
    
    def __init__(self):
        self.alerts_sent: List[Dict[str, Any]] = []
        self.email_config: Optional[Dict[str, str]] = None
        self.webhook_url: Optional[str] = None
    
    def configure_email(self, smtp_server: str, smtp_port: int, 
                        username: str, password: str, 
                        from_email: str, to_emails: List[str]):
        """Configura alertas por email."""
        self.email_config = {
            "smtp_server": smtp_server,
            "smtp_port": smtp_port,
            "username": username,
            "password": password,
            "from_email": from_email,
            "to_emails": to_emails
        }
    
    def configure_webhook(self, url: str):
        """Configura webhook para alertas."""
        self.webhook_url = url
    
    def send_email_alert(self, subject: str, message: str) -> bool:
        """Envía alerta por email."""
        if not self.email_config:
            return False
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config["from_email"]
            msg['To'] = ", ".join(self.email_config["to_emails"])
            msg['Subject'] = subject
            
            msg.attach(MIMEText(message, 'plain'))
            
            server = smtplib.SMTP(self.email_config["smtp_server"], 
                                 self.email_config["smtp_port"])
            server.starttls()
            server.login(self.email_config["username"], 
                       self.email_config["password"])
            server.send_message(msg)
            server.quit()
            
            self.alerts_sent.append({
                "type": "email",
                "subject": subject,
                "timestamp": datetime.now().isoformat()
            })
            
            return True
        except Exception as e:
            print(f"Error enviando email: {e}")
            return False
    
    def send_webhook_alert(self, alert_data: Dict[str, Any]) -> bool:
        """Envía alerta por webhook."""
        if not self.webhook_url:
            return False
        
        try:
            response = requests.post(
                self.webhook_url,
                json=alert_data,
                timeout=10
            )
            
            if response.status_code == 200:
                self.alerts_sent.append({
                    "type": "webhook",
                    "data": alert_data,
                    "timestamp": datetime.now().isoformat()
                })
                return True
        except Exception as e:
            print(f"Error enviando webhook: {e}")
        
        return False
    
    def check_api_health(self) -> Dict[str, Any]:
        """Verifica salud de la API y genera alertas si es necesario."""
        try:
            start = time.time()
            response = requests.get(f"{BASE_URL}/api/health", timeout=5)
            response_time = (time.time() - start) * 1000
            
            if response.status_code != 200:
                alert = {
                    "severity": "CRITICAL",
                    "type": "API_DOWN",
                    "message": f"API no responde correctamente: Status {response.status_code}",
                    "timestamp": datetime.now().isoformat()
                }
                self.send_alert(alert)
                return {"status": "unhealthy", "alert_sent": True}
            
            if response_time > 1000:
                alert = {
                    "severity": "WARNING",
                    "type": "SLOW_RESPONSE",
                    "message": f"Tiempo de respuesta alto: {response_time:.0f}ms",
                    "timestamp": datetime.now().isoformat()
                }
                self.send_alert(alert)
                return {"status": "healthy", "response_time": response_time, "alert_sent": True}
            
            return {"status": "healthy", "response_time": response_time, "alert_sent": False}
        
        except Exception as e:
            alert = {
                "severity": "CRITICAL",
                "type": "API_UNREACHABLE",
                "message": f"API no alcanzable: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
            self.send_alert(alert)
            return {"status": "error", "error": str(e), "alert_sent": True}
    
    def send_alert(self, alert_data: Dict[str, Any]):
        """Envía alerta por todos los canales configurados."""
        subject = f"[{alert_data['severity']}] {alert_data['type']}"
        message = f"""
Alerta de API BUL

Severidad: {alert_data['severity']}
Tipo: {alert_data['type']}
Mensaje: {alert_data['message']}
Timestamp: {alert_data['timestamp']}
"""
        
        if self.email_config:
            self.send_email_alert(subject, message)
        
        if self.webhook_url:
            self.send_webhook_alert(alert_data)
        
        severity_icon = "🔴" if alert_data["severity"] == "CRITICAL" else "🟡"
        print(f"{severity_icon} ALERTA: {alert_data['message']}")
    
    def get_alerts_summary(self) -> Dict[str, Any]:
        """Obtiene resumen de alertas."""
        return {
            "total_alerts": len(self.alerts_sent),
            "alerts": self.alerts_sent
        }

import time

def run_alert_monitor(duration_minutes: int = 60, check_interval: int = 60):
    """Ejecuta monitor de alertas."""
    alert_system = AlertSystem()
    
    # alert_system.configure_webhook("https://hooks.slack.com/services/YOUR/WEBHOOK/URL")
    
    print("🔔 Monitor de Alertas iniciado")
    print(f"Intervalo de verificación: {check_interval}s")
    print(f"Duración: {duration_minutes} minutos\n")
    
    end_time = datetime.now().timestamp() + (duration_minutes * 60)
    
    try:
        while datetime.now().timestamp() < end_time:
            result = alert_system.check_api_health()
            time.sleep(check_interval)
    except KeyboardInterrupt:
        print("\n⏹ Monitor detenido")
    
    summary = alert_system.get_alerts_summary()
    print(f"\n📊 Total de alertas enviadas: {summary['total_alerts']}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Sistema de Alertas")
    parser.add_argument("--duration", type=int, default=60, help="Duración en minutos")
    parser.add_argument("--interval", type=int, default=60, help="Intervalo en segundos")
    
    args = parser.parse_args()
    
    run_alert_monitor(args.duration, args.interval)
































