"""
MOEA Notifications - Sistema de notificaciones
==============================================
Sistema de notificaciones para eventos del sistema MOEA
"""
import json
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path


class MOEANotifier:
    """Sistema de notificaciones MOEA"""
    
    def __init__(self, config_file: str = ".moea_notifications.json"):
        self.config_file = Path(config_file)
        self.config = self._load_config()
        self.notification_history: List[Dict] = []
    
    def _load_config(self) -> Dict:
        """Cargar configuración"""
        default_config = {
            "enabled": True,
            "channels": {
                "console": True,
                "file": True,
                "email": False,
                "webhook": False
            },
            "events": {
                "project_generated": True,
                "project_failed": True,
                "benchmark_complete": True,
                "health_check_failed": True
            },
            "email": {
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "username": "",
                "password": "",
                "from_email": "",
                "to_emails": []
            },
            "webhook": {
                "url": "",
                "secret": ""
            },
            "file": {
                "path": "moea_notifications.log"
            }
        }
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    user_config = json.load(f)
                # Merge
                self._deep_update(default_config, user_config)
            except:
                pass
        
        return default_config
    
    def _deep_update(self, base: Dict, update: Dict):
        """Actualizar diccionario recursivamente"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value
    
    def notify(
        self,
        event: str,
        title: str,
        message: str,
        level: str = "info",
        data: Optional[Dict] = None
    ):
        """Enviar notificación"""
        if not self.config.get("enabled", True):
            return
        
        if not self.config.get("events", {}).get(event, True):
            return
        
        notification = {
            "timestamp": datetime.now().isoformat(),
            "event": event,
            "title": title,
            "message": message,
            "level": level,
            "data": data or {}
        }
        
        self.notification_history.append(notification)
        
        # Enviar por canales habilitados
        channels = self.config.get("channels", {})
        
        if channels.get("console", True):
            self._notify_console(notification)
        
        if channels.get("file", True):
            self._notify_file(notification)
        
        if channels.get("email", False):
            self._notify_email(notification)
        
        if channels.get("webhook", False):
            self._notify_webhook(notification)
    
    def _notify_console(self, notification: Dict):
        """Notificar por consola"""
        level_icons = {
            "info": "ℹ️",
            "success": "✅",
            "warning": "⚠️",
            "error": "❌"
        }
        
        icon = level_icons.get(notification["level"], "ℹ️")
        print(f"{icon} [{notification['event']}] {notification['title']}: {notification['message']}")
    
    def _notify_file(self, notification: Dict):
        """Notificar a archivo"""
        file_path = Path(self.config.get("file", {}).get("path", "moea_notifications.log"))
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(notification, ensure_ascii=False) + "\n")
    
    def _notify_email(self, notification: Dict):
        """Notificar por email"""
        email_config = self.config.get("email", {})
        
        if not email_config.get("username") or not email_config.get("to_emails"):
            return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = email_config.get("from_email", email_config.get("username"))
            msg['To'] = ", ".join(email_config.get("to_emails", []))
            msg['Subject'] = f"MOEA: {notification['title']}"
            
            body = f"""
Evento: {notification['event']}
Título: {notification['title']}
Mensaje: {notification['message']}
Nivel: {notification['level']}
Fecha: {notification['timestamp']}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(
                email_config.get("smtp_server", "smtp.gmail.com"),
                email_config.get("smtp_port", 587)
            )
            server.starttls()
            server.login(email_config["username"], email_config["password"])
            server.send_message(msg)
            server.quit()
        except Exception as e:
            print(f"⚠️  Error enviando email: {e}")
    
    def _notify_webhook(self, notification: Dict):
        """Notificar por webhook"""
        webhook_config = self.config.get("webhook", {})
        url = webhook_config.get("url")
        
        if not url:
            return
        
        try:
            requests.post(
                url,
                json=notification,
                timeout=5,
                headers={"Content-Type": "application/json"}
            )
        except Exception as e:
            print(f"⚠️  Error enviando webhook: {e}")
    
    def get_history(self, limit: int = 50) -> List[Dict]:
        """Obtener historial de notificaciones"""
        return self.notification_history[-limit:]
    
    def save_config(self):
        """Guardar configuración"""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MOEA Notifications")
    parser.add_argument(
        'event',
        help='Tipo de evento'
    )
    parser.add_argument(
        'title',
        help='Título de la notificación'
    )
    parser.add_argument(
        'message',
        help='Mensaje de la notificación'
    )
    parser.add_argument(
        '--level',
        choices=['info', 'success', 'warning', 'error'],
        default='info',
        help='Nivel de notificación'
    )
    parser.add_argument(
        '--config',
        default='.moea_notifications.json',
        help='Archivo de configuración'
    )
    
    args = parser.parse_args()
    
    notifier = MOEANotifier(args.config)
    notifier.notify(
        event=args.event,
        title=args.title,
        message=args.message,
        level=args.level
    )


if __name__ == "__main__":
    main()

