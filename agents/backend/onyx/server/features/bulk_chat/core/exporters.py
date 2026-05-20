"""
Exporters - Sistema de exportación
===================================

Sistema para exportar conversaciones en múltiples formatos.
"""

import json
import csv
import logging
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime
from io import StringIO

from .chat_session import ChatSession, ChatMessage

logger = logging.getLogger(__name__)


class ConversationExporter:
    """Exportador de conversaciones."""
    
    async def export_json(self, session: ChatSession, output_path: Optional[str] = None) -> str:
        """Exportar conversación a JSON."""
        data = {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "state": session.state.value,
            "messages": [
                {
                    "id": msg.id,
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                    "metadata": msg.metadata,
                }
                for msg in session.messages
            ],
        }
        
        json_str = json.dumps(data, indent=2, ensure_ascii=False)
        
        if output_path:
            Path(output_path).write_text(json_str, encoding="utf-8")
            logger.info(f"Exported session {session.session_id} to {output_path}")
        
        return json_str
    
    async def export_markdown(self, session: ChatSession, output_path: Optional[str] = None) -> str:
        """Exportar conversación a Markdown."""
        lines = [
            f"# Conversación - {session.session_id[:8]}",
            "",
            f"**Usuario**: {session.user_id or 'Anónimo'}",
            f"**Creada**: {session.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Estado**: {session.state.value}",
            "",
            "---",
            "",
        ]
        
        for msg in session.messages:
            role_emoji = "👤" if msg.role == "user" else "🤖"
            role_name = "Usuario" if msg.role == "user" else "Asistente"
            
            lines.extend([
                f"## {role_emoji} {role_name}",
                f"*{msg.timestamp.strftime('%Y-%m-%d %H:%M:%S')}*",
                "",
                msg.content,
                "",
                "---",
                "",
            ])
        
        markdown = "\n".join(lines)
        
        if output_path:
            Path(output_path).write_text(markdown, encoding="utf-8")
            logger.info(f"Exported session {session.session_id} to {output_path}")
        
        return markdown
    
    async def export_csv(self, session: ChatSession, output_path: Optional[str] = None) -> str:
        """Exportar conversación a CSV."""
        output = StringIO()
        writer = csv.writer(output)
        
        # Header
        writer.writerow(["timestamp", "role", "content", "message_id"])
        
        # Messages
        for msg in session.messages:
            writer.writerow([
                msg.timestamp.isoformat(),
                msg.role,
                msg.content,
                msg.id,
            ])
        
        csv_str = output.getvalue()
        
        if output_path:
            Path(output_path).write_text(csv_str, encoding="utf-8")
            logger.info(f"Exported session {session.session_id} to {output_path}")
        
        return csv_str
    
    async def export_txt(self, session: ChatSession, output_path: Optional[str] = None) -> str:
        """Exportar conversación a texto plano."""
        lines = [
            f"Conversación - {session.session_id[:8]}",
            f"Usuario: {session.user_id or 'Anónimo'}",
            f"Creada: {session.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Estado: {session.state.value}",
            "",
            "=" * 80,
            "",
        ]
        
        for msg in session.messages:
            role_name = "USUARIO" if msg.role == "user" else "ASISTENTE"
            lines.extend([
                f"[{role_name}] {msg.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                msg.content,
                "",
                "-" * 80,
                "",
            ])
        
        txt = "\n".join(lines)
        
        if output_path:
            Path(output_path).write_text(txt, encoding="utf-8")
            logger.info(f"Exported session {session.session_id} to {output_path}")
        
        return txt
    
    async def export_html(self, session: ChatSession, output_path: Optional[str] = None) -> str:
        """Exportar conversación a HTML."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Conversación - {session.session_id[:8]}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .message {{
            margin: 20px 0;
            padding: 15px;
            border-radius: 8px;
        }}
        .user {{
            background-color: #e3f2fd;
            text-align: right;
        }}
        .assistant {{
            background-color: #fff3e0;
        }}
        .timestamp {{
            font-size: 0.8em;
            color: #666;
            margin-bottom: 5px;
        }}
        .content {{
            line-height: 1.6;
        }}
        .header {{
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Conversación - {session.session_id[:8]}</h1>
        <p><strong>Usuario:</strong> {session.user_id or 'Anónimo'}</p>
        <p><strong>Creada:</strong> {session.created_at.strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Estado:</strong> {session.state.value}</p>
    </div>
"""
        
        for msg in session.messages:
            role_class = "user" if msg.role == "user" else "assistant"
            role_label = "Usuario" if msg.role == "user" else "Asistente"
            
            html += f"""
    <div class="message {role_class}">
        <div class="timestamp">{role_label} - {msg.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</div>
        <div class="content">{msg.content.replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br>')}</div>
    </div>
"""
        
        html += """
</body>
</html>
"""
        
        if output_path:
            Path(output_path).write_text(html, encoding="utf-8")
            logger.info(f"Exported session {session.session_id} to {output_path}")
        
        return html
































