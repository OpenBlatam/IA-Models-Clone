"""
Mock Server para pruebas sin servidor real
Permite probar el cliente TypeScript sin necesidad del servidor Python
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import time
from datetime import datetime
from typing import Dict, Any
import threading

PORT = 8001

class MockAPIHandler(BaseHTTPRequestHandler):
    """Handler para el mock server."""
    
    tasks: Dict[str, Dict[str, Any]] = {}
    task_counter = 0
    
    def do_GET(self):
        """Maneja requests GET."""
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({
                "message": "BUL API - Mock Server",
                "version": "1.0.0",
                "status": "operational",
                "timestamp": datetime.now().isoformat()
            }).encode())
        
        elif self.path == "/api/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "uptime": "0:05:00",
                "active_tasks": len([t for t in self.tasks.values() if t["status"] in ["queued", "processing"]]),
                "total_requests": 100,
                "version": "1.0.0"
            }).encode())
        
        elif self.path == "/api/stats":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            completed = len([t for t in self.tasks.values() if t["status"] == "completed"])
            total = len(self.tasks)
            self.wfile.write(json.dumps({
                "total_requests": 100,
                "active_tasks": len([t for t in self.tasks.values() if t["status"] in ["queued", "processing"]]),
                "completed_tasks": completed,
                "success_rate": completed / total if total > 0 else 1.0,
                "average_processing_time": 2.5,
                "uptime": "0:05:00"
            }).encode())
        
        elif self.path.startswith("/api/tasks/") and "/status" in self.path:
            task_id = self.path.split("/")[3]
            if task_id in self.tasks:
                task = self.tasks[task_id]
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "task_id": task_id,
                    "status": task["status"],
                    "progress": task["progress"],
                    "result": task.get("result"),
                    "error": task.get("error"),
                    "created_at": task["created_at"],
                    "updated_at": datetime.now().isoformat(),
                    "processing_time": task.get("processing_time")
                }).encode())
            else:
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"detail": "Task not found"}).encode())
        
        elif self.path.startswith("/api/tasks/") and "/document" in self.path:
            task_id = self.path.split("/")[3]
            if task_id in self.tasks and self.tasks[task_id]["status"] == "completed":
                task = self.tasks[task_id]
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "task_id": task_id,
                    "document": task["result"],
                    "metadata": task.get("request", {}),
                    "created_at": task["created_at"],
                    "completed_at": datetime.now().isoformat()
                }).encode())
            else:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"detail": "Task not completed"}).encode())
        
        elif self.path.startswith("/api/tasks"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            task_list = []
            for tid, task in self.tasks.items():
                task_list.append({
                    "task_id": tid,
                    "status": task["status"],
                    "progress": task["progress"],
                    "created_at": task["created_at"],
                    "updated_at": datetime.now().isoformat(),
                    "query_preview": task.get("request", {}).get("query", "")[:50]
                })
            self.wfile.write(json.dumps({
                "tasks": task_list,
                "total": len(task_list),
                "limit": 50,
                "offset": 0,
                "has_more": False
            }).encode())
        
        elif self.path.startswith("/api/documents"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            doc_list = []
            for tid, task in self.tasks.items():
                if task["status"] == "completed":
                    doc_list.append({
                        "task_id": tid,
                        "created_at": task["created_at"],
                        "query_preview": task.get("request", {}).get("query", "")[:50],
                        "business_area": task.get("request", {}).get("business_area"),
                        "document_type": task.get("request", {}).get("document_type")
                    })
            self.wfile.write(json.dumps({
                "documents": doc_list,
                "total": len(doc_list),
                "limit": 50,
                "offset": 0,
                "has_more": False
            }).encode())
        
        else:
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"detail": "Not found"}).encode())
    
    def do_POST(self):
        """Maneja requests POST."""
        if self.path == "/api/documents/generate":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            
            try:
                data = json.loads(body.decode())
                
                # Validación básica
                if not data.get("query") or len(data["query"]) < 10:
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        "detail": "Query must be at least 10 characters"
                    }).encode())
                    return
                
                # Crear tarea
                self.task_counter += 1
                task_id = f"mock_task_{self.task_counter}"
                
                self.tasks[task_id] = {
                    "status": "queued",
                    "progress": 0,
                    "request": data,
                    "created_at": datetime.now().isoformat(),
                    "result": None,
                    "error": None
                }
                
                # Simular procesamiento en background
                threading.Thread(
                    target=self._process_task,
                    args=(task_id,),
                    daemon=True
                ).start()
                
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "task_id": task_id,
                    "status": "queued",
                    "message": "Document generation started",
                    "estimated_time": 5,
                    "queue_position": 1,
                    "created_at": datetime.now().isoformat()
                }).encode())
            
            except json.JSONDecodeError:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"detail": "Invalid JSON"}).encode())
        
        else:
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"detail": "Not found"}).encode())
    
    def _process_task(self, task_id: str):
        """Simula procesamiento de tarea."""
        if task_id not in self.tasks:
            return
        
        task = self.tasks[task_id]
        
        # Simular progreso
        for progress in [10, 30, 50, 70, 90, 100]:
            time.sleep(0.5)
            task["status"] = "processing" if progress < 100 else "completed"
            task["progress"] = progress
            task["updated_at"] = datetime.now().isoformat()
        
        # Generar documento mock
        task["result"] = {
            "content": f"""# Documento Generado (Mock)

**Consulta:** {task["request"].get("query", "")}

**Área de Negocio:** {task["request"].get("business_area", "General")}

## Contenido

Este es un documento generado por el mock server para pruebas.

**Prioridad:** {task["request"].get("priority", 1)}

**Generado el:** {datetime.now().isoformat()}

---

*Nota: Este es un documento de prueba generado por el mock server.*
""",
            "format": "markdown",
            "word_count": 150,
            "generated_at": datetime.now().isoformat(),
            "using_bul_system": False
        }
        
        task["processing_time"] = 2.5
        task["updated_at"] = datetime.now().isoformat()
    
    def do_OPTIONS(self):
        """Maneja CORS preflight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
    
    def log_message(self, format, *args):
        """Suprime logs por defecto."""
        pass

def run_mock_server(port: int = PORT):
    """Ejecuta el mock server."""
    server = HTTPServer(("localhost", port), MockAPIHandler)
    print(f"🚀 Mock Server iniciado en http://localhost:{port}")
    print(f"📝 Útil para probar el cliente TypeScript sin servidor Python")
    print(f"⏹  Presiona Ctrl+C para detener\n")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n⏹ Mock Server detenido")
        server.shutdown()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Mock Server para pruebas")
    parser.add_argument("--port", type=int, default=PORT, help="Puerto del servidor")
    
    args = parser.parse_args()
    
    run_mock_server(args.port)
































