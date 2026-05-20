"""
BUL API Client - Python SDK
============================
Cliente Python completo para consumir la API de BUL
"""

import requests
import time
import json
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
import websocket
import threading
from dataclasses import dataclass, asdict


@dataclass
class DocumentRequest:
    """Request para generar documento."""
    query: str
    business_area: Optional[str] = None
    document_type: Optional[str] = None
    priority: int = 1
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TaskStatus:
    """Estado de una tarea."""
    task_id: str
    status: str
    progress: float
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    processing_time: Optional[float] = None


class BULApiClient:
    """Cliente Python para la API BUL."""
    
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        """
        Inicializa el cliente.
        
        Args:
            base_url: URL base de la API
            timeout: Timeout por defecto en segundos
        """
        self.base_url = base_url.rstrip('/')
        self.ws_base_url = self.base_url.replace('http://', 'ws://').replace('https://', 'wss://')
        self.timeout = timeout
        self.session = requests.Session()
        self.ws_connections: Dict[str, websocket.WebSocketApp] = {}
        self.ws_callbacks: Dict[str, Callable] = {}
    
    def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Realiza una petición HTTP."""
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                json=data,
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if hasattr(e.response, 'json'):
                error_data = e.response.json()
                raise Exception(error_data.get('detail', str(e)))
            raise Exception(f"Request failed: {str(e)}")
    
    def get_root(self) -> Dict[str, Any]:
        """Obtiene información raíz de la API."""
        return self._request('GET', '/')
    
    def get_health(self) -> Dict[str, Any]:
        """Verifica salud de la API."""
        return self._request('GET', '/api/health')
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de la API."""
        return self._request('GET', '/api/stats')
    
    def generate_document(self, request: DocumentRequest) -> Dict[str, Any]:
        """
        Genera un documento.
        
        Args:
            request: DocumentRequest con los parámetros
            
        Returns:
            Dict con task_id y status
        """
        data = asdict(request)
        data = {k: v for k, v in data.items() if v is not None}
        return self._request('POST', '/api/documents/generate', data=data)
    
    def get_task_document(self, task_id: str) -> Dict[str, Any]:
        """Obtiene el documento generado de una tarea."""
        return self._request('GET', f'/api/tasks/{task_id}/document')
    
    def list_documents(self, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        """Lista documentos generados."""
        return self._request('GET', '/api/documents', params={'limit': limit, 'offset': offset})
    
    def get_task_status(self, task_id: str) -> TaskStatus:
        """Obtiene el estado de una tarea."""
        response = self._request('GET', f'/api/tasks/{task_id}/status')
        return TaskStatus(**response)
    
    def list_tasks(
        self,
        status: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Lista tareas."""
        params = {'limit': limit, 'offset': offset}
        if status:
            params['status'] = status
        if user_id:
            params['user_id'] = user_id
        return self._request('GET', '/api/tasks', params=params)
    
    def delete_task(self, task_id: str) -> Dict[str, Any]:
        """Elimina una tarea."""
        return self._request('DELETE', f'/api/tasks/{task_id}')
    
    def cancel_task(self, task_id: str) -> Dict[str, Any]:
        """Cancela una tarea en progreso."""
        return self._request('POST', f'/api/tasks/{task_id}/cancel')
    
    def wait_for_task_completion(
        self,
        task_id: str,
        interval: int = 2,
        max_attempts: int = 150,
        on_progress: Optional[Callable[[TaskStatus], None]] = None
    ) -> TaskStatus:
        """
        Espera a que una tarea se complete usando polling.
        
        Args:
            task_id: ID de la tarea
            interval: Intervalo entre checks (segundos)
            max_attempts: Máximo número de intentos
            on_progress: Callback llamado en cada actualización
            
        Returns:
            TaskStatus final
            
        Raises:
            Exception si la tarea falla o expira el timeout
        """
        attempts = 0
        
        while attempts < max_attempts:
            status = self.get_task_status(task_id)
            
            if on_progress:
                on_progress(status)
            
            if status.status == 'completed':
                return status
            elif status.status == 'failed':
                raise Exception(status.error or 'Task failed')
            elif status.status == 'cancelled':
                raise Exception('Task was cancelled')
            
            time.sleep(interval)
            attempts += 1
        
        raise Exception('Task completion timeout')
    
    def connect_task_websocket(
        self,
        task_id: str,
        on_message: Callable[[Dict[str, Any]], None]
    ) -> websocket.WebSocketApp:
        """
        Conecta WebSocket para recibir actualizaciones de una tarea.
        
        Args:
            task_id: ID de la tarea
            on_message: Callback para mensajes recibidos
            
        Returns:
            WebSocketApp connection
        """
        ws_key = f"task_{task_id}"
        url = f"{self.ws_base_url}/api/ws/{task_id}"
        
        def on_ws_message(ws, message):
            try:
                data = json.loads(message)
                on_message(data)
            except json.JSONDecodeError as e:
                print(f"Error parsing WebSocket message: {e}")
        
        def on_ws_error(ws, error):
            print(f"WebSocket error: {error}")
        
        def on_ws_close(ws, close_status_code, close_msg):
            if ws_key in self.ws_connections:
                del self.ws_connections[ws_key]
            if ws_key in self.ws_callbacks:
                del self.ws_callbacks[ws_key]
        
        ws = websocket.WebSocketApp(
            url,
            on_message=on_ws_message,
            on_error=on_ws_error,
            on_close=on_ws_close
        )
        
        self.ws_connections[ws_key] = ws
        self.ws_callbacks[ws_key] = on_message
        
        ws_thread = threading.Thread(target=ws.run_forever, daemon=True)
        ws_thread.start()
        
        time.sleep(0.5)
        
        return ws
    
    def wait_for_task_completion_websocket(
        self,
        task_id: str,
        timeout: int = 300,
        on_progress: Optional[Callable[[TaskStatus], None]] = None
    ) -> Dict[str, Any]:
        """
        Espera a que una tarea se complete usando WebSocket.
        
        Args:
            task_id: ID de la tarea
            timeout: Timeout en segundos
            on_progress: Callback para actualizaciones
            
        Returns:
            Documento generado
            
        Raises:
            Exception si la tarea falla o expira el timeout
        """
        result = {'completed': False, 'document': None, 'error': None}
        event = threading.Event()
        
        def on_message(message: Dict[str, Any]):
            msg_type = message.get('type')
            
            if msg_type == 'task_update' and message.get('data'):
                data = message['data']
                status = TaskStatus(
                    task_id=task_id,
                    status=data.get('status', 'pending'),
                    progress=data.get('progress', 0),
                    result=data.get('result'),
                    error=data.get('error'),
                    created_at=message.get('timestamp'),
                    updated_at=message.get('timestamp'),
                    processing_time=data.get('processing_time')
                )
                
                if on_progress:
                    on_progress(status)
                
                if data.get('status') == 'completed':
                    result['completed'] = True
                    result['document'] = self.get_task_document(task_id)
                    event.set()
                elif data.get('status') == 'failed':
                    result['error'] = data.get('error', 'Task failed')
                    event.set()
            elif msg_type == 'initial_state' and message.get('data'):
                data = message['data']
                if data.get('status') == 'completed':
                    result['completed'] = True
                    result['document'] = self.get_task_document(task_id)
                    event.set()
            elif msg_type == 'error':
                result['error'] = message.get('message', 'WebSocket error')
                event.set()
        
        ws = self.connect_task_websocket(task_id, on_message)
        
        if event.wait(timeout=timeout):
            if result['error']:
                raise Exception(result['error'])
            return result['document']
        else:
            raise Exception('Task completion timeout')
    
    def generate_document_and_wait(
        self,
        request: DocumentRequest,
        use_websocket: bool = True,
        polling_interval: int = 2,
        max_attempts: int = 150,
        on_progress: Optional[Callable[[TaskStatus], None]] = None
    ) -> Dict[str, Any]:
        """
        Genera un documento y espera a que esté completo.
        
        Args:
            request: DocumentRequest
            use_websocket: Usar WebSocket si está disponible
            polling_interval: Intervalo para polling (si WebSocket falla)
            max_attempts: Máximo intentos para polling
            on_progress: Callback para actualizaciones
            
        Returns:
            Documento generado
        """
        response = self.generate_document(request)
        task_id = response['task_id']
        
        if use_websocket:
            try:
                return self.wait_for_task_completion_websocket(
                    task_id,
                    timeout=max_attempts * polling_interval,
                    on_progress=on_progress
                )
            except Exception as e:
                print(f"WebSocket failed, falling back to polling: {e}")
        
        status = self.wait_for_task_completion(
            task_id,
            interval=polling_interval,
            max_attempts=max_attempts,
            on_progress=on_progress
        )
        
        if status.status == 'failed':
            raise Exception(status.error or 'Document generation failed')
        
        return self.get_task_document(task_id)
    
    def disconnect_all_websockets(self):
        """Desconecta todas las conexiones WebSocket."""
        for ws in self.ws_connections.values():
            ws.close()
        self.ws_connections.clear()
        self.ws_callbacks.clear()

def create_bul_client(base_url: str = "http://localhost:8000", timeout: int = 30) -> BULApiClient:
    """Crea un cliente BUL API."""
    return BULApiClient(base_url, timeout)

if __name__ == "__main__":
    client = create_bul_client()
    
    health = client.get_health()
    print(f"API Health: {health}")
    
    request = DocumentRequest(
        query="Crear un plan de marketing digital",
        business_area="marketing",
        document_type="strategy"
    )
    
    def on_progress(status: TaskStatus):
        print(f"Progress: {status.progress}% - Status: {status.status}")
    
    try:
        document = client.generate_document_and_wait(
            request,
            use_websocket=True,
            on_progress=on_progress
        )
        print(f"\nDocument generated: {document['document']['title']}")
    except Exception as e:
        print(f"Error: {e}")
































