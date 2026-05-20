"""
Tests de WebSockets
"""

import pytest
from unittest.mock import Mock, patch
import json
import time


class TestWebSocketConnection:
    """Tests de conexión WebSocket"""
    
    def test_websocket_connect(self):
        """Test de conexión WebSocket"""
        def connect_websocket(client_id):
            return {
                "connected": True,
                "client_id": client_id,
                "connected_at": time.time(),
                "status": "connected"
            }
        
        result = connect_websocket("client123")
        
        assert result["connected"] == True
        assert result["client_id"] == "client123"
        assert "connected_at" in result
    
    def test_websocket_disconnect(self):
        """Test de desconexión WebSocket"""
        def disconnect_websocket(client_id, connection):
            connection["connected"] = False
            connection["disconnected_at"] = time.time()
            return connection
        
        connection = {
            "client_id": "client123",
            "connected": True
        }
        
        result = disconnect_websocket("client123", connection)
        
        assert result["connected"] == False
        assert "disconnected_at" in result
    
    def test_websocket_heartbeat(self):
        """Test de heartbeat WebSocket"""
        def send_heartbeat(connection):
            connection["last_heartbeat"] = time.time()
            return {
                "type": "pong",
                "timestamp": connection["last_heartbeat"]
            }
        
        connection = {"client_id": "client123"}
        response = send_heartbeat(connection)
        
        assert response["type"] == "pong"
        assert "timestamp" in response
        assert "last_heartbeat" in connection


class TestWebSocketMessages:
    """Tests de mensajes WebSocket"""
    
    def test_send_message(self):
        """Test de envío de mensaje"""
        def send_message(connection, message_type, data):
            message = {
                "type": message_type,
                "data": data,
                "timestamp": time.time()
            }
            return message
        
        message = send_message({}, "track_analysis", {"track_id": "123"})
        
        assert message["type"] == "track_analysis"
        assert message["data"]["track_id"] == "123"
        assert "timestamp" in message
    
    def test_receive_message(self):
        """Test de recepción de mensaje"""
        def receive_message(raw_message):
            try:
                message = json.loads(raw_message)
                return {
                    "valid": True,
                    "message": message
                }
            except json.JSONDecodeError:
                return {
                    "valid": False,
                    "error": "Invalid JSON"
                }
        
        valid_message = '{"type": "ping", "data": {}}'
        result = receive_message(valid_message)
        
        assert result["valid"] == True
        assert result["message"]["type"] == "ping"
        
        invalid_message = "invalid json"
        result = receive_message(invalid_message)
        assert result["valid"] == False
    
    def test_broadcast_message(self):
        """Test de broadcast de mensaje"""
        def broadcast_message(message, connections):
            sent_count = 0
            for connection in connections:
                if connection.get("connected"):
                    sent_count += 1
            return {
                "sent": sent_count,
                "total": len(connections)
            }
        
        connections = [
            {"client_id": "1", "connected": True},
            {"client_id": "2", "connected": True},
            {"client_id": "3", "connected": False}
        ]
        
        result = broadcast_message({"type": "update"}, connections)
        
        assert result["sent"] == 2
        assert result["total"] == 3


class TestWebSocketRooms:
    """Tests de salas WebSocket"""
    
    def test_join_room(self):
        """Test de unirse a sala"""
        def join_room(connection, room_id, rooms):
            if room_id not in rooms:
                rooms[room_id] = []
            
            if connection["client_id"] not in [c["client_id"] for c in rooms[room_id]]:
                rooms[room_id].append(connection)
            
            return {
                "joined": True,
                "room_id": room_id,
                "members": len(rooms[room_id])
            }
        
        rooms = {}
        connection = {"client_id": "client123", "connected": True}
        
        result = join_room(connection, "room1", rooms)
        
        assert result["joined"] == True
        assert result["room_id"] == "room1"
        assert result["members"] == 1
    
    def test_leave_room(self):
        """Test de salir de sala"""
        def leave_room(connection, room_id, rooms):
            if room_id in rooms:
                rooms[room_id] = [
                    c for c in rooms[room_id]
                    if c["client_id"] != connection["client_id"]
                ]
                return {
                    "left": True,
                    "room_id": room_id,
                    "members": len(rooms[room_id])
                }
            return {"left": False}
        
        rooms = {
            "room1": [
                {"client_id": "client123"},
                {"client_id": "client456"}
            ]
        }
        connection = {"client_id": "client123"}
        
        result = leave_room(connection, "room1", rooms)
        
        assert result["left"] == True
        assert len(rooms["room1"]) == 1
    
    def test_send_to_room(self):
        """Test de envío a sala"""
        def send_to_room(room_id, message, rooms):
            if room_id not in rooms:
                return {"sent": 0}
            
            sent_count = 0
            for connection in rooms[room_id]:
                if connection.get("connected"):
                    sent_count += 1
            
            return {
                "sent": sent_count,
                "room_id": room_id
            }
        
        rooms = {
            "room1": [
                {"client_id": "1", "connected": True},
                {"client_id": "2", "connected": True},
                {"client_id": "3", "connected": False}
            ]
        }
        
        result = send_to_room("room1", {"type": "update"}, rooms)
        
        assert result["sent"] == 2
        assert result["room_id"] == "room1"


class TestWebSocketErrorHandling:
    """Tests de manejo de errores WebSocket"""
    
    def test_handle_connection_error(self):
        """Test de manejo de error de conexión"""
        def handle_connection_error(error, connection):
            return {
                "error": True,
                "error_type": "connection_error",
                "message": str(error),
                "client_id": connection.get("client_id")
            }
        
        connection = {"client_id": "client123"}
        error = Exception("Connection timeout")
        
        result = handle_connection_error(error, connection)
        
        assert result["error"] == True
        assert result["error_type"] == "connection_error"
        assert result["client_id"] == "client123"
    
    def test_handle_message_error(self):
        """Test de manejo de error en mensaje"""
        def handle_message_error(error, message):
            return {
                "error": True,
                "error_type": "message_error",
                "message": str(error),
                "original_message": message
            }
        
        message = {"type": "invalid"}
        error = ValueError("Invalid message format")
        
        result = handle_message_error(error, message)
        
        assert result["error"] == True
        assert result["error_type"] == "message_error"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

