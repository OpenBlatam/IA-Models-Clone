"""
Test Result Streamer
Provides real-time streaming of test results via WebSocket
"""

import json
import asyncio
import websockets
from pathlib import Path
from typing import Dict, List, Set, Optional
from datetime import datetime
from collections import deque
import subprocess
import sys


class TestResultStreamer:
    """Stream test results in real-time"""
    
    def __init__(self, project_root: Path, port: int = 8765):
        self.project_root = project_root
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.result_buffer = deque(maxlen=100)  # Keep last 100 results
    
    async def register_client(self, websocket: websockets.WebSocketServerProtocol):
        """Register a new client"""
        self.clients.add(websocket)
        print(f"✅ Client connected. Total clients: {len(self.clients)}")
        
        # Send buffered results to new client
        for result in self.result_buffer:
            try:
                await websocket.send(json.dumps(result))
            except Exception:
                pass
    
    async def unregister_client(self, websocket: websockets.WebSocketServerProtocol):
        """Unregister a client"""
        self.clients.discard(websocket)
        print(f"❌ Client disconnected. Total clients: {len(self.clients)}")
    
    async def broadcast_result(self, result: Dict):
        """Broadcast test result to all connected clients"""
        self.result_buffer.append(result)
        
        message = json.dumps(result)
        disconnected = set()
        
        for client in self.clients:
            try:
                await client.send(message)
            except Exception:
                disconnected.add(client)
        
        # Remove disconnected clients
        for client in disconnected:
            await self.unregister_client(client)
    
    async def handle_client(self, websocket: websockets.WebSocketServerProtocol, path: str):
        """Handle client connection"""
        await self.register_client(websocket)
        
        try:
            # Keep connection alive and handle messages
            async for message in websocket:
                # Echo back or handle commands
                data = json.loads(message)
                if data.get('command') == 'ping':
                    await websocket.send(json.dumps({'type': 'pong', 'timestamp': datetime.now().isoformat()}))
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister_client(websocket)
    
    async def run_server(self):
        """Run WebSocket server"""
        print(f"🚀 Starting test result streamer on ws://localhost:{self.port}")
        print(f"📡 Clients can connect to: ws://localhost:{self.port}")
        
        async with websockets.serve(self.handle_client, "localhost", self.port):
            await asyncio.Future()  # Run forever
    
    async def stream_pytest_results(self, pytest_args: List[str] = None):
        """Stream pytest results in real-time"""
        if pytest_args is None:
            pytest_args = []
        
        # Start pytest process
        cmd = ["pytest", "-v", "--tb=short"] + pytest_args
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.project_root
        )
        
        current_test = None
        test_results = {}
        
        # Parse pytest output line by line
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            
            line_str = line.decode('utf-8').strip()
            
            # Parse test start
            if "test_" in line_str and "PASSED" not in line_str and "FAILED" not in line_str:
                # Extract test name
                parts = line_str.split()
                for part in parts:
                    if "test_" in part:
                        current_test = part.split("::")[-1]
                        break
            
            # Parse test result
            if "PASSED" in line_str or "FAILED" in line_str or "ERROR" in line_str:
                status = "passed" if "PASSED" in line_str else "failed"
                if "ERROR" in line_str:
                    status = "error"
                
                result = {
                    'type': 'test_result',
                    'test_name': current_test or 'unknown',
                    'status': status,
                    'timestamp': datetime.now().isoformat(),
                    'output': line_str
                }
                
                await self.broadcast_result(result)
                current_test = None
            
            # Parse summary
            if "passed" in line_str and "failed" in line_str:
                # Try to extract summary
                result = {
                    'type': 'summary',
                    'message': line_str,
                    'timestamp': datetime.now().isoformat()
                }
                await self.broadcast_result(result)
        
        # Final summary
        await process.wait()
        
        final_result = {
            'type': 'test_complete',
            'exit_code': process.returncode,
            'timestamp': datetime.now().isoformat()
        }
        await self.broadcast_result(final_result)


async def run_streamer(project_root: Path, port: int = 8765):
    """Run the streamer server"""
    streamer = TestResultStreamer(project_root, port)
    
    # Run server in background
    server_task = asyncio.create_task(streamer.run_server())
    
    # Wait a bit for server to start
    await asyncio.sleep(1)
    
    return streamer, server_task


def main():
    """CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Result Streamer')
    parser.add_argument('--port', type=int, default=8765, help='WebSocket port')
    parser.add_argument('--run-tests', action='store_true', help='Run tests and stream results')
    parser.add_argument('--pytest-args', nargs='*', help='Additional pytest arguments')
    parser.add_argument('--project-root', type=str, help='Project root directory')
    
    args = parser.parse_args()
    
    project_root = Path(args.project_root) if args.project_root else Path(__file__).parent
    
    async def main_async():
        streamer = TestResultStreamer(project_root, args.port)
        
        if args.run_tests:
            # Run tests and stream
            pytest_args = args.pytest_args or []
            await asyncio.gather(
                streamer.run_server(),
                streamer.stream_pytest_results(pytest_args)
            )
        else:
            # Just run server
            await streamer.run_server()
    
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n👋 Shutting down streamer...")


if __name__ == '__main__':
    main()

