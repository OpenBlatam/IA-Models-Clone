import { WebSocketServer } from 'ws';

const wss = new WebSocketServer({ port: 1234 });

wss.on('connection', (ws, req) => {
  console.log('WebSocket connection established');
  
  ws.on('message', (message) => {
    console.log('Received message:', message);
  });
  
  ws.on('close', () => {
    console.log('WebSocket connection closed');
  });
});

if (process.env.NODE_ENV !== 'production') {
  console.log('WebSocket server running on ws://localhost:1234');
}            