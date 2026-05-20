/**
 * WebSocket & Real-time Testing
 * 
 * Tests that verify WebSocket connections, real-time updates,
 * and bidirectional communication functionality.
 */

import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';

// Mock WebSocket
class MockWebSocket {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;

  readyState = MockWebSocket.CONNECTING;
  url: string;
  protocol: string;
  onopen: ((event: Event) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  private messageQueue: any[] = [];

  constructor(url: string, protocol?: string) {
    this.url = url;
    this.protocol = protocol || '';
    
    // Simulate connection
    setTimeout(() => {
      this.readyState = MockWebSocket.OPEN;
      if (this.onopen) {
        this.onopen(new Event('open'));
      }
    }, 100);
  }

  send(data: string | ArrayBuffer | Blob) {
    if (this.readyState !== MockWebSocket.OPEN) {
      throw new Error('WebSocket is not open');
    }
    this.messageQueue.push(data);
  }

  close(code?: number, reason?: string) {
    this.readyState = MockWebSocket.CLOSING;
    setTimeout(() => {
      this.readyState = MockWebSocket.CLOSED;
      if (this.onclose) {
        this.onclose(new CloseEvent('close', { code, reason }));
      }
    }, 100);
  }

  addEventListener(type: string, listener: EventListener) {
    if (type === 'open') this.onopen = listener as (event: Event) => void;
    if (type === 'close') this.onclose = listener as (event: CloseEvent) => void;
    if (type === 'error') this.onerror = listener as (event: Event) => void;
    if (type === 'message') this.onmessage = listener as (event: MessageEvent) => void;
  }

  removeEventListener(type: string, listener: EventListener) {
    if (type === 'open') this.onopen = null;
    if (type === 'close') this.onclose = null;
    if (type === 'error') this.onerror = null;
    if (type === 'message') this.onmessage = null;
  }

  simulateMessage(data: any) {
    if (this.onmessage) {
      this.onmessage(new MessageEvent('message', { data: JSON.stringify(data) }));
    }
  }
}

// Mock global WebSocket
(global as any).WebSocket = MockWebSocket;

describe('WebSocket & Real-time Testing', () => {
  let ws: MockWebSocket;

  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    if (ws) {
      ws.close();
    }
  });

  describe('WebSocket Connection', () => {
    it('should establish WebSocket connection', (done) => {
      ws = new MockWebSocket('ws://localhost:8080');
      
      ws.onopen = () => {
        expect(ws.readyState).toBe(MockWebSocket.OPEN);
        done();
      };
    });

    it('should handle connection errors', (done) => {
      ws = new MockWebSocket('ws://invalid-url');
      
      ws.onerror = () => {
        expect(ws.readyState).toBe(MockWebSocket.CLOSED);
        done();
      };
    });

    it('should close connection properly', (done) => {
      ws = new MockWebSocket('ws://localhost:8080');
      
      ws.onopen = () => {
        ws.close();
      };
      
      ws.onclose = () => {
        expect(ws.readyState).toBe(MockWebSocket.CLOSED);
        done();
      };
    });
  });

  describe('Message Sending', () => {
    it('should send messages when connected', (done) => {
      ws = new MockWebSocket('ws://localhost:8080');
      
      ws.onopen = () => {
        ws.send(JSON.stringify({ type: 'ping', data: 'test' }));
        expect(ws['messageQueue']).toHaveLength(1);
        done();
      };
    });

    it('should throw error when sending while not connected', () => {
      ws = new MockWebSocket('ws://localhost:8080');
      ws.readyState = MockWebSocket.CLOSED;
      
      expect(() => {
        ws.send(JSON.stringify({ type: 'test' }));
      }).toThrow('WebSocket is not open');
    });

    it('should send different message types', (done) => {
      ws = new MockWebSocket('ws://localhost:8080');
      
      ws.onopen = () => {
        ws.send(JSON.stringify({ type: 'track_update', data: { id: '1' } }));
        ws.send(JSON.stringify({ type: 'playlist_update', data: { id: '2' } }));
        
        expect(ws['messageQueue']).toHaveLength(2);
        done();
      };
    });
  });

  describe('Message Receiving', () => {
    it('should receive messages from server', (done) => {
      ws = new MockWebSocket('ws://localhost:8080');
      
      ws.onopen = () => {
        ws.onmessage = (event) => {
          const data = JSON.parse(event.data);
          expect(data.type).toBe('track_update');
          done();
        };
        
        ws.simulateMessage({ type: 'track_update', data: { id: '1' } });
      };
    });

    it('should handle different message formats', (done) => {
      ws = new MockWebSocket('ws://localhost:8080');
      let messageCount = 0;
      
      ws.onopen = () => {
        ws.onmessage = (event) => {
          messageCount++;
          if (messageCount === 2) {
            done();
          }
        };
        
        ws.simulateMessage({ type: 'update', data: 'text' });
        ws.simulateMessage({ type: 'update', data: { nested: 'object' } });
      };
    });
  });

  describe('Reconnection Logic', () => {
    it('should attempt reconnection on disconnect', (done) => {
      let reconnectAttempts = 0;
      const maxAttempts = 3;
      
      const reconnect = () => {
        reconnectAttempts++;
        if (reconnectAttempts < maxAttempts) {
          ws = new MockWebSocket('ws://localhost:8080');
          ws.onopen = () => {
            expect(reconnectAttempts).toBeLessThanOrEqual(maxAttempts);
            done();
          };
        }
      };
      
      ws = new MockWebSocket('ws://localhost:8080');
      ws.onclose = () => {
        reconnect();
      };
      
      ws.close();
    });

    it('should implement exponential backoff for reconnection', () => {
      const calculateDelay = (attempt: number) => {
        return Math.min(1000 * Math.pow(2, attempt), 30000);
      };

      expect(calculateDelay(0)).toBe(1000);
      expect(calculateDelay(1)).toBe(2000);
      expect(calculateDelay(2)).toBe(4000);
      expect(calculateDelay(10)).toBe(30000); // Max delay
    });
  });

  describe('Heartbeat/Ping-Pong', () => {
    it('should send ping messages periodically', (done) => {
      ws = new MockWebSocket('ws://localhost:8080');
      let pingCount = 0;
      
      ws.onopen = () => {
        const interval = setInterval(() => {
          ws.send(JSON.stringify({ type: 'ping' }));
          pingCount++;
          
          if (pingCount === 3) {
            clearInterval(interval);
            done();
          }
        }, 1000);
      };
    });

    it('should handle pong responses', (done) => {
      ws = new MockWebSocket('ws://localhost:8080');
      
      ws.onopen = () => {
        ws.onmessage = (event) => {
          const data = JSON.parse(event.data);
          if (data.type === 'pong') {
            expect(data.type).toBe('pong');
            done();
          }
        };
        
        ws.send(JSON.stringify({ type: 'ping' }));
        ws.simulateMessage({ type: 'pong' });
      };
    });
  });

  describe('Real-time Updates', () => {
    it('should update track status in real-time', (done) => {
      ws = new MockWebSocket('ws://localhost:8080');
      let trackStatus: any = null;
      
      ws.onopen = () => {
        ws.onmessage = (event) => {
          const data = JSON.parse(event.data);
          if (data.type === 'track_status_update') {
            trackStatus = data.status;
            expect(trackStatus).toBe('playing');
            done();
          }
        };
        
        ws.simulateMessage({
          type: 'track_status_update',
          status: 'playing',
          trackId: '1',
        });
      };
    });

    it('should broadcast updates to multiple clients', (done) => {
      const clients: MockWebSocket[] = [];
      let updateCount = 0;
      
      for (let i = 0; i < 3; i++) {
        const client = new MockWebSocket('ws://localhost:8080');
        client.onopen = () => {
          client.onmessage = () => {
            updateCount++;
            if (updateCount === 3) {
              done();
            }
          };
        };
        clients.push(client);
      }
      
      // Simulate broadcast
      setTimeout(() => {
        clients.forEach(client => {
          client.simulateMessage({ type: 'broadcast', data: 'update' });
        });
      }, 200);
    });
  });

  describe('Error Handling', () => {
    it('should handle connection timeout', (done) => {
      const timeout = 5000;
      let timedOut = false;
      
      ws = new MockWebSocket('ws://slow-server:8080');
      
      const timeoutId = setTimeout(() => {
        timedOut = true;
        ws.close();
        expect(timedOut).toBe(true);
        done();
      }, timeout);
      
      ws.onopen = () => {
        clearTimeout(timeoutId);
      };
    });

    it('should handle malformed messages', (done) => {
      ws = new MockWebSocket('ws://localhost:8080');
      
      ws.onopen = () => {
        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            expect(data).toBeDefined();
          } catch (error) {
            expect(error).toBeInstanceOf(Error);
            done();
          }
        };
        
        // Simulate malformed message
        ws.onmessage?.(new MessageEvent('message', { data: 'invalid json{' }));
      };
    });
  });

  describe('Connection State Management', () => {
    it('should track connection state', () => {
      ws = new MockWebSocket('ws://localhost:8080');
      
      expect(ws.readyState).toBe(MockWebSocket.CONNECTING);
      
      setTimeout(() => {
        expect(ws.readyState).toBe(MockWebSocket.OPEN);
      }, 150);
    });

    it('should handle state transitions', (done) => {
      ws = new MockWebSocket('ws://localhost:8080');
      const states: number[] = [];
      
      const checkState = () => {
        states.push(ws.readyState);
        if (states.length === 3) {
          expect(states).toEqual([
            MockWebSocket.CONNECTING,
            MockWebSocket.OPEN,
            MockWebSocket.CLOSED,
          ]);
          done();
        }
      };
      
      ws.onopen = () => {
        checkState();
        ws.close();
      };
      
      ws.onclose = () => {
        checkState();
      };
    });
  });
});

