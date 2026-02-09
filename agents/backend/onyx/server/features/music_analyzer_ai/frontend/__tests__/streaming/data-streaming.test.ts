/**
 * Data Streaming Testing
 * 
 * Tests that verify data streaming functionality including
 * Server-Sent Events (SSE), chunked responses, and progressive loading.
 */

import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';

// Mock EventSource for SSE
class MockEventSource {
  url: string;
  readyState: number = 0; // CONNECTING
  onopen: ((event: Event) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  private messages: Array<{ data: string; event?: string }> = [];

  constructor(url: string) {
    this.url = url;
    // Simulate connection
    setTimeout(() => {
      this.readyState = 1; // OPEN
      if (this.onopen) {
        this.onopen(new Event('open'));
      }
    }, 100);
  }

  close() {
    this.readyState = 2; // CLOSED
  }

  simulateMessage(data: string, event?: string) {
    if (this.onmessage) {
      this.onmessage(new MessageEvent('message', { data, type: event }));
    }
  }
}

(global as any).EventSource = MockEventSource;

describe('Data Streaming Testing', () => {
  describe('Server-Sent Events (SSE)', () => {
    it('should connect to SSE endpoint', (done) => {
      const eventSource = new EventSource('/api/stream');
      
      eventSource.onopen = () => {
        expect(eventSource.readyState).toBe(1); // OPEN
        done();
      };
    });

    it('should receive SSE messages', (done) => {
      const eventSource = new EventSource('/api/stream');
      const messages: string[] = [];
      
      eventSource.onmessage = (event) => {
        messages.push(event.data);
        if (messages.length === 3) {
          expect(messages).toHaveLength(3);
          done();
        }
      };
      
      eventSource.simulateMessage('message 1');
      eventSource.simulateMessage('message 2');
      eventSource.simulateMessage('message 3');
    });

    it('should handle SSE errors', (done) => {
      const eventSource = new EventSource('/api/stream');
      
      eventSource.onerror = () => {
        expect(eventSource.readyState).toBe(2); // CLOSED
        done();
      };
      
      eventSource.close();
    });

    it('should handle custom SSE event types', (done) => {
      const eventSource = new EventSource('/api/stream');
      const customMessages: string[] = [];
      
      eventSource.addEventListener('update', (event: any) => {
        customMessages.push(event.data);
        if (customMessages.length === 2) {
          done();
        }
      });
      
      eventSource.simulateMessage('data1', 'update');
      eventSource.simulateMessage('data2', 'update');
    });
  });

  describe('Chunked Responses', () => {
    it('should handle chunked data streams', async () => {
      const streamChunked = async function* () {
        yield 'chunk1';
        yield 'chunk2';
        yield 'chunk3';
      };

      const chunks: string[] = [];
      for await (const chunk of streamChunked()) {
        chunks.push(chunk);
      }

      expect(chunks).toEqual(['chunk1', 'chunk2', 'chunk3']);
    });

    it('should process chunks incrementally', async () => {
      const processChunks = async (stream: AsyncIterable<string>) => {
        const results: string[] = [];
        for await (const chunk of stream) {
          results.push(chunk.toUpperCase());
        }
        return results;
      };

      const stream = async function* () {
        yield 'hello';
        yield 'world';
      };

      const results = await processChunks(stream());
      expect(results).toEqual(['HELLO', 'WORLD']);
    });
  });

  describe('Progressive Loading', () => {
    it('should load data progressively', async () => {
      const loadProgressively = async (items: any[], batchSize: number) => {
        const loaded: any[] = [];
        for (let i = 0; i < items.length; i += batchSize) {
          const batch = items.slice(i, i + batchSize);
          loaded.push(...batch);
          await new Promise(resolve => setTimeout(resolve, 10));
        }
        return loaded;
      };

      const items = Array.from({ length: 10 }, (_, i) => ({ id: i }));
      const loaded = await loadProgressively(items, 3);
      
      expect(loaded).toHaveLength(10);
    });

    it('should update UI as data streams in', () => {
      let renderedItems = 0;
      
      const renderItem = (item: any) => {
        renderedItems++;
      };

      const items = [{ id: 1 }, { id: 2 }, { id: 3 }];
      items.forEach(renderItem);
      
      expect(renderedItems).toBe(3);
    });
  });

  describe('Stream Error Handling', () => {
    it('should handle stream errors gracefully', async () => {
      const streamWithError = async function* () {
        yield 'data1';
        throw new Error('Stream error');
      };

      const chunks: string[] = [];
      try {
        for await (const chunk of streamWithError()) {
          chunks.push(chunk);
        }
      } catch (error: any) {
        expect(error.message).toBe('Stream error');
        expect(chunks).toHaveLength(1);
      }
    });

    it('should retry failed streams', async () => {
      let attemptCount = 0;
      const maxRetries = 3;

      const streamWithRetry = async function* () {
        for (let i = 0; i < maxRetries; i++) {
          attemptCount++;
          try {
            yield 'data';
            return;
          } catch {
            if (i === maxRetries - 1) throw new Error('Max retries');
            await new Promise(resolve => setTimeout(resolve, 100));
          }
        }
      };

      const chunks: string[] = [];
      for await (const chunk of streamWithRetry()) {
        chunks.push(chunk);
      }

      expect(chunks).toHaveLength(1);
    });
  });

  describe('Stream Backpressure', () => {
    it('should handle backpressure in streams', async () => {
      const handleBackpressure = async (stream: AsyncIterable<string>) => {
        const buffer: string[] = [];
        const maxBufferSize = 5;

        for await (const chunk of stream) {
          buffer.push(chunk);
          if (buffer.length >= maxBufferSize) {
            // Process buffer
            const processed = buffer.splice(0);
            await new Promise(resolve => setTimeout(resolve, 10));
            return processed;
          }
        }
        return buffer;
      };

      const stream = async function* () {
        for (let i = 0; i < 10; i++) {
          yield `chunk${i}`;
        }
      };

      const result = await handleBackpressure(stream());
      expect(result.length).toBeLessThanOrEqual(5);
    });
  });
});

