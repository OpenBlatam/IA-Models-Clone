/**
 * Unit tests para stream-processor
 */
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { processStream, StreamProcessorOptions } from '../../app/api/tasks/utils/stream-processor';

describe('StreamProcessor', () => {
  let mockReader: any;
  let onContent: ReturnType<typeof vi.fn>;
  let onError: ReturnType<typeof vi.fn>;
  let checkStopped: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    onContent = vi.fn();
    onError = vi.fn();
    checkStopped = vi.fn().mockResolvedValue(false);
    
    mockReader = {
      read: vi.fn(),
      cancel: vi.fn(),
      releaseLock: vi.fn(),
    };
  });

  it('debería procesar chunks con contenido correctamente', async () => {
    const chunks = [
      { done: false, value: new TextEncoder().encode('{"content":"Hello","done":false}\n') },
      { done: false, value: new TextEncoder().encode('{"content":" World","done":false}\n') },
      { done: true, value: undefined },
    ];

    let chunkIndex = 0;
    mockReader.read.mockImplementation(() => Promise.resolve(chunks[chunkIndex++]));

    const result = await processStream({
      taskId: 'test-1',
      reader: mockReader,
      onContent,
      onError,
      checkStopped,
      minContentLength: 5,
    });

    expect(result.success).toBe(true);
    expect(result.contentLength).toBe(11); // "Hello World"
    expect(result.chunksReceived).toBe(2);
    expect(onContent).toHaveBeenCalledTimes(2);
    expect(onContent).toHaveBeenCalledWith('Hello');
    expect(onContent).toHaveBeenCalledWith(' World');
  });

  it('debería manejar chunks incompletos (sin newline)', async () => {
    const chunks = [
      { done: false, value: new TextEncoder().encode('{"content":"Hello","done":false}') },
      { done: false, value: new TextEncoder().encode('\n{"content":" World","done":false}\n') },
      { done: true, value: undefined },
    ];

    let chunkIndex = 0;
    mockReader.read.mockImplementation(() => Promise.resolve(chunks[chunkIndex++]));

    const result = await processStream({
      taskId: 'test-2',
      reader: mockReader,
      onContent,
      onError,
      checkStopped,
      minContentLength: 5,
    });

    expect(result.success).toBe(true);
    expect(result.contentLength).toBe(11);
    expect(onContent).toHaveBeenCalledTimes(2);
  });

  it('debería detectar cuando el stream termina sin contenido', async () => {
    const chunks = [
      { done: false, value: new TextEncoder().encode('{}\n') },
      { done: true, value: undefined },
    ];

    let chunkIndex = 0;
    mockReader.read.mockImplementation(() => Promise.resolve(chunks[chunkIndex++]));

    const result = await processStream({
      taskId: 'test-3',
      reader: mockReader,
      onContent,
      onError,
      checkStopped,
      minContentLength: 5,
    });

    expect(result.success).toBe(false);
    expect(result.contentLength).toBe(0);
    expect(result.chunksReceived).toBe(1);
    expect(onContent).not.toHaveBeenCalled();
  });

  it('debería procesar buffer restante cuando el stream termina', async () => {
    const chunks = [
      { done: false, value: new TextEncoder().encode('{"content":"Hello"') },
      { done: true, value: undefined },
    ];

    let chunkIndex = 0;
    mockReader.read.mockImplementation(() => Promise.resolve(chunks[chunkIndex++]));

    const result = await processStream({
      taskId: 'test-4',
      reader: mockReader,
      onContent,
      onError,
      checkStopped,
      minContentLength: 5,
    });

    // El buffer debería procesarse al final
    expect(result.chunksReceived).toBe(1);
  });

  it('debería detenerse si checkStopped devuelve true', async () => {
    checkStopped.mockResolvedValue(true);

    const chunks = [
      { done: false, value: new TextEncoder().encode('{"content":"Hello","done":false}\n') },
    ];

    mockReader.read.mockImplementation(() => Promise.resolve(chunks[0]));

    const result = await processStream({
      taskId: 'test-5',
      reader: mockReader,
      onContent,
      onError,
      checkStopped,
      minContentLength: 5,
    });

    expect(checkStopped).toHaveBeenCalled();
    expect(mockReader.cancel).toHaveBeenCalled();
    // El stream debería detenerse antes de procesar
    expect(onContent).not.toHaveBeenCalled();
  });

  it('debería verificar checkStopped periódicamente durante el procesamiento', async () => {
    let stopCheckCount = 0;
    checkStopped.mockImplementation(async () => {
      stopCheckCount++;
      // Devolver true después de 3 verificaciones
      return stopCheckCount >= 3;
    });

    const chunks = [
      { done: false, value: new TextEncoder().encode('{"content":"Chunk1","done":false}\n') },
      { done: false, value: new TextEncoder().encode('{"content":"Chunk2","done":false}\n') },
      { done: false, value: new TextEncoder().encode('{"content":"Chunk3","done":false}\n') },
    ];

    let chunkIndex = 0;
    mockReader.read.mockImplementation(() => {
      // Simular delay para permitir múltiples verificaciones
      return new Promise(resolve => {
        setTimeout(() => {
          resolve(chunks[chunkIndex++] || { done: true, value: undefined });
        }, 100);
      });
    });

    const result = await processStream({
      taskId: 'test-8',
      reader: mockReader,
      onContent,
      onError,
      checkStopped,
      minContentLength: 5,
    });

    // Debería haber verificado múltiples veces
    expect(checkStopped).toHaveBeenCalledTimes(3);
    expect(mockReader.cancel).toHaveBeenCalled();
  });

  it('debería manejar errores en el reader', async () => {
    mockReader.read.mockRejectedValue(new Error('Network error'));

    // El stream-processor captura el error y lo devuelve en el resultado
    const result = await processStream({
      taskId: 'test-6',
      reader: mockReader,
      onContent,
      onError,
      checkStopped,
      minContentLength: 5,
    });

    // Verificar que el resultado indica error
    expect(result.success).toBe(false);
    expect(result.error).toBe('Network error');
    expect(result.contentLength).toBe(0);
    expect(result.chunksReceived).toBe(0);
  });

  it('debería manejar líneas que no son JSON válido', async () => {
    const chunks = [
      { done: false, value: new TextEncoder().encode('not json\n{"content":"Hello","done":false}\n') },
      { done: true, value: undefined },
    ];

    let chunkIndex = 0;
    mockReader.read.mockImplementation(() => Promise.resolve(chunks[chunkIndex++]));

    const result = await processStream({
      taskId: 'test-7',
      reader: mockReader,
      onContent,
      onError,
      checkStopped,
      minContentLength: 5,
    });

    expect(result.success).toBe(true);
    expect(onContent).toHaveBeenCalled();
  });
});

