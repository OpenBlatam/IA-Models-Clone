/**
 * Data compression utilities
 * @module robot-3d-view/utils/compression
 */

/**
 * Compression options
 */
export interface CompressionOptions {
  level?: number; // 0-9, higher = better compression but slower
  method?: 'gzip' | 'deflate' | 'brotli';
}

/**
 * Compression Manager class
 */
export class CompressionManager {
  /**
   * Compresses data using CompressionStream API
   */
  async compress(data: string, options: CompressionOptions = {}): Promise<Uint8Array> {
    const { method = 'gzip' } = options;

    if (!('CompressionStream' in window)) {
      // Fallback: return original data as bytes
      return new TextEncoder().encode(data);
    }

    const stream = new CompressionStream(method);
    const writer = stream.writable.getWriter();
    const reader = stream.readable.getReader();

    // Write data
    writer.write(new TextEncoder().encode(data));
    writer.close();

    // Read compressed data
    const chunks: Uint8Array[] = [];
    let done = false;

    while (!done) {
      const { value, done: readerDone } = await reader.read();
      done = readerDone;
      if (value) {
        chunks.push(value);
      }
    }

    // Combine chunks
    const totalLength = chunks.reduce((acc, chunk) => acc + chunk.length, 0);
    const result = new Uint8Array(totalLength);
    let offset = 0;
    for (const chunk of chunks) {
      result.set(chunk, offset);
      offset += chunk.length;
    }

    return result;
  }

  /**
   * Decompresses data using DecompressionStream API
   */
  async decompress(data: Uint8Array, method: 'gzip' | 'deflate' | 'brotli' = 'gzip'): Promise<string> {
    if (!('DecompressionStream' in window)) {
      // Fallback: return data as string
      return new TextDecoder().decode(data);
    }

    const stream = new DecompressionStream(method);
    const writer = stream.writable.getWriter();
    const reader = stream.readable.getReader();

    // Write compressed data
    writer.write(data);
    writer.close();

    // Read decompressed data
    const chunks: Uint8Array[] = [];
    let done = false;

    while (!done) {
      const { value, done: readerDone } = await reader.read();
      done = readerDone;
      if (value) {
        chunks.push(value);
      }
    }

    // Combine chunks
    const totalLength = chunks.reduce((acc, chunk) => acc + chunk.length, 0);
    const result = new Uint8Array(totalLength);
    let offset = 0;
    for (const chunk of chunks) {
      result.set(chunk, offset);
      offset += chunk.length;
    }

    return new TextDecoder().decode(result);
  }

  /**
   * Compresses JSON data
   */
  async compressJSON(data: unknown): Promise<Uint8Array> {
    const json = JSON.stringify(data);
    return this.compress(json);
  }

  /**
   * Decompresses JSON data
   */
  async decompressJSON(data: Uint8Array): Promise<unknown> {
    const json = await this.decompress(data);
    return JSON.parse(json);
  }

  /**
   * Gets compression ratio
   */
  getCompressionRatio(original: number, compressed: number): number {
    return compressed / original;
  }

  /**
   * Checks if compression is supported
   */
  isCompressionSupported(): boolean {
    return 'CompressionStream' in window && 'DecompressionStream' in window;
  }
}

/**
 * Global compression manager instance
 */
export const compressionManager = new CompressionManager();



