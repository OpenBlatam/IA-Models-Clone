/**
 * Compression utility functions.
 * Provides helper functions for data compression.
 */

/**
 * Compresses string using simple RLE (Run-Length Encoding).
 */
export function compressRLE(str: string): string {
  let compressed = '';
  let count = 1;

  for (let i = 0; i < str.length; i++) {
    if (str[i] === str[i + 1]) {
      count++;
    } else {
      compressed += count > 1 ? `${count}${str[i]}` : str[i];
      count = 1;
    }
  }

  return compressed;
}

/**
 * Decompresses RLE string.
 */
export function decompressRLE(str: string): string {
  let decompressed = '';
  let i = 0;

  while (i < str.length) {
    if (/\d/.test(str[i])) {
      let count = '';
      while (/\d/.test(str[i])) {
        count += str[i];
        i++;
      }
      const num = parseInt(count, 10);
      decompressed += str[i].repeat(num);
    } else {
      decompressed += str[i];
    }
    i++;
  }

  return decompressed;
}

/**
 * Compresses JSON object by removing whitespace.
 */
export function compressJSON(obj: any): string {
  return JSON.stringify(obj);
}

/**
 * Estimates compression ratio.
 */
export function estimateCompressionRatio(original: string, compressed: string): number {
  if (original.length === 0) return 0;
  return (1 - compressed.length / original.length) * 100;
}

