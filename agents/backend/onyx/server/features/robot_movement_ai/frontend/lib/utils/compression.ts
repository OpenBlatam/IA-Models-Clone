/**
 * Compression utilities
 */

// Compress string (simple implementation using btoa)
export function compressString(str: string): string {
  if (typeof window === 'undefined') {
    return str;
  }
  
  try {
    // Simple base64 encoding (not real compression, but reduces some patterns)
    return btoa(unescape(encodeURIComponent(str)));
  } catch {
    return str;
  }
}

// Decompress string
export function decompressString(compressed: string): string {
  if (typeof window === 'undefined') {
    return compressed;
  }
  
  try {
    return decodeURIComponent(escape(atob(compressed)));
  } catch {
    return compressed;
  }
}

// Compress object to string
export function compressObject<T>(obj: T): string {
  const json = JSON.stringify(obj);
  return compressString(json);
}

// Decompress string to object
export function decompressObject<T>(compressed: string): T | null {
  try {
    const json = decompressString(compressed);
    return JSON.parse(json) as T;
  } catch {
    return null;
  }
}

// Calculate compression ratio
export function getCompressionRatio(original: string, compressed: string): number {
  if (original.length === 0) return 0;
  return (1 - compressed.length / original.length) * 100;
}



