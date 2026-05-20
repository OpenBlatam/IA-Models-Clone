/**
 * Hashing utilities
 */

// Simple hash function (djb2 algorithm)
export function hashString(str: string): number {
  let hash = 5381;
  for (let i = 0; i < str.length; i++) {
    hash = (hash << 5) + hash + str.charCodeAt(i);
  }
  return hash >>> 0; // Convert to unsigned 32-bit integer
}

// Hash object
export function hashObject(obj: any): number {
  const str = JSON.stringify(obj);
  return hashString(str);
}

// Generate hash from multiple values
export function hashMultiple(...values: any[]): number {
  const str = values.map(String).join('|');
  return hashString(str);
}

// Generate short hash (first 8 characters)
export function shortHash(str: string): string {
  const hash = hashString(str);
  return hash.toString(36).slice(0, 8);
}

// Generate ID from hash
export function generateIdFromHash(prefix: string, ...values: any[]): string {
  const hash = hashMultiple(...values);
  return `${prefix}-${hash.toString(36)}`;
}



