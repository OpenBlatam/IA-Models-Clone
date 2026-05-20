/**
 * Hash utility functions.
 * Provides helper functions for hashing data.
 */

/**
 * Simple hash function (djb2 algorithm).
 */
export function hash(str: string): number {
  let hash = 5381;
  for (let i = 0; i < str.length; i++) {
    hash = (hash << 5) + hash + str.charCodeAt(i);
  }
  return hash >>> 0; // Convert to unsigned 32-bit integer
}

/**
 * Generates a hash from an object.
 */
export function hashObject(obj: any): number {
  return hash(JSON.stringify(obj));
}

/**
 * Generates a simple hash string.
 */
export function hashString(str: string): string {
  return hash(str).toString(36);
}

/**
 * Generates a hash from an object as string.
 */
export function hashObjectString(obj: any): string {
  return hashString(JSON.stringify(obj));
}

/**
 * Generates a random hash.
 */
export function randomHash(): string {
  return hashString(Math.random().toString() + Date.now().toString());
}

