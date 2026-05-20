/**
 * ID utility functions.
 * Provides helper functions for generating IDs.
 */

let counter = 0;

/**
 * Generates a unique ID.
 */
export function generateId(prefix: string = 'id'): string {
  return `${prefix}-${Date.now()}-${++counter}-${Math.random().toString(36).substr(2, 9)}`;
}

/**
 * Generates a UUID v4.
 */
export function generateUUID(): string {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === 'x' ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

/**
 * Generates a short ID.
 */
export function generateShortId(): string {
  return Math.random().toString(36).substr(2, 9);
}

/**
 * Generates a numeric ID.
 */
export function generateNumericId(): number {
  return Date.now() * 1000 + Math.floor(Math.random() * 1000);
}

