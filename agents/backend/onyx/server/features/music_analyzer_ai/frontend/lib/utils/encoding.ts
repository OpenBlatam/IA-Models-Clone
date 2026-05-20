/**
 * Encoding utility functions.
 * Provides helper functions for encoding/decoding data.
 */

/**
 * Encodes string to base64.
 */
export function encodeBase64(str: string): string {
  if (typeof window !== 'undefined' && window.btoa) {
    return window.btoa(unescape(encodeURIComponent(str)));
  }
  // Fallback for Node.js
  return Buffer.from(str, 'utf-8').toString('base64');
}

/**
 * Decodes base64 string.
 */
export function decodeBase64(str: string): string {
  if (typeof window !== 'undefined' && window.atob) {
    return decodeURIComponent(escape(window.atob(str)));
  }
  // Fallback for Node.js
  return Buffer.from(str, 'base64').toString('utf-8');
}

/**
 * Encodes object to base64.
 */
export function encodeObjectBase64(obj: any): string {
  return encodeBase64(JSON.stringify(obj));
}

/**
 * Decodes base64 to object.
 */
export function decodeObjectBase64<T = any>(str: string): T {
  return JSON.parse(decodeBase64(str));
}

/**
 * Encodes string to URL-safe base64.
 */
export function encodeBase64URL(str: string): string {
  return encodeBase64(str)
    .replace(/\+/g, '-')
    .replace(/\//g, '_')
    .replace(/=/g, '');
}

/**
 * Decodes URL-safe base64 string.
 */
export function decodeBase64URL(str: string): string {
  let base64 = str.replace(/-/g, '+').replace(/_/g, '/');
  while (base64.length % 4) {
    base64 += '=';
  }
  return decodeBase64(base64);
}

