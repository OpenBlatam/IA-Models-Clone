/**
 * Security and input validation utilities
 * @module robot-3d-view/utils/security
 */

/**
 * Sanitization options
 */
export interface SanitizeOptions {
  allowHTML?: boolean;
  maxLength?: number;
  allowedTags?: string[];
  allowedAttributes?: string[];
}

/**
 * Security Manager class
 */
export class SecurityManager {
  /**
   * Sanitizes a string input
   */
  sanitizeString(input: string, options: SanitizeOptions = {}): string {
    const { maxLength = 10000 } = options;

    if (typeof input !== 'string') {
      return '';
    }

    // Limit length
    let sanitized = input.slice(0, maxLength);

    // Remove null bytes
    sanitized = sanitized.replace(/\0/g, '');

    // Remove control characters except newlines and tabs
    sanitized = sanitized.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, '');

    return sanitized;
  }

  /**
   * Sanitizes HTML content
   */
  sanitizeHTML(html: string, options: SanitizeOptions = {}): string {
    const {
      allowHTML = false,
      allowedTags = ['b', 'i', 'em', 'strong', 'a', 'p', 'br'],
      allowedAttributes = ['href', 'title'],
    } = options;

    if (!allowHTML) {
      return this.escapeHTML(html);
    }

    // Basic HTML sanitization (for production, use a library like DOMPurify)
    const div = document.createElement('div');
    div.textContent = html;
    return div.innerHTML;
  }

  /**
   * Escapes HTML special characters
   */
  escapeHTML(text: string): string {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  /**
   * Validates and sanitizes a number
   */
  sanitizeNumber(input: unknown, min?: number, max?: number): number | null {
    if (typeof input !== 'number' && typeof input !== 'string') {
      return null;
    }

    const num = typeof input === 'string' ? parseFloat(input) : input;

    if (isNaN(num) || !isFinite(num)) {
      return null;
    }

    if (min !== undefined && num < min) {
      return null;
    }

    if (max !== undefined && num > max) {
      return null;
    }

    return num;
  }

  /**
   * Validates and sanitizes an array
   */
  sanitizeArray<T>(
    input: unknown,
    validator: (item: unknown) => item is T,
    maxLength = 1000
  ): T[] {
    if (!Array.isArray(input)) {
      return [];
    }

    const sanitized = input
      .slice(0, maxLength)
      .filter(validator);

    return sanitized;
  }

  /**
   * Validates and sanitizes an object
   */
  sanitizeObject<T extends Record<string, unknown>>(
    input: unknown,
    schema: Record<string, (value: unknown) => unknown>
  ): Partial<T> | null {
    if (typeof input !== 'object' || input === null || Array.isArray(input)) {
      return null;
    }

    const obj = input as Record<string, unknown>;
    const sanitized: Record<string, unknown> = {};

    for (const [key, validator] of Object.entries(schema)) {
      if (key in obj) {
        try {
          sanitized[key] = validator(obj[key]);
        } catch (error) {
          // Skip invalid fields
        }
      }
    }

    return sanitized as Partial<T>;
  }

  /**
   * Validates URL
   */
  isValidURL(url: string): boolean {
    try {
      const parsed = new URL(url);
      return ['http:', 'https:'].includes(parsed.protocol);
    } catch {
      return false;
    }
  }

  /**
   * Validates email
   */
  isValidEmail(email: string): boolean {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
  }

  /**
   * Generates a secure random token
   */
  generateSecureToken(length = 32): string {
    const array = new Uint8Array(length);
    crypto.getRandomValues(array);
    return Array.from(array, (byte) => byte.toString(16).padStart(2, '0')).join('');
  }

  /**
   * Validates file type
   */
  isValidFileType(file: File, allowedTypes: string[]): boolean {
    return allowedTypes.some((type) => {
      if (type.endsWith('/*')) {
        return file.type.startsWith(type.slice(0, -2));
      }
      return file.type === type;
    });
  }

  /**
   * Validates file size
   */
  isValidFileSize(file: File, maxSize: number): boolean {
    return file.size <= maxSize;
  }
}

/**
 * Global security manager instance
 */
export const securityManager = new SecurityManager();



