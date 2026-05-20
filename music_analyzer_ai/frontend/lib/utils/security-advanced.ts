/**
 * Advanced security utilities.
 * Provides comprehensive security functions for the application.
 */

/**
 * Options for XSS sanitization.
 */
export interface XssSanitizeOptions {
  /**
   * Allowed HTML tags.
   */
  allowedTags?: string[];
  /**
   * Allowed HTML attributes.
   */
  allowedAttributes?: string[];
  /**
   * Whether to strip all HTML tags.
   */
  stripTags?: boolean;
}

/**
 * Sanitizes a string to prevent XSS attacks.
 * Removes potentially dangerous HTML/JavaScript.
 */
export function sanitizeXss(
  input: string,
  options: XssSanitizeOptions = {}
): string {
  const { stripTags = true } = options;

  if (stripTags) {
    // Remove all HTML tags
    return input
      .replace(/<[^>]*>/g, '')
      .replace(/&lt;/g, '<')
      .replace(/&gt;/g, '>')
      .replace(/&amp;/g, '&')
      .replace(/&quot;/g, '"')
      .replace(/&#x27;/g, "'")
      .replace(/&#x2F;/g, '/');
  }

  // Basic sanitization - remove script tags and event handlers
  let sanitized = input
    .replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '')
    .replace(/on\w+\s*=\s*["'][^"']*["']/gi, '')
    .replace(/javascript:/gi, '')
    .replace(/data:text\/html/gi, '');

  // Remove dangerous attributes
  sanitized = sanitized.replace(
    /\s*(on\w+|href|src|action|formaction)\s*=\s*["'][^"']*["']/gi,
    ''
  );

  return sanitized;
}

/**
 * Validates and sanitizes a URL to prevent open redirect attacks.
 */
export function sanitizeUrl(url: string, allowedDomains?: string[]): string | null {
  try {
    const urlObj = new URL(url);

    // Check protocol
    if (!['http:', 'https:'].includes(urlObj.protocol)) {
      return null;
    }

    // Check allowed domains
    if (allowedDomains && allowedDomains.length > 0) {
      const hostname = urlObj.hostname;
      const isAllowed = allowedDomains.some((domain) => {
        return hostname === domain || hostname.endsWith(`.${domain}`);
      });

      if (!isAllowed) {
        return null;
      }
    }

    // Remove dangerous protocols and data URIs
    if (urlObj.protocol === 'javascript:' || urlObj.protocol === 'data:') {
      return null;
    }

    return urlObj.toString();
  } catch {
    return null;
  }
}

/**
 * Generates a secure random token.
 */
export function generateSecureToken(length: number = 32): string {
  if (typeof window === 'undefined' || !window.crypto) {
    // Fallback for environments without crypto API
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    let token = '';
    for (let i = 0; i < length; i++) {
      token += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    return token;
  }

  const array = new Uint8Array(length);
  window.crypto.getRandomValues(array);
  return Array.from(array, (byte) => byte.toString(16).padStart(2, '0')).join('');
}

/**
 * Hashes a string using Web Crypto API.
 */
export async function hashString(
  input: string,
  algorithm: 'SHA-256' | 'SHA-384' | 'SHA-512' = 'SHA-256'
): Promise<string> {
  if (typeof window === 'undefined' || !window.crypto) {
    throw new Error('Crypto API not available');
  }

  const encoder = new TextEncoder();
  const data = encoder.encode(input);
  const hashBuffer = await window.crypto.subtle.digest(algorithm, data);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map((b) => b.toString(16).padStart(2, '0')).join('');
}

/**
 * Validates a Content Security Policy header.
 */
export function validateCSP(csp: string): {
  valid: boolean;
  errors: string[];
} {
  const errors: string[] = [];
  const directives = csp.split(';').map((d) => d.trim());

  // Required directives
  const requiredDirectives = ['default-src', 'script-src', 'style-src'];
  const presentDirectives = directives.map((d) => d.split(' ')[0]);

  requiredDirectives.forEach((required) => {
    if (!presentDirectives.includes(required)) {
      errors.push(`Missing required directive: ${required}`);
    }
  });

  // Check for unsafe-inline in script-src
  directives.forEach((directive) => {
    if (directive.startsWith('script-src') && directive.includes("'unsafe-inline'")) {
      errors.push("'unsafe-inline' in script-src is a security risk");
    }
  });

  return {
    valid: errors.length === 0,
    errors,
  };
}

/**
 * Checks if a string contains potentially dangerous content.
 */
export function containsDangerousContent(input: string): boolean {
  const dangerousPatterns = [
    /<script/i,
    /javascript:/i,
    /on\w+\s*=/i,
    /data:text\/html/i,
    /vbscript:/i,
    /expression\(/i,
    /@import/i,
  ];

  return dangerousPatterns.some((pattern) => pattern.test(input));
}

/**
 * Validates a file type based on extension and MIME type.
 */
export function validateFileType(
  filename: string,
  mimeType: string | null,
  allowedTypes: string[]
): boolean {
  const extension = filename.split('.').pop()?.toLowerCase();

  if (!extension) {
    return false;
  }

  // Check extension
  if (!allowedTypes.includes(extension)) {
    return false;
  }

  // Check MIME type if provided
  if (mimeType) {
    const allowedMimeTypes = allowedTypes.map((ext) => {
      const mimeMap: Record<string, string> = {
        jpg: 'image/jpeg',
        jpeg: 'image/jpeg',
        png: 'image/png',
        gif: 'image/gif',
        webp: 'image/webp',
        pdf: 'application/pdf',
        json: 'application/json',
        txt: 'text/plain',
        csv: 'text/csv',
      };
      return mimeMap[ext] || `application/${ext}`;
    });

    if (!allowedMimeTypes.includes(mimeType)) {
      return false;
    }
  }

  return true;
}

/**
 * Validates file size.
 */
export function validateFileSize(
  size: number,
  maxSize: number,
  unit: 'B' | 'KB' | 'MB' | 'GB' = 'MB'
): boolean {
  const multipliers: Record<string, number> = {
    B: 1,
    KB: 1024,
    MB: 1024 * 1024,
    GB: 1024 * 1024 * 1024,
  };

  const maxSizeBytes = maxSize * multipliers[unit];
  return size <= maxSizeBytes;
}

/**
 * Creates a secure file download.
 */
export function createSecureDownload(
  data: Blob | string,
  filename: string,
  mimeType?: string
): void {
  if (typeof window === 'undefined') {
    return;
  }

  const blob = typeof data === 'string' ? new Blob([data], { type: mimeType }) : data;
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = sanitizeXss(filename, { stripTags: true });
  link.style.display = 'none';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

/**
 * Rate limiting utility for client-side operations.
 */
export class ClientRateLimiter {
  private requests: Map<string, number[]> = new Map();
  private windowMs: number;
  private maxRequests: number;

  constructor(windowMs: number = 60000, maxRequests: number = 10) {
    this.windowMs = windowMs;
    this.maxRequests = maxRequests;
  }

  /**
   * Checks if a request is allowed.
   */
  isAllowed(key: string): boolean {
    const now = Date.now();
    const requests = this.requests.get(key) || [];

    // Remove old requests outside the window
    const validRequests = requests.filter((time) => now - time < this.windowMs);

    if (validRequests.length >= this.maxRequests) {
      return false;
    }

    // Add current request
    validRequests.push(now);
    this.requests.set(key, validRequests);

    return true;
  }

  /**
   * Resets the rate limit for a key.
   */
  reset(key: string): void {
    this.requests.delete(key);
  }

  /**
   * Gets remaining requests for a key.
   */
  getRemaining(key: string): number {
    const now = Date.now();
    const requests = this.requests.get(key) || [];
    const validRequests = requests.filter((time) => now - time < this.windowMs);
    return Math.max(0, this.maxRequests - validRequests.length);
  }
}




