/**
 * Input sanitization utilities to prevent XSS attacks
 * Follows OWASP guidelines for input sanitization
 */

/**
 * Sanitize string input by removing potentially dangerous characters
 */
export function sanitizeString(input: string): string {
  if (typeof input !== 'string') {
    return '';
  }

  return input
    .replace(/[<>]/g, '') // Remove angle brackets
    .replace(/javascript:/gi, '') // Remove javascript: protocol
    .replace(/on\w+=/gi, '') // Remove event handlers
    .trim();
}

/**
 * Sanitize URL to prevent XSS and protocol injection
 */
export function sanitizeUrl(url: string): string {
  if (typeof url !== 'string') {
    return '';
  }

  const trimmed = url.trim();

  // Only allow http, https, and mailto protocols
  const allowedProtocols = /^(https?|mailto):/i;
  if (!allowedProtocols.test(trimmed)) {
    return '';
  }

  return trimmed;
}

/**
 * Sanitize search query
 */
export function sanitizeSearchQuery(query: string): string {
  if (typeof query !== 'string') {
    return '';
  }

  return sanitizeString(query)
    .replace(/[^\w\s-]/g, '') // Only allow alphanumeric, spaces, and hyphens
    .substring(0, 100); // Limit length
}

/**
 * Validate and sanitize email
 */
export function sanitizeEmail(email: string): string {
  if (typeof email !== 'string') {
    return '';
  }

  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  const trimmed = email.trim().toLowerCase();

  if (!emailRegex.test(trimmed)) {
    return '';
  }

  return trimmed;
}

/**
 * Sanitize numeric input
 */
export function sanitizeNumber(input: unknown): number | null {
  if (typeof input === 'number' && !Number.isNaN(input)) {
    return input;
  }

  if (typeof input === 'string') {
    const parsed = parseFloat(input);
    if (!Number.isNaN(parsed)) {
      return parsed;
    }
  }

  return null;
}

/**
 * Sanitize object recursively
 */
export function sanitizeObject<T extends Record<string, unknown>>(
  obj: T
): Partial<T> {
  const sanitized: Partial<T> = {};

  for (const [key, value] of Object.entries(obj)) {
    if (typeof value === 'string') {
      sanitized[key as keyof T] = sanitizeString(value) as T[keyof T];
    } else if (typeof value === 'object' && value !== null) {
      sanitized[key as keyof T] = sanitizeObject(
        value as Record<string, unknown>
      ) as T[keyof T];
    } else {
      sanitized[key as keyof T] = value;
    }
  }

  return sanitized;
}

