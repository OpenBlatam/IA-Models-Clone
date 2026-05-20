/**
 * Input sanitization utilities.
 * Provides functions to sanitize user input and prevent XSS attacks.
 */

/**
 * Sanitizes a string by removing potentially dangerous characters.
 * @param input - String to sanitize
 * @returns Sanitized string
 */
export function sanitizeString(input: string): string {
  if (typeof input !== 'string') {
    return '';
  }

  // Remove HTML tags
  return input
    .replace(/<[^>]*>/g, '')
    .replace(/[<>]/g, '')
    .trim();
}

/**
 * Sanitizes a search query.
 * Removes special characters that could be used for injection attacks.
 * @param query - Search query to sanitize
 * @returns Sanitized query
 */
export function sanitizeSearchQuery(query: string): string {
  if (typeof query !== 'string') {
    return '';
  }

  // Remove potentially dangerous characters but keep basic search characters
  return query
    .replace(/[<>{}[\]\\|`]/g, '')
    .replace(/javascript:/gi, '')
    .replace(/on\w+=/gi, '')
    .trim()
    .slice(0, 200); // Limit length
}

/**
 * Validates and sanitizes a URL.
 * @param url - URL to validate and sanitize
 * @returns Sanitized URL or null if invalid
 */
export function sanitizeUrl(url: string): string | null {
  if (typeof url !== 'string') {
    return null;
  }

  try {
    const parsedUrl = new URL(url);
    
    // Only allow http and https protocols
    if (!['http:', 'https:'].includes(parsedUrl.protocol)) {
      return null;
    }

    return parsedUrl.toString();
  } catch {
    return null;
  }
}

/**
 * Escapes HTML special characters.
 * @param text - Text to escape
 * @returns Escaped text
 */
export function escapeHtml(text: string): string {
  if (typeof text !== 'string') {
    return '';
  }

  const map: Record<string, string> = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#039;',
  };

  return text.replace(/[&<>"']/g, (char) => map[char] || char);
}

/**
 * Validates email format.
 * @param email - Email to validate
 * @returns True if email is valid
 */
export function isValidEmail(email: string): boolean {
  if (typeof email !== 'string') {
    return false;
  }

  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
}

/**
 * Validates URL format.
 * @param url - URL to validate
 * @returns True if URL is valid
 */
export function isValidUrl(url: string): boolean {
  if (typeof url !== 'string') {
    return false;
  }

  try {
    new URL(url);
    return true;
  } catch {
    return false;
  }
}

