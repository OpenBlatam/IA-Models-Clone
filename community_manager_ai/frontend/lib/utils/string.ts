/**
 * String Utility Functions
 * Utility functions for string manipulation
 */

/**
 * Capitalizes the first letter of a string
 * @param str - String to capitalize
 * @returns Capitalized string
 */
export const capitalize = (str: string): string => {
  if (!str) return str;
  return str.charAt(0).toUpperCase() + str.slice(1).toLowerCase();
};

/**
 * Converts a string to camelCase
 * @param str - String to convert
 * @returns CamelCase string
 */
export const camelCase = (str: string): string => {
  return str
    .replace(/(?:^\w|[A-Z]|\b\w)/g, (word, index) => {
      return index === 0 ? word.toLowerCase() : word.toUpperCase();
    })
    .replace(/\s+/g, '');
};

/**
 * Converts a string to kebab-case
 * @param str - String to convert
 * @returns Kebab-case string
 */
export const kebabCase = (str: string): string => {
  return str
    .replace(/([a-z])([A-Z])/g, '$1-$2')
    .replace(/[\s_]+/g, '-')
    .toLowerCase();
};

/**
 * Converts a string to snake_case
 * @param str - String to convert
 * @returns Snake_case string
 */
export const snakeCase = (str: string): string => {
  return str
    .replace(/([a-z])([A-Z])/g, '$1_$2')
    .replace(/[\s-]+/g, '_')
    .toLowerCase();
};

/**
 * Converts a string to PascalCase
 * @param str - String to convert
 * @returns PascalCase string
 */
export const pascalCase = (str: string): string => {
  return str
    .replace(/(?:^\w|[A-Z]|\b\w)/g, (word) => {
      return word.toUpperCase();
    })
    .replace(/\s+/g, '');
};

/**
 * Removes HTML tags from a string
 * @param str - String with HTML
 * @returns String without HTML tags
 */
export const stripHtml = (str: string): string => {
  if (typeof window === 'undefined') {
    // Server-side: simple regex approach
    return str.replace(/<[^>]*>/g, '');
  }
  // Client-side: use DOM parser for better accuracy
  const tmp = document.createElement('DIV');
  tmp.innerHTML = str;
  return tmp.textContent || tmp.innerText || '';
};

/**
 * Escapes HTML special characters
 * @param str - String to escape
 * @returns Escaped string
 */
export const escapeHtml = (str: string): string => {
  const map: Record<string, string> = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#039;',
  };
  return str.replace(/[&<>"']/g, (m) => map[m]);
};

/**
 * Generates a random string
 * @param length - Length of the string
 * @param chars - Character set to use (default: alphanumeric)
 * @returns Random string
 */
export const randomString = (
  length: number,
  chars: string = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
): string => {
  let result = '';
  for (let i = 0; i < length; i++) {
    result += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return result;
};

/**
 * Checks if a string is a valid email
 * @param email - Email to validate
 * @returns True if email is valid
 */
export const isValidEmail = (email: string): boolean => {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
};

/**
 * Checks if a string is a valid URL
 * @param url - URL to validate
 * @returns True if URL is valid
 */
export const isValidUrl = (url: string): boolean => {
  try {
    new URL(url);
    return true;
  } catch {
    return false;
  }
};

/**
 * Truncates a string to a maximum length
 * @param str - String to truncate
 * @param maxLength - Maximum length
 * @param suffix - Suffix to append (default: '...')
 * @returns Truncated string
 */
export const truncate = (str: string, maxLength: number, suffix: string = '...'): string => {
  if (str.length <= maxLength) return str;
  return str.slice(0, maxLength - suffix.length) + suffix;
};

/**
 * Removes whitespace from both ends and collapses multiple spaces
 * @param str - String to clean
 * @returns Cleaned string
 */
export const cleanWhitespace = (str: string): string => {
  return str.trim().replace(/\s+/g, ' ');
};


