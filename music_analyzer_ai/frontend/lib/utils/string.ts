/**
 * String utility functions.
 * Provides helper functions for common string operations.
 */

/**
 * Converts a string to camelCase.
 * @param str - String to convert
 * @returns CamelCase string
 */
export function toCamelCase(str: string): string {
  if (typeof str !== 'string') {
    return '';
  }

  return str
    .replace(/(?:^\w|[A-Z]|\b\w)/g, (word, index) => {
      return index === 0 ? word.toLowerCase() : word.toUpperCase();
    })
    .replace(/\s+/g, '');
}

/**
 * Converts a string to kebab-case.
 * @param str - String to convert
 * @returns Kebab-case string
 */
export function toKebabCase(str: string): string {
  if (typeof str !== 'string') {
    return '';
  }

  return str
    .replace(/([a-z])([A-Z])/g, '$1-$2')
    .replace(/[\s_]+/g, '-')
    .toLowerCase();
}

/**
 * Converts a string to snake_case.
 * @param str - String to convert
 * @returns Snake_case string
 */
export function toSnakeCase(str: string): string {
  if (typeof str !== 'string') {
    return '';
  }

  return str
    .replace(/([a-z])([A-Z])/g, '$1_$2')
    .replace(/[\s-]+/g, '_')
    .toLowerCase();
}

/**
 * Converts a string to PascalCase.
 * @param str - String to convert
 * @returns PascalCase string
 */
export function toPascalCase(str: string): string {
  if (typeof str !== 'string') {
    return '';
  }

  return str
    .replace(/(?:^\w|[A-Z]|\b\w)/g, (word) => {
      return word.toUpperCase();
    })
    .replace(/\s+/g, '');
}

/**
 * Converts the first letter of a string to uppercase.
 * @param str - String to capitalize
 * @returns Capitalized string
 */
export function capitalizeFirst(str: string): string {
  if (typeof str !== 'string' || str.length === 0) {
    return str;
  }

  return str.charAt(0).toUpperCase() + str.slice(1);
}

/**
 * Removes whitespace from both ends of a string.
 * @param str - String to trim
 * @returns Trimmed string
 */
export function trim(str: string): string {
  if (typeof str !== 'string') {
    return '';
  }

  return str.trim();
}

/**
 * Removes all whitespace from a string.
 * @param str - String to remove whitespace from
 * @returns String without whitespace
 */
export function removeWhitespace(str: string): string {
  if (typeof str !== 'string') {
    return '';
  }

  return str.replace(/\s+/g, '');
}

/**
 * Generates a random string of specified length.
 * @param length - Length of the string
 * @param charset - Character set to use (default: alphanumeric)
 * @returns Random string
 */
export function randomString(
  length: number,
  charset: string = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
): string {
  let result = '';
  for (let i = 0; i < length; i++) {
    result += charset.charAt(Math.floor(Math.random() * charset.length));
  }
  return result;
}

/**
 * Checks if a string starts with a given substring.
 * @param str - String to check
 * @param searchString - Substring to search for
 * @param position - Position to start search (default: 0)
 * @returns True if string starts with substring
 */
export function startsWith(
  str: string,
  searchString: string,
  position: number = 0
): boolean {
  if (typeof str !== 'string' || typeof searchString !== 'string') {
    return false;
  }

  return str.startsWith(searchString, position);
}

/**
 * Checks if a string ends with a given substring.
 * @param str - String to check
 * @param searchString - Substring to search for
 * @param length - Length to search up to (default: str.length)
 * @returns True if string ends with substring
 */
export function endsWith(
  str: string,
  searchString: string,
  length?: number
): boolean {
  if (typeof str !== 'string' || typeof searchString !== 'string') {
    return false;
  }

  return str.endsWith(searchString, length);
}

/**
 * Replaces all occurrences of a substring in a string.
 * @param str - String to replace in
 * @param searchValue - Substring to replace
 * @param replaceValue - Replacement value
 * @returns String with replacements
 */
export function replaceAll(
  str: string,
  searchValue: string,
  replaceValue: string
): string {
  if (typeof str !== 'string') {
    return '';
  }

  return str.split(searchValue).join(replaceValue);
}

