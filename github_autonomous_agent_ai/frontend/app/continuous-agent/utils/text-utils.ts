/**
 * Text manipulation utilities
 */

/**
 * Truncates text to a maximum length with ellipsis
 * @param text - Text to truncate
 * @param maxLength - Maximum length before truncation
 * @param ellipsis - Ellipsis string (default: "...")
 * @returns Truncated text
 */
export const truncateText = (
  text: string,
  maxLength: number,
  ellipsis = "..."
): string => {
  if (text.length <= maxLength) {
    return text;
  }
  return text.substring(0, maxLength) + ellipsis;
};

/**
 * Checks if text should be truncated
 * @param text - Text to check
 * @param maxLength - Maximum length
 * @returns Whether text should be truncated
 */
export const shouldTruncate = (text: string, maxLength: number): boolean => {
  return text.length > maxLength;
};

/**
 * Formats character count with locale
 * @param count - Character count
 * @returns Formatted count string
 */
export const formatCharacterCount = (count: number): string => {
  return count.toLocaleString();
};



