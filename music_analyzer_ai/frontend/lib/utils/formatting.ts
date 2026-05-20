/**
 * Formatting utility functions.
 * Provides consistent formatting for dates, numbers, and text.
 */

/**
 * Formats a number with thousand separators.
 * @param num - Number to format
 * @param locale - Locale for formatting (default: 'es-ES')
 * @returns Formatted number string
 */
export function formatNumber(
  num: number,
  locale: string = 'es-ES'
): string {
  if (typeof num !== 'number' || isNaN(num)) {
    return '0';
  }

  return new Intl.NumberFormat(locale).format(num);
}

/**
 * Formats a date to a readable string.
 * @param date - Date to format
 * @param locale - Locale for formatting (default: 'es-ES')
 * @param options - Intl.DateTimeFormatOptions
 * @returns Formatted date string
 */
export function formatDate(
  date: Date | string | number,
  locale: string = 'es-ES',
  options?: Intl.DateTimeFormatOptions
): string {
  const dateObj = typeof date === 'string' || typeof date === 'number'
    ? new Date(date)
    : date;

  if (isNaN(dateObj.getTime())) {
    return 'Fecha inválida';
  }

  const defaultOptions: Intl.DateTimeFormatOptions = {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    ...options,
  };

  return new Intl.DateTimeFormat(locale, defaultOptions).format(dateObj);
}

/**
 * Formats a relative time (e.g., "2 hours ago").
 * @param date - Date to format
 * @param locale - Locale for formatting (default: 'es-ES')
 * @returns Formatted relative time string
 */
export function formatRelativeTime(
  date: Date | string | number,
  locale: string = 'es-ES'
): string {
  const dateObj = typeof date === 'string' || typeof date === 'number'
    ? new Date(date)
    : date;

  if (isNaN(dateObj.getTime())) {
    return 'Fecha inválida';
  }

  const rtf = new Intl.RelativeTimeFormat(locale, { numeric: 'auto' });
  const now = new Date();
  const diffInSeconds = Math.floor((dateObj.getTime() - now.getTime()) / 1000);

  const intervals = [
    { unit: 'year' as const, seconds: 31536000 },
    { unit: 'month' as const, seconds: 2592000 },
    { unit: 'week' as const, seconds: 604800 },
    { unit: 'day' as const, seconds: 86400 },
    { unit: 'hour' as const, seconds: 3600 },
    { unit: 'minute' as const, seconds: 60 },
    { unit: 'second' as const, seconds: 1 },
  ];

  for (const { unit, seconds } of intervals) {
    const interval = Math.floor(Math.abs(diffInSeconds) / seconds);
    if (interval >= 1) {
      return rtf.format(
        diffInSeconds < 0 ? -interval : interval,
        unit
      );
    }
  }

  return rtf.format(0, 'second');
}

/**
 * Formats file size in bytes to human-readable format.
 * @param bytes - Size in bytes
 * @param decimals - Number of decimal places (default: 2)
 * @returns Formatted file size string
 */
export function formatFileSize(bytes: number, decimals: number = 2): string {
  if (bytes === 0) return '0 Bytes';

  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'];

  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`;
}

/**
 * Truncates text to a maximum length with ellipsis.
 * @param text - Text to truncate
 * @param maxLength - Maximum length
 * @param suffix - Suffix to add (default: '...')
 * @returns Truncated text
 */
export function truncateText(
  text: string,
  maxLength: number,
  suffix: string = '...'
): string {
  if (typeof text !== 'string') {
    return '';
  }

  if (text.length <= maxLength) {
    return text;
  }

  return text.slice(0, maxLength - suffix.length) + suffix;
}

/**
 * Capitalizes the first letter of a string.
 * @param text - Text to capitalize
 * @returns Capitalized text
 */
export function capitalize(text: string): string {
  if (typeof text !== 'string' || text.length === 0) {
    return text;
  }

  return text.charAt(0).toUpperCase() + text.slice(1).toLowerCase();
}

/**
 * Formats a percentage value.
 * @param value - Value between 0 and 1
 * @param decimals - Number of decimal places (default: 0)
 * @returns Formatted percentage string
 */
export function formatPercent(value: number, decimals: number = 0): string {
  if (typeof value !== 'number' || isNaN(value)) {
    return '0%';
  }

  const clamped = Math.max(0, Math.min(1, value));
  const percent = (clamped * 100).toFixed(decimals);
  return `${percent}%`;
}

