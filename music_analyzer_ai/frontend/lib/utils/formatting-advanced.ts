/**
 * Advanced formatting utility functions.
 * Provides enhanced formatting for various data types.
 */

/**
 * Formats a number with currency symbol.
 * @param value - Number to format
 * @param currency - Currency code (default: 'USD')
 * @param locale - Locale string (default: 'en-US')
 * @returns Formatted currency string
 */
export function formatCurrency(
  value: number,
  currency: string = 'USD',
  locale: string = 'en-US'
): string {
  return new Intl.NumberFormat(locale, {
    style: 'currency',
    currency,
  }).format(value);
}

/**
 * Formats a number as percentage.
 * @param value - Number to format (0-1 or 0-100)
 * @param decimals - Number of decimals (default: 0)
 * @returns Formatted percentage string
 */
export function formatPercent(
  value: number,
  decimals: number = 0
): string {
  const percent = value > 1 ? value : value * 100;
  return `${percent.toFixed(decimals)}%`;
}

/**
 * Formats a number with thousand separators.
 * @param value - Number to format
 * @param locale - Locale string (default: 'en-US')
 * @returns Formatted number string
 */
export function formatNumber(
  value: number,
  locale: string = 'en-US'
): string {
  return new Intl.NumberFormat(locale).format(value);
}

/**
 * Formats a date range.
 * @param startDate - Start date
 * @param endDate - End date
 * @param locale - Locale string (default: 'en-US')
 * @returns Formatted date range string
 */
export function formatDateRange(
  startDate: Date | string | number,
  endDate: Date | string | number,
  locale: string = 'en-US'
): string {
  const start = new Date(startDate);
  const end = new Date(endDate);

  const startFormatted = new Intl.DateTimeFormat(locale, {
    month: 'short',
    day: 'numeric',
  }).format(start);

  const endFormatted = new Intl.DateTimeFormat(locale, {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  }).format(end);

  return `${startFormatted} - ${endFormatted}`;
}

/**
 * Formats a phone number.
 * @param phone - Phone number string
 * @param format - Format type (default: 'US')
 * @returns Formatted phone number
 */
export function formatPhone(
  phone: string,
  format: 'US' | 'international' = 'US'
): string {
  const cleaned = phone.replace(/\D/g, '');

  if (format === 'US' && cleaned.length === 10) {
    return `(${cleaned.slice(0, 3)}) ${cleaned.slice(3, 6)}-${cleaned.slice(6)}`;
  }

  if (format === 'international' && cleaned.length >= 10) {
    return `+${cleaned.slice(0, cleaned.length - 10)} ${cleaned.slice(-10).replace(/(\d{2})(\d{4})(\d{4})/, '$1 $2 $3')}`;
  }

  return phone;
}

/**
 * Formats a credit card number (masked).
 * @param cardNumber - Card number string
 * @param showLast - Number of last digits to show (default: 4)
 * @returns Masked card number
 */
export function formatCardNumber(
  cardNumber: string,
  showLast: number = 4
): string {
  const cleaned = cardNumber.replace(/\D/g, '');
  const last = cleaned.slice(-showLast);
  const masked = '*'.repeat(Math.max(0, cleaned.length - showLast));
  return `${masked}${last}`;
}

/**
 * Formats initials from a name.
 * @param name - Full name string
 * @param maxInitials - Maximum number of initials (default: 2)
 * @returns Initials string
 */
export function formatInitials(
  name: string,
  maxInitials: number = 2
): string {
  const parts = name.trim().split(/\s+/);
  if (parts.length === 0) return '';

  if (parts.length === 1) {
    return parts[0].charAt(0).toUpperCase();
  }

  const initials = parts
    .slice(0, maxInitials)
    .map((part) => part.charAt(0).toUpperCase())
    .join('');

  return initials;
}

/**
 * Formats a slug from a string.
 * @param text - Text to slugify
 * @returns Slug string
 */
export function formatSlug(text: string): string {
  return text
    .toLowerCase()
    .trim()
    .replace(/[^\w\s-]/g, '')
    .replace(/[\s_-]+/g, '-')
    .replace(/^-+|-+$/g, '');
}

/**
 * Formats a file size with unit.
 * @param bytes - Size in bytes
 * @param decimals - Number of decimals (default: 2)
 * @returns Formatted file size
 */
export function formatFileSize(
  bytes: number,
  decimals: number = 2
): string {
  if (bytes === 0) return '0 Bytes';

  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB'];

  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`;
}

