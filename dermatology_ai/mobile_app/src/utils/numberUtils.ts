/**
 * Number utilities
 */

/**
 * Format number with commas
 */
export const formatNumber = (num: number, decimals: number = 0): string => {
  return num.toLocaleString('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
};

/**
 * Format currency
 */
export const formatCurrency = (
  amount: number,
  currency: string = 'USD',
  locale: string = 'en-US'
): string => {
  return new Intl.NumberFormat(locale, {
    style: 'currency',
    currency,
  }).format(amount);
};

/**
 * Format percentage
 */
export const formatPercentage = (value: number, decimals: number = 0): string => {
  return `${value.toFixed(decimals)}%`;
};

/**
 * Clamp number between min and max
 */
export const clamp = (value: number, min: number, max: number): number => {
  return Math.min(Math.max(value, min), max);
};

/**
 * Round to decimal places
 */
export const round = (value: number, decimals: number = 0): number => {
  const factor = Math.pow(10, decimals);
  return Math.round(value * factor) / factor;
};

/**
 * Check if number is in range
 */
export const isInRange = (
  value: number,
  min: number,
  max: number,
  inclusive: boolean = true
): boolean => {
  if (inclusive) {
    return value >= min && value <= max;
  }
  return value > min && value < max;
};

/**
 * Generate random number
 */
export const random = (min: number, max: number): number => {
  return Math.floor(Math.random() * (max - min + 1)) + min;
};

/**
 * Parse number safely
 */
export const parseNumber = (value: string | number, defaultValue: number = 0): number => {
  if (typeof value === 'number') return value;
  const parsed = parseFloat(value);
  return isNaN(parsed) ? defaultValue : parsed;
};

/**
 * Format file size
 */
export const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${round(bytes / Math.pow(k, i), 2)} ${sizes[i]}`;
};

/**
 * Format duration (seconds to readable)
 */
export const formatDuration = (seconds: number): string => {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = seconds % 60;

  if (hours > 0) {
    return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  }
  return `${minutes}:${secs.toString().padStart(2, '0')}`;
};

