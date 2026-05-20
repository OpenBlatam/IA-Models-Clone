/**
 * Formatter utilities - consolidated formatting functions
 */

import { formatNumber, formatCurrency, formatDate, formatDuration, formatFileSize, formatPercentage, formatPosition } from './format';

// Re-export all formatters for convenience
export {
  formatNumber,
  formatCurrency,
  formatDate,
  formatDuration,
  formatFileSize,
  formatPercentage,
  formatPosition,
};

/**
 * Format bytes to human readable
 */
export function formatBytes(bytes: number, decimals: number = 2): string {
  return formatFileSize(bytes);
}

/**
 * Format timestamp to relative time
 */
export function formatRelativeTime(date: Date | string): string {
  const dateObj = typeof date === 'string' ? new Date(date) : date;
  const now = new Date();
  const diff = now.getTime() - dateObj.getTime();
  const seconds = Math.floor(diff / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);

  if (days > 0) return `hace ${days} día${days > 1 ? 's' : ''}`;
  if (hours > 0) return `hace ${hours} hora${hours > 1 ? 's' : ''}`;
  if (minutes > 0) return `hace ${minutes} minuto${minutes > 1 ? 's' : ''}`;
  return 'hace unos segundos';
}

/**
 * Format number with unit
 */
export function formatWithUnit(value: number, unit: string, decimals: number = 2): string {
  return `${formatNumber(value, decimals)} ${unit}`;
}

/**
 * Format range
 */
export function formatRange(min: number, max: number, unit?: string): string {
  const formatted = `${formatNumber(min)} - ${formatNumber(max)}`;
  return unit ? `${formatted} ${unit}` : formatted;
}



