/**
 * Utility Functions
 * Shared utility functions used throughout the application
 */

import { type ClassValue, clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';
import { PLATFORMS, POST_STATUS } from '@/lib/config/constants';

/**
 * Merges class names with Tailwind CSS conflict resolution
 * @param inputs - Class values to merge
 * @returns Merged class string
 */
export const cn = (...inputs: ClassValue[]): string => {
  return twMerge(clsx(inputs));
};

/**
 * Formats a date to a full date-time string
 * @param date - Date string or Date object
 * @param locale - Optional locale (default: 'es-ES')
 * @returns Formatted date string
 */
export const formatDate = (date: string | Date, locale: string = 'es-ES'): string => {
  const d = typeof date === 'string' ? new Date(date) : date;
  
  if (isNaN(d.getTime())) {
    return 'Invalid date';
  }
  
  return d.toLocaleDateString(locale, {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
};

/**
 * Formats a date to a short date string
 * @param date - Date string or Date object
 * @param locale - Optional locale (default: 'es-ES')
 * @returns Formatted short date string
 */
export const formatDateShort = (date: string | Date, locale: string = 'es-ES'): string => {
  const d = typeof date === 'string' ? new Date(date) : date;
  
  if (isNaN(d.getTime())) {
    return 'Invalid date';
  }
  
  return d.toLocaleDateString(locale, {
    month: 'short',
    day: 'numeric',
  });
};

/**
 * Formats a date to a relative time string (e.g., "2 hours ago")
 * @param date - Date string or Date object
 * @returns Relative time string
 */
export const formatRelativeTime = (date: string | Date): string => {
  const d = typeof date === 'string' ? new Date(date) : date;
  const now = new Date();
  const diffInSeconds = Math.floor((now.getTime() - d.getTime()) / 1000);

  if (diffInSeconds < 60) return 'just now';
  if (diffInSeconds < 3600) return `${Math.floor(diffInSeconds / 60)} minutes ago`;
  if (diffInSeconds < 86400) return `${Math.floor(diffInSeconds / 3600)} hours ago`;
  if (diffInSeconds < 604800) return `${Math.floor(diffInSeconds / 86400)} days ago`;
  
  return formatDateShort(d);
};

/**
 * Gets the platform icon emoji
 * @param platform - Platform name
 * @returns Platform icon emoji
 */
export const getPlatformIcon = (platform: string): string => {
  const icons: Record<string, string> = {
    [PLATFORMS.FACEBOOK]: '📘',
    [PLATFORMS.INSTAGRAM]: '📷',
    [PLATFORMS.TWITTER]: '🐦',
    [PLATFORMS.LINKEDIN]: '💼',
    [PLATFORMS.TIKTOK]: '🎵',
    [PLATFORMS.YOUTUBE]: '📺',
  };
  return icons[platform.toLowerCase()] || '🔗';
};

/**
 * Gets the platform color class
 * @param platform - Platform name
 * @returns Tailwind CSS color class
 */
export const getPlatformColor = (platform: string): string => {
  const colors: Record<string, string> = {
    [PLATFORMS.FACEBOOK]: 'bg-blue-600',
    [PLATFORMS.INSTAGRAM]: 'bg-gradient-to-r from-purple-500 to-pink-500',
    [PLATFORMS.TWITTER]: 'bg-blue-400',
    [PLATFORMS.LINKEDIN]: 'bg-blue-700',
    [PLATFORMS.TIKTOK]: 'bg-black',
    [PLATFORMS.YOUTUBE]: 'bg-red-600',
  };
  return colors[platform.toLowerCase()] || 'bg-gray-600';
};

/**
 * Gets the status color class
 * @param status - Post status
 * @returns Tailwind CSS color class
 */
export const getStatusColor = (status: string): string => {
  const colors: Record<string, string> = {
    [POST_STATUS.SCHEDULED]: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200',
    [POST_STATUS.PUBLISHED]: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200',
    [POST_STATUS.CANCELLED]: 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200',
  };
  return colors[status] || 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-200';
};

/**
 * Truncates text to a maximum length
 * @param text - Text to truncate
 * @param maxLength - Maximum length
 * @param suffix - Suffix to append (default: '...')
 * @returns Truncated text
 */
export const truncate = (text: string, maxLength: number, suffix: string = '...'): string => {
  if (text.length <= maxLength) return text;
  return text.slice(0, maxLength - suffix.length) + suffix;
};

/**
 * Debounce function
 * @param func - Function to debounce
 * @param wait - Wait time in milliseconds
 * @returns Debounced function
 */
export const debounce = <T extends (...args: unknown[]) => unknown>(
  func: T,
  wait: number
): ((...args: Parameters<T>) => void) => {
  let timeout: NodeJS.Timeout | null = null;
  
  return (...args: Parameters<T>) => {
    if (timeout) clearTimeout(timeout);
    timeout = setTimeout(() => func(...args), wait);
  };
};

/**
 * Validates if a string is a valid URL
 * @param url - URL string to validate
 * @returns True if valid URL
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
 * Formats a number with commas
 * @param num - Number to format
 * @returns Formatted number string
 */
export const formatNumber = (num: number): string => {
  return new Intl.NumberFormat('en-US').format(num);
};


