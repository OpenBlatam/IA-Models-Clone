/**
 * Utility functions for the application.
 * Provides common helper functions for formatting, styling, and debouncing.
 */

import { type ClassValue, clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

/**
 * Merges class names using clsx and tailwind-merge.
 * Useful for conditionally applying Tailwind CSS classes.
 * @param inputs - Class values to merge
 * @returns Merged class string
 */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

/**
 * Formats duration in milliseconds to MM:SS format.
 * @param ms - Duration in milliseconds
 * @returns Formatted duration string (e.g., "3:45")
 */
export function formatDuration(ms: number): string {
  if (ms < 0) {
    return '0:00';
  }
  const seconds = Math.floor(ms / 1000);
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
}

/**
 * Formats BPM (beats per minute) value.
 * @param bpm - BPM value
 * @returns Formatted BPM string (e.g., "120 BPM")
 */
export function formatBPM(bpm: number): string {
  if (bpm < 0) {
    return '0 BPM';
  }
  return `${Math.round(bpm)} BPM`;
}

/**
 * Formats a decimal value (0-1) as a percentage.
 * @param value - Decimal value between 0 and 1
 * @returns Formatted percentage string (e.g., "75%")
 * @deprecated Use formatPercent from './utils/formatting' instead
 */
export function formatPercentage(value: number): string {
  const clamped = Math.max(0, Math.min(1, value));
  return `${Math.round(clamped * 100)}%`;
}

/**
 * Creates a debounced version of a function.
 * The debounced function will delay its execution until after wait milliseconds
 * have elapsed since the last time it was invoked.
 * @param func - Function to debounce
 * @param wait - Number of milliseconds to delay
 * @returns Debounced function
 */
export function debounce<T extends (...args: unknown[]) => unknown>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout | null = null;
  return function executedFunction(...args: Parameters<T>) {
    const later = () => {
      timeout = null;
      func(...args);
    };
    if (timeout) clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}

